"""Async job manager — runs the (blocking) Orchestrator off the event loop.

The Orchestrator is synchronous and long-running (Gemma + ComfyUI take minutes),
so the UI/server can't call it directly. ``JobManager`` runs each job in a thread
executor, serialized by a single lock — only one generation at a time, because the
one GPU can't run two ComfyUI jobs at once (NOTES.md §4). Orchestrator progress
events are bridged onto a per-job ``Broker`` for the SSE stream (server.py).

GPU-free and framework-free: this is plain asyncio. The web layer is server.py.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable

from . import export as export_mod
from . import projects as projects_mod
from .orchestrator import Orchestrator
from .state import Job, JobStatus, Store
from .submission import resolve_submission

log = logging.getLogger("vorch.queue")


class Broker:
    """In-memory pub/sub over asyncio.Queues (one queue per subscriber).

    Every published event gets a per-job monotonic ``id`` and is kept in a
    small per-job ring buffer, so a subscriber that (re)connects can ask for
    everything ``after`` an id it has already seen (SSE ``Last-Event-ID``).
    Global subscribers (``subscribe_all``) see every job's events.
    """

    def __init__(self, history: int = 200) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = {}
        self._all: set[asyncio.Queue] = set()
        self._seq: dict[str, int] = {}
        self._history: dict[str, deque[dict]] = {}
        self._n = history

    def subscribe(self, job_id: str, *, after: int | None = None) -> asyncio.Queue:
        """Subscribe to one job; ``after=N`` first replays buffered events with id > N."""
        q: asyncio.Queue = asyncio.Queue()
        self._subs.setdefault(job_id, set()).add(q)
        if after is not None:
            for ev in self._history.get(job_id, ()):
                if ev["id"] > after:
                    q.put_nowait(ev)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        subs = self._subs.get(job_id)
        if subs:
            subs.discard(q)
            if not subs:
                self._subs.pop(job_id, None)

    def subscribe_all(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._all.add(q)
        return q

    def unsubscribe_all(self, q: asyncio.Queue) -> None:
        self._all.discard(q)

    def last_id(self, job_id: str) -> int:
        return self._seq.get(job_id, 0)

    def publish(self, job_id: str, event: dict) -> None:
        """Stamp, buffer and fan an event out (called on the loop thread)."""
        seq = self._seq.get(job_id, 0) + 1
        self._seq[job_id] = seq
        ev = {**event, "job_id": job_id, "id": seq}
        self._history.setdefault(job_id, deque(maxlen=self._n)).append(ev)
        for q in list(self._subs.get(job_id, ())):
            q.put_nowait(ev)
        for q in list(self._all):
            q.put_nowait(ev)


class JobManager:
    """Submit/approve jobs; run them serialized in a thread, streaming events."""

    def __init__(
        self,
        store: Store,
        *,
        orchestrator_factory: Callable[[Store, Job], Orchestrator] | None = None,
        env_check: Callable[..., object] | None = None,
        auto_export: bool = True,
    ):
        """``env_check(profile, project)`` (e.g. ``envcheck.gate``) runs before every
        submit unless the caller forces; ``auto_export`` materialises the handoff
        folder when a job reaches DONE."""
        self.store = store
        self.broker = Broker()
        self._env_check = env_check
        self._auto_export = auto_export
        self._lock = asyncio.Lock()  # one generation at a time (single GPU)
        self._tasks: set[asyncio.Task] = set()
        self._active: set[str] = set()  # jobs with a run segment in flight
        # How to rebuild an Orchestrator for a job (run-config lives on the job).
        self._factory = orchestrator_factory or Orchestrator.from_job

    # -- public API --

    async def submit(
        self,
        idea: str,
        *,
        style: str = "",
        master_face: str = "",
        autonomy: str = "semi",
        n_scenes: int | None = None,
        seed: int = 0,
        max_attempts: int | None = None,
        evaluate: bool | None = None,
        face_swap: bool | None = None,
        project_id: str | None = None,
        profile_id: str | None = None,
        profile_overrides: dict | None = None,
        mode: str = "t2i",
        reference_asset_ids: list[str] | None = None,
        reference_images: list[str] | None = None,
        force: bool = False,
    ) -> Job:
        """Resolve profile/project/assets, gate on the environment, create + run the job.

        Raises KeyError (unknown project/profile/asset), FileNotFoundError (master
        face / reference missing), ValueError (bad overrides or mode), and
        ``envcheck.EnvironmentNotReady`` when a check fails and ``force`` is off.
        """
        sub = resolve_submission(
            self.store, project_id=project_id, profile_id=profile_id,
            profile_overrides=profile_overrides, master_face=master_face,
            reference_asset_ids=reference_asset_ids or (), reference_images=reference_images or (),
            evaluate=evaluate, face_swap=face_swap, max_attempts=max_attempts,
        )
        if self._env_check is not None and not force:
            self._env_check(sub.profile, sub.project)
        orch = Orchestrator(self.store, profile=sub.profile, n_scenes=n_scenes, seed=seed)
        job = orch.create_job(
            idea, style=style, master_face=sub.master_face, autonomy=autonomy, mode=mode,
            reference_images=sub.reference_images, project_id=sub.project_id,
        )
        self._active.add(job.id)
        self._spawn(self._execute(job, lambda o, j: o.run(j)))
        return job

    def export_now(self, job_id: str, *, force: bool = False) -> export_mod.ExportRecord:
        """Materialise the handoff folder for a finished job (synchronous file I/O)."""
        if job_id in self._active:
            raise RuntimeError(f"job {job_id} is running")
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status is not JobStatus.DONE and not force:
            raise RuntimeError(f"job is {job.status.value}, not done (use force to export anyway)")
        project = projects_mod.load(self.store, job.project_id) if job.project_id else None
        rec = export_mod.export_job(self.store, job, project)
        job.export = rec.to_dict()
        self.store.save_job(job)
        self.broker.publish(job.id, {"type": "export", "path": rec.path, "files": rec.files, "ok": rec.ok})
        return rec

    async def approve(self, job_id: str, gate: str) -> Job:
        """Clear a gate and resume the job in the background; returns its current state.

        Idempotent while a run segment is in flight: a duplicate approve (double
        click, repeated request) for an already-running job is a no-op, so the
        pipeline can't be driven twice in parallel.
        """
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job_id in self._active:
            log.info("job %s already running; ignoring duplicate approve", job_id)
            return job
        self._active.add(job_id)
        self._spawn(self._execute(job, lambda o, j: o.approve(j, gate)))
        return job

    async def regenerate(self, job_id: str, index: int, *, what: str = "keyframe",
                         edits: dict | None = None) -> Job:
        """Redo one scene's keyframe (or clip) as a new run segment; the job pauses at
        the keyframes gate again for review (video jobs, semi autonomy)."""
        if job_id in self._active:
            raise RuntimeError(f"job {job_id} is running; wait for the current segment")
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status is JobStatus.RUNNING:
            raise RuntimeError("job is running")
        if not any(s.index == index for s in job.scenes):
            raise KeyError(f"scene {index}")
        self._active.add(job.id)
        self._spawn(self._execute(job, lambda o, j: o.regenerate(j, index, what=what, edits=edits)))
        return job

    def events(self, job_id: str, *, after: int | None = None) -> asyncio.Queue:
        """Subscribe to a job's event stream (drain until a {'type': 'end'} event)."""
        return self.broker.subscribe(job_id, after=after)

    def is_active(self, job_id: str) -> bool:
        """True while a run segment (run/approve) for this job is in flight."""
        return job_id in self._active

    def update_scenes(self, job_id: str, edits: list[dict]) -> Job:
        """Apply prompt/description edits to a paused job's scenes and persist them.

        Each edit is ``{"index": int, "description"?, "keyframe_prompt"?,
        "animation_prompt"?, "keyframe_negative"?, "animation_negative"?,
        "mode"?, "references"?}``. Only allowed while the job is **not** running
        (a run segment in flight would race the orchestrator's own writes).
        """
        if job_id in self._active:
            raise RuntimeError(f"job {job_id} is running; pause before editing")
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        by_index = {s.index: s for s in job.scenes}
        for edit in edits:
            scene = by_index.get(edit.get("index"))
            if scene is None:
                continue
            for field in ("description", "keyframe_prompt", "animation_prompt",
                          "keyframe_negative", "animation_negative", "mode", "references"):
                if edit.get(field) is not None:
                    setattr(scene, field, edit[field])
            chosen = edit.get("chosen_variant")
            if chosen is not None:
                if not 0 <= int(chosen) < len(scene.variants):
                    raise ValueError(f"scene {scene.index}: no variant {chosen}")
                v = scene.variants[int(chosen)]
                scene.keyframe_path, scene.keyframe_verdict = v.path, v.verdict
                scene.swapped_path = v.swapped_path
        return self.store.save_job(job)

    # -- internals --

    async def _execute(self, job: Job, fn: Callable[[Orchestrator, Job], None]) -> None:
        """Run a blocking orchestrator call in an executor, under the GPU lock."""
        loop = asyncio.get_running_loop()
        orch = self._factory(self.store, job)
        orch.emit = lambda event, jid=job.id: loop.call_soon_threadsafe(
            self.broker.publish, jid, event
        )
        async with self._lock:
            try:
                await loop.run_in_executor(None, fn, orch, job)
                if self._auto_export:
                    await self._maybe_export(loop, job)
            except Exception as e:  # surface failure as state + event, never crash the loop
                log.exception("job %s failed", job.id)
                job.status = JobStatus.FAILED
                self.store.save_job(job)
                self.broker.publish(job.id, {"job_id": job.id, "type": "error", "error": str(e)})
            finally:
                self._active.discard(job.id)
                self.broker.publish(job.id, {"job_id": job.id, "type": "end"})

    async def _maybe_export(self, loop, job: Job) -> None:
        """After a run segment: export once when the job just reached DONE."""
        fresh = self.store.load_job(job.id)
        if fresh is None or fresh.status is not JobStatus.DONE or fresh.export:
            return
        project = projects_mod.load(self.store, fresh.project_id) if fresh.project_id else None
        try:
            rec = await loop.run_in_executor(None, export_mod.export_job, self.store, fresh, project)
        except Exception as e:  # noqa: BLE001 - export must never fail a finished job
            log.warning("job %s: export failed: %s", job.id, e)
            return
        fresh.export = rec.to_dict()
        self.store.save_job(fresh)
        job.export = fresh.export
        self.broker.publish(job.id, {"type": "export", "path": rec.path, "files": rec.files, "ok": rec.ok})

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task
