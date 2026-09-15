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
import threading
from collections import deque
from collections.abc import Callable

from . import assembly as assembly_mod
from . import cancel
from . import export as export_mod
from . import projects as projects_mod
from . import scene_assets
from .orchestrator import Orchestrator
from .state import Decision, Job, JobStatus, Store
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
        # Stop signal per run segment (assembly also marks a job active but is not stoppable).
        self._stop: dict[str, threading.Event] = {}
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
        prev = self._start(job)
        self._spawn(self._execute(job, lambda o, j: o.run(j), prev))
        return job

    def _start(self, job: Job) -> JobStatus:
        """Mark a segment as queued: status RUNNING is persisted *before* the worker
        starts so a client that refetches right after the request already sees
        the job as active (and keeps/reopens its event stream).

        Returns the status the job had, so a segment stopped before it got the GPU
        lock can put the job back exactly as it was.
        """
        prev = job.status
        job.status = JobStatus.RUNNING
        self.store.save_job(job)
        self._active.add(job.id)
        self._stop[job.id] = threading.Event()
        self.broker.publish(job.id, {"type": "status", "status": job.status.value})
        return prev

    def stop(self, job_id: str) -> Job:
        """Ask a running segment to stop at its next safe point (idempotent).

        The worker thread checks the signal between persisted units of work and in
        the ComfyUI poll loop, where it also withdraws the prompt from ComfyUI. The job
        ends CANCELLED ("stopped") and can be edited and resumed with ``retry``.
        Raises RuntimeError when nothing stoppable is running (e.g. an assembly).
        """
        ev = self._stop.get(job_id)
        if ev is None:
            raise RuntimeError(f"job {job_id} has no run segment to stop")
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if not ev.is_set():
            ev.set()
            log.info("job %s: stop requested", job_id)
            self.broker.publish(job_id, {"job_id": job_id, "type": "stopping"})
        return job

    def is_stopping(self, job_id: str) -> bool:
        """True between a stop request and the segment actually ending."""
        ev = self._stop.get(job_id)
        return ev is not None and ev.is_set()

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
        prev = self._start(job)
        self._spawn(self._execute(job, lambda o, j: o.approve(j, gate), prev))
        return job

    async def retry(self, job_id: str) -> Job:
        """Resume a job from its persisted state; finished work is kept.

        Accepts a **failed** job, a **stopped** one (CANCELLED), and an **orphaned**
        one — persisted as RUNNING but with no segment in this process, which is what a
        server restart mid-run leaves behind. There is deliberately no startup sweep
        that marks such jobs stopped: ``vorch run``/``vorch approve`` drive jobs
        in-process against the same database, and a sweep would flag those live runs.
        No-op while a segment is in flight here.
        """
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job_id in self._active:
            log.info("job %s already running; ignoring duplicate retry", job_id)
            return job
        reasons = {JobStatus.FAILED: "resumed after failure", JobStatus.CANCELLED: "resumed after stop",
                   JobStatus.RUNNING: "resumed an orphaned run (no worker in this server)"}
        if job.status not in reasons:
            raise RuntimeError(f"job is {job.status.value}; only failed, stopped or orphaned jobs resume")
        self.store.log_decision(Decision(job_id=job.id, scene_index=None, phase="job", attempt=1,
                                         action="retry", reason=reasons[job.status]))
        prev = self._start(job)
        self._spawn(self._execute(job, lambda o, j: o.run(j), prev))
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
        prev = self._start(job)
        self._spawn(self._execute(job, lambda o, j: o.regenerate(j, index, what=what, edits=edits), prev))
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
        project = (projects_mod.load(self.store, job.project_id)
                   if job.project_id and any(scene_assets.has_asset_edits(e) for e in edits) else None)
        for edit in edits:
            scene = by_index.get(edit.get("index"))
            if scene is None:
                continue
            for field in ("description", "keyframe_prompt", "animation_prompt",
                          "keyframe_negative", "animation_negative", "mode", "references"):
                if edit.get(field) is not None:
                    setattr(scene, field, edit[field])
            if scene_assets.has_asset_edits(edit):
                scene_assets.apply_asset_edits(scene, edit, project)  # ValueError -> 400
            chosen = edit.get("chosen_variant")
            if chosen is not None:
                if not 0 <= int(chosen) < len(scene.variants):
                    raise ValueError(f"scene {scene.index}: no variant {chosen}")
                v = scene.variants[int(chosen)]
                scene.keyframe_path, scene.keyframe_verdict = v.path, v.verdict
                scene.swapped_path = v.swapped_path
        return self.store.save_job(job)

    # -- internals --

    async def _execute(self, job: Job, fn: Callable[[Orchestrator, Job], None],
                       prev: JobStatus = JobStatus.PENDING) -> None:
        """Run a blocking orchestrator call in an executor, under the GPU lock.

        The call runs with this segment's stop event bound (``cancel.call_bound``), so
        every path — run, approve, regenerate — honours Stop without extra plumbing.
        The job stays in ``_active`` until the executor returns: the worker thread
        cannot be killed, and edits must not race its saves.
        """
        loop = asyncio.get_running_loop()
        ev = self._stop.setdefault(job.id, threading.Event())
        async with self._lock:
            try:
                if ev.is_set():
                    # Stopped while waiting for the GPU: nothing ran, put the job back as it
                    # was (a new job has no earlier state, so it becomes stopped).
                    job.status = (JobStatus.CANCELLED if prev in (JobStatus.PENDING, JobStatus.RUNNING)
                                  else prev)
                    self.store.save_job(job)
                    self.store.log_decision(Decision(job_id=job.id, scene_index=None, phase="job", attempt=1,
                                                     action="stopped", reason="stopped before it started"))
                    self.broker.publish(job.id, {"job_id": job.id, "type": "status", "status": job.status.value})
                    return
                orch = self._factory(self.store, job)
                orch.emit = lambda event, jid=job.id: loop.call_soon_threadsafe(
                    self.broker.publish, jid, event
                )
                await loop.run_in_executor(None, cancel.call_bound, ev, fn, orch, job)
                if self._auto_export:
                    await self._maybe_export(loop, job)
            except cancel.JobStopped:
                log.info("job %s stopped", job.id)
                job.status = JobStatus.CANCELLED
                self.store.save_job(job)
                self.store.log_decision(Decision(job_id=job.id, scene_index=None, phase="job", attempt=1,
                                                 action="stopped", reason="stopped by the user"))
                self.broker.publish(job.id, {"job_id": job.id, "type": "status", "status": job.status.value})
            except Exception as e:  # surface failure as state + event, never crash the loop
                log.exception("job %s failed", job.id)
                job.status = JobStatus.FAILED
                self.store.save_job(job)
                self.broker.publish(job.id, {"job_id": job.id, "type": "error", "error": str(e)})
            finally:
                self._active.discard(job.id)
                self._stop.pop(job.id, None)
                self.broker.publish(job.id, {"job_id": job.id, "type": "end"})

    async def _maybe_export(self, loop, job: Job) -> None:
        """After a run segment: export once when the job just reached DONE."""
        fresh = self.store.load_job(job.id)
        if fresh is None or fresh.status is not JobStatus.DONE or fresh.export:
            return
        await self._reexport(loop, fresh)
        job.export = fresh.export

    async def _reexport(self, loop, job: Job) -> None:
        """(Re)build the handoff folder for ``job`` in the executor; warnings only."""
        project = projects_mod.load(self.store, job.project_id) if job.project_id else None
        try:
            rec = await loop.run_in_executor(None, export_mod.export_job, self.store, job, project)
        except Exception as e:  # noqa: BLE001 - export must never fail a finished job
            log.warning("job %s: export failed: %s", job.id, e)
            return
        job.export = rec.to_dict()
        self.store.save_job(job)
        self.broker.publish(job.id, {"type": "export", "path": rec.path, "files": rec.files, "ok": rec.ok})

    # -- assembly (ffmpeg, no GPU) --

    async def assemble(self, job_id: str, spec: dict | None = None, *, force: bool = False) -> Job:
        """Join selected clips into one video as a background task.

        Refused while a segment is active or (unless ``force``) when the job is not
        done. The job's status does not change; the result (or the error) lands in
        ``job.assembly``. Raises KeyError / RuntimeError / ValueError (bad spec).
        """
        if job_id in self._active:
            raise RuntimeError(f"job {job_id} is running; wait for the current segment")
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status is not JobStatus.DONE and not force:
            raise RuntimeError(f"job is {job.status.value}, not done (use force to assemble anyway)")
        resolved, paths = assembly_mod.resolve(job, spec)
        self._active.add(job.id)
        self._spawn(self._assemble_task(job, resolved, paths))
        return job

    def select_assembly_version(self, job_id: str, version: int) -> Job:
        """Make an earlier assembly version the current one (and re-export)."""
        job = self._assembly_job(job_id)
        assembly_mod.select_version(job, version)
        self.store.save_job(job)
        self._reexport_sync(job)
        return job

    def delete_assembly_version(self, job_id: str, version: int) -> Job:
        job = self._assembly_job(job_id)
        assembly_mod.delete_version(job, version)
        self.store.save_job(job)
        self._reexport_sync(job)
        return job

    def _assembly_job(self, job_id: str) -> Job:
        if job_id in self._active:
            raise RuntimeError(f"job {job_id} is running; wait for the current segment")
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def _reexport_sync(self, job: Job) -> None:
        if not (job.export or job.status is JobStatus.DONE):
            return
        project = projects_mod.load(self.store, job.project_id) if job.project_id else None
        try:
            rec = export_mod.export_job(self.store, job, project)
        except Exception as e:  # noqa: BLE001
            log.warning("job %s: export failed: %s", job.id, e)
            return
        job.export = rec.to_dict()
        self.store.save_job(job)
        self.broker.publish(job.id, {"type": "export", "path": rec.path, "files": rec.files, "ok": rec.ok})

    async def _assemble_task(self, job: Job, spec, paths) -> None:
        loop = asyncio.get_running_loop()
        emit = lambda ev, jid=job.id: loop.call_soon_threadsafe(self.broker.publish, jid, ev)  # noqa: E731
        try:
            rec = await loop.run_in_executor(None, lambda: assembly_mod.run(job, spec, paths, emit=emit))
            fresh = self.store.load_job(job.id) or job
            assembly_mod.register_version(fresh, rec)
            self.store.save_job(fresh)
            self.broker.publish(job.id, {"type": "assembly", "path": rec.path, "duration": rec.duration,
                                         "n_clips": rec.n_clips, "ok": rec.ok, "error": rec.error,
                                         "version": rec.version})
            if rec.ok and (fresh.export or fresh.status is JobStatus.DONE):
                await self._reexport(loop, fresh)
        except Exception as e:  # noqa: BLE001 - never crash the loop; record the failure
            log.exception("job %s: assembly task failed", job.id)
            fresh = self.store.load_job(job.id) or job
            assembly_mod.register_version(fresh, assembly_mod.AssemblyRecord(
                path=str(assembly_mod.output_path(job)), created_at="", ok=False, error=str(e),
                spec=spec.to_dict(), n_clips=len(paths)))
            self.store.save_job(fresh)
            self.broker.publish(job.id, {"type": "error", "error": f"assembly: {e}"})
        finally:
            self._active.discard(job.id)
            self.broker.publish(job.id, {"job_id": job.id, "type": "end"})

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task
