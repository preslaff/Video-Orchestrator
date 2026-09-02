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
from collections.abc import Callable

from .orchestrator import Orchestrator
from .state import Job, JobStatus, Store

log = logging.getLogger("vorch.queue")


class Broker:
    """In-memory per-job pub/sub over asyncio.Queues (one queue per subscriber)."""

    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.setdefault(job_id, set()).add(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        subs = self._subs.get(job_id)
        if subs:
            subs.discard(q)
            if not subs:
                self._subs.pop(job_id, None)

    def publish(self, job_id: str, event: dict) -> None:
        """Fan an event out to every current subscriber (called on the loop thread)."""
        for q in list(self._subs.get(job_id, ())):
            q.put_nowait(event)


class JobManager:
    """Submit/approve jobs; run them serialized in a thread, streaming events."""

    def __init__(
        self,
        store: Store,
        *,
        orchestrator_factory: Callable[[Store, Job], Orchestrator] | None = None,
    ):
        self.store = store
        self.broker = Broker()
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
        max_attempts: int = 2,
        evaluate: bool = True,
        face_swap: bool = True,
    ) -> Job:
        """Create + persist a job (validates the master face when swapping) and run it."""
        orch = Orchestrator(
            self.store, n_scenes=n_scenes, seed=seed, max_attempts=max_attempts,
            evaluate=evaluate, face_swap=face_swap,
        )
        job = orch.create_job(idea, style=style, master_face=master_face, autonomy=autonomy)
        self._active.add(job.id)
        self._spawn(self._execute(job, lambda o, j: o.run(j)))
        return job

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

    def events(self, job_id: str) -> asyncio.Queue:
        """Subscribe to a job's event stream (drain until a {'type': 'end'} event)."""
        return self.broker.subscribe(job_id)

    def is_active(self, job_id: str) -> bool:
        """True while a run segment (run/approve) for this job is in flight."""
        return job_id in self._active

    def update_scenes(self, job_id: str, edits: list[dict]) -> Job:
        """Apply prompt/description edits to a paused job's scenes and persist them.

        Each edit is ``{"index": int, "description"?, "keyframe_prompt"?,
        "animation_prompt"?}``. Only allowed while the job is **not** running
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
            for field in ("description", "keyframe_prompt", "animation_prompt"):
                if edit.get(field) is not None:
                    setattr(scene, field, edit[field])
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
            except Exception as e:  # surface failure as state + event, never crash the loop
                log.exception("job %s failed", job.id)
                job.status = JobStatus.FAILED
                self.store.save_job(job)
                self.broker.publish(job.id, {"job_id": job.id, "type": "error", "error": str(e)})
            finally:
                self._active.discard(job.id)
                self.broker.publish(job.id, {"job_id": job.id, "type": "end"})

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task
