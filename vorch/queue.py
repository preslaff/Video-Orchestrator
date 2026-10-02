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
import contextlib
import copy
import logging
import threading
from collections import deque
from pathlib import Path
from collections.abc import Callable

from . import budget, assembly as assembly_mod
from . import cancel
from . import export as export_mod
from . import projects as projects_mod
from . import scene_assets
from . import scenes as scenes_mod
from . import storyboard
from . import versions as versions_mod
from .orchestrator import GATE_KEYFRAMES, Orchestrator
from .state import Decision, Job, JobStatus, SceneStatus, Store
from .submission import resolve_submission

log = logging.getLogger("vorch.queue")

# How long a cancel waits for an in-flight worker to leave before recording itself anyway. The
# worker owns the job's saves, so cancelling underneath it would simply be overwritten; but a
# wedged ComfyUI call is exactly why cancel exists, so the wait cannot be unbounded.
_CANCEL_WAIT_TICK = 0.25
_CANCEL_WAIT_TICKS = 40          # 10 s


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
        keyframe_asset_ids: list[str] | None = None,
        budget_usd: float | None = None,
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
        animate = list(keyframe_asset_ids or [])
        if animate:
            # These images are the keyframes. A face swap would ReActor a face onto them, and
            # an unknown id must fail here rather than half-way through the run.
            if sub.project is None:
                raise ValueError("animating assets needs a project")
            if orch_face_swap(sub):
                raise ValueError("turn face swap off to animate your own images "
                                 "(it would paste a face onto them)")
            unknown = [a for a in animate if sub.project.asset(a) is None]
            if unknown:
                raise KeyError(unknown[0])
        orch = Orchestrator(self.store, profile=sub.profile, n_scenes=n_scenes, seed=seed)
        job = orch.create_job(
            idea, style=style, master_face=sub.master_face, autonomy=autonomy, mode=mode,
            reference_images=sub.reference_images, keyframe_asset_ids=animate,
            project_id=sub.project_id,
        )
        job.budget_usd = budget_usd
        self.store.save_job(job)
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
        ends STOPPED and can be edited and resumed with ``retry``.
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

    async def cancel(self, job_id: str) -> Job:
        """Finish with a job for good: terminal, and ``retry`` refuses it afterwards.

        Available in any state, which is the point — a job paused at a gate had no exit at all
        before, so half-finished runs accumulated indefinitely. Nothing is deleted: the scenes,
        clips and export folder stay, and the job can still be viewed, assembled and exported.

        On a job whose segment is in flight this first asks it to stop and then waits for the
        worker to actually leave, because that worker owns the job's saves — writing `cancelled`
        underneath it would be overwritten the moment it persisted its own ending. If it will not
        leave (a wedged ComfyUI call is the reason this exists), the cancel is still recorded, so
        the job stops presenting itself as resumable even while the thread winds down.
        """
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status is JobStatus.CANCELLED:
            return job                                  # idempotent
        if job_id in self._active:
            with contextlib.suppress(RuntimeError):     # nothing stoppable: fall through
                self.stop(job_id)
            for _ in range(_CANCEL_WAIT_TICKS):
                if job_id not in self._active:
                    break
                await asyncio.sleep(_CANCEL_WAIT_TICK)
            else:
                log.warning("job %s: still running after a stop; recording the cancel anyway", job_id)
        job = self.store.load_job(job_id) or job        # the worker may have saved on its way out
        job.status = JobStatus.CANCELLED
        self.store.save_job(job)
        self.store.log_decision(Decision(job_id=job.id, scene_index=None, phase="job", attempt=1,
                                         action="cancelled", reason="cancelled by the user"))
        log.info("job %s cancelled", job_id)
        self.broker.publish(job_id, {"job_id": job_id, "type": "status", "status": job.status.value})
        return job

    def reopen(self, job_id: str) -> Job:
        """Undo a cancel, so a mis-click costs nothing. The job becomes stopped, hence resumable."""
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status is not JobStatus.CANCELLED:
            raise RuntimeError(f"job is {job.status.value}, not cancelled")
        job.status = JobStatus.STOPPED
        self.store.save_job(job)
        self.store.log_decision(Decision(job_id=job.id, scene_index=None, phase="job", attempt=1,
                                         action="reopened", reason="cancel undone"))
        self.broker.publish(job_id, {"job_id": job_id, "type": "status", "status": job.status.value})
        return job

    def busy(self) -> bool:
        """True while any job run segment is queued or in flight (not counting assemblies).

        Background work that shares Gemma — the asset describer — waits for this to clear,
        so it never slows a storyboard or the evaluator on the CPU-pinned model.
        """
        return bool(self._stop)

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

        Accepts a **failed** job, a **stopped** one, and an **orphaned**
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
        reasons = {JobStatus.FAILED: "resumed after failure", JobStatus.STOPPED: "resumed after stop",
                   JobStatus.RUNNING: "resumed an orphaned run (no worker in this server)"}
        if job.status is JobStatus.CANCELLED:
            raise RuntimeError("job is cancelled; reopen it first if you want to carry on")
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
        scene = next((s for s in job.scenes if s.index == index), None)
        if scene is None:
            raise KeyError(f"scene {index}")
        self._rehearse_regenerate(job, scene, what, edits or {})  # ValueError -> 400, nothing started
        prev = self._start(job)
        # A clip is always rendered for *this* scene: going through `run()` would pause at the
        # keyframes gate (rendering nothing) or render every other scene missing a clip too.
        segment = ((lambda o, j: o.reanimate(j, index, edits, unfinished=prev)) if what == "video"
                   else (lambda o, j: o.regenerate(j, index, what=what, edits=edits)))
        self._spawn(self._execute(job, segment, prev))
        return job

    def delete_version(self, job_id: str, index: int, version_id: str) -> Job:
        """Remove one keyframe or clip version from a scene's history.

        The file stays on disk (with its provenance sidecar) — only the reference goes.
        Deleting the active version falls back to the newest remaining one. Refused while a
        run segment is in flight, like every other edit.
        """
        if job_id in self._active:
            raise RuntimeError(f"job {job_id} is running; pause before editing")
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        scene = next((s for s in job.scenes if s.index == index), None)
        if scene is None:
            raise KeyError(f"scene {index}")
        versions_mod.adopt_legacy(scene)
        kind = versions_mod.delete_version(scene, version_id, kind=job.kind)  # ValueError -> 400
        job.export = None
        self.store.log_decision(Decision(job_id=job.id, scene_index=index, phase=kind, attempt=1,
                                         action="deleted", reason=f"{kind} version {version_id}"))
        return self.store.save_job(job)

    def _rehearse_regenerate(self, job: Job, scene, what: str, edits: dict) -> None:
        """Apply a regenerate's edits to a copy of the scene and validate the result.

        ``Orchestrator.reopen_scene`` applies them inside the run segment, after RUNNING
        is persisted — any error there would fail the whole job. Rehearsing here turns a
        bad edit into a ValueError before anything starts.
        """
        if what not in ("keyframe", "video"):
            raise ValueError("what must be 'keyframe' or 'video'")
        if what == "video" and job.kind != "video":
            raise ValueError("stills jobs have no clip to regenerate")
        asset_id = edits.get("keyframe_asset_id")
        if asset_id is not None:
            if what != "keyframe":
                raise ValueError("an image as keyframe needs what='keyframe'")
            project = projects_mod.load(self.store, job.project_id) if job.project_id else None
            if project is None:
                raise ValueError("the job has no project: add the image to a project first")
            asset = project.asset(asset_id)
            if asset is None:
                raise ValueError(f"unknown project asset {asset_id!r}")
            if not Path(asset.path).is_file():
                raise ValueError(f"the asset's file is missing: {asset.path}")
        trial = copy.deepcopy(scene)
        for field_name in Orchestrator.SCENE_EDIT_FIELDS:
            if edits.get(field_name) is not None:
                setattr(trial, field_name, edits[field_name])
        if scene_assets.has_asset_edits(edits):
            project = projects_mod.load(self.store, job.project_id) if job.project_id else None
            scene_assets.apply_asset_edits(trial, edits, project)
        if what == "keyframe" and asset_id is None:  # a chosen image is not generated: mode is moot
            scene_assets.validate_scene(trial)

    def events(self, job_id: str, *, after: int | None = None) -> asyncio.Queue:
        """Subscribe to a job's event stream (drain until a {'type': 'end'} event)."""
        return self.broker.subscribe(job_id, after=after)

    def is_active(self, job_id: str) -> bool:
        """True while a run segment (run/approve) for this job is in flight."""
        return job_id in self._active

    def update_scenes(self, job_id: str, edits: list[dict]) -> Job:
        """Apply scene edits to a job that is not running and persist them.

        Each edit is ``{"index": int, "description"?, "keyframe_prompt"?,
        "animation_prompt"?, "keyframe_negative"?, "animation_negative"?,
        "mode"?, "references"?, "reference_asset_ids"?, "identity_asset_id"?,
        "chosen_variant"?}``. Allowed at any stage except while a segment is in flight
        (it would race the orchestrator's own writes). All-or-nothing: an invalid edit
        raises ValueError before anything is saved.

        Edits that change what was already built invalidate it, so a resumed or
        re-approved job never keeps a stale output: a new face drops the swap and the
        clip built on the old one; choosing a different keyframe variant on a video job
        drops its clip and reopens the keyframes gate.
        """
        if job_id in self._active:
            raise RuntimeError(f"job {job_id} is running; pause before editing")
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        by_index = {s.index: s for s in job.scenes}
        for s in job.scenes:
            versions_mod.adopt_legacy(s)  # a job from before versions can still have one selected
        project = (projects_mod.load(self.store, job.project_id)
                   if job.project_id and any(scene_assets.has_asset_edits(e) for e in edits) else None)
        reopen_keyframes = False
        drop_narration: list[int] = []
        for edit in edits:
            scene = by_index.get(edit.get("index"))
            if scene is None:
                continue
            old_identity = (scene.identity, scene.identity_asset_id)
            for field in ("description", "keyframe_prompt", "animation_prompt",
                          "keyframe_negative", "animation_negative", "mode", "references",
                          "edit_workflow", "keyframe_workflow", "edit_base", "narration"):
                if edit.get(field) is not None:
                    setattr(scene, field, edit[field])
            if edit.get("narration") is not None and not scene.narration.strip():
                # No line means no voice. Assembly keys off narration_path, so leaving the wav
                # behind would keep playing a take whose words have been deleted — and on a clip
                # with its own audio, there would be no way back to the original.
                drop_narration.append(scene.index)
            if edit.get("edit_base") is not None:
                scene_assets.resolve_edit_base(scene)
            if scene_assets.has_asset_edits(edit):
                scene_assets.apply_asset_edits(scene, edit, project)  # ValueError -> 400
            if any(edit.get(k) is not None for k in scene_assets.MODE_FIELDS):
                scene_assets.validate_scene(scene)
            if (scene.identity, scene.identity_asset_id) != old_identity:
                scene_assets.invalidate_after_identity_change(scene, job.kind)
                job.export = None
            chosen, chosen_id = edit.get("chosen_variant"), edit.get("chosen_variant_id")
            if chosen is not None or chosen_id is not None:
                v = versions_mod.variant_by(scene, id=chosen_id, index=chosen)
                outcome = versions_mod.select_variant(scene, v, kind=job.kind)
                if outcome != "unchanged":
                    job.export = None
                    # A clip rendered from this keyframe is still in the history: it becomes the
                    # deliverable again, so there is nothing to re-render and no gate to reopen.
                    reopen_keyframes = reopen_keyframes or outcome == "clip_cleared"
            if (clip_id := edit.get("chosen_clip_id")) is not None:
                if versions_mod.select_clip(scene, versions_mod.clip_by(scene, clip_id)):
                    job.export = None
        if drop_narration:
            o = Orchestrator.from_job(self.store, job)
            for index in drop_narration:
                o.clear_narration(job, index, keep_text=True)   # the text is already empty
        if reopen_keyframes and job.autonomy != "full":
            # The new keyframe must be reviewed before its clip is rendered again.
            self.store.log_decision(Decision(job_id=job.id, scene_index=None, phase=GATE_KEYFRAMES,
                                             attempt=1, action="reopened"))
            if job.status is JobStatus.DONE:
                job.status = JobStatus.NEEDS_APPROVAL
        return self.store.save_job(job)

    # -- structural edits: add / delete / reorder scenes --

    def _editable_job(self, job_id: str) -> Job:
        """A job nothing is generating right now — structural edits would race its writes.

        Note what this is *not*: it guards one job's saves against its own worker thread, and says
        nothing about the GPU. Anything that needs the card must take ``self._lock`` as well.
        """
        if job_id in self._active:
            raise RuntimeError(f"job {job_id} is running; pause before editing")
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def _after_structural_change(self, job: Job, *, moved: dict, removed=(), action: str,
                                 scene_id: str | None = None, reason: str = "") -> Job:
        """Renumber, keep a saved assembly pointing at the right clips, log it, save.

        A finished job goes back to the keyframes gate: an inserted scene has no keyframe, and a
        new order is something a person should see before the clips are re-joined.
        """
        scenes_mod.remap_assembly(job, moved, removed)
        job.export = None
        self.store.log_decision(Decision(job_id=job.id, scene_index=None, scene_id=scene_id,
                                         phase="scenes", attempt=1, action=action, reason=reason))
        if job.status is JobStatus.DONE and job.autonomy != "full":
            self.store.log_decision(Decision(job_id=job.id, scene_index=None, phase=GATE_KEYFRAMES,
                                             attempt=1, action="reopened"))
            job.status = JobStatus.NEEDS_APPROVAL
        return self.store.save_job(job)

    def add_scene(self, job_id: str, position: int, brief: dict, *, generate: bool = True) -> Job:
        """Insert a scene and (by default) have Gemma write its prompts in the background.

        The row appears at once — marked ``writing_prompts`` — so the person who typed the
        description sees it land; the model call follows on its own.
        """
        job = self._editable_job(job_id)
        before = {s.id: s.index for s in job.scenes}
        scene = scenes_mod.insert(job, position, writing_prompts=generate, **brief)
        job = self._after_structural_change(
            job, moved=_moved(before, job), action="added", scene_id=scene.id,
            reason=f"scene {scene.index + 1}: {scene.description[:120]}")
        if generate:
            self._spawn(self._write_prompts_task(job.id, scene.id))
        return job

    def delete_scene(self, job_id: str, index: int, *, delete_files: bool = False) -> Job:
        """Remove a scene; its rendered files stay on disk unless ``delete_files``."""
        job = self._editable_job(job_id)
        scene = next((s for s in job.scenes if s.index == index), None)
        if scene is None:
            raise KeyError(f"scene {index}")
        label = f"scene {index + 1}: {scene.description[:120]}"
        before = {s.id: s.index for s in job.scenes}
        removed, was = scenes_mod.remove(job, scene.id, delete_files=delete_files)
        return self._after_structural_change(
            job, moved=_moved(before, job), removed=[was], action="deleted", scene_id=removed.id,
            reason=label + (" (files deleted)" if delete_files else ""))

    async def speak_scene(self, job_id: str, index: int) -> Job:
        """Synthesise one scene's line now — under the GPU lock, off the event loop.

        Both of those are corrections. It used to run inline in an ``async def`` route, which meant
        the whole API and every SSE stream froze for the ~90 s model load; and its only guard was
        ``_editable_job``, which refuses when *this* job is running and says nothing about any
        other — so a Speak could load 11268 MiB on top of another job's Wan render on a 16303 MiB
        card. Taking ``self._lock`` is what actually serialises it against renders; waiting is the
        correct behaviour, and the arbiter frees the card once we hold it.
        """
        job = self._editable_job(job_id)
        loop = asyncio.get_running_loop()
        async with self._lock:
            o = Orchestrator.from_job(self.store, job)
            # KeyError -> 404, ValueError -> 400, TTSError -> 400 (see api/jobs.py)
            await loop.run_in_executor(None, o.speak_scene, job, index)
        return job

    def clear_narration(self, job_id: str, index: int, *, keep_text: bool = True) -> Job:
        """Remove a scene's spoken take, so its clip is heard as it was recorded."""
        job = self._editable_job(job_id)
        o = Orchestrator.from_job(self.store, job)
        o.clear_narration(job, index, keep_text=keep_text)   # KeyError -> 404
        return job

    def set_locked_regions(self, job_id: str, index: int, regions: list[dict]) -> Job:
        """Mark the parts of a scene's frame the animator must not redraw — a brand label,
        a monogram — and put them back on the clip that already exists."""
        job = self._editable_job(job_id)
        o = Orchestrator.from_job(self.store, job)
        o.set_locked_regions(job, index, regions)     # KeyError -> 404, ValueError -> 400
        return job

    def set_budget(self, job_id: str, budget_usd: float | None) -> Job:
        """Raise (or lift) a job's hosted-spend ceiling.

        Not an ``_editable_job``: the whole point is to unstick a job that is paused **on** its
        budget, and approving that gate without raising the ceiling only stops it again at the
        very next call.
        """
        job = self.store.load_job(job_id)
        if job is None:
            raise KeyError(job_id)                 # -> 404
        if budget_usd is not None:
            if budget_usd < 0:
                raise ValueError("budget_usd must be >= 0")
            spent = budget.spent(job).max_usd
            if budget_usd < spent:
                raise ValueError(
                    f"budget_usd ${budget_usd:.2f} is below the ${spent:.2f} already spent"
                )
        job.budget_usd = budget_usd
        self.store.save_job(job)
        log.info("job %s: budget set to %s", job_id,
                 "uncapped" if budget_usd is None else f"${budget_usd:.2f}")
        return job

    def reorder_scenes(self, job_id: str, scene_ids: list[str]) -> Job:
        """Put the scenes in this order (every scene, once) and renumber them."""
        job = self._editable_job(job_id)
        before = {s.id: s.index for s in job.scenes}
        scenes_mod.reorder(job, scene_ids)      # ValueError -> 400
        return self._after_structural_change(job, moved=_moved(before, job), action="reordered",
                                             reason=", ".join(str(s.index + 1) for s in job.scenes)[:200])

    def write_prompts(self, job_id: str, index: int) -> Job:
        """(Re)write one scene's keyframe/animation prompts from its description, with Gemma."""
        job = self._editable_job(job_id)
        scene = next((s for s in job.scenes if s.index == index), None)
        if scene is None:
            raise KeyError(f"scene {index}")
        if not scene.description.strip():
            raise ValueError(f"scene {index + 1} has no description to write prompts from")
        scene.writing_prompts = True
        job = self.store.save_job(job)
        self._spawn(self._write_prompts_task(job.id, scene.id))
        return job

    async def _write_prompts_task(self, job_id: str, scene_id: str) -> None:
        """Gemma writes one scene's two prompts, off the GPU lock.

        Gemma is CPU-pinned (invariant #1), so this follows the assembly task rather than
        ``_execute``: no lock, no RUNNING status, just events. A failure leaves the scene
        editable by hand and says so in the decision log.
        """
        loop = asyncio.get_running_loop()
        self.broker.publish(job_id, {"type": "step", "step": "prompts", "detail": "Gemma writes the scene's prompts"})
        try:
            job = self.store.load_job(job_id)
            if job is None:
                return
            scene = scenes_mod.by_id(job, scene_id)
            # The real orchestrator, not self._factory: this needs the job's frozen profile
            # (format, style, negatives) and touches Gemma only — no GPU, nothing to stub.
            orch = Orchestrator.from_job(self.store, job)
            pair = await loop.run_in_executor(None, lambda: _scene_prompts(orch, job, scene))
            fresh = self.store.load_job(job_id) or job
            target = scenes_mod.by_id(fresh, scene_id)
            target.keyframe_prompt, target.keyframe_negative = pair[0].positive, pair[0].negative
            target.animation_prompt, target.animation_negative = pair[1].positive, pair[1].negative
            target.writing_prompts = False
            self.store.save_job(fresh)
            self.store.log_decision(Decision(job_id=job_id, scene_index=target.index, scene_id=scene_id,
                                             phase="scenes", attempt=1, action="prompts",
                                             reason=target.keyframe_prompt[:200]))
            self.broker.publish(job_id, {"type": "storyboard", "scenes": len(fresh.scenes)})
        except Exception as e:  # noqa: BLE001 - a failed write must not cost the scene
            log.exception("job %s: writing prompts for scene %s failed", job_id, scene_id)
            fresh = self.store.load_job(job_id)
            if fresh is not None:
                try:
                    scenes_mod.by_id(fresh, scene_id).writing_prompts = False
                    self.store.save_job(fresh)
                except KeyError:
                    pass
            self.store.log_decision(Decision(job_id=job_id, scene_index=None, scene_id=scene_id,
                                             phase="scenes", attempt=1, action="warning",
                                             reason=f"writing prompts failed: {e}"[:500]))
            self.broker.publish(job_id, {"type": "error", "error": f"writing prompts: {e}"})
        finally:
            self.broker.publish(job_id, {"job_id": job_id, "type": "end"})

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
                    job.status = (JobStatus.STOPPED if prev in (JobStatus.PENDING, JobStatus.RUNNING)
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
                job.status = JobStatus.STOPPED
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

    def save_cut(self, job_id: str, index: int, *, clip_id: str | None, start: float | None,
                 end: float | None, make_active: bool) -> Job:
        """Keep a timeline's trim as a real take of that scene (and optionally as its clip)."""
        job = self._editable_job(job_id)
        clip = assembly_mod.save_cut(job, index, clip_id=clip_id, start=start, end=end,
                                     make_active=make_active)
        self.store.log_decision(Decision(
            job_id=job.id, scene_index=index, scene_id=None, phase="clip", attempt=1, action="cut",
            reason=f"{round((end or 0) - (start or 0), 2)}s kept as a take"
                   + (" and made this scene's clip" if make_active else ""),
        ))
        self.store.save_job(job)
        if make_active:
            self._reexport_sync(job)
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


def _moved(before: dict[str, int], job: Job) -> dict[int, int]:
    """``{old index: new index}`` for the scenes a structural change shifted."""
    return {old: s.index for s in job.scenes if (old := before.get(s.id)) is not None and old != s.index}


def _scene_prompts(orch: Orchestrator, job: Job, scene):
    """The keyframe and animation prompts for one scene, written the way the storyboard writes them.

    Same context as ``_phase_storyboard``: the job's profile decides format, style and negatives,
    and the project's assets are offered only when the job is in edit mode.
    """
    project = projects_mod.load(orch.store, job.project_id) if job.project_id else None
    asset_list = tuple(
        (a.id, a.kind, a.label or a.name) for a in (project.assets if project else [])
        if a.kind in ("master_face", "reference", "product", "logo")
    )
    context = storyboard.PromptContext.from_params(
        orch.params, assets=asset_list if (orch.face_swap or job.mode == "edit") else (),
        edit_mode=job.mode == "edit",
    )
    brief = storyboard.SceneBrief(scene=scene.index + 1, description=scene.description,
                                  character=scene.character, setting=scene.setting, mood=scene.mood)
    return (storyboard.keyframe_prompt(brief, style=job.style, context=context),
            storyboard.animation_prompt(brief, context=context))


def orch_face_swap(sub) -> bool:
    """Whether this submission would run the face swap (profile default, or the explicit flag)."""
    swap = getattr(sub.profile, "face_swap", None)
    return bool(getattr(swap, "enabled", False))
