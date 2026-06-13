"""Job-level phase machine — drives an idea to finished clips (DESIGN.md §8).

Where ``pipeline.py`` runs *one* scene's chain, the Orchestrator runs a whole
**Job**: storyboard -> per-scene prompts -> keyframes (+eval/retry) -> videos,
persisting every step to a ``state.Store`` and logging each verdict/decision. It
is **deterministic** (no autonomous LLM planning — DESIGN.md §11) and **resumable**:
phases are entered based on the persisted scene state, so a reloaded job continues
where it stopped.

Semi-autonomous gates (DESIGN.md §8): in ``autonomy="semi"`` the run pauses after
the storyboard and again before the expensive Wan phase, setting the job to
``NEEDS_APPROVAL``. The caller (CLI/UI) inspects, then ``approve(job, gate)`` to
continue. ``autonomy="full"`` runs straight through. An approval is itself logged
as a decision, which is how a resumed ``run`` knows a gate was already cleared.

Ordering note: like the skeleton, identity is imposed by swapping the *keyframe*
then animating (Klein -> ReActor -> Wan), since the ReActor template is image-based
(DESIGN.md §9). Continuity (``run_sequence``) remains a separate, sequential path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from . import config, evaluator, pipeline, storyboard
from .clients import comfy
from .state import Decision, Job, JobStatus, Scene, SceneStatus, Store, new_job

log = logging.getLogger("vorch.orchestrator")

GATE_STORYBOARD = "storyboard"
GATE_KEYFRAMES = "keyframes"


class Orchestrator:
    """Drives Jobs through the phased pipeline, persisting to a Store."""

    def __init__(
        self,
        store: Store,
        *,
        n_scenes: int | None = None,
        seed: int = config.SEED_RANDOM,
        max_attempts: int = 2,
        evaluate: bool = True,
        face_swap: bool = True,
    ):
        self.store = store
        self.n_scenes = n_scenes
        self.seed = seed
        self.max_attempts = max_attempts
        self.evaluate = evaluate
        self.face_swap = face_swap
        # Progress sink for live UIs (set by queue.py). No-op by default so the
        # CLI path is unaffected. Each event is a small JSON-able dict.
        self.emit: Callable[[dict], None] = lambda event: None

    def _event(self, job: Job, type: str, **data) -> None:
        self.emit({"job_id": job.id, "type": type, **data})

    @classmethod
    def from_job(cls, store: Store, job: Job) -> "Orchestrator":
        """Rebuild an Orchestrator from a job's persisted run-config (for resume)."""
        return cls(
            store,
            n_scenes=job.n_scenes,
            seed=job.seed,
            max_attempts=job.max_attempts,
            evaluate=job.evaluate,
            face_swap=job.face_swap,
        )

    # -- job lifecycle --

    def create_job(
        self,
        idea: str,
        *,
        style: str = "",
        master_face: str | Path = "",
        autonomy: str = "semi",
    ) -> Job:
        """Create and persist a fresh Job.

        The run-config (n_scenes/seed/max_attempts/evaluate/face_swap) is stored
        on the job so a later ``approve``/resume reconstructs it exactly (see
        ``from_job``). The master face is required and validated only when
        ``face_swap`` is on; otherwise it may be empty.
        """
        # Master face only matters (and is only validated) when swapping.
        master_str = ""
        if self.face_swap:
            master = Path(master_face).expanduser()
            if not master.is_file():
                raise FileNotFoundError(f"master face not found: {master}")
            master_str = str(master)
        elif master_face:
            master_str = str(Path(master_face).expanduser())
        job = new_job(
            idea=idea,
            style=style,
            master_face=master_str,
            autonomy=autonomy,
            n_scenes=self.n_scenes,
            seed=self.seed,
            max_attempts=self.max_attempts,
            evaluate=self.evaluate,
            face_swap=self.face_swap,
        )
        return self.store.save_job(job)

    def run(self, job: Job) -> Job:
        """Advance the job as far as autonomy allows; pause at unapproved gates."""
        job.status = JobStatus.RUNNING
        self.store.save_job(job)
        self._event(job, "status", status=job.status.value)

        if not job.scenes:
            self._phase_storyboard(job)
        if self._gate(job, GATE_STORYBOARD):
            return job

        if self._needs_keyframes(job):
            self._phase_keyframes(job)
        if self._gate(job, GATE_KEYFRAMES):
            return job

        if self._needs_videos(job):
            self._phase_videos(job)

        job.status = JobStatus.DONE
        self.store.save_job(job)
        log.info("job %s complete (%d scenes)", job.id, len(job.scenes))
        self._event(job, "status", status=job.status.value)
        return job

    def approve(self, job: Job, gate: str) -> Job:
        """Clear a semi-auto gate (logged as a decision) and continue the run."""
        self.store.log_decision(
            Decision(job_id=job.id, scene_index=None, phase=gate, attempt=1, action="approved")
        )
        log.info("job %s: gate %r approved", job.id, gate)
        return self.run(job)

    # -- phases --

    def _phase_storyboard(self, job: Job) -> None:
        log.info("job %s: storyboard (idea=%r)", job.id, job.idea)
        briefs = storyboard.generate_storyboard(job.idea, n_scenes=self.n_scenes)
        scenes: list[Scene] = []
        for i, brief in enumerate(briefs):
            kp = storyboard.keyframe_prompt(brief, style=job.style)
            ap = storyboard.animation_prompt(brief)
            scenes.append(
                Scene(
                    index=i,
                    description=brief.description,
                    character=brief.character,
                    setting=brief.setting,
                    mood=brief.mood,
                    keyframe_prompt=kp.positive,
                    animation_prompt=ap.positive,
                )
            )
        job.scenes = scenes
        self.store.log_decision(
            Decision(
                job_id=job.id, scene_index=None, phase="storyboard", attempt=1,
                action="generated", reason=f"{len(scenes)} scenes",
            )
        )
        self.store.save_job(job)
        self._event(job, "storyboard", scenes=len(scenes))

    def _phase_keyframes(self, job: Job) -> None:
        for scene in job.scenes:
            if scene.keyframe_path and scene.status is not SceneStatus.PENDING:
                continue  # resume: already done
            log.info("job %s: keyframe scene %d", job.id, scene.index)
            if self.evaluate:
                def log_attempt(attempt, _path, verdict, _idx=scene.index):
                    # Every attempt -> decision log (DESIGN §8), not just the last.
                    self.store.record_verdict(job.id, _idx, "keyframe", verdict, attempt=attempt)
                    self._flag_verdict(job, _idx, verdict, attempt=attempt)
                    self._event(job, "verdict", index=_idx, attempt=attempt,
                                decision=verdict.decision, score=verdict.score)

                path, verdict = pipeline.generate_keyframe_evaluated(
                    scene.keyframe_prompt, scene.keyframe_prompt,
                    seed=self.seed, max_attempts=self.max_attempts, on_attempt=log_attempt,
                )
                scene.keyframe_verdict = verdict
            else:
                path = pipeline.keyframe(scene.keyframe_prompt, seed=self.seed)
                self.store.log_decision(
                    Decision(job_id=job.id, scene_index=scene.index, phase="keyframe",
                             attempt=1, action="generated")
                )
            scene.keyframe_path = str(path)
            scene.status = SceneStatus.KEYFRAME
            self.store.save_job(job)  # persist per scene — resumable
            self._event(job, "keyframe", index=scene.index, path=scene.keyframe_path)

    def _flag_verdict(self, job: Job, scene_index: int, verdict, *, attempt: int = 1) -> None:
        """Log advisory flags alongside a recorded verdict (never blocks)."""
        if verdict.disagreement:
            self.store.log_decision(
                Decision(
                    job_id=job.id, scene_index=scene_index, phase="keyframe", attempt=attempt,
                    action="disagreement",
                    reason=f"gemma said {verdict.verdict!r}; objective gate -> {verdict.decision!r}",
                )
            )
        if verdict.needs_review:
            self.store.log_decision(
                Decision(
                    job_id=job.id, scene_index=scene_index, phase="keyframe", attempt=attempt,
                    action="review", score=verdict.score,
                    reason=f"accepted but score={verdict.score} (< {evaluator.REVIEW_SCORE_THRESHOLD})",
                )
            )

    def _phase_videos(self, job: Job) -> None:
        do_swap = self.face_swap and bool(job.master_face)
        master_name = comfy.upload_image(job.master_face) if do_swap else None
        for scene in job.scenes:
            if scene.video_path:
                continue  # resume: already done
            if not scene.keyframe_path:
                log.warning("job %s: scene %d has no keyframe; skipping", job.id, scene.index)
                continue
            if do_swap:
                log.info("job %s: video scene %d (swap -> animate)", job.id, scene.index)
                swapped = pipeline.swap_master_onto(scene.keyframe_path, master_name)
                scene.swapped_path = str(swapped)
                scene.status = SceneStatus.SWAP
                self.store.save_job(job)
                frame = swapped
            else:
                log.info("job %s: video scene %d (animate keyframe, no swap)", job.id, scene.index)
                frame = scene.keyframe_path

            video = pipeline.animate_frame(frame, scene.animation_prompt, seed=self.seed)
            scene.video_path = str(video)
            scene.status = SceneStatus.COMPLETE
            self.store.log_decision(
                Decision(job_id=job.id, scene_index=scene.index, phase="video",
                         attempt=1, action="generated")
            )
            self.store.save_job(job)
            self._event(job, "video", index=scene.index, path=scene.video_path)

    # -- gates & phase predicates --

    def _gate(self, job: Job, name: str) -> bool:
        """Return True (and pause the job) if this gate blocks progress."""
        if job.autonomy == "full" or self._approved(job, name):
            return False
        job.status = JobStatus.NEEDS_APPROVAL
        self.store.save_job(job)
        log.info("job %s: paused at gate %r (awaiting approval)", job.id, name)
        self._event(job, "gate", gate=name, status=job.status.value)
        return True

    def _approved(self, job: Job, name: str) -> bool:
        return any(
            d.phase == name and d.action == "approved"
            for d in self.store.decisions(job.id)
        )

    @staticmethod
    def _needs_keyframes(job: Job) -> bool:
        return any(s.keyframe_path is None for s in job.scenes)

    @staticmethod
    def _needs_videos(job: Job) -> bool:
        return any(s.video_path is None for s in job.scenes)
