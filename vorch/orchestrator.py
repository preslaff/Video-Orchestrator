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

from . import config, pipeline, profiles, provenance, storyboard
from .clients import comfy
from .genparams import DEFAULT, GenParams
from .steps.postprocess import run_chain
from .state import Decision, Job, JobStatus, Scene, SceneStatus, Store, Variant, new_job

log = logging.getLogger("vorch.orchestrator")

GATE_STORYBOARD = "storyboard"
GATE_KEYFRAMES = "keyframes"


def choose_variant(variants: list[Variant]) -> Variant:
    """Best candidate: accepted with the highest score, else highest score, else the last."""
    if not variants:
        raise ValueError("no variants to choose from")
    scored = [v for v in variants if v.verdict is not None]
    accepted = [v for v in scored if v.verdict.decision == "accept"]
    pool = accepted or scored
    if pool:
        return max(pool, key=lambda v: v.verdict.score)
    return variants[-1]


def output_prefix(job: Job, scene_index: int, kind: str, variant: int | None = None) -> str:
    """ComfyUI filename_prefix that groups a job's outputs: output/vorch/<job8>/s01_keyframe."""
    tag = f"s{scene_index:02d}_{kind}"
    if variant is not None:
        tag += f"_v{variant + 1}"
    return f"vorch/{job.id[:8]}/{tag}"


def _seed_used(path, requested: int) -> int | None:
    """The seed that actually produced ``path`` (manifest), else the requested one."""
    m = provenance.read_manifest(path)
    if m and isinstance(m.get("params"), dict) and m["params"].get("seed") is not None:
        return m["params"]["seed"]
    return requested if requested != config.SEED_RANDOM else None


class Orchestrator:
    """Drives Jobs through the phased pipeline, persisting to a Store."""

    def __init__(
        self,
        store: Store,
        *,
        profile: profiles.OutputProfile | None = None,
        params: GenParams | None = None,
        n_scenes: int | None = None,
        seed: int = config.SEED_RANDOM,
        max_attempts: int | None = None,
        evaluate: bool | None = None,
        face_swap: bool | None = None,
    ):
        """``profile`` (or ``params``) carries the generation settings; the explicit
        ``max_attempts`` / ``evaluate`` / ``face_swap`` override it (legacy CLI/tests).
        Without either, DEFAULT reproduces the pre-profile constants."""
        self.store = store
        self.profile = profile
        self.params = params or (GenParams.from_profile(profile) if profile is not None else DEFAULT)
        self.n_scenes = n_scenes
        self.seed = seed
        self.max_attempts = max_attempts if max_attempts is not None else self.params.max_attempts
        self.evaluate = evaluate if evaluate is not None else self.params.evaluate
        self.face_swap = face_swap if face_swap is not None else self.params.face_swap
        # Progress sink for live UIs (set by queue.py). No-op by default so the
        # CLI path is unaffected. Each event is a small JSON-able dict.
        self.emit: Callable[[dict], None] = lambda event: None

    def _event(self, job: Job, type: str, **data) -> None:
        self.emit({"job_id": job.id, "type": type, **data})

    @classmethod
    def from_job(cls, store: Store, job: Job) -> "Orchestrator":
        """Rebuild an Orchestrator from a job's persisted run-config (for resume).

        Jobs carry a frozen profile snapshot; older rows without one get the
        default profile plus their stored evaluate/face_swap/max_attempts.
        """
        if job.profile:
            profile = profiles.OutputProfile.model_validate(job.profile)
        else:
            profile = profiles.legacy_profile_for_job(job)
        return cls(
            store,
            profile=profile,
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
        mode: str = "t2i",
        reference_images: list[str] | None = None,
        project_id: str | None = None,
    ) -> Job:
        """Create and persist a fresh Job.

        The profile snapshot + run-config (n_scenes/seed/max_attempts/evaluate/
        face_swap) are stored on the job so a later ``approve``/resume
        reconstructs it exactly (see ``from_job``). The master face is required
        and validated only when ``face_swap`` is on; otherwise it may be empty.
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
        refs = [str(Path(r).expanduser()) for r in (reference_images or [])]
        if mode == "edit":
            if not refs:
                raise ValueError("mode='edit' needs at least one reference image")
            for r in refs:
                if not Path(r).is_file():
                    raise FileNotFoundError(f"reference image not found: {r}")
        elif mode != "t2i":
            raise ValueError(f"unknown mode {mode!r} (t2i | edit)")
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
            mode=mode,
            reference_images=refs,
            project_id=project_id,
            profile_id=self.profile.id if self.profile is not None else None,
            profile=self.profile.snapshot() if self.profile is not None else None,
            kind=self.params.kind,
        )
        return self.store.save_job(job)

    def run(self, job: Job) -> Job:
        """Advance the job as far as autonomy allows; pause at unapproved gates."""
        token = provenance.set_context(job_id=job.id, profile=self.params.profile_name or None)
        try:
            return self._run(job)
        finally:
            provenance.reset_context(token)

    def _run(self, job: Job) -> Job:
        job.status = JobStatus.RUNNING
        self.store.save_job(job)
        self._event(job, "status", status=job.status.value)

        if not job.scenes:
            self._phase_storyboard(job)
        if self._gate(job, GATE_STORYBOARD):
            return job

        if self._needs_keyframes(job):
            self._phase_keyframes(job)

        if job.kind == "stills":
            # Everything after the keyframes is cheap: no second gate.
            if self._needs_stills_finish(job):
                self._phase_finish_stills(job)
        else:
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
        self._event(job, "phase", phase="storyboard", total=self.n_scenes)
        self._event(job, "step", step="storyboard", detail="Gemma writes the storyboard and prompts")
        context = storyboard.PromptContext.from_params(self.params)
        briefs = storyboard.generate_storyboard(job.idea, n_scenes=self.n_scenes, context=context)
        scenes: list[Scene] = []
        for i, brief in enumerate(briefs):
            kp = storyboard.keyframe_prompt(brief, style=job.style, context=context)
            ap = storyboard.animation_prompt(brief, context=context)
            scenes.append(
                Scene(
                    index=i,
                    description=brief.description,
                    character=brief.character,
                    setting=brief.setting,
                    mood=brief.mood,
                    keyframe_prompt=kp.positive,
                    keyframe_negative=kp.negative,
                    animation_prompt=ap.positive,
                    animation_negative=ap.negative,
                    mode=job.mode,
                    references=list(job.reference_images) if job.mode == "edit" else [],
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

    @property
    def variants_per_scene(self) -> int:
        return max(1, self.params.variants_per_scene)

    def _phase_keyframes(self, job: Job) -> None:
        """One keyframe per scene — or ``variants_per_scene`` candidates, the best
        of which becomes ``keyframe_path``. Persisted after every variant, so a
        resumed run continues mid-scene."""
        ref_cache: dict = {}  # reference path -> staged name (upload once per run)
        n = self.variants_per_scene
        self._event(job, "phase", phase="keyframes", total=len(job.scenes), variants=n)
        for scene in job.scenes:
            if not self._scene_needs_keyframes(scene, n):
                continue  # resume: already done
            log.info("job %s: keyframes scene %d (%d variant(s))", job.id, scene.index, n)
            refs = (
                pipeline.stage_references(scene.references, self.params, cache=ref_cache)
                if scene.mode == "edit" else ()
            )
            for v in range(len(scene.variants), n):
                provenance.set_context(scene=scene.index, variant=v if n > 1 else None)
                seed = config.derive_seed(self.seed, scene=scene.index, variant=v)
                path, verdict = self._one_keyframe(job, scene, seed=seed, refs=refs, variant=v)
                scene.variants.append(Variant(path=str(path), seed=_seed_used(path, seed), verdict=verdict))
                self.store.save_job(job)  # persist per variant — resumable
                if n > 1:
                    self._event(job, "variant", index=scene.index, variant=v, path=str(path))
            chosen = choose_variant(scene.variants)
            scene.keyframe_path = chosen.path
            scene.keyframe_verdict = chosen.verdict
            scene.status = SceneStatus.KEYFRAME
            self.store.save_job(job)  # persist per scene — resumable
            self._event(job, "keyframe", index=scene.index, path=scene.keyframe_path)

    def _one_keyframe(self, job: Job, scene: Scene, *, seed: int, refs, variant: int):
        prefix = output_prefix(job, scene.index, "keyframe", variant if self.variants_per_scene > 1 else None)
        n_var = self.variants_per_scene
        # Announce the step before generation (progress UI); retries re-announce via on_start.
        self._event(job, "step", step="keyframe", index=scene.index, variant=variant, variants=n_var,
                    attempt=1, max_attempts=self.max_attempts if self.evaluate else 1, mode=scene.mode)
        if self.evaluate:
            def on_start(attempt, _seed, _idx=scene.index):
                if attempt > 1:
                    self._event(job, "step", step="keyframe", index=_idx, variant=variant, variants=n_var,
                                attempt=attempt, max_attempts=self.max_attempts, mode=scene.mode)

            def log_attempt(attempt, _path, verdict, _idx=scene.index):
                # Every attempt -> decision log (DESIGN §8), not just the last.
                self.store.record_verdict(job.id, _idx, "keyframe", verdict, attempt=attempt)
                self._flag_verdict(job, _idx, verdict, attempt=attempt)
                self._event(job, "verdict", index=_idx, attempt=attempt, variant=variant,
                            decision=verdict.decision, score=verdict.score)

            return pipeline.generate_keyframe_evaluated(
                scene.keyframe_prompt, scene.keyframe_prompt,
                seed=seed, max_attempts=self.max_attempts, on_attempt=log_attempt,
                params=self.params, negative=scene.keyframe_negative or None,
                mode=scene.mode, references=refs, filename_prefix=prefix, on_start=on_start,
            )
        path, _ = pipeline._fresh_keyframe(
            scene.keyframe_prompt, seed=seed, evaluate=False, max_attempts=1,
            params=self.params, negative=scene.keyframe_negative or None,
            mode=scene.mode, references=refs, filename_prefix=prefix,
        )
        self.store.log_decision(
            Decision(job_id=job.id, scene_index=scene.index, phase="keyframe",
                     attempt=1, action="generated", reason=f"variant {variant}" if self.variants_per_scene > 1 else "")
        )
        return path, None

    @staticmethod
    def _scene_needs_keyframes(scene: Scene, n: int) -> bool:
        if scene.keyframe_path is None or scene.status is SceneStatus.PENDING:
            return True
        return n > 1 and len(scene.variants) < n

    def _phase_finish_stills(self, job: Job) -> None:
        """Stills jobs: impose the master face on every variant (when enabled) and
        mark the scenes complete. Post-steps for stills hook in here."""
        do_swap = self.face_swap and bool(job.master_face)
        master_name = comfy.upload_image(job.master_face) if do_swap else None
        self._event(job, "phase", phase="stills", total=len(job.scenes))
        for scene in job.scenes:
            if scene.status is SceneStatus.COMPLETE:
                continue
            if not scene.keyframe_path:
                log.warning("job %s: scene %d has no keyframe; skipping", job.id, scene.index)
                continue
            provenance.set_context(scene=scene.index)
            if do_swap:
                for v, variant in enumerate(scene.variants):
                    if variant.swapped_path:
                        continue
                    provenance.set_context(variant=v if len(scene.variants) > 1 else None)
                    self._event(job, "step", step="swap", index=scene.index, variant=v, variants=len(scene.variants))
                    swapped = pipeline.swap_master_onto(
                        variant.path, master_name, params=self.params,
                        label=f"scene {scene.index} variant {v} faceswap",
                        filename_prefix=output_prefix(job, scene.index, "swapped", v if len(scene.variants) > 1 else None),
                    )
                    variant.swapped_path = str(swapped)
                    self.store.save_job(job)
                chosen = next((x for x in scene.variants if x.path == scene.keyframe_path), None)
                if chosen is not None:
                    scene.swapped_path = chosen.swapped_path
                elif not scene.swapped_path:
                    scene.swapped_path = str(pipeline.swap_master_onto(
                        scene.keyframe_path, master_name, params=self.params,
                        filename_prefix=output_prefix(job, scene.index, "swapped")))
                scene.status = SceneStatus.SWAP
            if self.params.post_steps:
                for v, variant in enumerate(scene.variants):
                    if variant.post_paths:
                        continue
                    self._event(job, "step", step="post", index=scene.index, variant=v,
                                chain=[st.workflow for st in self.params.post_steps])
                    src = variant.swapped_path or variant.path
                    outs = run_chain(
                        src, self.params.post_steps, seed=variant.seed,
                        filename_prefix=output_prefix(job, scene.index, "post", v if len(scene.variants) > 1 else None),
                    )
                    variant.post_paths = [str(o) for o in outs]
                    self.store.save_job(job)
                chosen = next((x for x in scene.variants if x.path == scene.keyframe_path), None)
                if chosen is not None:
                    scene.post_paths = list(chosen.post_paths)
            scene.status = SceneStatus.COMPLETE
            self.store.log_decision(
                Decision(job_id=job.id, scene_index=scene.index, phase="stills",
                         attempt=1, action="generated", reason=f"{len(scene.variants)} variant(s)")
            )
            self.store.save_job(job)
            self._event(job, "stills", index=scene.index, variants=len(scene.variants))

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
                    reason=f"accepted but score={verdict.score} (< {verdict.review_threshold})",
                )
            )

    def _phase_videos(self, job: Job) -> None:
        do_swap = self.face_swap and bool(job.master_face)
        master_name = comfy.upload_image(job.master_face) if do_swap else None
        self._event(job, "phase", phase="videos", total=len(job.scenes),
                    length=self.params.length, fps=self.params.fps)
        for scene in job.scenes:
            if scene.video_path:
                continue  # resume: already done
            if not scene.keyframe_path:
                log.warning("job %s: scene %d has no keyframe; skipping", job.id, scene.index)
                continue
            provenance.set_context(scene=scene.index)
            if do_swap:
                log.info("job %s: video scene %d (swap -> animate)", job.id, scene.index)
                self._event(job, "step", step="swap", index=scene.index)
                swapped = pipeline.swap_master_onto(
                    scene.keyframe_path, master_name, params=self.params,
                    filename_prefix=output_prefix(job, scene.index, "swapped"),
                )
                scene.swapped_path = str(swapped)
                scene.status = SceneStatus.SWAP
                self.store.save_job(job)
                frame = swapped
            else:
                log.info("job %s: video scene %d (animate keyframe, no swap)", job.id, scene.index)
                frame = scene.keyframe_path

            seed = config.derive_seed(self.seed, scene=scene.index)
            self._event(job, "step", step="animate", index=scene.index,
                        length=self.params.length, fps=self.params.fps, seconds=self.params.clip_seconds)
            video = pipeline.animate_frame(
                frame, scene.animation_prompt, seed=seed,
                params=self.params, negative=scene.animation_negative or None,
                filename_prefix=output_prefix(job, scene.index, "clip"),
            )
            scene.raw_video_path = str(video)
            scene.video_path = str(video)
            self.store.log_decision(
                Decision(job_id=job.id, scene_index=scene.index, phase="video",
                         attempt=1, action="generated")
            )
            self.store.save_job(job)
            self._event(job, "video", index=scene.index, path=scene.video_path)
            if self.params.post_steps:
                self._event(job, "step", step="post", index=scene.index,
                            chain=[st.workflow for st in self.params.post_steps])
                outs = run_chain(
                    video, self.params.post_steps, fps=self.params.fps, seed=seed,
                    filename_prefix=output_prefix(job, scene.index, "post"),
                )
                if outs:
                    scene.post_paths = [str(o) for o in outs]
                    scene.video_path = str(outs[-1])
                    self.store.log_decision(
                        Decision(job_id=job.id, scene_index=scene.index, phase="post", attempt=1,
                                 action="generated", reason=", ".join(s.workflow for s in self.params.post_steps))
                    )
                    for i, o in enumerate(outs):
                        self._event(job, "postprocess", index=scene.index, step=i, path=str(o))
            scene.status = SceneStatus.COMPLETE
            self.store.save_job(job)

    # -- gates & phase predicates --

    @staticmethod
    def gates(job: Job) -> list[str]:
        """The approval gates this job passes through, in order (by kind)."""
        return profiles.gates_for({"kind": getattr(job, "kind", "video")})

    @classmethod
    def next_gate(cls, job: Job, store: Store) -> str | None:
        """The first gate not yet approved (regardless of the job's status)."""
        approved = {d.phase for d in store.decisions(job.id) if d.action == "approved"}
        for g in cls.gates(job):
            if g not in approved:
                return g
        return None

    @classmethod
    def pending_gate(cls, job: Job, store: Store) -> str | None:
        """The gate a paused job is waiting on; None unless NEEDS_APPROVAL."""
        if job.status is not JobStatus.NEEDS_APPROVAL:
            return None
        return cls.next_gate(job, store)

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

    def _needs_keyframes(self, job: Job) -> bool:
        n = self.variants_per_scene
        return any(self._scene_needs_keyframes(s, n) for s in job.scenes)

    @staticmethod
    def _needs_videos(job: Job) -> bool:
        return any(s.video_path is None for s in job.scenes)

    @staticmethod
    def _needs_stills_finish(job: Job) -> bool:
        return any(s.status is not SceneStatus.COMPLETE for s in job.scenes)
