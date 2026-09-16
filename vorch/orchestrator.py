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
from datetime import datetime, timezone
from pathlib import Path

from . import cancel, config, media, pipeline, profiles, provenance, storyboard
from . import projects as projects_mod
from . import scene_assets
from .genparams import DEFAULT, GenParams
from .steps.edit import reference_slots
from .steps.postprocess import run_chain
from .state import Clip, Decision, Job, JobStatus, Scene, SceneStatus, Store, Variant, new_job

log = logging.getLogger("vorch.orchestrator")

GATE_STORYBOARD = "storyboard"
GATE_KEYFRAMES = "keyframes"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def batch_of(scene: Scene) -> int:
    """Which generation run the scene is on; every version records the batch that made it."""
    return scene.regenerations


def current_batch(scene: Scene) -> list[Variant]:
    """The keyframes made by this run — what `choose_variant` may pick from.

    History is kept, so choosing across *all* variants would let an old high scorer
    silently override a fresh run (or the user's explicit pick).
    """
    return [v for v in scene.variants if v.batch == batch_of(scene)]


def active_variant(scene: Scene) -> Variant | None:
    return next((v for v in scene.variants if v.path == scene.keyframe_path), None)


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
            master = Path(master_face).expanduser() if master_face else None
            if master is not None and not master.is_file():
                raise FileNotFoundError(f"master face not found: {master}")
            if master is None:
                # No default face: the storyboard must be able to pick one per scene.
                project = projects_mod.load(self.store, project_id) if project_id else None
                if not projects_mod.identity_assets(project):
                    raise FileNotFoundError("master face not found: face swap is on but no master face was "
                                            "given and the project has no master_face assets")
            master_str = str(master) if master is not None else ""
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

        # Stop checkpoints sit between phases; there is none before DONE — once the last
        # phase returns, all the work is done and a late stop must not discard it.
        if not job.scenes:
            cancel.check()
            self._phase_storyboard(job)
        if self._gate(job, GATE_STORYBOARD):
            return job

        if self._needs_keyframes(job):
            cancel.check()
            self._phase_keyframes(job)

        if job.kind == "stills":
            # Everything after the keyframes is cheap: no second gate.
            if self._needs_stills_finish(job):
                cancel.check()
                self._phase_finish_stills(job)
        else:
            if self._gate(job, GATE_KEYFRAMES):
                return job
            if self._needs_videos(job):
                cancel.check()
                self._phase_videos(job)

        job.status = JobStatus.DONE
        self.store.save_job(job)
        log.info("job %s complete (%d scenes)", job.id, len(job.scenes))
        self._event(job, "status", status=job.status.value)
        return job

    SCENE_EDIT_FIELDS = ("description", "keyframe_prompt", "keyframe_negative", "animation_prompt",
                         "animation_negative", "mode", "references")

    def reopen_scene(self, job: Job, index: int, *, what: str = "keyframe", edits: dict | None = None) -> Scene:
        """Discard a scene's keyframe (and everything built on it) or just its clip,
        apply prompt edits, and reopen the keyframes gate so the result is reviewed.

        ``what="keyframe"``: keyframe, variants, swap, clip, post-steps are cleared and
        the scene is regenerated from the keyframes phase; ``what="video"``: only the
        clip/post outputs are cleared (video jobs). The job's export is cleared so the
        handoff folder is rebuilt when the job completes again.

        ``edits["keyframe_asset_id"]`` (with ``what="keyframe"``) uses that project image
        as the keyframe instead of generating one: it is fitted to the job's keyframe
        size inside COMFY_OUTPUT and **appended** as a ``source="user"`` variant (the
        generated candidates stay selectable), everything built on the old keyframe is
        cleared, and the gate is reopened exactly as for a regenerated keyframe. The
        keyframes phase never generates over it and the evaluator never judges it.
        """
        scene = next((s for s in job.scenes if s.index == index), None)
        if scene is None:
            raise KeyError(f"job {job.id} has no scene {index}")
        if what not in ("keyframe", "video"):
            raise ValueError("what must be 'keyframe' or 'video'")
        if what == "video" and job.kind != "video":
            raise ValueError("stills jobs have no clip to regenerate")
        for field_name in self.SCENE_EDIT_FIELDS:
            if edits and edits.get(field_name) is not None:
                setattr(scene, field_name, edits[field_name])
        asset_changes: list[str] = []
        if scene_assets.has_asset_edits(edits):
            project = projects_mod.load(self.store, job.project_id) if job.project_id else None
            asset_changes = scene_assets.apply_asset_edits(scene, edits or {}, project)
        user_asset = (edits or {}).get("keyframe_asset_id") if what == "keyframe" else None
        applied = ""
        if what == "keyframe" and edits and edits.get("apply_suggestion"):
            fix = self.last_fix_suggestion(job, scene)
            if fix and fix not in scene.keyframe_prompt:
                scene.keyframe_prompt = f"{scene.keyframe_prompt.rstrip('. ')}. Correction: {fix}"
                applied = fix
        if what == "keyframe":
            scene.keyframe_verdict = None
            scene.swapped_path = None
            scene.swap_verdict = None
            if user_asset:
                path, asset = self._install_user_keyframe(job, scene, user_asset)
                scene.variants.append(Variant(path=str(path), source="user", asset_id=user_asset,
                                              created_at=_now(), batch=scene.regenerations + 1))
                scene.keyframe_path = str(path)
            else:
                # History is kept: only the pointer is cleared, which is what makes the next
                # run generate a new batch. Every file stays on disk with its provenance.
                scene.keyframe_path = None
        if what == "video":
            scene.swapped_path = None   # the swap is part of the video phase: redo it with the current identity
            scene.swap_verdict = None
        scene.video_path = None      # the clip projection; scene.clips keeps every rendered clip
        scene.raw_video_path = None
        scene.video_verdict = None
        scene.post_paths = []
        scene.status = SceneStatus.PENDING if (what == "keyframe" and not user_asset) else SceneStatus.KEYFRAME
        scene.regenerations += 1
        job.export = None
        changed = [k for k in self.SCENE_EDIT_FIELDS if edits and edits.get(k) is not None] + asset_changes
        if applied:
            changed.append("applied evaluator suggestion")
        if user_asset:
            self.store.log_decision(Decision(
                job_id=job.id, scene_index=index, phase="keyframe", attempt=scene.regenerations,
                action="user_keyframe", reason=f"image: {asset.label or asset.name}",
            ))
        else:
            self.store.log_decision(Decision(
                job_id=job.id, scene_index=index, phase=what, attempt=scene.regenerations,
                action="regenerate", reason=f"edited: {', '.join(changed)}" if changed else "same prompts, new seed",
            ))
        if what == "keyframe" and job.kind == "video" and job.autonomy != "full":
            # The keyframes gate must be passed again for the new frame.
            self.store.log_decision(Decision(job_id=job.id, scene_index=None, phase=GATE_KEYFRAMES,
                                             attempt=1, action="reopened"))
        self.store.save_job(job)
        return scene

    def _install_user_keyframe(self, job: Job, scene: Scene, asset_id: str):
        """Fit a project image to the job's keyframe size as this scene's next keyframe file.

        Written under COMFY_OUTPUT (the only tree ``/media`` serves) and at the keyframe
        size, so the UI can show it and Wan never has to crop it. ffmpeg reads WebP/AVIF
        directly — no ``ensure_png`` round trip, which would shrink it first.
        """
        project = projects_mod.load(self.store, job.project_id) if job.project_id else None
        asset = project.asset(asset_id) if project is not None else None
        if asset is None:
            raise ValueError(f"unknown project asset {asset_id!r}")
        dest = config.COMFY_OUTPUT / f"{output_prefix(job, scene.index, 'keyframe')}_user_{scene.regenerations + 1}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        path = media.fit_image(asset.path, *self.params.keyframe_size, dest)
        # The provenance context is not bound outside run(): name the job explicitly.
        provenance.write_manifest(path, "user_keyframe", {"asset_id": asset_id, "source": asset.path},
                                  {"job_id": job.id, "scene": scene.index, "kind": "user_keyframe"})
        log.info("job %s: scene %d keyframe <- %s", job.id, scene.index, asset.label or asset.name)
        return path, asset

    def last_fix_suggestion(self, job: Job, scene: Scene) -> str:
        """The evaluator's most recent fix suggestion for a scene: the stored verdict,
        else the last 'retry' decision's reason."""
        if scene.keyframe_verdict is not None and scene.keyframe_verdict.fix_suggestion:
            return scene.keyframe_verdict.fix_suggestion
        for d in reversed(self.store.decisions(job.id)):
            if d.scene_index == scene.index and d.phase == "keyframe" and d.action == "retry" and d.reason:
                return d.reason
        return ""

    def regenerate(self, job: Job, index: int, *, what: str = "keyframe", edits: dict | None = None) -> Job:
        """``reopen_scene`` + ``run``: rerun only what was cleared (other scenes keep their files)."""
        self.reopen_scene(job, index, what=what, edits=edits)
        log.info("job %s: regenerating %s of scene %d", job.id, what, index)
        return self.run(job)

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
        project = projects_mod.load(self.store, job.project_id) if job.project_id else None
        asset_list = tuple(
            (a.id, a.kind, a.label or a.name) for a in (project.assets if project else [])
            if a.kind in ("master_face", "reference", "product", "logo")
        )
        context = storyboard.PromptContext.from_params(
            self.params, assets=asset_list if (self.face_swap or job.mode == "edit") else (), edit_mode=job.mode == "edit",
        )
        briefs = storyboard.generate_storyboard(job.idea, n_scenes=self.n_scenes, context=context)
        if self.n_scenes and len(briefs) > self.n_scenes:
            # n_scenes is a hard cap: the model sometimes returns more than asked for
            log.info("job %s: storyboard returned %d scenes, keeping the first %d", job.id, len(briefs), self.n_scenes)
            briefs = briefs[: self.n_scenes]
        scenes: list[Scene] = []
        for i, brief in enumerate(briefs):
            kp = storyboard.keyframe_prompt(brief, style=job.style, context=context)
            ap = storyboard.animation_prompt(brief, context=context)
            scene = Scene(
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
                identity=job.master_face if self.face_swap else "",
            )
            self._assign_scene_assets(job, scene, brief, project)
            scenes.append(scene)
        job.scenes = scenes
        self.store.log_decision(
            Decision(
                job_id=job.id, scene_index=None, phase="storyboard", attempt=1,
                action="generated", reason=f"{len(scenes)} scenes",
            )
        )
        self.store.save_job(job)
        self._event(job, "storyboard", scenes=len(scenes))

    def _assign_scene_assets(self, job: Job, scene: Scene, brief, project) -> None:
        """Apply the storyboard's per-scene identity / reference picks (project asset
        ids). Unknown ids are dropped with a decision-log warning; without a pick the
        scene keeps the job defaults (master face, reference images)."""
        if project is None or not (self.face_swap or job.mode == "edit"):
            return
        picks: dict = {}
        if self.face_swap and brief.identity is not None:
            picks["identity_asset_id"] = brief.identity
        if job.mode == "edit" and brief.reference_ids:
            picks["reference_ids"] = list(brief.reference_ids)
        if not picks and brief.identity is None and self.face_swap and "identity" in getattr(brief, "model_fields_set", set()):
            picks["identity_asset_id"] = ""  # the model explicitly said: no face in this scene
        if not picks:
            return
        edits: dict = {}
        if "identity_asset_id" in picks:
            aid = picks["identity_asset_id"]
            if aid == "" or project.asset(aid) is not None and project.asset(aid).kind == "master_face":
                edits["identity_asset_id"] = aid
            else:
                self.store.log_decision(Decision(job_id=job.id, scene_index=scene.index, phase="storyboard",
                                                 attempt=1, action="warning",
                                                 reason=f"storyboard picked unknown identity {aid!r}; keeping the default"))
        if "reference_ids" in picks:
            known = [r for r in picks["reference_ids"] if project.asset(r) is not None]
            unknown = [r for r in picks["reference_ids"] if r not in known]
            if unknown:
                self.store.log_decision(Decision(job_id=job.id, scene_index=scene.index, phase="storyboard",
                                                 attempt=1, action="warning",
                                                 reason=f"storyboard picked unknown references {unknown}"))
            if known:
                edits["reference_asset_ids"] = known
        if not edits:
            return
        changed = scene_assets.apply_asset_edits(scene, edits, project)
        label = scene_assets.identity_label(project, scene) or "none"
        self.store.log_decision(Decision(
            job_id=job.id, scene_index=scene.index, phase="storyboard", attempt=1, action="assigned",
            reason=", ".join(f"identity: {label}" if c == "identity" else f"references: {len(scene.reference_asset_ids)}"
                             for c in changed),
        ))

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
            cancel.check()
            log.info("job %s: keyframes scene %d (%d variant(s))", job.id, scene.index, n)
            refs = (
                pipeline.stage_references(scene.references, self.params, cache=ref_cache)
                if scene.mode == "edit" else ()
            )
            batch = batch_of(scene)
            done = len(current_batch(scene))   # resume: how many of this run's variants exist
            slots = reference_slots(self.params.edit_workflow) if scene.mode == "edit" else 0
            if slots and len(scene.references) > slots and not done:
                self.store.log_decision(Decision(
                    job_id=job.id, scene_index=scene.index, phase="keyframe", attempt=1, action="warning",
                    reason=f"{len(scene.references)} references but {self.params.edit_workflow} uses the first {slots}",
                ))
            for v in range(done, n):
                cancel.check()  # every earlier variant is already saved
                provenance.set_context(scene=scene.index, variant=v if n > 1 else None)
                seed = config.derive_seed(self.seed, scene=scene.index, variant=v,
                                          attempt=1 + 10 * scene.regenerations)
                path, verdict = self._one_keyframe(job, scene, seed=seed, refs=refs, variant=v)
                scene.variants.append(Variant(path=str(path), seed=_seed_used(path, seed), verdict=verdict,
                                              created_at=_now(), batch=batch))
                self.store.save_job(job)  # persist per variant — resumable
                if n > 1:
                    self._event(job, "variant", index=scene.index, variant=v, path=str(path))
            chosen = choose_variant(current_batch(scene))
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
                # Before generating (not after judging): a stop here never discards a
                # keyframe that was already made and scored.
                cancel.check()
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
                identity_image=(scene.identity or None) if self.face_swap else None,
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
        active = active_variant(scene)
        if active is not None and active.source == "user":
            return False  # the active keyframe is the user's image: never generate over it
        if scene.keyframe_path is None or scene.status is SceneStatus.PENDING:
            return True
        # Only this run's variants count: history from earlier runs must not look "already done".
        return n > 1 and len(current_batch(scene)) < n

    def _phase_finish_stills(self, job: Job) -> None:
        """Stills jobs: impose the master face on every variant (when enabled) and
        mark the scenes complete. Post-steps for stills hook in here."""
        face_cache: dict = {}  # (backend, face path) -> staged name
        self._event(job, "phase", phase="stills", total=len(job.scenes))
        for scene in job.scenes:
            if scene.status is SceneStatus.COMPLETE:
                continue
            if not scene.keyframe_path:
                log.warning("job %s: scene %d has no keyframe; skipping", job.id, scene.index)
                continue
            cancel.check()
            provenance.set_context(scene=scene.index)
            face = scene.identity if self.face_swap else ""
            do_swap = bool(face)
            master_name = pipeline.stage_identity(face, self.params, cache=face_cache) if do_swap else None
            # Only this run's variants are finished; older ones keep the outputs they had.
            batch = current_batch(scene) or scene.variants
            many = len(batch) > 1
            if do_swap:
                for v, variant in enumerate(batch):
                    if variant.swapped_path:
                        continue
                    cancel.check()
                    provenance.set_context(variant=v if many else None)
                    self._event(job, "step", step="swap", index=scene.index, variant=v, variants=len(batch))
                    swapped = pipeline.swap_master_onto(
                        variant.path, master_name, params=self.params,
                        label=f"scene {scene.index} variant {v} faceswap",
                        filename_prefix=output_prefix(job, scene.index, "swapped", v if many else None),
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
                for v, variant in enumerate(batch):
                    if variant.post_paths:
                        continue
                    cancel.check()
                    self._event(job, "step", step="post", index=scene.index, variant=v,
                                chain=[st.workflow for st in self.params.post_steps])
                    src = variant.swapped_path or variant.path
                    outs = run_chain(
                        src, self.params.post_steps, seed=variant.seed,
                        filename_prefix=output_prefix(job, scene.index, "post", v if many else None),
                        master_face=scene.identity or None,
                        prompts={"positive": scene.keyframe_prompt},
                    )
                    variant.post_paths = [str(o) for o in outs]
                    self.store.save_job(job)
                chosen = next((x for x in scene.variants if x.path == scene.keyframe_path), None)
                if chosen is not None:
                    scene.post_paths = list(chosen.post_paths)
            scene.status = SceneStatus.COMPLETE
            self.store.log_decision(
                Decision(job_id=job.id, scene_index=scene.index, phase="stills",
                         attempt=1, action="generated", reason=f"{len(batch)} variant(s)")
            )
            self.store.save_job(job)
            self._event(job, "stills", index=scene.index, variants=len(batch))

    @staticmethod
    def _record_clip(scene: Scene, seed: int) -> Clip:
        """Append the clip just finished to the scene's history (the scene's clip fields
        stay the projection of the active one). Re-recording the same file is a no-op, so
        a resumed run cannot duplicate an entry."""
        existing = next((c for c in scene.clips if c.video_path == scene.video_path), None)
        if existing is not None:
            return existing
        clip = Clip(video_path=str(scene.video_path), raw_video_path=scene.raw_video_path,
                    post_paths=list(scene.post_paths), seed=seed, verdict=scene.video_verdict,
                    keyframe_path=scene.keyframe_path, created_at=_now(), batch=scene.regenerations)
        scene.clips.append(clip)
        return clip

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
        """Swap (when the scene has a face), animate, then post-process each scene.

        Resumable at every persisted step, which a user stop relies on: a scene whose
        swap is saved is not swapped again; a scene whose raw clip is saved but whose
        post-steps did not finish runs only the chain. ``video_path`` (the deliverable)
        is set only once the chain has finished — or together with the raw clip when
        the profile has no post-steps.

        Each finished clip is appended to ``scene.clips`` and the scene's clip fields are
        the projection of the active one, so re-animating keeps every earlier take.
        """
        face_cache: dict = {}  # (backend, face path) -> staged name; faces differ per scene
        self._event(job, "phase", phase="videos", total=len(job.scenes),
                    length=self.params.length, fps=self.params.fps)
        for scene in job.scenes:
            if scene.video_path:
                continue  # resume: already done
            if not scene.keyframe_path:
                log.warning("job %s: scene %d has no keyframe; skipping", job.id, scene.index)
                continue
            cancel.check()
            provenance.set_context(scene=scene.index)
            seed = config.derive_seed(self.seed, scene=scene.index, attempt=1 + 10 * scene.regenerations)
            if not scene.raw_video_path:
                face = scene.identity if self.face_swap else ""
                if face and scene.swapped_path:
                    log.info("job %s: video scene %d (swap already done -> animate)", job.id, scene.index)
                    frame = scene.swapped_path
                elif face:
                    master_name = pipeline.stage_identity(face, self.params, cache=face_cache)
                    log.info("job %s: video scene %d (swap %s -> animate)", job.id, scene.index, Path(face).name)
                    self._event(job, "step", step="swap", index=scene.index, identity=Path(face).name)
                    swapped = pipeline.swap_master_onto(
                        scene.keyframe_path, master_name, params=self.params,
                        filename_prefix=output_prefix(job, scene.index, "swapped"),
                    )
                    scene.swapped_path = str(swapped)
                    scene.status = SceneStatus.SWAP
                    self.store.save_job(job)
                    frame = swapped
                    cancel.check()  # the swap is saved
                else:
                    log.info("job %s: video scene %d (animate keyframe, no swap)", job.id, scene.index)
                    frame = scene.keyframe_path

                self._event(job, "step", step="animate", index=scene.index,
                            length=self.params.length, fps=self.params.fps, seconds=self.params.clip_seconds)
                video = pipeline.animate_frame(
                    frame, scene.animation_prompt, seed=seed,
                    params=self.params, negative=scene.animation_negative or None,
                    filename_prefix=output_prefix(job, scene.index, "clip"),
                )
                scene.raw_video_path = str(video)
                if not self.params.post_steps:
                    scene.video_path = str(video)
                    self._record_clip(scene, seed)
                self.store.log_decision(
                    Decision(job_id=job.id, scene_index=scene.index, phase="video",
                             attempt=1, action="generated")
                )
                self.store.save_job(job)
                self._event(job, "video", index=scene.index, path=scene.raw_video_path)
            else:
                log.info("job %s: video scene %d (clip saved -> post-steps only)", job.id, scene.index)
            if self.params.post_steps:
                cancel.check()  # the raw clip is saved
                self._event(job, "step", step="post", index=scene.index,
                            chain=[st.workflow for st in self.params.post_steps])
                outs = run_chain(
                    Path(scene.raw_video_path), self.params.post_steps, fps=self.params.fps, seed=seed,
                    filename_prefix=output_prefix(job, scene.index, "post"),
                    master_face=scene.identity or None,
                    prompts={"positive": scene.keyframe_prompt, "motion": scene.animation_prompt},
                )
                if outs:
                    scene.post_paths = [str(o) for o in outs]
                    scene.video_path = str(outs[-1])
                    self._record_clip(scene, seed)
                    self.store.log_decision(
                        Decision(job_id=job.id, scene_index=scene.index, phase="post", attempt=1,
                                 action="generated", reason=", ".join(s.workflow for s in self.params.post_steps))
                    )
                    for i, o in enumerate(outs):
                        self._event(job, "postprocess", index=scene.index, step=i, path=str(o))
                else:
                    scene.video_path = scene.raw_video_path  # every step was skipped
                    self._record_clip(scene, seed)
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
        state: dict[str, bool] = {}
        for d in store.decisions(job.id):
            if d.action == "approved":
                state[d.phase] = True
            elif d.action == "reopened":
                state[d.phase] = False
        for g in cls.gates(job):
            if not state.get(g):
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
        """A gate counts as approved until a later 'reopened' entry (scene regeneration)."""
        approved = False
        for d in self.store.decisions(job.id):
            if d.phase != name:
                continue
            if d.action == "approved":
                approved = True
            elif d.action == "reopened":
                approved = False
        return approved

    def _needs_keyframes(self, job: Job) -> bool:
        n = self.variants_per_scene
        return any(self._scene_needs_keyframes(s, n) for s in job.scenes)

    @staticmethod
    def _needs_videos(job: Job) -> bool:
        return any(s.video_path is None for s in job.scenes)

    @staticmethod
    def _needs_stills_finish(job: Job) -> bool:
        return any(s.status is not SceneStatus.COMPLETE for s in job.scenes)
