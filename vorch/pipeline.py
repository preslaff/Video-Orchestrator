"""Scene drivers — single scene and sequential (continuity) chains.

Single scene proves the ComfyUI chain end-to-end:

    Klein keyframe -> ReActor (swap master face onto keyframe) -> Wan animate

``run_sequence`` chains scenes for visual continuity: the last frame of scene N's
clip can seed scene N+1 (``continuous=True``), re-locking identity each step. See
docs/sequential-continuity-plan.md. Each output is staged into ComfyUI input/
before the next step's LoadImage reads it, and VRAM is reclaimed before each Wan
phase. State persistence and the UI remain later Phase-2 increments.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import backends, config, formats, vram
from .clients import comfy
from .evaluator import Verdict, evaluate_image
from .genparams import DEFAULT, GenParams
from .media import last_frame
from .media import fit_image
from .steps.animate import animate
from .steps.edit import edit
from .steps.faceswap import faceswap
from .steps.keyframe import keyframe

log = logging.getLogger("vorch.pipeline")


@dataclass
class Scene:
    """One unit of a sequence.

    ``continuous=True`` seeds this scene from the previous clip's last frame
    instead of a fresh Klein keyframe (ignored for scene 0 — nothing precedes it).
    """

    keyframe_prompt: str
    animation_prompt: str
    continuous: bool = False
    keyframe_negative: str | None = None   # overrides params.keyframe_negative
    animation_negative: str | None = None  # overrides params.animate_negative
    mode: str = "t2i"                      # t2i | edit (reference-locked keyframe)
    references: tuple[str, ...] = ()       # local image paths for mode="edit"


@dataclass
class SceneResult:
    video: Path
    start_image: Path  # the still actually fed to Wan (post-swap when a swap ran)
    keyframe: Path | None = None  # the fresh Klein keyframe (None for continuous scenes)
    swapped: Path | None = None  # ReActor output (None when continuous + relock disabled)
    continued_from_previous: bool = False
    keyframe_verdict: Verdict | None = None


def stage_references(
    paths: Sequence[str | Path], params: GenParams = DEFAULT, *, fit: bool = True,
    cache: dict | None = None,
) -> list[str]:
    """Fit each reference to the keyframe size (cover + crop) and stage it in ComfyUI input/.

    Returns the staged names in order. ``cache`` ((backend, path) -> name) lets
    a job upload each reference once across scenes. References go to the edit
    workflow's backend (the workflow that reads them).
    """
    backend = backends.for_workflow(params.edit_workflow)
    names: list[str] = []
    for raw in paths:
        key = (backend.name, str(raw))
        if cache is not None and key in cache:
            names.append(cache[key])
            continue
        src = Path(raw).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"reference image not found: {src}")
        staged = fit_image(src, *params.keyframe_size) if fit else src
        name = comfy.upload_image(staged, backend=backend)
        if cache is not None:
            cache[key] = name
        names.append(name)
    return names


def upload_master(master_face: str | Path, params: GenParams = DEFAULT) -> str:
    """Stage the master face on the face-swap workflow's backend; return its name."""
    return comfy.upload_image(master_face, backend=backends.for_workflow(params.faceswap_workflow))


def stage_identity(face: str | Path, params: GenParams = DEFAULT, *, cache: dict | None = None) -> str:
    """Stage a scene's identity face once per (backend, path); return the staged name.

    Scenes may carry different faces (the protagonist at different ages) — the
    cache lets a job upload each face once per run.
    """
    backend = backends.for_workflow(params.faceswap_workflow)
    key = (backend.name, str(face))
    if cache is not None and key in cache:
        return cache[key]
    src = Path(face).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"identity face not found: {src}")
    name = comfy.upload_image(src, backend=backend)
    if cache is not None:
        cache[key] = name
    return name


def swap_master_onto(
    frame_path: Path, master_name: str, *, label: str = "faceswap", params: GenParams = DEFAULT,
    filename_prefix: str | None = None,
) -> Path:
    """Stage ``frame_path`` (on the face-swap workflow's backend) and impose the
    already-staged master face onto it."""
    frame_name = comfy.upload_image(frame_path, backend=backends.for_workflow(params.faceswap_workflow))
    log.info("%s: imposing master face", label)
    swapped = faceswap(frame_name, master_name, filename_prefix=filename_prefix, **params.faceswap_kwargs())
    log.info("%s: %s", label, swapped)
    return swapped


def _animate_need_mb(params: GenParams) -> int:
    """VRAM that must be free before a local animate, from the manifest's ComfyUI entry.

    Falls back to ``VRAM_GUARD_MB`` when the manifest cannot be read — a broken services file must
    not stop a render that would have worked.
    """
    from . import services

    try:
        need = services.get("comfy").vram.needs_mb
    except services.ServiceError:
        need = 0
    return need or config.VRAM_GUARD_MB


def animate_frame(
    frame_path: Path,
    animation_prompt: str,
    *,
    seed: int,
    params: GenParams = DEFAULT,
    negative: str | None = None,
    filename_prefix: str | None = None,
) -> Path:
    """Stage ``frame_path``, reclaim VRAM, and animate it into a clip.

    Wan (dual-model) is the heaviest step, so reclaim ComfyUI VRAM at this phase
    boundary and confirm headroom before dispatching (NOTES.md §4) — unless the
    animate workflow runs on a remote backend, which has its own GPU. Size,
    length and fps come from ``params`` (the job's profile).
    """
    backend = backends.for_workflow(params.animate_workflow)
    frame_name = comfy.upload_image(frame_path, backend=backend)
    if backend.is_local:
        # The Klein/ReActor -> Wan boundary, and the heaviest moment in the whole pipeline. Freeing
        # ComfyUI's models is no longer enough on its own: another process may hold the card (a
        # narration server, most likely), and the old code would free 222 MB, log the result and
        # walk into an OOM. ensure_headroom frees what the manifest allows, re-measures, and raises
        # VramUnavailable naming the holder if it still does not fit.
        from . import supervisor

        supervisor.ensure_headroom(_animate_need_mb(params))
        vram.log_state("before Wan")
    else:
        vram.reclaim(backend=backend)   # remote: a no-op, kept for the log line
    kw = params.animate_kwargs()
    if negative is not None:
        kw["negative"] = negative
    log.info("animate: %sx%s, %s frames @ %sfps (%ss)", kw["width"], kw["height"],
             kw["length"], kw["fps"], params.clip_seconds)
    video_path = animate(frame_name, animation_prompt, seed=seed, filename_prefix=filename_prefix, **kw)
    log.info("animate: %s", video_path)
    return video_path


def generate_keyframe_evaluated(
    prompt: str,
    task: str,
    *,
    seed: int = config.SEED_RANDOM,
    max_attempts: int = 2,
    on_attempt: Callable[[int, Path, Verdict], None] | None = None,
    params: GenParams = DEFAULT,
    negative: str | None = None,
    filename_prefix: str | None = None,
    mode: str = "t2i",
    references: Sequence[str] = (),
    on_start: Callable[[int, int], None] | None = None,
    identity_image: str | Path | None = None,
) -> tuple[Path, Verdict]:
    """Generate a keyframe, evaluate it, and retry with the fix up to N times.

    Returns the (path, verdict) of the accepted attempt, or of the final attempt
    if none were accepted — the caller's human gate decides what to do then.
    ``on_attempt(attempt, path, verdict)`` is invoked after each evaluation so the
    caller can log every attempt to its decision log (DESIGN.md §8), not just the
    final one; ``on_start(attempt, seed)`` fires before each generation (progress UI).
    """
    attempt_prompt = prompt
    kf_path: Path
    verdict: Verdict
    for attempt in range(1, max_attempts + 1):
        # seed 0 = random (each retry differs); a fixed seed is bumped per attempt.
        attempt_seed = config.derive_seed(seed, attempt=attempt)
        log.info("keyframe: attempt %d/%d (seed=%s, mode=%s)", attempt, max_attempts, attempt_seed, mode)
        if on_start is not None:
            on_start(attempt, attempt_seed)
        kf_path = _make_keyframe(
            attempt_prompt, seed=attempt_seed, params=params, negative=negative,
            filename_prefix=filename_prefix, mode=mode, references=references,
        )
        verdict = evaluate_image(
            kf_path, task, orientation=params.orientation,
            framing_hint=formats.framing_hint(params.aspect),
            review_threshold=params.review_threshold,
            identity_image=identity_image,
        )
        if on_attempt is not None:
            on_attempt(attempt, kf_path, verdict)
        if verdict.decision == "accept":
            return kf_path, verdict
        # Feed the fix back into the prompt for the next attempt (NOTES.md §6).
        attempt_prompt = f"{prompt}. Correction: {verdict.fix_suggestion}".strip()
    log.warning("keyframe: no attempt accepted after %d tries; using last", max_attempts)
    return kf_path, verdict


def _make_keyframe(
    prompt: str,
    *,
    seed: int,
    params: GenParams,
    negative: str | None,
    filename_prefix: str | None,
    mode: str,
    references: Sequence[str],
) -> Path:
    """One keyframe: text-to-image, or a reference-locked edit when ``mode == "edit"``."""
    if mode == "edit":
        if not references:
            raise ValueError("mode='edit' needs staged reference images")
        kw = params.edit_kwargs()
        if negative is not None:
            kw["negative"] = negative
        return edit(prompt, list(references), seed=seed, filename_prefix=filename_prefix, **kw)
    if mode != "t2i":
        raise ValueError(f"unknown keyframe mode {mode!r} (t2i | edit)")
    kw = params.keyframe_kwargs()
    if negative is not None:
        kw["negative"] = negative
    return keyframe(prompt, seed=seed, filename_prefix=filename_prefix, **kw)


def _fresh_keyframe(
    prompt: str,
    *,
    seed: int,
    evaluate: bool,
    max_attempts: int,
    params: GenParams = DEFAULT,
    negative: str | None = None,
    filename_prefix: str | None = None,
    mode: str = "t2i",
    references: Sequence[str] = (),
) -> tuple[Path, Verdict | None]:
    """Build a fresh keyframe (t2i or edit), optionally with the evaluate/retry loop."""
    if evaluate:
        return generate_keyframe_evaluated(
            prompt, prompt, seed=seed, max_attempts=max_attempts, params=params,
            negative=negative, filename_prefix=filename_prefix, mode=mode, references=references,
        )
    log.info("keyframe: generating %sx%s (seed=%s, mode=%s)", *params.keyframe_size, seed, mode)
    kf_path = _make_keyframe(
        prompt, seed=seed, params=params, negative=negative, filename_prefix=filename_prefix,
        mode=mode, references=references,
    )
    log.info("keyframe: %s", kf_path)
    return kf_path, None


def run_scene(
    keyframe_prompt: str,
    animation_prompt: str,
    master_face: str | Path = "",
    *,
    seed: int = config.SEED_RANDOM,
    evaluate: bool = False,
    task: str | None = None,
    max_attempts: int = 2,
    face_swap: bool = True,
    params: GenParams = DEFAULT,
) -> SceneResult:
    """Run one scene through the full chain and return the artifact paths.

    With ``evaluate=True`` the keyframe is judged by Gemma and retried with the
    fix suggestion (up to ``max_attempts``); ``task`` is the intent to score
    against (defaults to the keyframe prompt). With ``face_swap=False`` the
    keyframe is animated directly and ``master_face`` is ignored.
    """
    master_name = None
    if face_swap:
        master_face = Path(master_face).expanduser()
        if not master_face.is_file():
            raise FileNotFoundError(f"master face not found: {master_face}")
        master_name = upload_master(master_face, params)

    if evaluate:
        kf_path, verdict = generate_keyframe_evaluated(
            keyframe_prompt, task or keyframe_prompt, seed=seed, max_attempts=max_attempts,
            params=params,
        )
    else:
        kf_path, verdict = _fresh_keyframe(
            keyframe_prompt, seed=seed, evaluate=False, max_attempts=max_attempts, params=params
        )

    swapped_path = swap_master_onto(kf_path, master_name, params=params) if face_swap else None
    video_path = animate_frame(swapped_path or kf_path, animation_prompt, seed=seed, params=params)

    return SceneResult(
        video=video_path,
        start_image=swapped_path or kf_path,
        keyframe=kf_path,
        swapped=swapped_path,
        continued_from_previous=False,
        keyframe_verdict=verdict,
    )


def run_sequence(
    scenes: Sequence[Scene],
    master_face: str | Path = "",
    *,
    seed: int = config.SEED_RANDOM,
    evaluate: bool = False,
    relock_identity: bool = True,
    max_attempts: int = 2,
    face_swap: bool = True,
    params: GenParams = DEFAULT,
) -> list[SceneResult]:
    """Run a sequence of scenes, chaining continuity where requested.

    For each scene: a fresh Klein keyframe (identity-locked to ``master_face``)
    unless the scene is ``continuous`` (and not the first), in which case the
    previous clip's last frame seeds it. Carried frames are re-locked to the
    master face when ``relock_identity`` is set (default), countering Wan drift.

    With ``face_swap=False`` no swap runs at all (fresh keyframes and carried
    frames animate raw) and ``relock_identity`` / ``master_face`` are ignored.

    Continuity is inherently sequential (scene N depends on N-1's video), so this
    diverges from the phase-by-stage batching of NOTES.md §5.
    """
    master_name = None
    if face_swap:
        master_face = Path(master_face).expanduser()
        if not master_face.is_file():
            raise FileNotFoundError(f"master face not found: {master_face}")
        master_name = upload_master(master_face, params)
    relock = relock_identity and face_swap
    ref_cache: dict = {}

    results: list[SceneResult] = []
    prev_video: Path | None = None

    for i, scene in enumerate(scenes):
        continuous = scene.continuous and i > 0  # scene 0 is always fresh
        if continuous:
            log.info("scene %d/%d: continuous — carrying last frame forward", i + 1, len(scenes))
            assert prev_video is not None  # i > 0 guarantees a previous clip
            frame = last_frame(prev_video)
            kf_path = None
            verdict = None
            if relock:
                swapped = swap_master_onto(frame, master_name, label=f"scene {i + 1} relock", params=params)
                start_image = swapped
            else:
                swapped = None
                start_image = frame
        else:
            log.info("scene %d/%d: fresh keyframe", i + 1, len(scenes))
            refs = stage_references(scene.references, params, cache=ref_cache) if scene.mode == "edit" else ()
            kf_path, verdict = _fresh_keyframe(
                scene.keyframe_prompt, seed=seed, evaluate=evaluate, max_attempts=max_attempts,
                params=params, negative=scene.keyframe_negative, mode=scene.mode, references=refs,
            )
            if face_swap:
                swapped = swap_master_onto(kf_path, master_name, label=f"scene {i + 1} faceswap", params=params)
                start_image = swapped
            else:
                swapped = None
                start_image = kf_path

        video_path = animate_frame(
            start_image, scene.animation_prompt, seed=seed, params=params,
            negative=scene.animation_negative,
        )
        results.append(
            SceneResult(
                video=video_path,
                start_image=start_image,
                keyframe=kf_path,
                swapped=swapped,
                continued_from_previous=continuous,
                keyframe_verdict=verdict,
            )
        )
        prev_video = video_path

    return results
