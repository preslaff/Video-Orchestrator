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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import config
from .clients import comfy
from .evaluator import Verdict, evaluate_image
from .media import last_frame
from .steps.animate import animate
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


@dataclass
class SceneResult:
    video: Path
    start_image: Path  # the still actually fed to Wan (post-swap when a swap ran)
    keyframe: Path | None = None  # the fresh Klein keyframe (None for continuous scenes)
    swapped: Path | None = None  # ReActor output (None when continuous + relock disabled)
    continued_from_previous: bool = False
    keyframe_verdict: Verdict | None = None


def _swap_master_onto(frame_path: Path, master_name: str, *, label: str = "faceswap") -> Path:
    """Stage ``frame_path`` and impose the (already-staged) master face onto it."""
    frame_name = comfy.upload_image(frame_path)
    log.info("%s: imposing master face", label)
    swapped = faceswap(frame_name, master_name)
    log.info("%s: %s", label, swapped)
    return swapped


def _animate_frame(frame_path: Path, animation_prompt: str, *, seed: int) -> Path:
    """Stage ``frame_path``, reclaim VRAM, and animate it into a clip."""
    frame_name = comfy.upload_image(frame_path)
    log.info("free: reclaiming VRAM before Wan")
    comfy.free()
    log.info("animate: %s frames @ %sfps", config.WAN_LENGTH, config.WAN_FPS)
    video_path = animate(frame_name, animation_prompt, seed=seed)
    log.info("animate: %s", video_path)
    return video_path


def generate_keyframe_evaluated(
    prompt: str,
    task: str,
    *,
    seed: int = config.SEED_RANDOM,
    max_attempts: int = 2,
) -> tuple[Path, Verdict]:
    """Generate a keyframe, evaluate it, and retry with the fix up to N times.

    Returns the (path, verdict) of the accepted attempt, or of the final attempt
    if none were accepted — the caller's human gate decides what to do then.
    """
    attempt_prompt = prompt
    kf_path: Path
    verdict: Verdict
    for attempt in range(1, max_attempts + 1):
        # seed 0 = random (each retry differs); a fixed seed is bumped per attempt.
        attempt_seed = seed if seed == config.SEED_RANDOM else seed + attempt - 1
        log.info("keyframe: attempt %d/%d (seed=%s)", attempt, max_attempts, attempt_seed)
        kf_path = keyframe(attempt_prompt, seed=attempt_seed)
        verdict = evaluate_image(kf_path, task)
        if verdict.verdict == "accept":
            return kf_path, verdict
        # Feed the fix back into the prompt for the next attempt (NOTES.md §6).
        attempt_prompt = f"{prompt}. Correction: {verdict.fix_suggestion}".strip()
    log.warning("keyframe: no attempt accepted after %d tries; using last", max_attempts)
    return kf_path, verdict


def _fresh_keyframe(
    prompt: str,
    *,
    seed: int,
    evaluate: bool,
    max_attempts: int,
) -> tuple[Path, Verdict | None]:
    """Build a fresh Klein keyframe, optionally with the evaluate/retry loop."""
    if evaluate:
        return generate_keyframe_evaluated(prompt, prompt, seed=seed, max_attempts=max_attempts)
    log.info("keyframe: generating (seed=%s)", seed)
    kf_path = keyframe(prompt, seed=seed)
    log.info("keyframe: %s", kf_path)
    return kf_path, None


def run_scene(
    keyframe_prompt: str,
    animation_prompt: str,
    master_face: str | Path,
    *,
    seed: int = config.SEED_RANDOM,
    evaluate: bool = False,
    task: str | None = None,
    max_attempts: int = 2,
) -> SceneResult:
    """Run one scene through the full chain and return the artifact paths.

    With ``evaluate=True`` the keyframe is judged by Gemma and retried with the
    fix suggestion (up to ``max_attempts``); ``task`` is the intent to score
    against (defaults to the keyframe prompt).
    """
    master_face = Path(master_face).expanduser()
    if not master_face.is_file():
        raise FileNotFoundError(f"master face not found: {master_face}")
    master_name = comfy.upload_image(master_face)

    if evaluate:
        kf_path, verdict = generate_keyframe_evaluated(
            keyframe_prompt, task or keyframe_prompt, seed=seed, max_attempts=max_attempts
        )
    else:
        kf_path, verdict = _fresh_keyframe(
            keyframe_prompt, seed=seed, evaluate=False, max_attempts=max_attempts
        )

    swapped_path = _swap_master_onto(kf_path, master_name)
    video_path = _animate_frame(swapped_path, animation_prompt, seed=seed)

    return SceneResult(
        video=video_path,
        start_image=swapped_path,
        keyframe=kf_path,
        swapped=swapped_path,
        continued_from_previous=False,
        keyframe_verdict=verdict,
    )


def run_sequence(
    scenes: Sequence[Scene],
    master_face: str | Path,
    *,
    seed: int = config.SEED_RANDOM,
    evaluate: bool = False,
    relock_identity: bool = True,
    max_attempts: int = 2,
) -> list[SceneResult]:
    """Run a sequence of scenes, chaining continuity where requested.

    For each scene: a fresh Klein keyframe (identity-locked to ``master_face``)
    unless the scene is ``continuous`` (and not the first), in which case the
    previous clip's last frame seeds it. Carried frames are re-locked to the
    master face when ``relock_identity`` is set (default), countering Wan drift.

    Continuity is inherently sequential (scene N depends on N-1's video), so this
    diverges from the phase-by-stage batching of NOTES.md §5.
    """
    master_face = Path(master_face).expanduser()
    if not master_face.is_file():
        raise FileNotFoundError(f"master face not found: {master_face}")
    master_name = comfy.upload_image(master_face)

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
            if relock_identity:
                swapped = _swap_master_onto(frame, master_name, label=f"scene {i + 1} relock")
                start_image = swapped
            else:
                swapped = None
                start_image = frame
        else:
            log.info("scene %d/%d: fresh keyframe", i + 1, len(scenes))
            kf_path, verdict = _fresh_keyframe(
                scene.keyframe_prompt, seed=seed, evaluate=evaluate, max_attempts=max_attempts
            )
            swapped = _swap_master_onto(kf_path, master_name, label=f"scene {i + 1} faceswap")
            start_image = swapped

        video_path = _animate_frame(start_image, scene.animation_prompt, seed=seed)
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
