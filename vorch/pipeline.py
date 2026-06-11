"""Single-scene driver — the Phase-1 walking skeleton (no evaluation).

Proves the ComfyUI chain end-to-end:

    Klein keyframe -> ReActor (swap master face onto keyframe) -> Wan animate

Each output is staged into ComfyUI input/ before the next step's LoadImage reads
it, and VRAM is reclaimed before the heavy Wan phase. The evaluator, retry loop,
state persistence, and multi-scene phasing are Phase 2 (DESIGN.md §9/§11).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from . import config
from .clients import comfy
from .evaluator import Verdict, evaluate_image
from .steps.animate import animate
from .steps.faceswap import faceswap
from .steps.keyframe import keyframe

log = logging.getLogger("vorch.pipeline")


@dataclass
class SceneResult:
    keyframe: Path
    swapped: Path
    video: Path
    keyframe_verdict: Verdict | None = None


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

    verdict: Verdict | None = None
    if evaluate:
        kf_path, verdict = generate_keyframe_evaluated(
            keyframe_prompt, task or keyframe_prompt, seed=seed, max_attempts=max_attempts
        )
    else:
        log.info("keyframe: generating (seed=%s)", seed)
        kf_path = keyframe(keyframe_prompt, seed=seed)
    log.info("keyframe: %s", kf_path)

    kf_name = comfy.upload_image(kf_path)
    master_name = comfy.upload_image(master_face)

    log.info("faceswap: imposing master face")
    swapped_path = faceswap(kf_name, master_name)
    log.info("faceswap: %s", swapped_path)

    swapped_name = comfy.upload_image(swapped_path)

    log.info("free: reclaiming VRAM before Wan")
    comfy.free()

    log.info("animate: %s frames @ %sfps", config.WAN_LENGTH, config.WAN_FPS)
    video_path = animate(swapped_name, animation_prompt, seed=seed)
    log.info("animate: %s", video_path)

    return SceneResult(
        keyframe=kf_path,
        swapped=swapped_path,
        video=video_path,
        keyframe_verdict=verdict,
    )
