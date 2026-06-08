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
from .steps.animate import animate
from .steps.faceswap import faceswap
from .steps.keyframe import keyframe

log = logging.getLogger("vorch.pipeline")


@dataclass
class SceneResult:
    keyframe: Path
    swapped: Path
    video: Path


def run_scene(
    keyframe_prompt: str,
    animation_prompt: str,
    master_face: str | Path,
    *,
    seed: int = config.SEED_RANDOM,
) -> SceneResult:
    """Run one scene through the full chain and return the artifact paths."""
    master_face = Path(master_face).expanduser()
    if not master_face.is_file():
        raise FileNotFoundError(f"master face not found: {master_face}")

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

    return SceneResult(keyframe=kf_path, swapped=swapped_path, video=video_path)
