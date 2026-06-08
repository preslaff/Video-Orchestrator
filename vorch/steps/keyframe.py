"""Keyframe generation — FLUX.2 Klein text-to-image."""

from __future__ import annotations

from pathlib import Path

from .. import config
from . import run_step


def keyframe(
    prompt: str,
    *,
    width: int = config.KEYFRAME_WIDTH,
    height: int = config.KEYFRAME_HEIGHT,
    steps: int = config.KEYFRAME_STEPS,
    cfg: float = config.KEYFRAME_CFG,
    seed: int = config.SEED_RANDOM,
    unet: str | None = None,
    timeout: int | None = None,
) -> Path:
    """Generate one keyframe image; return its output path."""
    params = dict(
        positive=prompt, width=width, height=height, steps=steps, cfg=cfg, seed=seed
    )
    if unet is not None:
        params["unet"] = unet
    return run_step("keyframe", timeout=timeout, **params)[0]
