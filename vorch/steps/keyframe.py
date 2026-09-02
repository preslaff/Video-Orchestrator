"""Keyframe generation — FLUX.2 Klein text-to-image (or any `keyframe`-kind workflow)."""

from __future__ import annotations

from pathlib import Path

from .. import config
from . import run_step


def keyframe(
    prompt: str,
    *,
    workflow: str = "keyframe",
    width: int = config.KEYFRAME_WIDTH,
    height: int = config.KEYFRAME_HEIGHT,
    steps: int = config.KEYFRAME_STEPS,
    cfg: float = config.KEYFRAME_CFG,
    seed: int = config.SEED_RANDOM,
    unet: str | None = None,
    negative: str | None = None,
    batch_size: int | None = None,
    filename_prefix: str | None = None,
    timeout: int | None = None,
) -> Path:
    """Generate one keyframe image; return its output path."""
    seed = config.resolve_seed(seed)  # materialize random seed so it can be recorded
    params = dict(
        positive=prompt, width=width, height=height, steps=steps, cfg=cfg, seed=seed
    )
    for name, value in (("unet", unet), ("negative", negative),
                        ("batch_size", batch_size), ("filename_prefix", filename_prefix)):
        if value is not None:
            params[name] = value
    return run_step(workflow, timeout=timeout, **params)[0]
