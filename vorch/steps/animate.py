"""Animation — Wan 2.2 image-to-video (dual-model cascade)."""

from __future__ import annotations

from pathlib import Path

from .. import config
from . import run_step

DEFAULT_NEGATIVE = "distorted, morphing, jitter, flicker, extra limbs, warped face"


def animate(
    start_image_name: str,
    prompt: str,
    *,
    negative: str = DEFAULT_NEGATIVE,
    width: int = config.WAN_WIDTH,
    height: int = config.WAN_HEIGHT,
    length: int = config.WAN_LENGTH,
    fps: int = config.WAN_FPS,
    seed: int = config.SEED_RANDOM,
    timeout: int | None = None,
) -> Path:
    """Animate a staged keyframe into a video clip; return its output path.

    ``start_image_name`` must already be staged in ComfyUI input/ (the LoadImage
    node reads by name) — use comfy.upload_image() first. ``length`` must be 4n+1;
    the registry validates this.
    """
    seed = config.resolve_seed(seed)  # materialize random seed so it can be recorded
    return run_step(
        "animate",
        timeout=timeout,
        start_image=start_image_name,
        positive=prompt,
        negative=negative,
        width=width,
        height=height,
        length=length,
        fps=fps,
        seed_high=seed,
        seed_low=seed,
    )[0]
