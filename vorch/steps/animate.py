"""Animation — Wan 2.2 image-to-video (or any `animate`-kind workflow)."""

from __future__ import annotations

from pathlib import Path

from .. import config
from . import run_step

DEFAULT_NEGATIVE = "distorted, morphing, jitter, flicker, extra limbs, warped face"


def animate(
    start_image_name: str,
    prompt: str,
    *,
    workflow: str = "animate",
    negative: str = DEFAULT_NEGATIVE,
    width: int = config.WAN_WIDTH,
    height: int = config.WAN_HEIGHT,
    length: int = config.WAN_LENGTH,
    fps: int = config.WAN_FPS,
    seed: int = config.SEED_RANDOM,
    steps: int | None = None,
    cfg: float | None = None,
    split_step: int | None = None,
    filename_prefix: str | None = None,
    timeout: int | None = None,
) -> Path:
    """Animate a staged keyframe into a video clip; return its output path.

    ``start_image_name`` must already be staged in ComfyUI input/ (the LoadImage
    node reads by name) — use comfy.upload_image() first. ``length`` must be 4n+1;
    the registry validates this. Longer clips scale the workflow timeout.
    """
    seed = config.resolve_seed(seed)  # materialize random seed so it can be recorded
    params = dict(
        start_image=start_image_name, positive=prompt, negative=negative,
        width=width, height=height, length=length, fps=fps,
        seed=seed,  # fans out to both cascade samplers (registry targets)
    )
    for name, value in (("steps", steps), ("cfg", cfg), ("split_step", split_step),
                        ("filename_prefix", filename_prefix)):
        if value is not None:
            params[name] = value
    if timeout is None:
        from .. import registry

        base = registry.load_registry().get(workflow).timeout or config.RUN_TIMEOUT
        timeout = int(base * max(1.0, length / config.WAN_LENGTH))
    return run_step(workflow, timeout=timeout, **params)[0]
