"""Keyframe generation — FLUX.2 Klein text-to-image (or any `keyframe`-kind workflow).

Only ``positive`` and ``seed`` are sent unconditionally; every other param goes out
only when the workflow declares it (``spec.accepts``), exactly as ``steps.edit`` does.
Local samplers take literal ``width``/``height``, but a hosted API node often expresses
size as an enum instead (``size``, ``aspect_ratio``) and has no sampler at all, so
sending ints it never declared would mean a KeyError or a silently dropped param.
"""

from __future__ import annotations

from pathlib import Path

from .. import config, registry
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
    aspect: str | None = None,
    unet: str | None = None,
    negative: str | None = None,
    batch_size: int | None = None,
    filename_prefix: str | None = None,
    timeout: int | None = None,
) -> Path:
    """Generate one keyframe image; return its output path."""
    spec = registry.load_registry().get(workflow)
    # A workflow whose size is an enum renders its template default when no aspect arrives —
    # a 9:16 job would come back square with no error at all. Refuse instead.
    ap = spec.params.get("aspect")
    if aspect is None and ap is not None and ap.from_aspect:
        raise ValueError(
            f"workflow {workflow!r} takes its size from the aspect, so `aspect` is required "
            f"(it offers {', '.join(sorted(ap.from_aspect))})"
        )
    seed = config.resolve_seed(seed)  # materialize random seed so it can be recorded
    params: dict = {"positive": prompt, "seed": seed}
    for name, value in (("width", width), ("height", height), ("aspect", aspect),
                        ("steps", steps), ("cfg", cfg),
                        ("unet", unet), ("negative", negative),
                        ("batch_size", batch_size), ("filename_prefix", filename_prefix)):
        if value is not None and spec.accepts(name):
            params[name] = value
    return run_step(workflow, timeout=timeout, **params)[0]
