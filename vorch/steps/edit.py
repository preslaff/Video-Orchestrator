"""Reference-locked keyframe — Klein identity-edit / KV-edit (kind ``edit``).

The reference images must already be staged in ComfyUI input/ (LoadImage reads
by name); ``pipeline.stage_references`` fits + uploads them. References are
mapped onto the workflow's declared image inputs in order; when fewer are
given than the workflow needs, the last one is repeated (identity_edit takes
two). Size params are sent only when the workflow declares them (KV-edit
derives its output size from the first reference instead).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .. import config, registry
from . import run_step


def edit(
    prompt: str,
    reference_names: Sequence[str],
    *,
    workflow: str = "identity_edit",
    seed: int = config.SEED_RANDOM,
    width: int | None = None,
    height: int | None = None,
    negative: str | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    unet: str | None = None,
    filename_prefix: str | None = None,
    timeout: int | None = None,
) -> Path:
    if not reference_names:
        raise ValueError("edit needs at least one staged reference image")
    spec = registry.load_registry().get(workflow)
    if spec.kind != "edit":
        raise ValueError(f"workflow {workflow!r} is kind {spec.kind!r}, not 'edit'")
    inputs = spec.image_inputs()
    seed = config.resolve_seed(seed)
    params: dict = {"positive": prompt, "seed": seed}
    for i, pname in enumerate(inputs):
        params[pname] = reference_names[min(i, len(reference_names) - 1)]
    for name, value in (("width", width), ("height", height), ("negative", negative),
                        ("steps", steps), ("cfg", cfg), ("unet", unet),
                        ("filename_prefix", filename_prefix)):
        if value is not None and spec.accepts(name):
            params[name] = value
    return run_step(workflow, timeout=timeout, **params)[0]
