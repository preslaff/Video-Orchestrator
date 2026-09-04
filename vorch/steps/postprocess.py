"""Post-processing chains — registered ``postprocess`` workflows applied in order.

A profile lists ``post_steps: [{workflow, params}]``; after each clip (or
still) the orchestrator feeds the file through them: stage the current file
into ComfyUI ``input/``, fill ``fps``/``seed`` from the run when the workflow
declares them, run, and hand the output to the next step. Steps whose input
type does not match the media are skipped (an interpolator on a still), and
VRAM is reclaimed whenever the workflow changes (different models). Each step
stages its input on, and runs on, its own workflow's backend.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from .. import backends, registry, vram
from ..clients import comfy
from . import run_step

log = logging.getLogger("vorch.postprocess")

AUTO_FILL = ("fps", "seed")


def run_chain(
    path: str | Path,
    chain: Sequence,
    *,
    fps: int | None = None,
    seed: int | None = None,
    filename_prefix: str | None = None,
) -> list[Path]:
    """Apply ``chain`` (objects with ``.workflow`` / ``.params``) to ``path``.

    Returns every produced file in order (empty when nothing applied); the
    final deliverable is ``produced[-1]`` if any.
    """
    reg = registry.load_registry()
    current = Path(path)
    produced: list[Path] = []
    prev_workflow: str | None = None
    for i, step in enumerate(chain):
        spec = reg.get(step.workflow)
        if spec.kind != "postprocess":
            raise ValueError(f"post step {step.workflow!r} is kind {spec.kind!r}, not 'postprocess'")
        is_video = current.suffix.lower() in registry.VIDEO_EXT
        inputs = spec.video_inputs() if is_video else spec.image_inputs()
        if not inputs:
            log.info("post %s: skipped (no %s input)", step.workflow, "video" if is_video else "image")
            continue
        backend = backends.for_spec(spec)
        if step.workflow != prev_workflow:
            vram.reclaim(backend=backend)
            if backend.is_local:
                vram.log_state(f"before {step.workflow}")
        name = comfy.upload_image(current, backend=backend)  # /upload/image stages any file
        values: dict = {inputs[0]: name}
        auto = {"fps": fps, "seed": seed}
        for key in AUTO_FILL:
            if auto[key] is not None and spec.accepts(key) and key not in step.params:
                values[key] = auto[key]
        values.update(step.params)
        if filename_prefix and spec.accepts("filename_prefix") and "filename_prefix" not in step.params:
            values["filename_prefix"] = f"{filename_prefix}_{step.workflow}"
        outs = run_step(step.workflow, **values)
        current = outs[0]
        produced.append(current)
        prev_workflow = step.workflow
        log.info("post %s (%d/%d): %s", step.workflow, i + 1, len(chain), current)
    return produced
