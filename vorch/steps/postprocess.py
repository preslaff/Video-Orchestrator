"""Post-processing chains — registered ``postprocess`` workflows applied in order.

A profile lists ``post_steps: [{workflow, params}]``; after each clip (or
still) the orchestrator feeds the file through them: stage the current file
into ComfyUI ``input/``, fill ``fps``/``seed`` from the run when the workflow
declares them, run, and hand the output to the next step. A step declaring a
``source_image`` param (video face swap, character replacement) gets the job's
master face, staged on that step's backend. Steps whose input
type does not match the media are skipped (an interpolator on a still), and
VRAM is reclaimed whenever the workflow changes (different models). Each step
stages its input on, and runs on, its own workflow's backend.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from .. import backends, cancel, registry, vram
from ..clients import comfy
from . import run_step

log = logging.getLogger("vorch.postprocess")

AUTO_FILL = ("fps", "seed")
MASTER_PARAM = "source_image"  # filled with the staged master face when the step declares it
# Scene prompts a generative post-step may declare: `positive` (the scene's keyframe prompt —
# character + setting) and `motion` (its animation prompt). Explicit step params win.
PROMPT_PARAMS = ("positive", "motion")


def run_chain(
    path: str | Path,
    chain: Sequence,
    *,
    fps: int | None = None,
    seed: int | None = None,
    filename_prefix: str | None = None,
    master_face: str | Path | None = None,
    prompts: dict[str, str] | None = None,
) -> list[Path]:
    """Apply ``chain`` (objects with ``.workflow`` / ``.params``) to ``path``.

    ``master_face`` (a local image) is staged once per backend and passed as
    ``source_image`` to steps that declare it; ``prompts`` (``positive`` /
    ``motion``) fill the same-named params of generative steps (Wan-Animate-2)
    so the replaced character keeps the scene's setting and action. Returns
    every produced file in order (empty when nothing applied); the final
    deliverable is ``produced[-1]``.
    """
    reg = registry.load_registry()
    current = Path(path)
    produced: list[Path] = []
    prev_workflow: str | None = None
    staged_master: dict[str, str] = {}  # backend name -> uploaded name
    for i, step in enumerate(chain):
        cancel.check()  # a stop between steps; the chain reruns whole on resume
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
        for key in PROMPT_PARAMS:
            text = (prompts or {}).get(key)
            if text and spec.accepts(key) and key not in step.params:
                values[key] = text
        if master_face and spec.accepts(MASTER_PARAM) and MASTER_PARAM not in step.params:
            if backend.name not in staged_master:
                staged_master[backend.name] = comfy.upload_image(master_face, backend=backend)
            values[MASTER_PARAM] = staged_master[backend.name]
        values.update(step.params)
        if filename_prefix and spec.accepts("filename_prefix") and "filename_prefix" not in step.params:
            values["filename_prefix"] = f"{filename_prefix}_{step.workflow}"
        outs = run_step(step.workflow, **values)
        current = outs[0]
        produced.append(current)
        prev_workflow = step.workflow
        log.info("post %s (%d/%d): %s", step.workflow, i + 1, len(chain), current)
    return produced
