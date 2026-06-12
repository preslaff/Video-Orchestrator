"""Pipeline steps — thin wrappers that turn semantic params into a ComfyUI run.

Each step: build the workflow via the registry -> dispatch -> return output path(s).
"""

from __future__ import annotations

from pathlib import Path

from .. import provenance, workflows
from ..clients import comfy


def run_step(step: str, *, timeout: int | None = None, **params) -> list[Path]:
    """Apply ``params`` to ``step``'s template, run it, return output file paths.

    Writes a provenance manifest (prompts + seed + params) beside each output for
    reproducibility — see ``provenance.write_manifest``.
    """
    wf = workflows.apply_params(step, **params)
    descriptors = comfy.run(wf, timeout=timeout)
    if not descriptors:
        raise comfy.ComfyError(f"Step {step!r} produced no output files")
    paths = [comfy.output_path(d) for d in descriptors]
    for p in paths:
        provenance.write_manifest(p, step, params)
    return paths
