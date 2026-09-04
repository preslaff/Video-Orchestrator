"""Pipeline steps — thin wrappers that turn semantic params into a ComfyUI run.

Each step: look the workflow up in the registry -> apply params by node id ->
dispatch -> select the declared output files -> write provenance -> return paths.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import backends, provenance, registry
from ..clients import comfy

log = logging.getLogger("vorch.steps")


def run_step(step: str, *, timeout: int | None = None, **params) -> list[Path]:
    """Apply ``params`` to workflow ``step``, run it, return its output file paths.

    Params the workflow does not declare are dropped (with a warning) when the
    kind contract lists them as optional, so a leaner user-installed workflow
    still runs. The run goes to the workflow's declared backend; a remote
    backend's outputs are downloaded into the local output layout. Writes a
    provenance manifest (incl. ``backend``) beside each output.
    """
    reg = registry.load_registry()
    spec = reg.get(step)
    wf = reg.apply(step, strict=False, **params)
    if timeout is None:
        timeout = spec.timeout
    backend = backends.for_spec(spec)
    descriptors = _select_outputs(spec, comfy.run(wf, timeout=timeout, backend=backend))
    if not descriptors:
        raise comfy.ComfyError(f"Step {step!r} produced no output files on {backend.name}")
    paths = [comfy.output_path(d, backend) for d in descriptors]
    extra = {"workflow": spec.name, "kind": spec.kind, "template": spec.template.name,
             "backend": backend.name, "gpu": backend.gpu}
    for p in paths:
        provenance.write_manifest(p, step, params, extra)
    return paths


def _select_outputs(spec: registry.WorkflowSpec, descriptors: list[dict]) -> list[dict]:
    """Keep the files from the declared output nodes / media type (fallback: all)."""
    chosen = descriptors
    if spec.output_nodes:
        by_node = [d for d in chosen if d.get("node_id") in spec.output_nodes]
        if by_node:
            chosen = by_node
    if spec.output_type:
        def is_video(d: dict) -> bool:
            return Path(d.get("filename", "")).suffix.lower() in registry.VIDEO_EXT
        want_video = spec.output_type == "video"
        by_type = [d for d in chosen if is_video(d) == want_video]
        if by_type:
            chosen = by_type
    return chosen
