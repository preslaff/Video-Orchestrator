"""Workflow registry — the single source of truth for ComfyUI node surgery.

Each ComfyUI workflow in ``workflows/`` is an API-format template: a flat dict
keyed by node ID. The orchestrator loads a template, replaces ONLY the fields
listed here (by node ID), and dispatches it (SKILL.md invariant 2). The rest of
the codebase uses the semantic parameter names below and never touches raw node
IDs like ``"75:73"``.

Node IDs and field names were verified against the actual JSON files; every
target is a literal value (not a link reference), so plain assignment is safe.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from . import config

# semantic name -> (node_id, input_field)
Registry = dict[str, dict[str, Any]]

REGISTRY: Registry = {
    "keyframe": {
        "template": "image_flux2_klein_text_to_image.json",
        "output_node": "9",  # SaveImage
        "params": {
            "positive": ("76", "value"),
            "seed": ("75:73", "noise_seed"),
            "width": ("75:68", "value"),
            "height": ("75:69", "value"),
            "steps": ("75:62", "steps"),
            "cfg": ("75:63", "cfg"),
            "unet": ("75:70", "unet_name"),
        },
    },
    "animate": {
        "template": "wan22_i2v_gguf.json",
        "output_node": "13",  # SaveVideo
        "params": {
            "start_image": ("5", "image"),
            "positive": ("6", "text"),
            "negative": ("7", "text"),
            "width": ("8", "width"),
            "height": ("8", "height"),
            "length": ("8", "length"),
            "seed_high": ("9", "noise_seed"),
            "seed_low": ("10", "noise_seed"),
            "fps": ("12", "fps"),
        },
    },
    "faceswap": {
        "template": "ReActor_Face_Swap.json",
        "output_node": "4",  # SaveImage
        "params": {
            "input_image": ("1", "image"),
            "source_image": ("2", "image"),
            "restore_model": ("3", "face_restore_model"),
        },
    },
}


def load_template(step: str) -> dict:
    """Load a fresh copy of the workflow JSON for ``step``."""
    if step not in REGISTRY:
        raise KeyError(f"Unknown workflow step {step!r}. Known: {sorted(REGISTRY)}")
    path = Path(config.WORKFLOW_DIR) / REGISTRY[step]["template"]
    with open(path) as f:
        return json.load(f)


def apply_params(step: str, wf: dict | None = None, **values: Any) -> dict:
    """Return a workflow dict for ``step`` with ``values`` applied by node ID.

    Pass only the semantic params you want to override; unspecified fields keep
    the template's defaults. Loads a fresh template if ``wf`` is not given.
    """
    spec = REGISTRY[step]
    wf = copy.deepcopy(wf) if wf is not None else load_template(step)

    if step == "animate" and "length" in values:
        _validate_wan_length(values["length"])

    for name, value in values.items():
        if name not in spec["params"]:
            raise KeyError(
                f"{step!r} has no param {name!r}. Known: {sorted(spec['params'])}"
            )
        node_id, field = spec["params"][name]
        wf[node_id]["inputs"][field] = value
    return wf


def output_node(step: str) -> str:
    """The node ID whose history outputs hold the result file(s) for ``step``."""
    return REGISTRY[step]["output_node"]


def _validate_wan_length(length: int) -> None:
    """Wan WanImageToVideo accepts only length = 4n+1 (SKILL.md invariant 3)."""
    if not isinstance(length, int) or length < 1 or length % 4 != 1:
        raise ValueError(
            f"Wan length must be 4n+1 (49, 81, 121, ...), got {length!r}"
        )
