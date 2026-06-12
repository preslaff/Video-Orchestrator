"""Generation provenance — a sidecar manifest next to each output for reproducibility.

Every generation flows through ``steps.run_step`` with its semantic params (prompts,
seed, dimensions, …) and produces a media file. For each output we write a
``<output>.json`` manifest capturing exactly what produced it — the prompt(s), the
(materialized) seed, the template, and every param — so any frame or clip can be
regenerated later. Best-effort: a failed write logs a warning, never breaks a run.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import workflows

log = logging.getLogger("vorch.provenance")


def manifest_path(media_path: str | Path) -> Path:
    """The sidecar path for a media file: ``foo.png`` -> ``foo.png.json``."""
    media_path = Path(media_path)
    return media_path.with_name(media_path.name + ".json")


def read_manifest(media_path: str | Path) -> dict | None:
    """Load the manifest beside ``media_path``; None if absent/unreadable."""
    try:
        return json.loads(manifest_path(media_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_manifest(media_path: str | Path, step: str, params: dict) -> Path | None:
    """Write the generation manifest beside ``media_path``; return its path or None."""
    media_path = Path(media_path)
    template = workflows.REGISTRY.get(step, {}).get("template")
    data = {
        "step": step,
        "template": template,
        "output": media_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,  # prompts, seed, width/height/steps/cfg/length/fps, …
    }
    dest = manifest_path(media_path)
    try:
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("provenance: could not write manifest for %s: %s", media_path.name, e)
        return None
    return dest
