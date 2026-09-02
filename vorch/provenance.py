"""Generation provenance — a sidecar manifest next to each output for reproducibility.

Every generation flows through ``steps.run_step`` with its semantic params (prompts,
seed, dimensions, …) and produces a media file. For each output we write a
``<output>.json`` manifest capturing exactly what produced it — the prompt(s), the
(materialized) seed, the template, and every param — so any frame or clip can be
regenerated later. Best-effort: a failed write logs a warning, never breaks a run.
"""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import registry

log = logging.getLogger("vorch.provenance")

# Run-level context (job id, profile, scene, variant...) stamped into every
# manifest written while it is set — so steps don't need it threaded through.
_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar("vorch_provenance_ctx", default={})


def set_context(**kv) -> contextvars.Token:
    """Merge ``kv`` into the provenance context; returns a token for ``reset_context``."""
    return _ctx.set({**_ctx.get(), **{k: v for k, v in kv.items() if v is not None}})


def reset_context(token: contextvars.Token) -> None:
    _ctx.reset(token)


def context() -> dict:
    return dict(_ctx.get())


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


def write_manifest(
    media_path: str | Path, step: str, params: dict, extra: dict | None = None
) -> Path | None:
    """Write the generation manifest beside ``media_path``; return its path or None."""
    media_path = Path(media_path)
    try:
        spec = registry.load_registry().get(step)
        template, kind = spec.template.name, spec.kind
    except (KeyError, registry.RegistryError):
        template, kind = None, None
    data = {
        "step": step,
        "workflow": step,
        "kind": kind,
        "template": template,
        "output": media_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,  # prompts, seed, width/height/steps/cfg/length/fps, …
        **(extra or {}),
    }
    ctx = context()
    if ctx:
        data["context"] = ctx
    dest = manifest_path(media_path)
    try:
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("provenance: could not write manifest for %s: %s", media_path.name, e)
        return None
    return dest
