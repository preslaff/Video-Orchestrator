"""Registered ComfyUI workflows (built-in + user), for the Setup page and pickers."""

from __future__ import annotations

from fastapi import APIRouter, Query

from .. import registry
from .errors import ApiError

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


@router.get("")
async def list_workflows(kind: str | None = None, reload: bool = Query(False)) -> dict:
    reg = registry.load_registry(force=reload)
    return {
        "workflows": [s.to_dict() for s in reg.list(kind)],
        "errors": dict(reg.errors),
        "kinds": list(registry.KINDS),
        "user_dir": str(registry.config.USER_WORKFLOW_DIR),
    }


@router.get("/{name}")
async def get_workflow(name: str) -> dict:
    reg = registry.load_registry()
    try:
        spec = reg.get(name)
    except KeyError:
        raise ApiError(404, "not_found", f"no such workflow {name!r}")
    d = spec.to_dict()
    d["problems"] = reg.validate(spec)
    return d
