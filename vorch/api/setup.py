"""Setup routes: environment check + effective configuration."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from .. import config, db, envcheck, profiles
from ..state import Store
from .deps import get_store
from .errors import ApiError

router = APIRouter(prefix="/api/setup", tags=["setup"])


@router.get("/env")
async def env(
    deep: bool = Query(False, description="also probe Gemma vision (slow: one model call)"),
    force: bool = Query(False, description="bypass the 30s cache"),
    profile_id: str | None = Query(None, description="check this profile's workflows strictly"),
    store: Store = Depends(get_store),
) -> dict:
    profile = None
    if profile_id:
        try:
            profile = profiles.resolve(store, profile_id)
        except KeyError:
            raise ApiError(404, "not_found", f"no such profile {profile_id!r}")
    return envcheck.run_checks(profile=profile, store=store, deep=deep, force=force).to_dict()


def _masked_dsn(dsn: str) -> str:
    try:
        parts = conninfo_to_dict(dsn)
    except Exception:  # noqa: BLE001
        return dsn
    if parts.get("password"):
        parts["password"] = "***"
    return make_conninfo(**parts)


@router.get("/config")
async def effective_config(store: Store = Depends(get_store)) -> dict:
    return {
        "comfy_url": config.COMFY_URL,
        "gemma_url": config.GEMMA_URL,
        "comfy_input": str(config.COMFY_INPUT),
        "comfy_output": str(config.COMFY_OUTPUT),
        "workflow_dir": str(config.WORKFLOW_DIR),
        "user_workflow_dir": str(config.USER_WORKFLOW_DIR),
        "projects_root": str(config.PROJECTS_ROOT),
        "export_root": str(config.EXPORT_ROOT),
        "database_url": _masked_dsn(config.DATABASE_URL),
        "schema_version": store.schema_version(),
        "latest_schema_version": db.latest_version(),
        "ui_url": config.UI_URL,
        "vram_guard_mb": config.VRAM_GUARD_MB,
    }
