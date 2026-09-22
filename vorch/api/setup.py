"""Setup routes: environment check + effective configuration."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from .. import backends, config, db, envcheck, profiles
from ..state import Store
from .deps import get_store
from .errors import ApiError

router = APIRouter(prefix="/api/setup", tags=["setup"])

# When this process started, against which the code on disk is compared: an edited module is
# only live after a restart, and a UI calling a route this build does not have gets a bare 404.
_STARTED_AT = time.time()


def _code_changed_since_start() -> bool:
    """True when a ``vorch`` module on disk is newer than this process — i.e. restart me.

    A development guard (an installed, unedited checkout never reports true): the frontend and
    the server come from one working tree, so "the file changed after the server started" is
    exactly the mismatch that makes a new route look like a missing feature.
    """
    root = Path(__file__).resolve().parent.parent
    try:
        return any(f.stat().st_mtime > _STARTED_AT for f in root.rglob("*.py"))
    except OSError:
        return False


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


def _backends() -> tuple[list[dict], str | None]:
    # Not force=True: the setup page polls this every 30 s, and forcing re-read the YAML and
    # logged a full backends line on every poll — which reads like a runaway loop in the log.
    # ``load`` already invalidates its cache when the file's mtime or size changes, so edits
    # still show up without a restart; it simply stays quiet when nothing has changed.
    try:
        return [b.to_dict() for b in backends.load().values()], None
    except backends.BackendError as e:
        return [backends.local().to_dict()], str(e)


@router.get("/config")
async def effective_config(store: Store = Depends(get_store)) -> dict:
    known, backends_error = _backends()
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
        "backends_file": str(config.BACKENDS_FILE),
        "backends": known,  # never carries auth headers
        "backends_error": backends_error,
        "started_at": datetime.fromtimestamp(_STARTED_AT, timezone.utc).isoformat(),
        "code_changed_since_start": _code_changed_since_start(),
    }
