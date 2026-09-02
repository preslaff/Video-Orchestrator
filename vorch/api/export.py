"""Handoff export: run / inspect the per-job deliverables folder."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from .. import export as export_mod
from ..queue import JobManager
from ..state import Store
from .deps import get_manager, get_store
from .errors import ApiError

router = APIRouter(prefix="/api/jobs", tags=["export"])


@router.post("/{job_id}/export")
async def run_export(
    job_id: str,
    force: bool = Query(False, description="export even if the job is not done"),
    manager: JobManager = Depends(get_manager),
) -> dict:
    try:
        rec = manager.export_now(job_id, force=force)
    except KeyError:
        raise ApiError(404, "not_found", "no such job")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    return rec.to_dict()


@router.get("/{job_id}/export")
async def export_status(job_id: str, store: Store = Depends(get_store)) -> dict:
    job = store.load_job(job_id)
    if job is None:
        raise ApiError(404, "not_found", "no such job")
    return export_mod.status(job)
