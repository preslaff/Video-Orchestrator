"""Assemble a job's clips into one video (GET status / POST run)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from .. import assembly as assembly_mod
from ..queue import JobManager
from ..state import Store
from .deps import get_manager, get_store
from .errors import ApiError
from .public import media_url
from .schemas import AssemblySpecIn, AssemblyStatus

router = APIRouter(prefix="/api/jobs", tags=["assembly"])


def _status(store: Store, manager: JobManager, job_id: str) -> AssemblyStatus:
    job = store.load_job(job_id)
    if job is None:
        raise ApiError(404, "not_found", "no such job")
    st = assembly_mod.status(job, active=manager.is_active(job_id))
    for c in st["clips"]:
        c["url"] = media_url(c.get("path"))
    if st["record"]:
        st["record"]["url"] = media_url(st["record"].get("path"))
    return AssemblyStatus.model_validate(st)


@router.get("/{job_id}/assemble", response_model=AssemblyStatus)
async def assembly_status(
    job_id: str, manager: JobManager = Depends(get_manager), store: Store = Depends(get_store)
) -> AssemblyStatus:
    return _status(store, manager, job_id)


@router.post("/{job_id}/assemble", response_model=AssemblyStatus)
async def run_assembly(
    job_id: str,
    spec: AssemblySpecIn | None = None,
    force: bool = Query(False, description="assemble even if the job is not done"),
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> AssemblyStatus:
    body = spec.model_dump(exclude_none=True) if spec is not None else None
    try:
        await manager.assemble(job_id, body, force=force)
    except KeyError:
        raise ApiError(404, "not_found", "no such job")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))
    return _status(store, manager, job_id)
