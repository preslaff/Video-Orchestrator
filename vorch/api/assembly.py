"""Assemble a job's clips into one video (GET status / POST run)."""

from __future__ import annotations

from pathlib import Path

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
    for v in st.get("versions", []):
        v["url"] = _busted(media_url(v.get("path")), v)
    if st["record"]:
        url = media_url(st["record"].get("path"))
        # Same path every time -> browsers would replay the previous assembly from
        # cache; the creation time makes each assembly a distinct URL.
        st["record"]["url"] = _busted(url, st["record"])
    return AssemblyStatus.model_validate(st)


def _busted(url: str | None, rec: dict) -> str | None:
    stamp = "".join(ch for ch in (rec.get("created_at") or "") if ch.isdigit())
    if not stamp:
        try:
            info = Path(rec["path"]).stat()
            stamp = f"{int(info.st_mtime)}{info.st_size}"
        except (OSError, KeyError):
            stamp = ""
    return f"{url}?v={stamp}" if url and stamp else url


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


@router.post("/{job_id}/assemble/versions/{version}/select", response_model=AssemblyStatus)
async def select_version(
    job_id: str, version: int, manager: JobManager = Depends(get_manager), store: Store = Depends(get_store)
) -> AssemblyStatus:
    try:
        manager.select_assembly_version(job_id, version)
    except KeyError as e:
        raise ApiError(404, "not_found", str(e).strip("'"))
    except FileNotFoundError as e:
        raise ApiError(409, "conflict", str(e))
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    return _status(store, manager, job_id)


@router.delete("/{job_id}/assemble/versions/{version}", response_model=AssemblyStatus)
async def delete_version(
    job_id: str, version: int, manager: JobManager = Depends(get_manager), store: Store = Depends(get_store)
) -> AssemblyStatus:
    try:
        manager.delete_assembly_version(job_id, version)
    except KeyError as e:
        raise ApiError(404, "not_found", str(e).strip("'"))
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    return _status(store, manager, job_id)
