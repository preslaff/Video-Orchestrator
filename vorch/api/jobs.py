"""Job routes: create / list / get / delete / decisions / provenance / edit scenes / approve."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..envcheck import EnvironmentNotReady
from ..queue import JobManager
from ..state import JobStatus, Store
from .deps import get_manager, get_store
from .errors import ApiError
from .public import job_to_public, provenance_rows
from .schemas import (
    DecisionOut, Gate, JobPublic, JobSpec, JobSummaryOut, Page, ProvenanceOut, SceneEdits, SceneRegenerate,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _load(store: Store, job_id: str):
    job = store.load_job(job_id)
    if job is None:
        raise ApiError(404, "not_found", "no such job")
    return job


@router.post("", response_model=JobPublic)
async def create_job(
    spec: JobSpec,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    try:
        job = await manager.submit(
            spec.idea,
            style=spec.style,
            master_face=spec.master_face,
            autonomy=spec.autonomy,
            n_scenes=spec.n_scenes,
            seed=spec.seed,
            max_attempts=spec.max_attempts,
            evaluate=spec.evaluate,
            face_swap=spec.face_swap,
            project_id=spec.project_id,
            profile_id=spec.profile_id,
            profile_overrides=spec.profile_overrides,
            mode=spec.mode,
            reference_asset_ids=spec.reference_asset_ids,
            reference_images=spec.reference_images,
            force=spec.force,
        )
    except EnvironmentNotReady as e:
        raise ApiError(409, "environment_not_ready", str(e), details=e.report.to_dict())
    except (FileNotFoundError, KeyError, ValueError) as e:
        raise ApiError(400, "bad_request", str(e).strip("'"))
    return job_to_public(job, store)


@router.get("", response_model=Page[JobSummaryOut])
async def list_jobs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: JobStatus | None = None,
    project_id: str | None = None,
    store: Store = Depends(get_store),
) -> Page[JobSummaryOut]:
    items = [
        JobSummaryOut(id=s.id, status=s.status.value, idea=s.idea, updated_at=s.updated_at,
                      project_id=s.project_id, kind=s.kind, profile_id=s.profile_id, profile_name=s.profile_name)
        for s in store.list_jobs(limit=limit, offset=offset, status=status, project_id=project_id)
    ]
    total = store.count_jobs(status=status, project_id=project_id)
    return Page(items=items, total=total, limit=limit, offset=offset)


@router.get("/{job_id}", response_model=JobPublic)
async def get_job(job_id: str, store: Store = Depends(get_store)) -> JobPublic:
    return job_to_public(_load(store, job_id), store)


@router.delete("/{job_id}", status_code=204)
async def delete_job(
    job_id: str,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> None:
    _load(store, job_id)
    if manager.is_active(job_id):
        raise ApiError(409, "conflict", f"job {job_id} is running")
    store.delete_job(job_id)


@router.get("/{job_id}/decisions", response_model=list[DecisionOut])
async def get_decisions(job_id: str, store: Store = Depends(get_store)) -> list[DecisionOut]:
    _load(store, job_id)
    return [
        DecisionOut(
            scene_index=d.scene_index, phase=d.phase, attempt=d.attempt,
            action=d.action, score=d.score, reason=d.reason, created_at=d.created_at,
        )
        for d in store.decisions(job_id)
    ]


@router.get("/{job_id}/provenance", response_model=list[ProvenanceOut])
async def get_provenance(job_id: str, store: Store = Depends(get_store)) -> list[ProvenanceOut]:
    return provenance_rows(_load(store, job_id))


@router.put("/{job_id}/scenes", response_model=JobPublic)
async def update_scenes(
    job_id: str,
    edits: SceneEdits,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    try:
        job = manager.update_scenes(job_id, [e.model_dump() for e in edits.scenes])
    except KeyError:
        raise ApiError(404, "not_found", "no such job")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))
    return job_to_public(job, store)


@router.post("/{job_id}/scenes/{index}/regenerate", response_model=JobPublic)
async def regenerate_scene(
    job_id: str,
    index: int,
    body: SceneRegenerate,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    _load(store, job_id)
    edits = body.model_dump(exclude={"what"}, exclude_none=True)
    try:
        job = await manager.regenerate(job_id, index, what=body.what, edits=edits)
    except KeyError as e:
        raise ApiError(404, "not_found", f"no such {str(e).strip(chr(39))}")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))
    return job_to_public(job, store)


@router.post("/{job_id}/retry", response_model=JobPublic)
async def retry(
    job_id: str,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    """Resume a failed job from where it stopped (e.g. after fixing the environment)."""
    _load(store, job_id)
    try:
        job = await manager.retry(job_id)
    except KeyError:
        raise ApiError(404, "not_found", "no such job")
    except RuntimeError as e:
        raise ApiError(409, "conflict", str(e))
    return job_to_public(job, store)


@router.post("/{job_id}/approve/{gate}", response_model=JobPublic)
async def approve(
    job_id: str,
    gate: Gate,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
) -> JobPublic:
    job = _load(store, job_id)
    if job.status is not JobStatus.NEEDS_APPROVAL:
        raise ApiError(409, "conflict", f"job is {job.status.value}, not awaiting approval")
    try:
        job = await manager.approve(job_id, gate)
    except KeyError:
        raise ApiError(404, "not_found", "no such job")
    return job_to_public(job, store)
