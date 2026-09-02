"""Job routes: create / list / get / decisions / edit scenes / approve."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..queue import JobManager
from ..state import Store, job_to_dict
from .deps import get_manager, get_store
from .schemas import JobSpec, SceneEdits

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("")
async def create_job(spec: JobSpec, manager: JobManager = Depends(get_manager)) -> dict:
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
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return job_to_dict(job)


@router.get("")
async def list_jobs(store: Store = Depends(get_store)) -> list[dict]:
    return [
        {"id": s.id, "status": s.status.value, "idea": s.idea, "updated_at": s.updated_at}
        for s in store.list_jobs()
    ]


@router.get("/{job_id}")
async def get_job(job_id: str, store: Store = Depends(get_store)) -> dict:
    job = store.load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return job_to_dict(job)


@router.get("/{job_id}/decisions")
async def get_decisions(job_id: str, store: Store = Depends(get_store)) -> list[dict]:
    return [
        {
            "scene_index": d.scene_index, "phase": d.phase, "attempt": d.attempt,
            "action": d.action, "score": d.score, "reason": d.reason,
            "created_at": d.created_at,
        }
        for d in store.decisions(job_id)
    ]


@router.put("/{job_id}/scenes")
async def update_scenes(
    job_id: str, edits: SceneEdits, manager: JobManager = Depends(get_manager)
) -> dict:
    try:
        job = manager.update_scenes(job_id, [e.model_dump() for e in edits.scenes])
    except KeyError:
        raise HTTPException(status_code=404, detail="no such job")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return job_to_dict(job)


@router.post("/{job_id}/approve/{gate}")
async def approve(job_id: str, gate: str, manager: JobManager = Depends(get_manager)) -> dict:
    try:
        job = await manager.approve(job_id, gate)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such job")
    return job_to_dict(job)
