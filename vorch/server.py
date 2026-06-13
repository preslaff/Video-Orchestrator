"""HTTP API + SSE progress stream (DESIGN §11).

Thin FastAPI layer over ``queue.JobManager``: create/list/get/approve jobs and
stream their live progress as Server-Sent Events. ``register_routes`` attaches the
API to any FastAPI app so the NiceGUI dashboard (app/) can host it in one process;
``create_app`` builds a standalone app for testing or ``vorch serve``.
"""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import config
from .queue import JobManager
from .state import Store, job_to_dict

log = logging.getLogger("vorch.server")


class JobSpec(BaseModel):
    """Request body for creating a job (mirrors the YAML job spec / DESIGN §7)."""

    idea: str
    master_face: str
    style: str = ""
    autonomy: str = "semi"
    n_scenes: int | None = None
    seed: int = 0
    max_attempts: int = 2
    evaluate: bool = True


def register_routes(app: FastAPI, manager: JobManager, store: Store) -> None:
    """Attach the /api routes (and SSE) to an existing FastAPI app."""

    @app.post("/api/jobs")
    async def create_job(spec: JobSpec) -> dict:
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
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return job_to_dict(job)

    @app.get("/api/jobs")
    async def list_jobs() -> list[dict]:
        return [
            {"id": s.id, "status": s.status.value, "idea": s.idea, "updated_at": s.updated_at}
            for s in store.list_jobs()
        ]

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        job = store.load_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job_to_dict(job)

    @app.get("/api/jobs/{job_id}/decisions")
    async def get_decisions(job_id: str) -> list[dict]:
        return [
            {
                "scene_index": d.scene_index, "phase": d.phase, "attempt": d.attempt,
                "action": d.action, "score": d.score, "reason": d.reason,
                "created_at": d.created_at,
            }
            for d in store.decisions(job_id)
        ]

    @app.post("/api/jobs/{job_id}/approve/{gate}")
    async def approve(job_id: str, gate: str) -> dict:
        try:
            job = await manager.approve(job_id, gate)
        except KeyError:
            raise HTTPException(status_code=404, detail="no such job")
        return job_to_dict(job)

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str):
        q = manager.events(job_id)

        async def stream():
            try:
                while True:
                    event = await q.get()
                    yield {"data": json.dumps(event)}
                    if event.get("type") == "end":
                        break
            finally:
                manager.broker.unsubscribe(job_id, q)

        return EventSourceResponse(stream())


def create_app(store: Store | None = None, manager: JobManager | None = None) -> FastAPI:
    """Build a standalone FastAPI app with the job API wired to a JobManager."""
    if store is None:
        config.STATE_DB.parent.mkdir(parents=True, exist_ok=True)
        store = Store(config.STATE_DB)
    manager = manager or JobManager(store)
    app = FastAPI(title="Video Orchestrator")
    app.state.store = store
    app.state.manager = manager
    register_routes(app, manager, store)
    return app
