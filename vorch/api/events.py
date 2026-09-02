"""Server-Sent Events: live progress for one job."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from ..queue import JobManager
from .deps import get_manager

router = APIRouter(prefix="/api", tags=["events"])


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, manager: JobManager = Depends(get_manager)):
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
