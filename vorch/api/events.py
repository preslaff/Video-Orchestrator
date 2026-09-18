"""Server-Sent Events: live progress for one job, or for all jobs.

Events are named (``event: <type>``) and carry a per-job ``id`` so a browser
reconnect (``Last-Event-ID``) or an explicit ``?after=N`` replays what was
missed from the Broker's ring buffer. A terminal job with nothing to replay
gets a synthetic ``end`` so the client stops instead of reconnecting forever.

The ring buffer holds **every** segment of a job, so a replay can contain the
``end`` of an earlier segment. Each stream therefore opens with an ``open``
event carrying ``last_id`` — the watermark where the replay stops and live
events begin — so a client can tell "this segment finished" from "a previous
segment finished, keep listening". Without it a client stops at the first
replayed ``end`` and silently misses everything a new run emits.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from ..queue import JobManager
from ..state import JobStatus, Store
from .deps import get_manager, get_store
from .errors import ApiError

router = APIRouter(prefix="/api", tags=["events"])

PING_SECONDS = 15
_TERMINAL = {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED}


def _after(request: Request) -> int:
    raw = request.headers.get("last-event-id") or request.query_params.get("after") or "0"
    try:
        return int(raw)
    except ValueError:
        raise ApiError(400, "bad_request", "Last-Event-ID / after must be an integer")


def _sse(ev: dict) -> ServerSentEvent:
    return ServerSentEvent(
        data=json.dumps(ev), event=ev.get("type"), id=str(ev["id"]) if "id" in ev else None
    )


@router.get("/jobs/{job_id}/events")
async def job_events(
    job_id: str,
    request: Request,
    manager: JobManager = Depends(get_manager),
    store: Store = Depends(get_store),
):
    job = store.load_job(job_id)
    if job is None:
        raise ApiError(404, "not_found", "no such job")
    after = _after(request)
    watermark = manager.broker.last_id(job_id)
    q = manager.events(job_id, after=after)
    if (
        job.status in _TERMINAL
        and not manager.is_active(job_id)  # the run's own `end` is still coming
        and watermark <= after
    ):
        # Nothing buffered beyond what the client has (e.g. after a restart).
        q.put_nowait({"job_id": job_id, "type": "end", "id": after, "final": True})
    # `last_id` says where the replay ends and `active` whether a segment is running now.
    # While one is, a buffered `end` is an earlier segment's history and the stream stays open
    # for the live one; with nothing running, a replayed `end` is still the end of the story.
    active = manager.is_active(job_id)
    opening = {"job_id": job_id, "type": "open", "last_id": watermark,
               "active": active, "status": job.status.value}

    def terminal(ev: dict) -> bool:
        return ev.get("type") == "end" and bool(
            ev.get("final") or not active or ev.get("id", 0) > watermark
        )

    async def stream():
        yield _sse(opening)
        try:
            while True:
                ev = await q.get()
                yield _sse(ev)
                if terminal(ev):
                    break
        finally:
            manager.broker.unsubscribe(job_id, q)

    return EventSourceResponse(stream(), ping=PING_SECONDS)


@router.get("/events")
async def all_events(manager: JobManager = Depends(get_manager)):
    """Every job's events, for the jobs list page. Ids are per job, so none is sent."""
    q = manager.broker.subscribe_all()

    async def stream():
        try:
            while True:
                ev = await q.get()
                yield ServerSentEvent(data=json.dumps(ev), event=ev.get("type"))
        finally:
            manager.broker.unsubscribe_all(q)

    return EventSourceResponse(stream(), ping=PING_SECONDS)
