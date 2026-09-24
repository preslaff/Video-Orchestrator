"""Read-only view of the services vorch depends on, for Setup → Services.

Everything here is a question, never an instruction: what is running, whether it is ours, whether
it is the right instance, who holds the GPU. Starting and stopping over HTTP comes later, on
purpose — the panel should first let a person *see* an 11 GB orphan and a server on the wrong voice
before vorch is given a button that kills things.

Both handlers hand the work to a thread. Probing walks /proc and makes real HTTP requests to other
local servers; doing that on the event loop would stall every SSE stream, which is exactly the
mistake ``speak_scene`` makes today and which the enforcement stage fixes.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from .. import services as services_mod
from .. import supervisor
from .errors import ApiError

router = APIRouter(prefix="/api/services", tags=["services"])


@router.get("")
async def list_services() -> dict:
    """Every service in dependency order, plus the GPU holder table."""
    try:
        return await asyncio.to_thread(supervisor.summary)
    except services_mod.ServiceError as e:
        raise ApiError(400, "bad_request", f"the service manifest is unusable: {e}")


@router.get("/{name}/logs")
async def service_logs(name: str, tail: int = Query(200, ge=1, le=5000)) -> dict:
    """The tail of one service's log file."""
    try:
        spec = services_mod.get(name)
    except services_mod.ServiceError as e:
        raise ApiError(404, "not_found", str(e))
    text = await asyncio.to_thread(supervisor.tail, name, tail)
    return {"name": name, "path": str(spec.log_path), "tail": tail, "text": text}
