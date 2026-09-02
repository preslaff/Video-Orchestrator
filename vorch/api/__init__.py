"""HTTP API + SSE progress stream (DESIGN §11).

Thin FastAPI layer over ``queue.JobManager``. ``register_routes`` attaches every
router to an existing FastAPI app (the NiceGUI dashboard hosts it in-process);
``create_app`` builds a standalone app for ``vorch serve`` and the tests.
Handlers reach the Store / JobManager through ``app.state`` (see ``deps.py``).
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from ..queue import JobManager
from ..state import Store
from . import events, jobs
from .errors import ApiError, install_error_handlers
from .schemas import JobSpec, SceneEdit, SceneEdits

__all__ = ["ApiError", "JobSpec", "SceneEdit", "SceneEdits", "create_app", "register_routes"]

log = logging.getLogger("vorch.api")


def register_routes(app: FastAPI, manager: JobManager, store: Store) -> None:
    """Attach the /api routes (and SSE) to an existing FastAPI app."""
    app.state.store = store
    app.state.manager = manager
    app.include_router(jobs.router)
    app.include_router(events.router)


def create_app(store: Store | None = None, manager: JobManager | None = None) -> FastAPI:
    """Build a standalone FastAPI app with the job API wired to a JobManager."""
    if store is None:
        store = Store.open()
    manager = manager or JobManager(store)
    app = FastAPI(title="Video Orchestrator")
    install_error_handlers(app)
    register_routes(app, manager, store)
    return app
