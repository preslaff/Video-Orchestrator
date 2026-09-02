"""HTTP API + SSE progress stream (DESIGN §11).

Thin FastAPI layer over ``queue.JobManager``. ``register_routes`` attaches every
router to an existing FastAPI app (the NiceGUI dashboard hosts it in-process);
``create_app`` builds a standalone app for ``vorch serve`` and the tests.
Handlers reach the Store / JobManager through ``app.state`` (see ``deps.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from .. import __version__, config
from ..queue import JobManager
from ..state import Store
from . import events, export, jobs, profiles, projects, setup, workflows
from .cors import install_cors
from .errors import ApiError, install_error_handlers
from .media import mount_media
from .schemas import JobSpec, SceneEdit, SceneEdits

__all__ = ["ApiError", "JobSpec", "SceneEdit", "SceneEdits", "create_app", "openapi_schema", "register_routes"]

log = logging.getLogger("vorch.api")


def register_routes(app: FastAPI, manager: JobManager, store: Store) -> None:
    """Attach the /api routes (and SSE) to an existing FastAPI app."""
    app.state.store = store
    app.state.manager = manager
    app.include_router(jobs.router)
    app.include_router(events.router)
    app.include_router(export.router)
    app.include_router(setup.router)
    app.include_router(workflows.router)
    app.include_router(profiles.router)
    app.include_router(projects.router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"service": "vorch", "ui": config.UI_URL, "api": "/api", "docs": "/docs"}


def openapi_schema() -> dict:
    """The API's OpenAPI document without touching the database (for `vorch openapi`)."""
    app = FastAPI(title="Video Orchestrator")
    register_routes(app, None, None)  # type: ignore[arg-type]  # handlers never run
    return app.openapi()


def create_app(
    store: Store | None = None,
    manager: JobManager | None = None,
    *,
    media_dir: Path | None = None,
) -> FastAPI:
    """Build a standalone FastAPI app: job API + SSE + ``/media`` static files.

    ``media_dir`` defaults to ComfyUI's output directory (``config.COMFY_OUTPUT``).
    """
    if store is None:
        store = Store.open()
    if manager is None:
        from ..envcheck import gate

        manager = JobManager(store, env_check=lambda profile, project: gate(profile, project, store=store))
    app = FastAPI(title="Video Orchestrator")
    install_cors(app)
    install_error_handlers(app)
    register_routes(app, manager, store)
    media_dir = media_dir or config.COMFY_OUTPUT
    if not media_dir.exists():
        log.warning("media dir %s does not exist yet; /media will 404 until ComfyUI writes there", media_dir)
    mount_media(app, media_dir)
    return app
