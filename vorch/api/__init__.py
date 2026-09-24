"""HTTP API + SSE progress stream (DESIGN §11).

Thin FastAPI layer over ``queue.JobManager``. ``register_routes`` attaches every
router to an existing FastAPI app (the NiceGUI dashboard hosts it in-process);
``create_app`` builds a standalone app for ``vorch serve`` and the tests.
Handlers reach the Store / JobManager through ``app.state`` (see ``deps.py``).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from .. import __version__, config, logsetup
from ..describer import AssetDescriber
from ..queue import JobManager
from ..state import Store
from . import assembly, events, export, jobs, profiles, projects, services, setup, workflows
from .cors import install_cors
from .errors import ApiError, install_error_handlers
from .media import mount_media
from .schemas import JobSpec, SceneEdit, SceneEdits

__all__ = ["ApiError", "JobSpec", "SceneEdit", "SceneEdits", "create_app", "openapi_schema", "register_routes"]

log = logging.getLogger("vorch.api")


def register_routes(app: FastAPI, manager: JobManager, store: Store,
                    describer: AssetDescriber | None = None) -> None:
    """Attach the /api routes (and SSE) to an existing FastAPI app."""
    app.state.store = store
    app.state.manager = manager
    app.state.describer = describer
    app.include_router(jobs.router)
    app.include_router(events.router)
    app.include_router(export.router)
    app.include_router(assembly.router)
    app.include_router(setup.router)
    app.include_router(workflows.router)
    app.include_router(profiles.router)
    app.include_router(projects.router)
    app.include_router(services.router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {"service": "vorch", "ui": config.UI_URL, "api": "/api", "docs": "/docs"}


def openapi_schema() -> dict:
    """The API's OpenAPI document without touching the database (for `vorch openapi`)."""
    app = FastAPI(title="Video Orchestrator")
    register_routes(app, None, None)  # type: ignore[arg-type]  # handlers never run
    return app.openapi()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Reap warm services this process started, on a timer and again on the way out.

    The reaper is what keeps ``lifetime: warm`` honest: the narration server stays up while you are
    writing lines, and gives the card back on its own once you stop. Only ever a server *vorch*
    started — one launched by hand is its owner's to close.
    """
    from .. import supervisor

    supervisor.install_exit_handlers()

    async def reaper() -> None:
        while True:
            await asyncio.sleep(config.REAP_INTERVAL_SEC)
            try:
                await asyncio.to_thread(supervisor.reap_idle)
            except Exception:                       # a housekeeping task must never die
                log.exception("idle reaper failed")

    task = asyncio.create_task(reaper())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        await asyncio.to_thread(supervisor._stop_ours)


def create_app(
    store: Store | None = None,
    manager: JobManager | None = None,
    *,
    media_dir: Path | None = None,
    describer: AssetDescriber | None = None,
) -> FastAPI:
    """Build a standalone FastAPI app: job API + SSE + ``/media`` static files.

    ``media_dir`` defaults to ComfyUI's output directory (``config.COMFY_OUTPUT``).
    """
    # A no-op when the CLI already configured logging; the point is the other entry points —
    # `uvicorn vorch.api:app`, an embedded app, a notebook — which otherwise get nothing and
    # silently drop every INFO line the orchestrator writes about what it decided.
    logsetup.configure()
    if store is None:
        store = Store.open()
    if manager is None:
        from ..envcheck import gate

        manager = JobManager(store, env_check=lambda profile, project: gate(profile, project, store=store))
    app = FastAPI(title="Video Orchestrator", lifespan=_lifespan)
    install_cors(app)
    install_error_handlers(app)
    if describer is None:
        describer = AssetDescriber(store, busy=manager.busy)
    register_routes(app, manager, store, describer)
    media_dir = media_dir or config.COMFY_OUTPUT
    if not media_dir.exists():
        log.warning("media dir %s does not exist yet; /media will 404 until ComfyUI writes there", media_dir)
    mount_media(app, media_dir)
    return app
