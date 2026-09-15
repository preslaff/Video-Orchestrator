"""Request-scoped accessors for the objects ``register_routes`` stores on ``app.state``."""

from __future__ import annotations

from fastapi import Request

from ..describer import AssetDescriber
from ..queue import JobManager
from ..state import Store


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_manager(request: Request) -> JobManager:
    return request.app.state.manager


def get_describer(request: Request) -> AssetDescriber | None:
    return getattr(request.app.state, "describer", None)
