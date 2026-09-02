"""Compatibility shim: the HTTP API now lives in ``vorch.api``.

Kept so ``from vorch.server import create_app`` (tests, dashboard) keeps working
until the NiceGUI dashboard is removed.
"""

from __future__ import annotations

from .api import JobSpec, SceneEdit, SceneEdits, create_app, register_routes

__all__ = ["JobSpec", "SceneEdit", "SceneEdits", "create_app", "register_routes"]
