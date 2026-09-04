"""Serve generated media (ComfyUI's output dir) at ``/media/<relative path>``.

Starlette's ``StaticFiles`` already does the two things that matter here:
Range requests (video seeking) via ``FileResponse`` and path-traversal
protection (``realpath`` + ``commonpath`` against the root). ``check_dir=False``
lets the app start before ComfyUI has produced anything.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope


class _RevalidatingStaticFiles(StaticFiles):
    """Generated files are rewritten in place (re-assembly, regeneration), so the
    browser must revalidate instead of replaying a cached copy."""

    def file_response(self, *args, **kwargs) -> Response:  # type: ignore[override]
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    async def get_response(self, path: str, scope: Scope) -> Response:
        resp = await super().get_response(path, scope)
        resp.headers.setdefault("Cache-Control", "no-cache")
        return resp


def mount_media(app: FastAPI, directory: Path) -> None:
    app.mount("/media", _RevalidatingStaticFiles(directory=str(directory), check_dir=False), name="media")
