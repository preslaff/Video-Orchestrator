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


def mount_media(app: FastAPI, directory: Path) -> None:
    app.mount("/media", StaticFiles(directory=str(directory), check_dir=False), name="media")
