"""CORS for the Next.js dev/prod origin.

Normally unnecessary: the Next app proxies ``/api`` and ``/media`` through its
own origin (``next.config`` rewrites). It matters only when the browser talks
to FastAPI directly (``NEXT_PUBLIC_API_ORIGIN``), e.g. if proxied SSE ever
buffers. Origins come from ``VORCH_CORS_ORIGINS`` (comma-separated).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .. import config


def install_cors(app: FastAPI) -> None:
    if not config.CORS_ORIGINS:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.CORS_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges"],
    )
