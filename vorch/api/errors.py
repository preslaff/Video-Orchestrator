"""Uniform JSON error envelope: ``{"error": {"code", "message", "details"}}``."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

_DEFAULT_CODES = {
    400: "bad_request",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    422: "validation_error",
    500: "internal",
}


class ApiError(HTTPException):
    """HTTPException with a stable machine-readable ``code``."""

    def __init__(
        self, status_code: int, code: str, message: str, details: Any | None = None
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.details = details


def envelope(status_code: int, code: str, message: str, details: Any | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details}},
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http(request, exc: StarletteHTTPException):
        code = getattr(exc, "code", None) or _DEFAULT_CODES.get(exc.status_code, "http_error")
        resp = envelope(exc.status_code, code, str(exc.detail), getattr(exc, "details", None))
        if exc.headers:
            resp.headers.update(exc.headers)
        return resp

    @app.exception_handler(RequestValidationError)
    async def _validation(request, exc: RequestValidationError):
        return envelope(422, "validation_error", "invalid request", jsonable_encoder(exc.errors()))
