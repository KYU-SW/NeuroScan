"""API errors: stable codes, HTTP status, user-safe messages (no paths, no tracebacks)."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

LOGGER = logging.getLogger("neuroscan_api")


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.status, self.code, self.message = status, code, message
        self.details, self.retryable, self.headers = details or {}, retryable, headers


# Phase 5A engine codes -> (HTTP-equivalent status, retryable). Engine failures are asynchronous:
# they appear as run.status = failed with error.code; this table documents their meaning.
ENGINE_CODES: dict[str, tuple[int, bool]] = {
    **{
        c: (422, False)
        for c in (
            "E_INVALID_ARGUMENT",
            "E_INPUT_FORMAT",
            "E_INPUT_CORRUPT",
            "E_INPUT_NOT_3D",
            "E_GRID_MISMATCH",
            "E_NONFINITE",
            "E_EMPTY_IMAGE",
            "E_SHAPE_TOO_SMALL",
            "E_SPACING_INVALID",
            "E_BRAIN_MASK",
            "E_PREPROCESS",
            "E_STRICT_WARNING",
        )
    },
    "E_SEALED_TEST_CASE": (403, False),
    "E_INPUT_NOT_FOUND": (500, False),
    "E_CHECKPOINT": (503, False),
    "E_CONFIG_MISMATCH": (503, False),
    **{
        c: (503, True)
        for c in (
            "E_DEVICE",
            "E_INFERENCE_OOM",
            "E_INTERRUPTED",
            "E_WORKER_LOST",
            "E_WORKER_RESTARTED",
            "E_TIMEOUT",
            "E_MODEL_RELOAD_FAILED",
        )
    },
    **{
        c: (500, False)
        for c in (
            "E_INFERENCE",
            "E_INTERNAL",
            "E_OUTPUT_VERIFY",
            "E_OUTPUT_EXISTS",
            "E_REGISTRATION",
        )
    },
    "E_OUTPUT_WRITE": (507, True),
}


def engine_retryable(code: str | None) -> bool:
    return ENGINE_CODES.get(code or "", (500, False))[1]


def safe_traceback(exc: BaseException) -> str:
    frames = traceback.extract_tb(exc.__traceback__)
    return (
        type(exc).__name__
        + " | "
        + " < ".join(f"{Path(f.filename).name}:{f.lineno}" for f in reversed(frames))
    )


def error_body(request: Request, code: str, message: str, *, details=None, retryable=False):
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
            "retryable": retryable,
            "details": details or {},
        }
    }


def install_handlers(app) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        body = error_body(
            request, exc.code, exc.message, details=exc.details, retryable=exc.retryable
        )
        return JSONResponse(body, status_code=exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        # field locations only — never echo the submitted values
        fields = [".".join(str(p) for p in e.get("loc", ())) for e in exc.errors()]
        body = error_body(
            request, "VALIDATION_ERROR", "Request is invalid.", details={"fields": fields}
        )
        return JSONResponse(body, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        msg = {404: "Not found.", 405: "Method not allowed."}.get(
            exc.status_code, "Request failed."
        )
        return JSONResponse(error_body(request, code, msg), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _internal(request: Request, exc: Exception):
        LOGGER.error(
            "internal error request_id=%s %s",
            getattr(request.state, "request_id", None),
            safe_traceback(exc),
        )
        body = error_body(request, "INTERNAL_ERROR", "An internal error occurred.")
        return JSONResponse(body, status_code=500)
