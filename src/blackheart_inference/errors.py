"""Stable error envelope, copy-shape from the orchestrator.

Callers (notably the orchestrator's inference proxy) branch on
``error_code`` — keeping the shape identical means the orchestrator
can forward the envelope verbatim and the researcher sees the same
contract regardless of which service originated the failure.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

_unhandled_logger = logging.getLogger(__name__)


class NextAction(BaseModel):
    kind: Literal["retry", "call", "read_doc", "contact_human", "note"]
    method: str | None = None
    path: str | None = None
    wait_s: float | None = None
    doc_anchor: str | None = None
    hint: str | None = None


class ErrorEnvelope(BaseModel):
    error_code: str = Field(..., description="Stable snake_case identifier.")
    message: str
    retryable: bool
    hint: str | None = None
    next_action: NextAction | None = None
    details: dict[str, Any] | None = None


class InferenceError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        error_code: str,
        message: str,
        retryable: bool = False,
        hint: str | None = None,
        next_action: NextAction | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.envelope = ErrorEnvelope(
            error_code=error_code,
            message=message,
            retryable=retryable,
            hint=hint,
            next_action=next_action,
            details=details,
        )


def _envelope_response(status_code: int, env: ErrorEnvelope) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=env.model_dump(exclude_none=True))


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(InferenceError)
    async def _own(_: Request, exc: InferenceError) -> JSONResponse:
        return _envelope_response(exc.status_code, exc.envelope)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _envelope_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorEnvelope(
                error_code="validation_failed",
                message="Request body or query parameters failed validation.",
                retryable=False,
                hint="Fix the fields listed in details.errors and resubmit.",
                details={"errors": [dict(e) for e in exc.errors()]},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _envelope_response(
            exc.status_code,
            ErrorEnvelope(
                error_code=f"http_{exc.status_code}",
                message=str(exc.detail) if exc.detail else "HTTP error.",
                retryable=exc.status_code in (429, 502, 503, 504),
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        _unhandled_logger.exception(
            "unhandled exception in %s %s", request.method, request.url.path,
        )
        return _envelope_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorEnvelope(
                error_code="internal_error",
                message="The inference service crashed handling this request.",
                retryable=False,
                hint="Check inference logs for the traceback.",
                next_action=NextAction(kind="contact_human"),
                details={"exception_class": type(exc).__name__},
            ),
        )
