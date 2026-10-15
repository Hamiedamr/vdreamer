"""The one error body (PRD §2.5): ``{error: {code, message, details?}}``.

Every non-2xx response the gateway produces — its own checks, FastAPI's
request validation, Starlette's routing 404/405, an unhandled exception —
leaves through here, so a client parses exactly one shape and switches on a
stable ``code``. :class:`ErrorCode` is the catalogue; codes are part of the
protocol contract and never change meaning (new ones may be added).
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

_log = logging.getLogger(__name__)


class ErrorCode(enum.StrEnum):
    """Stable error codes (PRD §2.5: "stable codes (e.g., ``run_conflict``)")."""

    # 400
    MISSING_IDENTITY = "missing_identity"
    PROTOCOL_VERSION_MISMATCH = "protocol_version_mismatch"
    MODE_NOT_SUPPORTED = "mode_not_supported"
    INVALID_LAST_EVENT_ID = "invalid_last_event_id"
    MISSING_IDEMPOTENCY_KEY = "missing_idempotency_key"
    # 401 / 403
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN_ROLE = "forbidden_role"
    # 404 / 405
    NOT_FOUND = "not_found"
    AGENT_NOT_FOUND = "agent_not_found"
    THREAD_NOT_FOUND = "thread_not_found"
    RUN_NOT_FOUND = "run_not_found"
    INTERRUPT_NOT_FOUND = "interrupt_not_found"
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    # 409
    RUN_CONFLICT = "run_conflict"
    RUN_NOT_ACTIVE = "run_not_active"
    VERSION_CONFLICT = "version_conflict"
    INTERRUPT_NOT_PENDING = "interrupt_not_pending"
    # 410
    INTERRUPT_EXPIRED = "interrupt_expired"
    ARTIFACT_EXPIRED = "artifact_expired"
    # 422
    INVALID_REQUEST = "invalid_request"
    INVALID_INPUT = "invalid_input"
    INVALID_ACTION = "invalid_action"
    INVALID_EDIT = "invalid_edit"
    CALLBACK_NOT_ALLOWED = "callback_not_allowed"
    UNSUPPORTED_FORMAT = "unsupported_format"
    # 5xx
    NOT_READY = "not_ready"
    INTERNAL_ERROR = "internal_error"


class ApiError(Exception):
    """Raise anywhere in a route to answer with the one error body."""

    def __init__(
        self,
        status_code: int,
        code: ErrorCode,
        message: str,
        details: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = dict(details) if details is not None else None
        self.headers = dict(headers) if headers is not None else None


def error_body(code: str, message: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``{"error": {"code", "message", "details"?}}`` — ``details`` omitted when empty."""
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = jsonable_encoder(dict(details))
    return {"error": error}


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """The one error body as a response."""
    return JSONResponse(
        error_body(code, message, details),
        status_code=status_code,
        headers=dict(headers) if headers is not None else None,
    )


_STATUS_CODES: dict[int, ErrorCode] = {
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.METHOD_NOT_ALLOWED,
}


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(exc.status_code, exc.code, exc.message, exc.details, exc.headers)


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _STATUS_CODES.get(exc.status_code, ErrorCode.INVALID_REQUEST)
    message = str(exc.detail) if exc.detail else "request failed"
    headers = dict(exc.headers) if exc.headers else None
    return error_response(exc.status_code, code, message, headers=headers)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [
        {key: value for key, value in error.items() if key in ("loc", "msg", "type")}
        for error in exc.errors()
    ]
    return error_response(
        422, ErrorCode.INVALID_REQUEST, "request validation failed", {"errors": errors}
    )


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    # The message never echoes the exception: internals stay in the log (§5.4).
    _log.exception("unhandled error serving %s %s", request.method, request.url.path)
    return error_response(500, ErrorCode.INTERNAL_ERROR, "internal server error")


def install_error_handlers(app: FastAPI) -> None:
    """Route every failure path through the one error body."""
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled)
