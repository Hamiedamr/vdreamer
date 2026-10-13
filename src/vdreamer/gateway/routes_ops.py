"""Ops and contract documents (PRD §2.5): health, readiness, metrics, schemas, docs.

``/openapi.json`` itself is FastAPI's (``openapi_url``); everything else is
here. All of these are :data:`~vdreamer.gateway.auth.PUBLIC_PATHS`.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Final

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, JSONResponse, Response

from vdreamer import metrics
from vdreamer.gateway.auth import API_PREFIX
from vdreamer.gateway.backend import GatewayBackend
from vdreamer.gateway.common import get_backend
from vdreamer.gateway.errors import ErrorCode, error_response
from vdreamer.gateway.schemas import ErrorResponse, HealthOut, ReadyOut
from vdreamer.protocol_schema import event_json_schema
from vdreamer.queue import RunState
from vdreamer.vendored import STATIC_DIR

router = APIRouter(prefix=API_PREFIX, tags=["ops"])

Backend = Annotated[GatewayBackend, Depends(get_backend)]

#: The docs page CSP (§5.2): same-origin only. Swagger UI needs inline
#: *styles* (React style attributes), never inline scripts — the init code
#: is a same-origin file — and ``connect-src 'self'`` lets it load the
#: OpenAPI document and nothing else.
DOCS_CSP: Final[str] = (
    "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; font-src 'self' data:; connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)

#: Per-probe budget on ``/readyz``.
PROBE_TIMEOUT_S: Final[float] = 2.0


@router.get("/healthz", response_model=HealthOut)
async def healthz() -> HealthOut:
    """Liveness: the process serves requests."""
    return HealthOut(status="ok")


@router.get("/readyz", response_model=ReadyOut, responses={503: {"model": ErrorResponse}})
async def readyz(backend: Backend) -> Any:
    """Readiness: every backend probe answers within its budget (``503`` otherwise)."""
    checks: dict[str, str] = {}
    for name, probe in backend.probes.items():
        try:
            await asyncio.wait_for(probe(), PROBE_TIMEOUT_S)
        except Exception as error:
            checks[name] = f"failed: {type(error).__name__}"
        else:
            checks[name] = "ok"
    if any(result != "ok" for result in checks.values()):
        return error_response(
            503, ErrorCode.NOT_READY, "a backend is not ready", {"checks": checks}
        )
    return ReadyOut(status="ready", checks=checks)


@router.get(
    "/metrics",
    response_class=Response,
    responses={200: {"content": {metrics.CONTENT_TYPE: {}}}},
)
async def prometheus_metrics(backend: Backend) -> Response:
    """Prometheus text exposition (§5.4); queue depth is refreshed per scrape."""
    try:
        counts = await backend.queue.count_by_status()
    except Exception:  # a scrape must not fail because the queue is briefly down
        counts = None
    if counts is not None:
        for state in RunState:
            metrics.QUEUE_DEPTH.labels(status=state.value).set(counts.get(state, 0))
    return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)


@router.get("/protocol/schema.json")
async def protocol_schema() -> JSONResponse:
    """The protocol v1.0 event JSON Schema (§3.3, §6.1)."""
    return JSONResponse(event_json_schema())


@router.get("/docs", response_class=HTMLResponse, include_in_schema=False)
async def docs() -> HTMLResponse:
    """Swagger UI from vendored assets only (§5.2, A13), under :data:`DOCS_CSP`."""
    html = (STATIC_DIR / "vdreamer" / "docs.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Content-Security-Policy": DOCS_CSP})
