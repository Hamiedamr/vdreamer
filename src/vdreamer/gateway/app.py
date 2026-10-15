"""``create_app`` — the FastAPI gateway over one :class:`GatewayBackend` (D20).

FastAPI is the implementation, not the contract (D20): the contract is the
REST/SSE protocol (§2.5, §3.3) and its OpenAPI 3.1 document. FastAPI's own
docs routes are disabled (they load Swagger UI from a CDN — §5.2 forbids it);
``/api/v1/docs`` serves the vendored copy instead, and ``/static`` serves the
vendored assets from the wheel.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from vdreamer import __version__
from vdreamer.events import PROTOCOL_VERSION
from vdreamer.gateway import (
    routes_artifacts,
    routes_interrupts,
    routes_ops,
    routes_runs,
    routes_trigger,
)
from vdreamer.gateway.auth import API_PREFIX, InternalAuthMiddleware
from vdreamer.gateway.backend import GatewayBackend
from vdreamer.gateway.errors import install_error_handlers
from vdreamer.gateway.settings import GatewaySettings
from vdreamer.vendored import STATIC_DIR

#: Extra process-lifetime work (``vdreamer dev`` runs its worker loop here;
#: ``vdreamer serve`` its ``LISTEN`` loop).
Lifespan = Callable[[], AbstractAsyncContextManager[None]]

OPENAPI_URL = f"{API_PREFIX}/openapi.json"


def create_app(
    backend: GatewayBackend,
    settings: GatewaySettings,
    *,
    lifespan: Lifespan | None = None,
) -> FastAPI:
    """Build the gateway app (§2.5) over ``backend``."""

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        if lifespan is None:
            yield
            return
        async with lifespan():
            yield

    app = FastAPI(
        title="vdreamer",
        version=__version__,
        summary=f"LangGraph agents served over REST/SSE — protocol {PROTOCOL_VERSION}.",
        description=(
            "The vdreamer REST/SSE contract (PRD §2.5). Authenticate with the internal "
            "bearer token plus X-User-Id / X-User-Roles (D13); upstream services use a "
            "service token on /agents/{name}/trigger (A16). Every error response is "
            "{error: {code, message, details?}}. Events stream as SSE; their JSON "
            f"Schema is at {API_PREFIX}/protocol/schema.json."
        ),
        openapi_url=OPENAPI_URL,
        docs_url=None,
        redoc_url=None,
        swagger_ui_oauth2_redirect_url=None,
        lifespan=_lifespan,
    )
    app.state.backend = backend
    app.state.settings = settings

    install_error_handlers(app)
    app.add_middleware(
        InternalAuthMiddleware, tokens=settings.tokens, service_tokens=settings.service_tokens
    )

    app.include_router(routes_runs.router)
    app.include_router(routes_trigger.router)
    app.include_router(routes_interrupts.router)
    app.include_router(routes_artifacts.router)
    app.include_router(routes_ops.router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
