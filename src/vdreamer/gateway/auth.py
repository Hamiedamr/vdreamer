"""Internal auth and the protocol-version handshake.

The gateway is never exposed to end users: the Web Backend calls it with
``Authorization: Bearer <internal_token>`` plus the end user's identity in
``X-User-Id`` / ``X-User-Roles``. This module is a pure ASGI middleware
so the checks run *before* FastAPI parses bodies or validates parameters —
The token is checked first. In order, for every ``/api/v1`` request outside
:data:`PUBLIC_PATHS`:

1. **Token** — one of the (at most two, for rotation) configured internal
   tokens, compared in constant time → else ``401 unauthorized``.
2. **Protocol version** — a client-sent ``X-Protocol-Version`` must share the
   server's major (``1``) → else ``400 protocol_version_mismatch``. The header
   is optional on requests; every response carries the server's ``1.0``.
3. **Identity** — ``X-User-Id`` must be present → else ``400
   missing_identity``. ``X-User-Roles`` is a comma-separated list (may be
   empty).

A service token (``VDREAMER_SERVICE_TOKENS``) replaces steps 1 and 3 on
:func:`is_service_route` only: the caller is ``service:<name>`` and sends no
identity headers.

The resolved :class:`Identity` rides ``scope["state"]["identity"]``; routes
read it through :func:`current_identity`. Every response also gets
``X-Content-Type-Options: nosniff``.

:data:`PUBLIC_PATHS` — probes, metrics, and the contract documents — skip the
token and identity steps: orchestrator probes and the Prometheus scraper carry
no user, and the docs page loads in a browser that cannot attach a bearer
token. The gateway binds to the internal interface only, which is what
protects them.
"""

from __future__ import annotations

import hmac
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Final

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vdreamer.events import PROTOCOL_VERSION, PROTOCOL_VERSION_HEADER
from vdreamer.gateway.errors import ErrorCode, error_response

API_PREFIX: Final[str] = "/api/v1"

#: Routes served without the internal token or an identity (see module doc).
PUBLIC_PATHS: Final[frozenset[str]] = frozenset(
    {
        f"{API_PREFIX}/healthz",
        f"{API_PREFIX}/readyz",
        f"{API_PREFIX}/metrics",
        f"{API_PREFIX}/openapi.json",
        f"{API_PREFIX}/protocol/schema.json",
        f"{API_PREFIX}/docs",
    }
)

USER_ID_HEADER: Final[str] = "X-User-Id"
USER_ROLES_HEADER: Final[str] = "X-User-Roles"

_PROTOCOL_MAJOR: Final[str] = PROTOCOL_VERSION.split(".", 1)[0]


@dataclass(frozen=True)
class Identity:
    """The end user a request acts for — as asserted by the Web Backend.

    An upstream service acts as ``service:<name>`` with ``service`` set.
    """

    user_id: str
    roles: tuple[str, ...] = ()
    service: bool = False

    def has_any_role(self, roles: Sequence[str]) -> bool:
        """Whether the user holds at least one of ``roles``."""
        return any(role in self.roles for role in roles)


def parse_roles(raw: str | None) -> tuple[str, ...]:
    """``"a, b,,c"`` → ``("a", "b", "c")`` — order kept, blanks and duplicates dropped."""
    seen: dict[str, None] = {}
    for part in (raw or "").split(","):
        role = part.strip()
        if role:
            seen.setdefault(role, None)
    return tuple(seen)


def _header(scope: Scope, name: str) -> str | None:
    wanted = name.lower().encode("latin-1")
    for key, value in scope.get("headers", ()):
        if key.lower() == wanted:
            return bytes(value).decode("latin-1")
    return None


def _protocol_major(value: str) -> str:
    return value.strip().split(".", 1)[0]


def _presented_token(authorization: str | None) -> bytes | None:
    if authorization is None:
        return None
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credentials.strip():
        return None
    return credentials.strip().encode("utf-8")


def is_service_route(method: str, path: str) -> bool:
    """What a service token may call: trigger, and read the runs it started."""
    parts = path.removeprefix(API_PREFIX).strip("/").split("/")
    if method == "POST":
        return len(parts) == 3 and parts[0] == "agents" and parts[2] == "trigger"
    if method == "GET":
        if parts[0] != "runs":
            return False
        return len(parts) == 2 or (len(parts) == 3 and parts[2] == "events")
    return False


class InternalAuthMiddleware:
    """Token → protocol version → identity, ahead of routing.

    A service token is accepted on :func:`is_service_route` only; its
    caller is ``service:<name>`` and carries no ``X-User-Id``.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        tokens: Sequence[str],
        service_tokens: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.app = app
        self._tokens = tuple(token.encode("utf-8") for token in tokens)
        self._service_tokens = tuple(
            (name, token.encode("utf-8")) for name, token in service_tokens
        )

    def _token_ok(self, authorization: str | None) -> bool:
        presented = _presented_token(authorization)
        if presented is None:
            return False
        # Compare against every token (no early exit): constant time overall.
        matches = [hmac.compare_digest(presented, token) for token in self._tokens]
        return any(matches)

    def _service_name(self, authorization: str | None) -> str | None:
        presented = _presented_token(authorization)
        if presented is None:
            return None
        found: str | None = None
        for name, token in self._service_tokens:
            if hmac.compare_digest(presented, token):
                found = name
        return found

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                names = {bytes(key).lower() for key, _ in headers}
                if PROTOCOL_VERSION_HEADER.lower().encode() not in names:
                    headers.append(
                        (PROTOCOL_VERSION_HEADER.encode("latin-1"), PROTOCOL_VERSION.encode())
                    )
                if b"x-content-type-options" not in names:
                    headers.append((b"x-content-type-options", b"nosniff"))
                message["headers"] = headers
            await send(message)

        path: str = scope.get("path", "")
        if not (path == API_PREFIX or path.startswith(f"{API_PREFIX}/")):
            await self.app(scope, receive, send_with_headers)
            return

        public = path in PUBLIC_PATHS
        authorization = _header(scope, "authorization")
        internal = not public and self._token_ok(authorization)
        service: str | None = None
        if not public and not internal and is_service_route(scope.get("method", ""), path):
            service = self._service_name(authorization)
        if not public and not internal and service is None:
            response = error_response(
                401,
                ErrorCode.UNAUTHORIZED,
                "a valid internal bearer token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send_with_headers)
            return

        client_version = _header(scope, PROTOCOL_VERSION_HEADER)
        if client_version is not None and _protocol_major(client_version) != _PROTOCOL_MAJOR:
            response = error_response(
                400,
                ErrorCode.PROTOCOL_VERSION_MISMATCH,
                f"protocol major version {_protocol_major(client_version)!r} is not supported",
                {"server_version": PROTOCOL_VERSION, "client_version": client_version},
            )
            await response(scope, receive, send_with_headers)
            return

        state: dict[str, Any] = scope.setdefault("state", {})
        if service is not None:
            state["identity"] = Identity(user_id=f"service:{service}", service=True)
        elif not public:
            user_id = (_header(scope, USER_ID_HEADER) or "").strip()
            if not user_id:
                response = error_response(
                    400,
                    ErrorCode.MISSING_IDENTITY,
                    f"the {USER_ID_HEADER} header is required",
                )
                await response(scope, receive, send_with_headers)
                return
            state["identity"] = Identity(
                user_id=user_id, roles=parse_roles(_header(scope, USER_ROLES_HEADER))
            )

        await self.app(scope, receive, send_with_headers)


#: Declares the bearer scheme in OpenAPI; the middleware already enforced it.
_bearer = HTTPBearer(auto_error=False, description="The internal token.")


def current_identity(
    request: Request,
    _credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    _user_id: Annotated[
        str | None,
        Header(
            alias=USER_ID_HEADER,
            description="The end user's id, asserted by the Web Backend. Required.",
        ),
    ] = None,
    _user_roles: Annotated[
        str | None,
        Header(alias=USER_ROLES_HEADER, description="The end user's roles, comma-separated."),
    ] = None,
) -> Identity:
    """The request's :class:`Identity` (set by :class:`InternalAuthMiddleware`).

    The header parameters exist so OpenAPI documents the contract; the values
    were already validated by the middleware, which answered before routing
    when anything was missing.
    """
    identity = request.scope.get("state", {}).get("identity")
    if not isinstance(identity, Identity):  # pragma: no cover - middleware always sets it
        raise RuntimeError("InternalAuthMiddleware is not installed")
    return identity
