"""The REST/SSE gateway (PRD §2.5, D20) — ``vdreamer serve``.

FastAPI lives behind this package and never in the agent-author API (D20):
``import vdreamer`` does not import it. Build an app with :func:`create_app`
over a :class:`GatewayBackend`; ``vdreamer.entrypoints`` wires the Postgres
(``serve``) and in-memory (``dev``) backends.
"""

from vdreamer.gateway.app import create_app
from vdreamer.gateway.auth import API_PREFIX, PUBLIC_PATHS, Identity
from vdreamer.gateway.backend import GatewayBackend, Probe
from vdreamer.gateway.errors import ApiError, ErrorCode
from vdreamer.gateway.settings import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    PREVIOUS_TOKEN_ENV_VAR,
    TOKEN_ENV_VAR,
    GatewayConfigError,
    GatewaySettings,
    tokens_from_env,
    validate_bind_host,
)

__all__ = [
    "API_PREFIX",
    "ApiError",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "ErrorCode",
    "GatewayBackend",
    "GatewayConfigError",
    "GatewaySettings",
    "Identity",
    "PREVIOUS_TOKEN_ENV_VAR",
    "PUBLIC_PATHS",
    "Probe",
    "TOKEN_ENV_VAR",
    "create_app",
    "tokens_from_env",
    "validate_bind_host",
]
