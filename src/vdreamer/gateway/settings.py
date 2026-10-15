"""Gateway settings: internal tokens, bind address, tuning (D13, A14, A23).

Everything comes from the environment (A23) — tokens are secrets and never
appear on the command line or in YAML:

- ``VDREAMER_INTERNAL_TOKEN`` — the current internal token (required);
- ``VDREAMER_INTERNAL_TOKEN_PREVIOUS`` — the token being rotated out
  (optional). A14: "two valid for rotation" — set the new token as current,
  keep the old one as previous until every caller has switched, then drop it.

The gateway binds to the internal interface only (A14, §5.1): wildcard
addresses (``0.0.0.0``, ``::``) are refused — name the internal interface's
address explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from vdreamer.events import HEARTBEAT_INTERVAL_S

#: The current internal token's environment variable (A14).
TOKEN_ENV_VAR: Final[str] = "VDREAMER_INTERNAL_TOKEN"

#: The rotated-out token's environment variable — optional (A14).
PREVIOUS_TOKEN_ENV_VAR: Final[str] = "VDREAMER_INTERNAL_TOKEN_PREVIOUS"

#: The role that reads every artifact regardless of thread ownership (A8).
ARTIFACT_READ_ALL_ROLE: Final[str] = "artifact:read:all"

#: The default bind address: loopback until the operator names an interface.
DEFAULT_HOST: Final[str] = "127.0.0.1"

DEFAULT_PORT: Final[int] = 8000

#: Addresses that bind every interface — refused (A14: internal interface only).
_WILDCARD_HOSTS: Final[frozenset[str]] = frozenset({"0.0.0.0", "::", "[::]", "*", ""})

#: Tokens shorter than this are refused: an internal token is a secret, not a password.
MIN_TOKEN_LENGTH: Final[int] = 16


class GatewayConfigError(RuntimeError):
    """The gateway's configuration is missing or unsafe."""


#: Four-eyes scopes (A32): comma-separated ``fnmatch`` patterns, e.g. ``db:write:*``.
STRICT_SCOPES_ENV_VAR: Final[str] = "HITL_STRICT_SCOPES"


@dataclass(frozen=True)
class GatewaySettings:
    """Runtime settings the app is built with.

    ``strict_scopes`` are the ``HITL_STRICT_SCOPES`` patterns (A32): an
    interrupt whose ``scope`` matches one needs an approver who is not the run's
    initiator.
    """

    tokens: tuple[str, ...]
    heartbeat_s: float = HEARTBEAT_INTERVAL_S
    artifact_read_all_role: str = ARTIFACT_READ_ALL_ROLE
    strict_scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tokens:
            raise GatewayConfigError("at least one internal token is required (A14)")
        if len(self.tokens) > 2:
            raise GatewayConfigError("at most two internal tokens are valid at once (A14)")
        for token in self.tokens:
            if len(token) < MIN_TOKEN_LENGTH:
                raise GatewayConfigError(
                    f"internal tokens must be at least {MIN_TOKEN_LENGTH} characters"
                )


def tokens_from_env() -> tuple[str, ...]:
    """The valid internal tokens: current first, then the rotated-out one (A14, A23)."""
    current = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if not current:
        raise GatewayConfigError(
            f"environment variable {TOKEN_ENV_VAR} is not set "
            "(A23: secrets come from env, never YAML or the command line)"
        )
    previous = os.environ.get(PREVIOUS_TOKEN_ENV_VAR, "").strip()
    return (current, previous) if previous and previous != current else (current,)


def strict_scopes_from_env() -> tuple[str, ...]:
    """The four-eyes scope patterns from ``HITL_STRICT_SCOPES`` (A32); empty when unset."""
    raw = os.environ.get(STRICT_SCOPES_ENV_VAR, "")
    return tuple(pattern.strip() for pattern in raw.split(",") if pattern.strip())


def validate_bind_host(host: str) -> str:
    """Refuse wildcard binds (A14: the gateway binds to the internal interface only)."""
    if host.strip() in _WILDCARD_HOSTS:
        raise GatewayConfigError(
            f"refusing to bind {host!r}: the gateway binds to the internal interface only "
            "(A14) — pass that interface's address with --host"
        )
    return host
