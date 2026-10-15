"""Shared route plumbing: backend access, ownership, access roles, encoding."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pydantic
from fastapi import Request

from vdreamer.agents import BaseAgent
from vdreamer.gateway.auth import Identity
from vdreamer.gateway.backend import GatewayBackend
from vdreamer.gateway.errors import ApiError, ErrorCode
from vdreamer.gateway.schemas import AgentOut, RunConfig, RunOut
from vdreamer.gateway.settings import GatewaySettings
from vdreamer.queue import RunRecord
from vdreamer.runtime import is_reserved_configurable_key
from vdreamer.threads import ThreadRecord


def get_backend(request: Request) -> GatewayBackend:
    backend = request.app.state.backend
    assert isinstance(backend, GatewayBackend)
    return backend


def get_settings(request: Request) -> GatewaySettings:
    settings = request.app.state.settings
    assert isinstance(settings, GatewaySettings)
    return settings


def iso(ts: float | None) -> str | None:
    """Epoch seconds → UTC ISO-8601 (``None`` stays ``None``)."""
    return datetime.fromtimestamp(ts, UTC).isoformat() if ts is not None else None


def iso_required(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


# -- agents ---------------------------------------------------------------------


def agent_or_404(backend: GatewayBackend, name: str) -> BaseAgent:
    agent = backend.agents.get(name)
    if agent is None:
        raise ApiError(404, ErrorCode.AGENT_NOT_FOUND, f"no agent named {name!r}")
    return agent


def can_use(agent: BaseAgent, identity: Identity) -> bool:
    """Agent RBAC: ``access_roles`` empty = any user, else any one of them.

    An upstream service is vouched for by its token, not by roles.
    """
    roles = tuple(agent.access_roles)
    return not roles or identity.service or identity.has_any_role(roles)


def require_access(agent: BaseAgent, identity: Identity) -> None:
    if not can_use(agent, identity):
        raise ApiError(
            403,
            ErrorCode.FORBIDDEN_ROLE,
            f"agent {agent.name!r} requires one of the roles {list(agent.access_roles)}",
            {"required_roles": list(agent.access_roles)},
        )


def agent_out(agent: BaseAgent) -> AgentOut:
    schema = agent.input_schema.model_json_schema() if agent.input_schema is not None else None
    return AgentOut(
        name=agent.name,
        mode=agent.mode.value,
        input_schema=schema,
        expose_thinking=bool(agent.expose_thinking),
        access_roles=list(agent.access_roles),
    )


# -- threads and runs ---------------------------------------------------------------


async def owned_thread_or_404(
    backend: GatewayBackend, thread_id: str, identity: Identity
) -> ThreadRecord:
    """The caller's live thread. Someone else's thread is a 404, never a 403:
    owner scoping must not leak which ids exist."""
    thread = await backend.threads.get(thread_id)
    if thread is None or thread.owner_id != identity.user_id:
        raise ApiError(404, ErrorCode.THREAD_NOT_FOUND, f"no thread {thread_id!r}")
    return thread


async def owned_run_or_404(
    backend: GatewayBackend, run_id: str, identity: Identity
) -> tuple[RunRecord, ThreadRecord]:
    """A run on one of the caller's live threads (owner-scoped like threads)."""
    run = await backend.queue.get(run_id)
    if run is None or run.thread_id is None:
        raise ApiError(404, ErrorCode.RUN_NOT_FOUND, f"no run {run_id!r}")
    thread = await backend.threads.get(run.thread_id)
    if thread is None or thread.owner_id != identity.user_id:
        raise ApiError(404, ErrorCode.RUN_NOT_FOUND, f"no run {run_id!r}")
    return run, thread


def validated_input(
    agent_name: str, schema: type[pydantic.BaseModel] | None, value: Any
) -> Any:
    """The run input against the agent's ``input_schema`` (``422 invalid_input``)."""
    if schema is None:
        return value
    try:
        return schema.model_validate(value).model_dump(mode="json")
    except pydantic.ValidationError as exc:
        errors = [
            {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
            for error in exc.errors(include_url=False)
        ]
        raise ApiError(
            422,
            ErrorCode.INVALID_INPUT,
            f"input does not match agent {agent_name!r}'s input_schema",
            {"errors": errors},
        ) from None


def run_config(config: RunConfig | None) -> dict[str, Any] | None:
    """The client's run config as stored on the row; reserved keys are ``422``."""
    if config is None:
        return None
    reserved = sorted(key for key in config.configurable if is_reserved_configurable_key(key))
    if reserved:
        raise ApiError(
            422,
            ErrorCode.INVALID_REQUEST,
            "config.configurable sets reserved keys",
            {"reserved_keys": reserved},
        )
    return config.model_dump(exclude_none=True)


def run_out(run: RunRecord) -> RunOut:
    return RunOut(
        run_id=run.id,
        thread_id=run.thread_id or run.id,
        agent=run.agent,
        status=run.status.value,
        attempt=run.attempt,
        created_at=iso_required(run.created_at),
        updated_at=iso_required(run.updated_at),
    )


# -- encoding ------------------------------------------------------------------------


def jsonable(value: Any, _depth: int = 0) -> Any:
    """Best-effort JSON encoding of checkpointed state (``GET /threads/{id}``).

    Pydantic models (every LangChain message is one) dump in JSON mode,
    containers recurse, dataclasses become dicts; anything else falls back to
    its ``repr`` — a history read must never 500 on an exotic channel value.
    """
    if _depth > 64:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return jsonable(value.value, _depth + 1)
    if isinstance(value, pydantic.BaseModel):
        try:
            return value.model_dump(mode="json")
        except Exception:
            return repr(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value), _depth + 1)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item, _depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item, _depth + 1) for item in value]
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    if isinstance(value, datetime):
        return value.isoformat()
    return repr(value)


def state_channels(channel_values: Mapping[str, Any]) -> dict[str, Any]:
    """The user-facing state channels — LangGraph's internal ones dropped."""
    return {
        key: value
        for key, value in channel_values.items()
        if not key.startswith("__") and not key.startswith("branch:")
    }
