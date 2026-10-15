"""``POST /agents/{name}/trigger`` — the headless entry.

A trigger is a run started by a system, not a person in a chat: the caller is
the Web Backend acting for a user, or an upstream service (a service
token). It differs from ``POST /threads/{id}/runs`` in four ways:

- ``Idempotency-Key`` is required. The run id is derived from
  ``(agent, caller, key)``, so a retry finds the run the first request made and
  answers ``200`` with it instead of starting a second one. Concurrent
  retries race on that id; the loser sees the winner's run.
- the thread is created for the caller unless ``thread_id`` names one they own;
- the run is *headless*: approvals need an ``approver_role`` and a
  ``reject`` ends it ``rejected``;
- an optional ``callback`` URL is told when the run completes, fails, or raises
  an interrupt — through the outbox, to hosts on ``CALLBACK_ALLOWLIST``.
"""

from __future__ import annotations

import hashlib
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Response

from vdreamer.agents import Mode
from vdreamer.gateway.auth import API_PREFIX, Identity, current_identity
from vdreamer.gateway.backend import GatewayBackend
from vdreamer.gateway.common import (
    agent_or_404,
    get_backend,
    get_settings,
    owned_thread_or_404,
    require_access,
    run_config,
    validated_input,
)
from vdreamer.gateway.errors import ApiError, ErrorCode
from vdreamer.gateway.schemas import ErrorResponse, RunCreated, TriggerCreate
from vdreamer.gateway.settings import GatewaySettings
from vdreamer.network import (
    EndpointError,
    EndpointNotAllowed,
    EndpointUnavailable,
    check_url,
    default_port,
)
from vdreamer.queue import ActiveRunConflict, RunRecord
from vdreamer.threads import ThreadRecord

router = APIRouter(prefix=API_PREFIX)

Backend = Annotated[GatewayBackend, Depends(get_backend)]
Settings = Annotated[GatewaySettings, Depends(get_settings)]
Caller = Annotated[Identity, Depends(current_identity)]

_E = {"model": ErrorResponse}

#: Longest ``Idempotency-Key`` accepted.
MAX_KEY_LENGTH = 255

REPLAYED_HEADER = "Idempotent-Replayed"


def trigger_run_id(agent: str, caller: str, key: str) -> str:
    """The run id an idempotency key maps to: the same key, from the same
    caller, for the same agent, is the same run."""
    digest = hashlib.sha256(f"{agent}\0{caller}\0{key}".encode()).hexdigest()
    return f"run_{digest[:32]}"


def _require_key(value: str | None) -> str:
    key = (value or "").strip()
    if not key:
        raise ApiError(
            400,
            ErrorCode.MISSING_IDEMPOTENCY_KEY,
            "the Idempotency-Key header is required on /trigger",
        )
    if len(key) > MAX_KEY_LENGTH:
        raise ApiError(
            422,
            ErrorCode.INVALID_REQUEST,
            f"the Idempotency-Key is longer than {MAX_KEY_LENGTH} characters",
        )
    return key


async def _check_callback(url: str, settings: GatewaySettings) -> None:
    """A callback goes only to a host on ``CALLBACK_ALLOWLIST``.

    A name off the host globs must resolve, now, only into the allowed networks
    — the same check delivery repeats — so a bad callback is a ``422`` here, not
    a delivery that retries until it is parked ``dead``.
    """
    allowlist = settings.callback_allowlist
    if allowlist is None:
        raise ApiError(
            422,
            ErrorCode.CALLBACK_NOT_ALLOWED,
            "callbacks are off: CALLBACK_ALLOWLIST is not set on the gateway",
        )
    try:
        parsed = check_url("callback", url, allowlist)
        await allowlist.check_connect(parsed.hostname or "", default_port(parsed))
    except (EndpointNotAllowed, EndpointUnavailable):
        raise ApiError(
            422,
            ErrorCode.CALLBACK_NOT_ALLOWED,
            "the callback host is not on CALLBACK_ALLOWLIST",
        ) from None
    except EndpointError:
        raise ApiError(
            422,
            ErrorCode.CALLBACK_NOT_ALLOWED,
            "the callback must be an http(s) URL with a host and no credentials",
        ) from None


def _created(run: RunRecord, response: Response, *, replayed: bool) -> RunCreated:
    response.headers["Location"] = f"{API_PREFIX}/runs/{run.id}"
    if replayed:
        response.status_code = 200
        response.headers[REPLAYED_HEADER] = "true"
    return RunCreated(run_id=run.id, thread_id=run.thread_id or run.id, status=run.status.value)


@router.post(
    "/agents/{name}/trigger",
    status_code=202,
    response_model=RunCreated,
    responses={
        200: {"model": RunCreated, "description": "The Idempotency-Key was seen: the first run."},
        400: _E,
        401: _E,
        403: _E,
        404: _E,
        409: _E,
        422: _E,
    },
    tags=["runs"],
)
async def trigger_agent(
    name: str,
    body: TriggerCreate,
    backend: Backend,
    settings: Settings,
    caller: Caller,
    response: Response,
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            description="Required. A retry with the same key returns the original run.",
        ),
    ] = None,
) -> RunCreated:
    """``{input, config?, thread_id?, callback?}`` → ``202 {run_id}``.

    ``200`` with the original run when the ``Idempotency-Key`` was already
    used. ``400`` for a chat-only agent or a missing key; ``409`` when
    ``thread_id`` names a thread with an active run.
    """
    agent = agent_or_404(backend, name)
    require_access(agent, caller)
    if agent.mode is Mode.CHAT:
        raise ApiError(
            400,
            ErrorCode.MODE_NOT_SUPPORTED,
            f"agent {agent.name!r} is chat-only: start runs with POST "
            f"{API_PREFIX}/agents/{agent.name}/threads and /threads/{{id}}/runs",
        )
    key = _require_key(idempotency_key)
    run_id = trigger_run_id(agent.name, caller.user_id, key)
    existing = await backend.queue.get(run_id)
    if existing is not None:
        return _created(existing, response, replayed=True)

    run_input = validated_input(agent.name, agent.input_schema, body.input)
    config = run_config(body.config)
    if body.callback is not None:
        await _check_callback(body.callback, settings)

    created: ThreadRecord | None = None
    if body.thread_id is not None:
        thread = await owned_thread_or_404(backend, body.thread_id, caller)
        if thread.agent != agent.name:
            raise ApiError(404, ErrorCode.THREAD_NOT_FOUND, f"no thread {body.thread_id!r}")
    else:
        thread = created = await backend.threads.create(agent.name, caller.user_id)

    conflict: ActiveRunConflict | None = None
    try:
        run = await backend.queue.enqueue(
            agent.name,
            run_input,
            thread_id=thread.id,
            run_id=run_id,
            user_id=caller.user_id,
            roles=caller.roles,
            config=config,
            callback=body.callback,
            headless=True,
        )
    except ActiveRunConflict as exc:
        conflict = exc
    except ValueError:
        # Not a conflict on the thread: the id is taken (checked below).
        pass
    else:
        return _created(run, response, replayed=False)

    # A concurrent request with the same key won the race for the run id.
    winner = await backend.queue.get(run_id)
    if created is not None:
        await backend.threads.soft_delete(created.id)
    if winner is not None:
        return _created(winner, response, replayed=True)
    if conflict is not None:
        raise ApiError(
            409,
            ErrorCode.RUN_CONFLICT,
            "the thread already has an active run",
            {"active_run_id": conflict.active_run_id, "reason": conflict.reason},
        )
    raise ApiError(500, ErrorCode.INTERNAL_ERROR, "the run could not be created")


__all__ = ["REPLAYED_HEADER", "router", "trigger_run_id"]
