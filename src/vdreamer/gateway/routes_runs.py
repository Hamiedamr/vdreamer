"""Agents, threads, runs, and their event streams (PRD §2.5, §3.1–§3.3)."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Query, Response
from fastapi.responses import StreamingResponse

from vdreamer.agents import Mode
from vdreamer.events import EventEnvelope, EventType, RunStatusPayload
from vdreamer.gateway.auth import API_PREFIX, Identity, current_identity
from vdreamer.gateway.backend import GatewayBackend
from vdreamer.gateway.common import (
    agent_or_404,
    agent_out,
    can_use,
    get_backend,
    get_settings,
    iso_required,
    jsonable,
    owned_run_or_404,
    owned_thread_or_404,
    require_access,
    run_config,
    run_out,
    state_channels,
    validated_input,
)
from vdreamer.gateway.errors import ApiError, ErrorCode
from vdreamer.gateway.schemas import (
    AgentList,
    AgentOut,
    ErrorResponse,
    RunCreate,
    RunCreated,
    RunOut,
    ThreadCreate,
    ThreadCreated,
    ThreadDetail,
    ThreadList,
    ThreadOut,
)
from vdreamer.gateway.settings import GatewaySettings
from vdreamer.gateway.sse import (
    SSE_HEADERS,
    SSE_MEDIA_TYPE,
    LastEventIdError,
    choose_thread_run,
    parse_last_event_id,
    run_event_stream,
)
from vdreamer.queue import ACTIVE_STATES, ActiveRunConflict, RunRecord, RunState
from vdreamer.threads import ThreadRecord

_log = logging.getLogger(__name__)

router = APIRouter(prefix=API_PREFIX)

Backend = Annotated[GatewayBackend, Depends(get_backend)]
Settings = Annotated[GatewaySettings, Depends(get_settings)]
Caller = Annotated[Identity, Depends(current_identity)]

_E = {"model": ErrorResponse}
_AUTH_ERRORS: dict[int | str, dict[str, Any]] = {400: _E, 401: _E}


def _thread_out(thread: ThreadRecord) -> ThreadOut:
    return ThreadOut(
        thread_id=thread.id,
        agent=thread.agent,
        created_at=iso_required(thread.created_at),
        metadata=dict(thread.metadata),
    )


def _require_chat(agent_name: str, mode: Mode) -> None:
    """Threads and runs are the chat path; headless agents use ``/trigger`` (D15)."""
    if mode is Mode.HEADLESS:
        raise ApiError(
            400,
            ErrorCode.MODE_NOT_SUPPORTED,
            f"agent {agent_name!r} is headless: start runs with POST "
            f"{API_PREFIX}/agents/{agent_name}/trigger (D15)",
        )


# -- agents ----------------------------------------------------------------------------


@router.get("/agents", response_model=AgentList, responses=_AUTH_ERRORS, tags=["agents"])
async def list_agents(backend: Backend, caller: Caller) -> AgentList:
    """The registry (§2.5): every agent the caller's roles may use."""
    agents = sorted(backend.agents.values(), key=lambda agent: agent.name)
    return AgentList(agents=[agent_out(agent) for agent in agents if can_use(agent, caller)])


@router.get(
    "/agents/{name}",
    response_model=AgentOut,
    responses={**_AUTH_ERRORS, 403: _E, 404: _E},
    tags=["agents"],
)
async def get_agent(name: str, backend: Backend, caller: Caller) -> AgentOut:
    """One agent: mode and input JSON Schema (§2.5)."""
    agent = agent_or_404(backend, name)
    require_access(agent, caller)
    return agent_out(agent)


# -- threads ----------------------------------------------------------------------------


@router.post(
    "/agents/{name}/threads",
    status_code=201,
    response_model=ThreadCreated,
    responses={**_AUTH_ERRORS, 403: _E, 404: _E, 422: _E},
    tags=["threads"],
)
async def create_thread(
    name: str,
    backend: Backend,
    caller: Caller,
    response: Response,
    body: ThreadCreate | None = None,
) -> ThreadCreated:
    """``201 {thread_id}`` — a new session owned by the caller (§2.5, §3.1)."""
    agent = agent_or_404(backend, name)
    require_access(agent, caller)
    _require_chat(agent.name, agent.mode)
    thread = await backend.threads.create(
        agent.name, caller.user_id, metadata=(body.metadata if body is not None else None)
    )
    response.headers["Location"] = f"{API_PREFIX}/threads/{thread.id}"
    return ThreadCreated(thread_id=thread.id)


@router.get("/threads", response_model=ThreadList, responses=_AUTH_ERRORS, tags=["threads"])
async def list_threads(
    backend: Backend,
    caller: Caller,
    agent: str | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ThreadList:
    """The caller's threads, newest first (owner-scoped, §2.5)."""
    threads = await backend.threads.list(caller.user_id, agent=agent, limit=limit, offset=offset)
    return ThreadList(
        threads=[_thread_out(thread) for thread in threads], limit=limit, offset=offset
    )


@router.get(
    "/threads/{thread_id}",
    response_model=ThreadDetail,
    responses={**_AUTH_ERRORS, 404: _E},
    tags=["threads"],
)
async def get_thread(thread_id: str, backend: Backend, caller: Caller) -> ThreadDetail:
    """The thread, its runs, and its history from the checkpoint (§2.5)."""
    thread = await owned_thread_or_404(backend, thread_id, caller)
    runs = await backend.queue.list_for_thread(thread.id)
    active = next((run for run in runs if run.status in ACTIVE_STATES), None)
    values: Any = None
    checkpoint_id: str | None = None
    if backend.checkpointer is not None:
        checkpoint = await backend.checkpointer.aget_tuple(
            {"configurable": {"thread_id": thread.id, "checkpoint_ns": ""}}
        )
        if checkpoint is not None:
            channel_values = checkpoint.checkpoint.get("channel_values") or {}
            values = jsonable(state_channels(channel_values))
            checkpoint_id = checkpoint.checkpoint.get("id")
    base = _thread_out(thread)
    return ThreadDetail(
        **base.model_dump(),
        runs=[run_out(run) for run in runs],
        active_run_id=active.id if active is not None else None,
        values=values,
        checkpoint_id=checkpoint_id,
    )


@router.delete(
    "/threads/{thread_id}",
    status_code=204,
    responses={**_AUTH_ERRORS, 404: _E, 409: _E},
    tags=["threads"],
)
async def delete_thread(thread_id: str, backend: Backend, caller: Caller) -> Response:
    """Soft delete (§2.5). A thread with an active run must be cancelled first."""
    thread = await owned_thread_or_404(backend, thread_id, caller)
    runs = await backend.queue.list_for_thread(thread.id)
    active = next((run for run in runs if run.status in ACTIVE_STATES), None)
    if active is not None:
        raise ApiError(
            409,
            ErrorCode.RUN_CONFLICT,
            "the thread has an active run; cancel it before deleting the thread",
            {"active_run_id": active.id, "reason": "thread_has_active_run"},
        )
    await backend.threads.soft_delete(thread.id)
    return Response(status_code=204)


# -- runs --------------------------------------------------------------------------------


@router.post(
    "/threads/{thread_id}/runs",
    status_code=202,
    response_model=RunCreated,
    responses={**_AUTH_ERRORS, 403: _E, 404: _E, 409: _E, 422: _E},
    tags=["runs"],
)
async def create_run(
    thread_id: str,
    body: RunCreate,
    backend: Backend,
    caller: Caller,
    response: Response,
    wait_for_idle: bool = False,
) -> RunCreated:
    """``{input, config?}`` → ``202 {run_id}``; ``409`` while the thread is busy (D12, A11).

    ``?wait_for_idle=true`` queues behind a ``running``/``awaiting_approval``
    run instead of conflicting — never behind another ``queued`` one (§2.5).
    """
    thread = await owned_thread_or_404(backend, thread_id, caller)
    agent = agent_or_404(backend, thread.agent)
    require_access(agent, caller)
    _require_chat(agent.name, agent.mode)
    run_input = validated_input(agent.name, agent.input_schema, body.input)
    config = run_config(body.config)
    try:
        run = await backend.queue.enqueue(
            agent.name,
            run_input,
            thread_id=thread.id,
            run_id=None,
            wait_for_idle=wait_for_idle,
            user_id=caller.user_id,
            roles=caller.roles,
            config=config,
        )
    except ActiveRunConflict as conflict:
        raise ApiError(
            409,
            ErrorCode.RUN_CONFLICT,
            "the thread already has an active run",
            {"active_run_id": conflict.active_run_id, "reason": conflict.reason},
        ) from None
    response.headers["Location"] = f"{API_PREFIX}/runs/{run.id}"
    return RunCreated(run_id=run.id, thread_id=thread.id, status=run.status.value)


@router.get(
    "/runs/{run_id}",
    response_model=RunOut,
    responses={**_AUTH_ERRORS, 404: _E},
    tags=["runs"],
)
async def get_run(run_id: str, backend: Backend, caller: Caller) -> RunOut:
    """The run's status (§2.5, §3.2)."""
    run, _ = await owned_run_or_404(backend, run_id, caller)
    return run_out(run)


@router.post(
    "/runs/{run_id}/cancel",
    status_code=202,
    response_model=RunOut,
    responses={**_AUTH_ERRORS, 404: _E, 409: _E},
    tags=["runs"],
)
async def cancel_run(run_id: str, backend: Backend, caller: Caller) -> RunOut:
    """Cancel (§2.5, §3.2): the row is ``cancelled`` at once; a running graph drains
    cooperatively at its next superstep, once its worker's heartbeat notices
    (within half a lease window), leaving a resumable checkpoint."""
    run, _ = await owned_run_or_404(backend, run_id, caller)
    prior = await backend.queue.cancel(run.id)
    if prior is None:
        current = await backend.queue.get(run.id)
        status = current.status.value if current is not None else "unknown"
        raise ApiError(
            409, ErrorCode.RUN_NOT_ACTIVE, f"run {run.id!r} is {status}", {"status": status}
        )
    await backend.interrupts.cancel_for_run(run.id)
    if prior in (RunState.QUEUED, RunState.AWAITING_APPROVAL):
        # No worker holds the run, so the gateway writes its final event.
        await backend.event_writer.write(
            EventEnvelope(
                type=EventType.RUN_STATUS,
                run_id=run.id,
                thread_id=run.thread_id or run.id,
                agent=run.agent,
                seq=0,  # the log allocates the real seq
                payload=RunStatusPayload(status="cancelled", attempt=run.attempt).event_payload(),
            )
        )
    current = await backend.queue.get(run.id)
    return run_out(current if current is not None else run)


# -- event streams ------------------------------------------------------------------------

_LAST_EVENT_ID = Header(
    default=None,
    alias="Last-Event-ID",
    description="Resume after this event id ('{run_id}:{seq}', §3.3).",
)

_SSE_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {"content": {SSE_MEDIA_TYPE: {"schema": {"type": "string"}}}, "description": "SSE"},
    **_AUTH_ERRORS,
    404: _E,
}


def _stream(
    run: RunRecord, after_seq: int, backend: GatewayBackend, settings: GatewaySettings
) -> StreamingResponse:
    return StreamingResponse(
        run_event_stream(
            run=run,
            after_seq=after_seq,
            reader=backend.event_reader,
            hub=backend.hub,
            get_run=backend.queue.get,
            heartbeat_s=settings.heartbeat_s,
        ),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )


@router.get(
    "/runs/{run_id}/events",
    response_class=StreamingResponse,
    responses={**_SSE_RESPONSES, 400: _E},
    tags=["events"],
)
async def run_events(
    run_id: str,
    backend: Backend,
    settings: Settings,
    caller: Caller,
    last_event_id: str | None = _LAST_EVENT_ID,
) -> StreamingResponse:
    """SSE (§2.5, §3.3): replay ``seq > Last-Event-ID``, then tail live."""
    run, _ = await owned_run_or_404(backend, run_id, caller)
    try:
        after = parse_last_event_id(last_event_id, run.id)
    except LastEventIdError as exc:
        raise ApiError(400, ErrorCode.INVALID_LAST_EVENT_ID, str(exc)) from None
    return _stream(run, after, backend, settings)


@router.get(
    "/threads/{thread_id}/events",
    response_class=StreamingResponse,
    responses=_SSE_RESPONSES,
    tags=["events"],
)
async def thread_events(
    thread_id: str,
    backend: Backend,
    settings: Settings,
    caller: Caller,
    last_event_id: str | None = _LAST_EVENT_ID,
) -> StreamingResponse:
    """SSE joining the thread's active — else latest — run (reattach after refresh, §2.5).

    A ``Last-Event-ID`` naming that run resumes after it; one naming another
    (older) run replays the joined run from the start.
    """
    thread = await owned_thread_or_404(backend, thread_id, caller)
    run = choose_thread_run(await backend.queue.list_for_thread(thread.id))
    if run is None:
        raise ApiError(404, ErrorCode.RUN_NOT_FOUND, f"thread {thread.id!r} has no runs yet")
    try:
        after = parse_last_event_id(last_event_id, run.id)
    except LastEventIdError:
        after = 0
    return _stream(run, after, backend, settings)
