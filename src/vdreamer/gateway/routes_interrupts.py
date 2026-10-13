"""The approver inbox and the one resolve endpoint (PRD §4.2, §4.3, A12, A44).

Authority (§4.3): the run's initiator resolves by default; an
``approver_role`` on the interrupt replaces that with "any holder of the
role". A caller who can neither resolve nor initiated the run does not see the
record at all (``404``); the initiator of a role-gated interrupt sees it but
gets ``403 forbidden_role`` on resolve.

Resolution, in order: visibility (``404``) → authority (``403``) → expiry
(``410``, lazily — even before the M8 sweeper marks it) → still pending and
``expected_version`` current (``409``; A12, the first resolver wins) → the
body against the record's ``actions`` (``422``) → the compare-and-swap →
``interrupt.resolved`` → once the run has no pending interrupt left, the
resume map ``{interrupt_id: value}`` is planted on the run and it is requeued
("resolution enqueues a resume", §3.2).

Milestone 8 completes the policy: ``edit_schema`` validation, four-eyes
(``HITL_STRICT_SCOPES``, A32), the expiry sweeper, headless ``reject →
rejected``, and the ``interrupt_policies`` of non-vdreamer sources.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from vdreamer.events import EventEnvelope, EventType, InterruptResolvedPayload
from vdreamer.gateway.auth import API_PREFIX, Identity, current_identity
from vdreamer.gateway.backend import GatewayBackend
from vdreamer.gateway.common import get_backend, iso, iso_required, jsonable
from vdreamer.gateway.errors import ApiError, ErrorCode
from vdreamer.gateway.schemas import (
    Decision,
    ErrorResponse,
    InterruptList,
    InterruptOut,
    ResolveRequest,
)
from vdreamer.interrupts import (
    InterruptConflict,
    InterruptRecord,
    InterruptStatus,
    Viewer,
    visible_to,
)
from vdreamer.queue import InvalidTransition

_log = logging.getLogger(__name__)

router = APIRouter(prefix=API_PREFIX, tags=["interrupts"])

Backend = Annotated[GatewayBackend, Depends(get_backend)]
Caller = Annotated[Identity, Depends(current_identity)]

_E = {"model": ErrorResponse}

#: Actions whose resolution carries a ``text`` (§4.3 table).
_TEXT_ACTIONS = frozenset({"feedback", "respond"})


def can_resolve(record: InterruptRecord, identity: Identity) -> bool:
    """§4.3 authority: the approver role when set, else the run's initiator."""
    if record.approver_role is not None:
        return record.approver_role in identity.roles
    return record.initiator_id is not None and record.initiator_id == identity.user_id


def viewer(identity: Identity) -> Viewer:
    return Viewer(user_id=identity.user_id, roles=identity.roles)


def can_view(record: InterruptRecord, identity: Identity) -> bool:
    """The approver role's holders and the initiator see it (``interrupts.visible_to``)."""
    return visible_to(record, viewer(identity))


#: ``GET /interrupts`` page bound.
MAX_INBOX_LIMIT = 500


def interrupt_out(record: InterruptRecord) -> InterruptOut:
    return InterruptOut(
        interrupt_id=record.interrupt_id,
        run_id=record.run_id,
        thread_id=record.thread_id,
        agent=record.agent,
        kind=record.kind,
        scope=record.scope,
        approver_role=record.approver_role,
        actions=list(record.actions),
        payload=jsonable(dict(record.payload)),
        edit_schema=dict(record.edit_schema) if record.edit_schema is not None else None,
        status=record.status.value,
        version=record.version,
        expires_at=iso(record.expires_at),
        created_at=iso_required(record.created_at),
        resolved_by=record.resolved_by,
        resolution=jsonable(record.resolution),
    )


async def _visible_or_404(
    backend: GatewayBackend, interrupt_id: str, identity: Identity
) -> InterruptRecord:
    record = await backend.interrupts.get(interrupt_id)
    if record is None or not can_view(record, identity):
        raise ApiError(404, ErrorCode.INTERRUPT_NOT_FOUND, f"no interrupt {interrupt_id!r}")
    return record


@router.get("/interrupts", response_model=InterruptList, responses={400: _E, 401: _E})
async def list_interrupts(
    backend: Backend,
    caller: Caller,
    status: Annotated[str, Query(pattern="^(pending|resolved|expired|cancelled|all)$")] = (
        "pending"
    ),
    limit: Annotated[int, Query(ge=1, le=MAX_INBOX_LIMIT)] = 100,
) -> InterruptList:
    """The approver inbox (§2.5): interrupts the caller may resolve or initiated."""
    wanted = None if status == "all" else InterruptStatus(status)
    records = await backend.interrupts.list(
        status=wanted, visible_to=viewer(caller), limit=limit
    )
    return InterruptList(interrupts=[interrupt_out(record) for record in records])


@router.get(
    "/interrupts/{interrupt_id}",
    response_model=InterruptOut,
    responses={400: _E, 401: _E, 404: _E},
)
async def get_interrupt(interrupt_id: str, backend: Backend, caller: Caller) -> InterruptOut:
    """One §4.2 record."""
    return interrupt_out(await _visible_or_404(backend, interrupt_id, caller))


def _decision_value(decision: Decision, record: InterruptRecord) -> dict[str, Any]:
    """Validate one decision against the record and build its resume entry (§4.3)."""
    if decision.action not in record.actions:
        raise ApiError(
            422,
            ErrorCode.INVALID_ACTION,
            f"action {decision.action!r} is not offered by this interrupt",
            {"actions": list(record.actions)},
        )
    if decision.action == "edit" and decision.payload is None:
        raise ApiError(422, ErrorCode.INVALID_EDIT, "an 'edit' resolution needs a payload")
    if decision.action in _TEXT_ACTIONS and not (decision.text or "").strip():
        raise ApiError(
            422, ErrorCode.INVALID_REQUEST, f"a {decision.action!r} resolution needs text"
        )
    value: dict[str, Any] = {"action": decision.action}
    if decision.payload is not None:
        value["payload"] = decision.payload
    if decision.text is not None:
        value["text"] = decision.text
    return value


def _resume_value(body: ResolveRequest, record: InterruptRecord) -> dict[str, Any]:
    """The resume value the gateway builds (§4.5): ``{action, payload?, text?}`` or
    ``{"decisions": [...]}`` in action order."""
    if body.decisions is not None:
        return {"decisions": [_decision_value(decision, record) for decision in body.decisions]}
    assert body.action is not None  # ResolveRequest enforces exactly one form
    return _decision_value(
        Decision(action=body.action, payload=body.payload, text=body.text), record
    )


@router.post(
    "/interrupts/{interrupt_id}/resolve",
    response_model=InterruptOut,
    responses={400: _E, 401: _E, 403: _E, 404: _E, 409: _E, 410: _E, 422: _E},
)
async def resolve_interrupt(
    interrupt_id: str, body: ResolveRequest, backend: Backend, caller: Caller
) -> InterruptOut:
    """Resolve (§4.3): ``200`` | ``403 forbidden_role`` | ``409`` stale | ``410`` | ``422``."""
    record = await _visible_or_404(backend, interrupt_id, caller)
    if not can_resolve(record, caller):
        raise ApiError(
            403,
            ErrorCode.FORBIDDEN_ROLE,
            (
                f"resolving needs the role {record.approver_role!r}"
                if record.approver_role is not None
                else "only the run's initiator may resolve this interrupt"
            ),
            {"approver_role": record.approver_role} if record.approver_role else None,
        )
    if record.status is InterruptStatus.EXPIRED or (
        record.status is InterruptStatus.PENDING and record.is_expired(time.time())
    ):
        raise ApiError(
            410,
            ErrorCode.INTERRUPT_EXPIRED,
            f"interrupt {interrupt_id!r} expired",
            {"expires_at": iso(record.expires_at)},
        )
    if record.status is InterruptStatus.RESOLVED:
        raise ApiError(
            409,
            ErrorCode.VERSION_CONFLICT,
            "the interrupt was already resolved (the first resolver wins, A12)",
            {"version": record.version, "resolved_by": record.resolved_by},
        )
    if record.status is not InterruptStatus.PENDING:
        raise ApiError(
            409,
            ErrorCode.INTERRUPT_NOT_PENDING,
            f"interrupt {interrupt_id!r} is {record.status.value}",
            {"status": record.status.value},
        )
    if body.expected_version != record.version:
        raise ApiError(
            409,
            ErrorCode.VERSION_CONFLICT,
            "expected_version is stale — refresh the interrupt and retry",
            {"version": record.version},
        )
    value = _resume_value(body, record)
    try:
        resolved = await backend.interrupts.resolve(
            record.interrupt_id,
            expected_version=body.expected_version,
            resolved_by=caller.user_id,
            resolution=value,
        )
    except InterruptConflict:
        current = await backend.interrupts.get(record.interrupt_id)
        raise ApiError(
            409,
            ErrorCode.VERSION_CONFLICT,
            "the interrupt changed under this resolution (the first resolver wins, A12)",
            {"version": current.version if current is not None else None},
        ) from None

    action = body.action if body.action is not None else "decisions"
    await backend.event_writer.write(
        EventEnvelope(
            type=EventType.INTERRUPT_RESOLVED,
            run_id=resolved.run_id,
            thread_id=resolved.thread_id,
            agent=resolved.agent,
            seq=0,  # the log allocates the real seq
            payload=InterruptResolvedPayload(
                interrupt_id=resolved.interrupt_id, action=action, resolved_by=caller.user_id
            ).event_payload(),
        )
    )
    await _resume_when_settled(backend, resolved.run_id)
    return interrupt_out(resolved)


async def _resume_when_settled(backend: GatewayBackend, run_id: str) -> None:
    """Requeue the run once none of its interrupts is pending (§3.2).

    The resume map carries every resolution the run has; the worker narrows it
    to the interrupts its checkpoint still awaits.
    """
    if await backend.interrupts.list(status=InterruptStatus.PENDING, run_id=run_id):
        return
    resolved = await backend.interrupts.list(status=InterruptStatus.RESOLVED, run_id=run_id)
    resume = {record.interrupt_id: record.resolution for record in resolved}
    try:
        await backend.queue.resume(run_id, resume=resume)
    except InvalidTransition:
        # The run is no longer awaiting approval (cancelled, expired, or a
        # concurrent resolver already requeued it): nothing to resume.
        _log.info("run %s: resolution recorded; the run was not awaiting approval", run_id)
