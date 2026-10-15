"""The approver inbox and the one resolve endpoint (PRD §4.2, §4.3, A12, A44).

Authority (§4.3): the run's initiator resolves by default; an
``approver_role`` on the interrupt replaces that with "any holder of the
role". A caller who can neither resolve nor initiated the run does not see the
record at all (``404``); the initiator of a role-gated interrupt sees it but
gets ``403 forbidden_role`` on resolve.

Resolution, in order: visibility (``404``) → authority (``403``: the role,
else the initiator; four-eyes under a ``HITL_STRICT_SCOPES`` scope, A32) →
expiry (``410``, lazily — even before the sweeper marks it) → still pending
and ``expected_version`` current (``409``; A12, the first resolver wins) → the
body against the record (``422``: offered actions, ``edit`` payloads against
``edit_schema``) → the compare-and-swap → ``interrupt.resolved`` → then either

- a headless ``reject`` of an approval ends the run ``rejected`` (§4.3), or
- once the run has no pending interrupt left, the resume map
  ``{interrupt_id: value}`` is planted on the run and it is requeued
  ("resolution enqueues a resume", §3.2).

The resume value is built per source (§4.5, A44): ``{action, payload?,
text?}`` for ``require_approval``; ``{"decisions": [...]}`` in the
``HumanInTheLoopMiddleware`` shape for ``tool_review``; ``{"responses":
{...}}`` for an MCP elicitation; the resolver payload verbatim for a raw
``interrupt()``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Annotated, Any

import jsonschema
from fastapi import APIRouter, Depends, Query

from vdreamer.agents import runs_headless
from vdreamer.events import EventEnvelope, EventType, InterruptResolvedPayload
from vdreamer.gateway.auth import API_PREFIX, Identity, current_identity
from vdreamer.gateway.backend import GatewayBackend
from vdreamer.gateway.common import get_backend, get_settings, iso, iso_required, jsonable
from vdreamer.gateway.errors import ApiError, ErrorCode
from vdreamer.gateway.schemas import (
    Decision,
    ErrorResponse,
    InterruptList,
    InterruptOut,
    ResolveRequest,
)
from vdreamer.gateway.settings import GatewaySettings
from vdreamer.interrupts import (
    GENERIC_KIND,
    MCP_ELICITATION_KIND,
    POLICY_KINDS,
    TOOL_REVIEW_KIND,
    InterruptConflict,
    InterruptRecord,
    InterruptStatus,
    Viewer,
    end_waiting_run,
    is_strict_scope,
    visible_to,
)
from vdreamer.queue import InvalidTransition, RunRecord, RunState

_log = logging.getLogger(__name__)

router = APIRouter(prefix=API_PREFIX, tags=["interrupts"])

Backend = Annotated[GatewayBackend, Depends(get_backend)]
Settings = Annotated[GatewaySettings, Depends(get_settings)]
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


def _schema_errors(schema: Mapping[str, Any], instance: Any) -> list[dict[str, Any]]:
    """``instance`` against a JSON Schema (an ``edit_schema``, a tool's ``args_schema``).

    An unusable schema refuses every edit: the gateway enforces the policy it
    was given (A15), it never skips it.
    """
    schema = dict(schema)
    validator_cls = jsonschema.validators.validator_for(
        schema, default=jsonschema.Draft202012Validator
    )
    try:
        validator_cls.check_schema(schema)
    except jsonschema.SchemaError:
        _log.warning("interrupt edit_schema is not a valid JSON Schema; refusing the edit")
        return [{"loc": [], "msg": "the interrupt's edit_schema is not a valid JSON Schema"}]
    errors = sorted(
        validator_cls(schema).iter_errors(instance), key=lambda error: list(error.absolute_path)
    )
    return [
        {"loc": list(error.absolute_path), "msg": error.message}
        for error in errors[:MAX_EDIT_ERRORS]
    ]


#: How many schema errors a ``422 invalid_edit`` lists.
MAX_EDIT_ERRORS = 20


def _check_edit(schema: Mapping[str, Any] | None, payload: Any) -> None:
    """§4.3: an ``edit`` payload must validate against the schema → else ``422``."""
    if schema is None:
        return
    errors = _schema_errors(schema, payload)
    if errors:
        raise ApiError(
            422,
            ErrorCode.INVALID_EDIT,
            "the edit payload does not match the interrupt's edit_schema",
            {"errors": errors},
        )


def _decision_value(decision: Decision, record: InterruptRecord) -> dict[str, Any]:
    """Validate one decision against the record and build its resume entry (§4.3)."""
    if decision.action not in record.actions:
        raise ApiError(
            422,
            ErrorCode.INVALID_ACTION,
            f"action {decision.action!r} is not offered by this interrupt",
            {"actions": list(record.actions)},
        )
    if decision.action == "edit":
        if decision.payload is None:
            raise ApiError(422, ErrorCode.INVALID_EDIT, "an 'edit' resolution needs a payload")
        _check_edit(record.edit_schema, decision.payload)
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


def _elicitation_resume(decision: Decision, record: InterruptRecord) -> dict[str, Any]:
    """MCP elicitation (§4.5): ``{"responses": {key: {action, content?}}}``.

    ``edit`` answers every request: ``payload`` maps each request ``key`` to
    its form content (or is already ``{"responses": {...}}``). ``reject``
    declines every request; the server decides how the call goes on.
    """
    requests = record.payload.get("requests") if isinstance(record.payload, dict) else None
    keys = [
        str(request["key"])
        for request in requests or []
        if isinstance(request, dict) and "key" in request
    ]
    if decision.action == "reject":
        return {"responses": {key: {"action": "decline"} for key in keys}}
    payload = decision.payload
    if isinstance(payload, dict) and isinstance(payload.get("responses"), dict):
        responses: dict[str, Any] = payload["responses"]
    elif isinstance(payload, dict):
        responses = {key: {"action": "accept", "content": value} for key, value in payload.items()}
    else:
        raise ApiError(422, ErrorCode.INVALID_EDIT, "an elicitation answer needs a payload object")
    missing = sorted(set(keys) - set(responses))
    if missing:
        raise ApiError(
            422,
            ErrorCode.INVALID_EDIT,
            "every elicitation request needs an answer",
            {"missing": missing},
        )
    return {"responses": responses}


def _review_decision(
    decision: Decision, request: Mapping[str, Any], config: Mapping[str, Any] | None, index: int
) -> dict[str, Any]:
    """One ``HumanInTheLoopMiddleware`` decision for one action request (§4.5)."""
    allowed = [str(item) for item in (config or {}).get("allowed_decisions") or ()]
    if decision.action not in allowed:
        raise ApiError(
            422,
            ErrorCode.INVALID_ACTION,
            f"decision {index}: {decision.action!r} is not allowed for "
            f"{request.get('name')!r}",
            {"index": index, "allowed_decisions": allowed},
        )
    if decision.action == "approve":
        return {"type": "approve"}
    if decision.action == "edit":
        if decision.payload is None:
            raise ApiError(
                422,
                ErrorCode.INVALID_EDIT,
                f"decision {index}: an 'edit' needs the edited args as its payload",
            )
        args_schema = (config or {}).get("args_schema")
        _check_edit(args_schema if isinstance(args_schema, Mapping) else None, decision.payload)
        return {
            "type": "edit",
            "edited_action": {"name": str(request.get("name")), "args": decision.payload},
        }
    if decision.action == "respond":
        if not (decision.text or "").strip():
            raise ApiError(
                422, ErrorCode.INVALID_REQUEST, f"decision {index}: a 'respond' needs text"
            )
        return {"type": "respond", "message": decision.text}
    value: dict[str, Any] = {"type": "reject"}
    if decision.text:
        value["message"] = decision.text
    return value


def _tool_review_resume(body: ResolveRequest, record: InterruptRecord) -> dict[str, Any]:
    """``HumanInTheLoopMiddleware`` (§4.5): ``{"decisions": [...]}`` in request order.

    One decision per action request; a review of a single call may also use
    the ``{action, payload?, text?}`` form. ``edit`` payloads are the edited
    args, validated against the tool's ``args_schema`` when it has one.
    """
    raw_requests = record.payload.get("action_requests")
    requests = [r for r in raw_requests or [] if isinstance(r, Mapping)]
    configs = {
        str(config.get("action_name")): config
        for config in record.payload.get("review_configs") or []
        if isinstance(config, Mapping)
    }
    decisions = (
        body.decisions
        if body.decisions is not None
        else [Decision(action=str(body.action), payload=body.payload, text=body.text)]
    )
    if len(decisions) != len(requests):
        raise ApiError(
            422,
            ErrorCode.INVALID_REQUEST,
            f"send one decision per action request ({len(requests)}), in order",
            {"action_requests": len(requests)},
        )
    return {
        "decisions": [
            _review_decision(decision, request, configs.get(str(request.get("name"))), index)
            for index, (decision, request) in enumerate(zip(decisions, requests, strict=True))
        ]
    }


def _generic_resume(body: ResolveRequest, record: InterruptRecord) -> Any:
    """A raw ``interrupt(value)`` (§4.5): the resolver payload, verbatim.

    ``respond`` with a ``payload`` resumes with that object; with only
    ``text``, with the text.
    """
    if body.decisions is not None or body.action not in record.actions:
        raise ApiError(
            422,
            ErrorCode.INVALID_ACTION,
            "a generic interrupt takes one 'respond' resolution",
            {"actions": list(record.actions)},
        )
    if body.payload is not None:
        return body.payload
    if not (body.text or "").strip():
        raise ApiError(
            422, ErrorCode.INVALID_REQUEST, "a 'respond' resolution needs a payload or text"
        )
    return body.text


def _resume_value(body: ResolveRequest, record: InterruptRecord) -> Any:
    """The resume value the gateway builds, per source (§4.5)."""
    if record.kind == TOOL_REVIEW_KIND:
        return _tool_review_resume(body, record)
    if record.kind == GENERIC_KIND:
        return _generic_resume(body, record)
    if record.kind == MCP_ELICITATION_KIND and body.action is not None:
        decision = Decision(action=body.action, payload=body.payload, text=body.text)
        _decision_value(decision, record)  # action offered, edit has a payload
        return _elicitation_resume(decision, record)
    if body.decisions is not None:
        return {"decisions": [_decision_value(decision, record) for decision in body.decisions]}
    assert body.action is not None  # ResolveRequest enforces exactly one form
    return _decision_value(
        Decision(action=body.action, payload=body.payload, text=body.text), record
    )


def _check_authority(record: InterruptRecord, caller: Identity, settings: GatewaySettings) -> None:
    """§4.3 authority (``403 forbidden_role``): the approver role, else the
    initiator; under a ``HITL_STRICT_SCOPES`` scope, never the initiator (A32)."""
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
    if (
        is_strict_scope(record.scope, settings.strict_scopes)
        and record.initiator_id is not None
        and record.initiator_id == caller.user_id
    ):
        raise ApiError(
            403,
            ErrorCode.FORBIDDEN_ROLE,
            f"scope {record.scope!r} needs four eyes: the run's initiator may not approve it",
            {"reason": "four_eyes", "scope": record.scope},
        )


def _ends_run_rejected(
    backend: GatewayBackend, record: InterruptRecord, body: ResolveRequest, run: RunRecord | None
) -> bool:
    """§4.3: a headless ``reject`` of an approval ends the run ``rejected``.

    Approvals are ``require_approval`` interrupts (incl. ``ApprovalGate`` and
    per-tool gates). A tool review's ``reject`` is the middleware's own
    decision (the model is told) and an elicitation's is a decline: those resume.
    ``run`` is the interrupt's run: a ``/trigger`` run of a ``BOTH`` agent is
    headless though the agent is not.
    """
    if body.action != "reject" or record.kind in POLICY_KINDS:
        return False
    agent = backend.agents.get(record.agent)
    return agent is not None and runs_headless(agent, run is not None and run.headless)


@router.post(
    "/interrupts/{interrupt_id}/resolve",
    response_model=InterruptOut,
    responses={400: _E, 401: _E, 403: _E, 404: _E, 409: _E, 410: _E, 422: _E},
)
async def resolve_interrupt(
    interrupt_id: str, body: ResolveRequest, backend: Backend, settings: Settings, caller: Caller
) -> InterruptOut:
    """Resolve (§4.3): ``200`` | ``403 forbidden_role`` | ``409`` stale | ``410`` | ``422``."""
    record = await _visible_or_404(backend, interrupt_id, caller)
    _check_authority(record, caller, settings)
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
    run = await backend.queue.get(resolved.run_id)
    if _ends_run_rejected(backend, resolved, body, run):
        if run is not None:
            await end_waiting_run(
                backend.queue, backend.interrupts, backend.event_writer, run, RunState.REJECTED
            )
    else:
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
