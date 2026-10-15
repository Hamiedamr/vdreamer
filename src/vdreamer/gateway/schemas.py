"""Request and response bodies of the REST contract.

These models are what OpenAPI 3.1 publishes (``/api/v1/openapi.json``).
Timestamps are UTC ISO-8601 strings; ids are opaque strings.
"""

from __future__ import annotations

from typing import Any

import pydantic
from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable error code, e.g. 'run_conflict'.")
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """The one error body: ``{error: {code, message, details?}}``."""

    error: ErrorDetail


# -- agents --------------------------------------------------------------------


class AgentOut(BaseModel):
    name: str
    mode: str = Field(description="chat | headless | both")
    input_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema of the run input, when the agent declares one."
    )
    expose_thinking: bool
    access_roles: list[str] = Field(description="Any of these roles may use it; empty = anyone.")


class AgentList(BaseModel):
    agents: list[AgentOut]


# -- threads -------------------------------------------------------------------


class ThreadCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any] = Field(default_factory=dict)


class ThreadCreated(BaseModel):
    thread_id: str


class ThreadOut(BaseModel):
    thread_id: str
    agent: str
    created_at: str
    metadata: dict[str, Any]


class ThreadList(BaseModel):
    threads: list[ThreadOut]
    limit: int
    offset: int


class RunOut(BaseModel):
    run_id: str
    thread_id: str
    agent: str
    status: str = Field(
        description="queued | running | awaiting_approval | completed | failed | rejected | "
        "expired | cancelled"
    )
    attempt: int
    created_at: str
    updated_at: str


class ThreadDetail(ThreadOut):
    runs: list[RunOut]
    active_run_id: str | None = None
    values: Any = Field(
        default=None, description="The thread's latest checkpointed state, JSON-encoded."
    )
    checkpoint_id: str | None = None


# -- runs ----------------------------------------------------------------------


class RunConfig(BaseModel):
    """The client-settable slice of the LangGraph run config."""

    model_config = ConfigDict(extra="forbid")

    configurable: dict[str, Any] = Field(
        default_factory=dict,
        description="Passed to the graph as config['configurable']; thread and "
        "checkpoint keys are reserved.",
    )
    recursion_limit: int | None = Field(default=None, ge=1)


class RunCreate(BaseModel):
    """``{input, config?}``."""

    model_config = ConfigDict(extra="forbid")

    input: Any = Field(
        default=None, description="The graph input; validated against the agent's input_schema."
    )
    config: RunConfig | None = None


class TriggerCreate(BaseModel):
    """``{input, config?, thread_id?, callback?}`` for ``POST /agents/{name}/trigger``."""

    model_config = ConfigDict(extra="forbid")

    input: Any = Field(
        default=None, description="The graph input; validated against the agent's input_schema."
    )
    config: RunConfig | None = None
    thread_id: str | None = Field(
        default=None,
        description="Continue this thread (one the caller owns); omitted, a new thread is created.",
    )
    callback: str | None = Field(
        default=None,
        max_length=2048,
        description="URL notified when the run completes, fails or raises an interrupt; "
        "its host must be on CALLBACK_ALLOWLIST.",
    )


class RunCreated(BaseModel):
    run_id: str
    thread_id: str
    status: str


# -- interrupts ------------------------------------------------------------------


class InterruptOut(BaseModel):
    """The record."""

    interrupt_id: str
    run_id: str
    thread_id: str
    agent: str
    kind: str
    scope: str | None = None
    approver_role: str | None = None
    actions: list[str]
    payload: dict[str, Any]
    edit_schema: dict[str, Any] | None = None
    status: str = Field(description="pending | resolved | expired | cancelled")
    version: int = Field(description="Send back as expected_version when resolving.")
    expires_at: str | None = None
    created_at: str
    resolved_by: str | None = None
    resolution: Any = None


class InterruptList(BaseModel):
    interrupts: list[InterruptOut]


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str
    payload: dict[str, Any] | None = None
    text: str | None = None


class ResolveRequest(BaseModel):
    """``{action, payload?, text?, expected_version}`` or ``{decisions, expected_version}``.

    The second form serves multi-action interrupts, one decision per action in
    order.
    """

    model_config = ConfigDict(extra="forbid")

    action: str | None = None
    payload: dict[str, Any] | None = None
    text: str | None = None
    decisions: list[Decision] | None = None
    expected_version: int = Field(description="The record's version you resolved against.")

    @pydantic.model_validator(mode="after")
    def _one_form(self) -> ResolveRequest:
        if (self.action is None) == (self.decisions is None):
            raise ValueError("send exactly one of 'action' or 'decisions'")
        if self.decisions is not None and not self.decisions:
            raise ValueError("'decisions' must not be empty")
        if self.decisions is not None and (self.payload is not None or self.text is not None):
            raise ValueError("'payload'/'text' belong inside each decision")
        return self


# -- artifacts -------------------------------------------------------------------


class ArtifactOut(BaseModel):
    """The ``artifact.created`` field set."""

    artifact_id: str
    question_id: str = Field(description="The run id.")
    session_id: str = Field(description="The thread id.")
    kind: str = Field(description="table | image | html | json | file")
    renderer: str | None = None
    name: str
    mime_type: str
    size_bytes: int | None = None
    preview: dict[str, Any]
    view_url: str
    download_url: str
    metadata: dict[str, Any]


class ArtifactPreview(BaseModel):
    """The stored preview; for tables also one page of rows (``/preview``)."""

    artifact_id: str
    kind: str
    preview: dict[str, Any]
    offset: int
    limit: int
    sort: str | None = None
    columns: list[dict[str, str]] | None = Field(
        default=None, description="Tables: ``[{name, type}]``."
    )
    rows: list[dict[str, Any]] | None = Field(
        default=None, description="Tables: the requested page of rows."
    )
    row_count: int | None = Field(default=None, description="Tables: total rows.")


# -- ops -------------------------------------------------------------------------


class HealthOut(BaseModel):
    status: str


class ReadyOut(BaseModel):
    status: str
    checks: dict[str, str]
