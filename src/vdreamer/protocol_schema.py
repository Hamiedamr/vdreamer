"""The protocol v1.0 event JSON Schema.

One JSON Schema (draft 2020-12) for the wire envelope: the envelope fields
(``v`` pinned to ``"1.0"``, ``type`` one of the twelve event types, …) plus,
per event type, the payload shape from the table — an ``if type == T
then payload is T's model`` clause per type. Payload shapes come straight
from the pydantic models in :mod:`vdreamer.events`, so the schema can never
drift from what the drivers emit.

Served at ``GET /api/v1/protocol/schema.json``;
``vdreamer export-schema`` writes the same document to disk. The ``$schema``
URI is an identifier, never fetched.

Clients ignore unknown event types, so the schema constrains the
payload only for the types it knows and leaves ``additionalProperties`` open
on the envelope and every payload — a minor protocol addition must not break a
validating client.
"""

from __future__ import annotations

from typing import Any, Final

import pydantic
from pydantic.json_schema import models_json_schema

from vdreamer.events import (
    PROTOCOL_VERSION,
    ArtifactCreatedPayload,
    CitationPayload,
    EventType,
    InterruptRequestedPayload,
    InterruptResolvedPayload,
    LogPayload,
    MessageCompletedPayload,
    ProgressPayload,
    RunStatusPayload,
    TokenDeltaPayload,
    ToolCallPayload,
    ToolResultPayload,
)

#: Each event type's payload model (the table).
PAYLOAD_MODELS: Final[dict[EventType, type[pydantic.BaseModel]]] = {
    EventType.RUN_STATUS: RunStatusPayload,
    EventType.TOKEN_DELTA: TokenDeltaPayload,
    EventType.THINKING_DELTA: TokenDeltaPayload,
    EventType.MESSAGE_COMPLETED: MessageCompletedPayload,
    EventType.TOOL_CALL: ToolCallPayload,
    EventType.TOOL_RESULT: ToolResultPayload,
    EventType.ARTIFACT_CREATED: ArtifactCreatedPayload,
    EventType.CITATION: CitationPayload,
    EventType.INTERRUPT_REQUESTED: InterruptRequestedPayload,
    EventType.INTERRUPT_RESOLVED: InterruptResolvedPayload,
    EventType.PROGRESS: ProgressPayload,
    EventType.LOG: LogPayload,
}

_DEFS_REF: Final[str] = "#/$defs/{model}"


def event_json_schema() -> dict[str, Any]:
    """Build the protocol event JSON Schema (draft 2020-12)."""
    models = sorted(set(PAYLOAD_MODELS.values()), key=lambda model: model.__name__)
    refs, top = models_json_schema(
        [(model, "serialization") for model in models], ref_template=_DEFS_REF
    )
    defs: dict[str, Any] = dict(top.get("$defs", {}))

    clauses: list[dict[str, Any]] = []
    for event_type, model in PAYLOAD_MODELS.items():
        clauses.append(
            {
                "if": {
                    "properties": {"type": {"const": event_type.value}},
                    "required": ["type"],
                },
                "then": {"properties": {"payload": refs[(model, "serialization")]}},
            }
        )

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"vdreamer protocol {PROTOCOL_VERSION} event envelope",
        "description": (
            "One SSE event. The SSE id is '{run_id}:{seq}'. Clients "
            "deduplicate by seq and ignore unknown event types."
        ),
        "type": "object",
        "required": ["v", "type", "run_id", "thread_id", "agent", "seq", "ts", "ns", "payload"],
        "properties": {
            "v": {"const": PROTOCOL_VERSION},
            "type": {"type": "string", "examples": [event.value for event in EventType]},
            "run_id": {"type": "string"},
            "thread_id": {"type": "string"},
            "agent": {"type": "string"},
            "seq": {"type": "integer", "minimum": 1},
            "ts": {"type": "string", "format": "date-time"},
            "ns": {"type": "array", "items": {"type": "string"}},
            "payload": {"type": "object"},
        },
        "allOf": clauses,
        "$defs": defs,
        "x-event-types": [event.value for event in EventType],
    }
