"""The Python reference fold (PRD §6.1) — test-only, never shipped (no client SDK, A45).

It is the executable reading of the kit's README: parse SSE text, dedup by
``(run_id, seq)``, ignore unknown event types, check the protocol major, and
fold every known event into the §6.1 client state::

    {status, error, messages[{id, text, thinking, citations[]}], toolCalls[],
     artifacts[], interrupts[], progress, logs}

Client teams port *this* behavior to their stack and prove it against the
same recordings; the conformance test proves this implementation does.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_MAJOR = "1"

#: Artifact kinds protocol 1.0 defines (§3.4); anything else renders as a download.
KNOWN_KINDS = frozenset({"table", "image", "html", "json", "file"})

#: Renderers this reference client implements; an unknown renderer is a download (§6.1).
KNOWN_RENDERERS = frozenset({"verdict.v1"})

TERMINAL = frozenset({"completed", "failed", "cancelled", "expired", "rejected"})


class ProtocolMismatch(ValueError):
    """An event of another protocol major: the client must refuse the stream (§6.1)."""


@dataclass
class SseEvent:
    id: str | None
    data: str


def parse_sse(text: str) -> list[SseEvent]:
    """The SSE wire format: ``id:``/``data:`` fields, blank-line dispatch, ``:`` comments."""
    events: list[SseEvent] = []
    event_id: str | None = None
    data: list[str] = []
    for line in text.splitlines():
        if line == "":
            if data:
                events.append(SseEvent(event_id, "\n".join(data)))
            event_id, data = None, []
        elif line.startswith(":"):
            continue  # heartbeat / comment
        else:
            name, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if name == "id":
                event_id = value
            elif name == "data":
                data.append(value)
    if data:
        events.append(SseEvent(event_id, "\n".join(data)))
    return events


def _message(state: dict[str, Any], message_id: str) -> dict[str, Any]:
    for message in state["messages"]:
        if message["id"] == message_id:
            return message
    message = {"id": message_id, "text": "", "thinking": "", "citations": [], "completed": False}
    state["messages"].append(message)
    return message


def _render(kind: str, renderer: str | None) -> str:
    if kind not in KNOWN_KINDS:
        return "download"
    if renderer is not None and renderer not in KNOWN_RENDERERS:
        return "download"
    return kind


@dataclass
class Fold:
    """One client's view of a stream (or a thread's streams)."""

    state: dict[str, Any] = field(
        default_factory=lambda: {
            "status": None,
            "error": None,
            "messages": [],
            "toolCalls": [],
            "artifacts": [],
            "interrupts": [],
            "progress": None,
            "logs": [],
        }
    )
    seen: set[tuple[str, int]] = field(default_factory=set)
    last_event_id: str | None = None
    ignored: list[str] = field(default_factory=list)

    def feed(self, envelope: dict[str, Any], sse_id: str | None = None) -> None:
        major = str(envelope.get("v", "")).split(".", 1)[0]
        if major != PROTOCOL_MAJOR:
            raise ProtocolMismatch(f"protocol {envelope.get('v')!r} is not {PROTOCOL_MAJOR}.x")
        key = (str(envelope["run_id"]), int(envelope["seq"]))
        if key in self.seen:
            return  # dedup by seq (per run)
        self.seen.add(key)
        if sse_id is not None:
            self.last_event_id = sse_id
        handler = getattr(self, "_on_" + str(envelope["type"]).replace(".", "_"), None)
        if handler is None:
            self.ignored.append(str(envelope["type"]))  # unknown type: ignore (§6.1)
            return
        handler(envelope["payload"])

    def feed_sse(self, text: str) -> None:
        for event in parse_sse(text):
            self.feed(json.loads(event.data), event.id)

    # -- handlers ------------------------------------------------------------------------

    def _on_run_status(self, payload: dict[str, Any]) -> None:
        self.state["status"] = payload["status"]
        self.state["error"] = payload.get("error")

    def _on_token_delta(self, payload: dict[str, Any]) -> None:
        message = _message(self.state, payload["message_id"])
        if not message["completed"]:  # message.completed is authoritative
            message["text"] += payload["text"]

    def _on_thinking_delta(self, payload: dict[str, Any]) -> None:
        message = _message(self.state, payload["message_id"])
        if not message["completed"]:
            message["thinking"] += payload["text"]

    def _on_message_completed(self, payload: dict[str, Any]) -> None:
        message = _message(self.state, payload["message_id"])
        message["text"] = payload["text"]
        if payload.get("thinking") is not None:
            message["thinking"] = payload["thinking"]
        message["completed"] = True

    def _on_citation(self, payload: dict[str, Any]) -> None:
        message = _message(self.state, payload["message_id"])
        if any(c["citation_id"] == payload["citation_id"] for c in message["citations"]):
            return
        message["citations"].append(
            {key: value for key, value in payload.items() if key != "message_id"}
        )

    def _on_tool_call(self, payload: dict[str, Any]) -> None:
        # Upsert: a resumed run re-executes its node from the top (§4), re-emitting
        # the same tool_call_id — the call is running again, not a second call.
        for call in self.state["toolCalls"]:
            if call["tool_call_id"] == payload["tool_call_id"]:
                call.update(
                    name=payload["name"], source=payload["source"], args=payload["args"],
                    status="running",
                )
                return
        self.state["toolCalls"].append(
            {
                "tool_call_id": payload["tool_call_id"],
                "name": payload["name"],
                "source": payload["source"],
                "args": payload["args"],
                "status": "running",
                "summary": None,
                "artifact_ids": [],
                "error": None,
            }
        )

    def _on_tool_result(self, payload: dict[str, Any]) -> None:
        call = next(
            (c for c in self.state["toolCalls"] if c["tool_call_id"] == payload["tool_call_id"]),
            None,
        )
        if call is None:  # a result without its call (a mid-run join): keep it anyway
            call = {"tool_call_id": payload["tool_call_id"], "name": None, "source": None,
                    "args": {}}
            self.state["toolCalls"].append(call)
        call.update(
            status=payload["status"],
            summary=payload["summary"],
            artifact_ids=list(payload.get("artifact_ids") or []),
            error=payload.get("error"),
        )

    def _on_artifact_created(self, payload: dict[str, Any]) -> None:
        if any(a["artifact_id"] == payload["artifact_id"] for a in self.state["artifacts"]):
            return
        self.state["artifacts"].append(
            {
                "artifact_id": payload["artifact_id"],
                "kind": payload["kind"],
                "renderer": payload.get("renderer"),
                "name": payload["name"],
                "mime_type": payload.get("mime_type"),
                "view_url": payload.get("view_url"),
                "download_url": payload.get("download_url"),
                "render": _render(str(payload["kind"]), payload.get("renderer")),
            }
        )

    def _on_interrupt_requested(self, payload: dict[str, Any]) -> None:
        if any(i["interrupt_id"] == payload["interrupt_id"] for i in self.state["interrupts"]):
            return
        self.state["interrupts"].append(
            {
                "interrupt_id": payload["interrupt_id"],
                "kind": payload["kind"],
                "actions": list(payload["actions"]),
                "approver_role": payload.get("approver_role"),
                "scope": payload.get("scope"),
                "status": "pending",
                "action": None,
                "resolved_by": None,
            }
        )

    def _on_interrupt_resolved(self, payload: dict[str, Any]) -> None:
        for interrupt in self.state["interrupts"]:
            if interrupt["interrupt_id"] == payload["interrupt_id"]:
                interrupt.update(
                    status="resolved", action=payload["action"], resolved_by=payload["resolved_by"]
                )

    def _on_progress(self, payload: dict[str, Any]) -> None:
        self.state["progress"] = {"label": payload["label"], "pct": payload.get("pct")}

    def _on_log(self, payload: dict[str, Any]) -> None:
        self.state["logs"].append({"level": payload["level"], "message": payload["message"]})

    def result(self) -> dict[str, Any]:
        """The client state, without the fold's own bookkeeping."""
        state = json.loads(json.dumps(self.state))
        for message in state["messages"]:
            message.pop("completed", None)
        return dict(state)


def fold_recording(recording: dict[str, Any]) -> dict[str, Any]:
    """Replay a kit recording's SSE steps (honoring ``reset``) into the final state."""
    fold = Fold()
    for step in recording["steps"]:
        if step["kind"] != "sse":
            continue
        if step.get("reset"):
            fold = Fold()  # a page refresh: the client starts over from the thread replay
        fold.feed_sse(step["body"])
    return fold.result()


def fold_events(envelopes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    fold = Fold()
    for envelope in envelopes:
        fold.feed(envelope)
    return fold.result()
