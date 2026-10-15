"""REST contract kit (Milestone 4): an in-process gateway, with or without workers.

Every stack runs over either store backend (``store``): the in-memory twins,
or the SQLite stores ``vdreamer dev`` persists to (``sqlite_state``) — the
fixtures run each test against both. Two stacks:

- ``client`` — the gateway alone: runs stay ``queued`` (no worker claims them),
  so route contracts are deterministic;
- ``live`` — the gateway plus one :class:`~vdreamer.worker.Worker` per agent
  running inside the app's lifespan (exactly the ``vdreamer dev`` wiring), for
  end-to-end flows through the REST surface.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import aiosqlite
import pydantic
from _hitl_agents import hitl_agents
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode, emit_progress, require_approval
from vdreamer.artifacts import ArtifactStore, InMemoryArtifactStore
from vdreamer.eventlog import InMemoryEventLog
from vdreamer.gateway import GatewayBackend, GatewaySettings, create_app
from vdreamer.interrupts import InMemoryInterruptStore
from vdreamer.network import NetworkAllowlist
from vdreamer.outbox import InMemoryOutboxStore, OutboxEventSink, OutboxSettings
from vdreamer.queue import InMemoryRunQueue
from vdreamer.runtime import AgentRunner
from vdreamer.sqlite_state import (
    SqliteArtifactStore,
    SqliteEventLog,
    SqliteInterruptStore,
    SqliteOutboxStore,
    SqliteRunQueue,
    SqliteThreadStore,
    open_state_db,
)
from vdreamer.threads import InMemoryThreadStore
from vdreamer.worker import Worker

TOKEN = "current-internal-token-0123456789"
#: ``HITL_STRICT_SCOPES`` of every kit gateway (A32): ``strict``'s scope matches.
STRICT_SCOPES = ("db:write:*",)
PREVIOUS_TOKEN = "previous-internal-token-012345678"
#: The kit's upstream service (A16): ``service:mail-worker``.
SERVICE = "mail-worker"
SERVICE_TOKEN = "mail-worker-service-token-0123456789"
#: The kit's ``CALLBACK_ALLOWLIST`` (A26).
CALLBACK_ALLOWLIST = "*.internal,10.20.0.0/16"
#: The kit's outbox channels (M9): the operator webhook and the review deep link.
OUTBOX_SETTINGS = OutboxSettings(
    webhook_url="https://hooks.internal/vdreamer",
    secret="outbox-secret-0123456789",
    web_base_url="https://web.internal",
)


def headers(
    user: str | None = "alice",
    roles: tuple[str, ...] = (),
    token: str | None = TOKEN,
    **extra: str,
) -> dict[str, str]:
    """Request headers as the Web Backend sends them (D13)."""
    out: dict[str, str] = {}
    if token is not None:
        out["Authorization"] = f"Bearer {token}"
    if user is not None:
        out["X-User-Id"] = user
    if roles:
        out["X-User-Roles"] = ",".join(roles)
    out.update(extra)
    return out


# -- agents ---------------------------------------------------------------------------


class EchoState(TypedDict, total=False):
    text: str
    reply: str


class EchoAgent(BaseAgent):
    """Chat: one node echoing the input, with a progress event."""

    name = "echo"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> Any:
        def echo(state: EchoState) -> dict[str, Any]:
            emit_progress("echoing", 50.0)
            return {"reply": f"echo:{state.get('text', '')}"}

        graph = StateGraph(EchoState)
        graph.add_node("echo", echo)
        graph.add_edge(START, "echo")
        graph.add_edge("echo", END)
        return graph


class FormInput(pydantic.BaseModel):
    text: str
    n: int = 1


class FormAgent(EchoAgent):
    """Both modes, with an input_schema the gateway validates against."""

    name = "form"
    mode = Mode.BOTH
    input_schema = FormInput


class BatchAgent(EchoAgent):
    """Headless only: no threads/runs from the chat routes (D15)."""

    name = "batch"
    mode = Mode.HEADLESS
    input_schema = FormInput


class SecretAgent(EchoAgent):
    """Role-gated (§2.2 ``access_roles``)."""

    name = "secret"
    access_roles = ("netops",)


class GateState(TypedDict, total=False):
    text: str
    decision: Any


class GateAgent(BaseAgent):
    """Chat: pauses on ``require_approval`` (initiator authority), then records it."""

    name = "gate"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> Any:
        def gate(state: GateState) -> dict[str, Any]:
            decision = require_approval(
                "review",
                payload={"draft": state.get("text", "")},
                actions=["approve", "reject", "edit", "feedback"],
                ttl_hours=1,
            )
            return {"decision": decision}

        graph = StateGraph(GateState)
        graph.add_node("gate", gate)
        graph.add_edge(START, "gate")
        graph.add_edge("gate", END)
        return graph


#: What the ``tooling`` agent's governed tool returns: far over 8,000 chars (§2.7).
BIG_RESULT = json.dumps({"rows": [{"i": i, "label": "row-" + "y" * 30} for i in range(400)]})


class ToolingAgent(BaseAgent):
    """Chat: one governed tool call whose result overflows into an artifact (M5)."""

    name = "tooling"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> Any:
        @vdreamer.tool(redact=["query"])
        def report(query: str, token: str) -> str:
            """Build a large report."""
            return BIG_RESULT

        def plan(state: MessagesState) -> dict[str, Any]:
            if any(getattr(m, "type", "") == "tool" for m in state["messages"]):
                return {"messages": [AIMessage("done", id="ai_done")]}
            call = {"name": "report", "args": {"query": "q1", "token": "t0k"}, "id": "call_r"}
            return {"messages": [AIMessage("", id="ai_call", tool_calls=[call])]}

        def route(state: MessagesState) -> str:
            last = state["messages"][-1]
            return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

        graph = StateGraph(MessagesState)
        graph.add_node("plan", plan)
        graph.add_node("tools", ToolNode(ctx.tools.govern([report])))
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", route)
        graph.add_edge("tools", "plan")
        return graph


class ElicitState(TypedDict, total=False):
    text: str
    answer: Any


class ElicitAgent(BaseAgent):
    """Chat: raises the interrupt ``langchain.mcp`` raises for an MCP elicitation."""

    name = "elicit"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> Any:
        def ask(state: ElicitState) -> dict[str, Any]:
            answer = interrupt(
                {
                    "type": "mcp_elicitation",
                    "tool_name": "fx_book",
                    "requests": [
                        {
                            "key": "k1",
                            "message": "Which date?",
                            "mode": "form",
                            "requested_schema": {
                                "type": "object",
                                "properties": {"date": {"type": "string"}},
                            },
                        }
                    ],
                }
            )
            return {"answer": answer}

        graph = StateGraph(ElicitState)
        graph.add_node("ask", ask)
        graph.add_edge(START, "ask")
        graph.add_edge("ask", END)
        return graph


def tiny_png(width: int = 3, height: int = 2) -> bytes:
    """A valid RGB PNG (no imaging library needed)."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


#: Rows of the ``reports`` agent's table artifact (more than one Parquet page).
REPORT_ROWS = 2500


class ReportState(TypedDict, total=False):
    text: str
    refs: dict[str, str]


class ReportsAgent(BaseAgent):
    """Chat: one node emitting every artifact kind (M6), the HTML linking the table."""

    name = "reports"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> Any:
        def report(state: ReportState) -> dict[str, Any]:
            from vdreamer import ArtifactKind, emit_artifact

            rows = (
                {"i": i, "city": f"c{i % 7}", "score": (i * 37) % 101} for i in range(REPORT_ROWS)
            )
            table = emit_artifact("scores", ArtifactKind.TABLE, rows)
            image = emit_artifact("chart", ArtifactKind.IMAGE, tiny_png())
            html = emit_artifact(
                "report",
                ArtifactKind.HTML,
                "<html><head><title>Q3 report</title>"
                '<meta name="description" content="Scores by city">'
                '<script src="/static/vendor/plotly/plotly.min.js"></script></head>'
                f'<body><a href="artifact://{table.artifact_id}">data</a>'
                f'<img src="artifact://{image.artifact_id}"></body></html>',
            )
            verdict = emit_artifact(
                "verdict", ArtifactKind.JSON, {"ok": True}, renderer="verdict.v1"
            )
            notes = emit_artifact(
                "notes", ArtifactKind.FILE, b"plain notes", mime_type="text/plain",
                filename="notes.txt",
            )
            refs = {
                "table": table.artifact_id,
                "image": image.artifact_id,
                "html": html.artifact_id,
                "json": verdict.artifact_id,
                "file": notes.artifact_id,
            }
            return {"refs": refs}

        graph = StateGraph(ReportState)
        graph.add_node("report", report)
        graph.add_edge(START, "report")
        graph.add_edge("report", END)
        return graph


class ReviewedAgent(BaseAgent):
    """Chat: emits a table, then waits for a ``reviewer`` (granted-approver access, A8)."""

    name = "reviewed"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> Any:
        async def draft(state: GateState) -> dict[str, Any]:
            from vdreamer import ArtifactKind, aemit_artifact

            ref = await aemit_artifact("draft", ArtifactKind.TABLE, [{"q": "SELECT 1"}])
            return {"text": ref.artifact_id}

        def gate(state: GateState) -> dict[str, Any]:
            decision = require_approval(
                "sql_review", payload={"artifact": state.get("text")}, approver_role="reviewer"
            )
            return {"decision": decision}

        graph = StateGraph(GateState)
        graph.add_node("draft", draft)
        graph.add_node("gate", gate)
        graph.add_edge(START, "draft")
        graph.add_edge("draft", "gate")
        graph.add_edge("gate", END)
        return graph


class PackageInput(pydantic.BaseModel):
    site_id: str
    package: vdreamer.FileRef


class UnpackState(TypedDict, total=False):
    site_id: str
    package: dict[str, Any]
    manifest: dict[str, Any]


class UnpackAgent(BaseAgent):
    """Both modes: ``FileRef`` input → ``safe_extract`` → a ``json`` manifest artifact (UC2)."""

    name = "unpack"
    mode = Mode.BOTH
    input_schema = PackageInput

    def build(self, ctx: BuildContext) -> Any:
        def unpack(state: UnpackState) -> dict[str, Any]:
            from vdreamer import ArtifactKind, FileRef, current_context, emit_artifact

            files = current_context().files
            ref = FileRef.model_validate(state["package"])
            manifest = files.safe_extract(ref, allow=["*.txt", "image/*"])
            first = files.read_bytes(manifest.files[0]) if manifest.files else b""
            emit_artifact(
                "manifest", ArtifactKind.JSON, manifest, metadata={"first": first.decode()}
            )
            return {"manifest": manifest.model_dump()}

        graph = StateGraph(UnpackState)
        graph.add_node("unpack", unpack)
        graph.add_edge(START, "unpack")
        graph.add_edge("unpack", END)
        return graph


def all_agents() -> dict[str, BaseAgent]:
    agents: list[BaseAgent] = [
        EchoAgent(),
        FormAgent(),
        BatchAgent(),
        SecretAgent(),
        GateAgent(),
        ToolingAgent(),
        ElicitAgent(),
        ReportsAgent(),
        ReviewedAgent(),
        UnpackAgent(),
        *hitl_agents(),
    ]
    return {agent.name: agent for agent in agents}


# -- stacks ------------------------------------------------------------------------------


@dataclass
class Stack:
    app: FastAPI
    backend: GatewayBackend
    queue: InMemoryRunQueue
    log: InMemoryEventLog | SqliteEventLog
    threads: InMemoryThreadStore
    interrupts: InMemoryInterruptStore
    artifacts: ArtifactStore
    checkpointer: InMemorySaver
    db: aiosqlite.Connection | None = None
    outbox: InMemoryOutboxStore | None = None

    def close(self) -> None:
        if self.db is not None:
            asyncio.run(self.db.close())


def make_stack(
    *,
    workers: bool,
    heartbeat_s: float = 0.2,
    sqlite_path: Path | None = None,
    extra_agents: Sequence[BaseAgent] = (),
    worker_agents: Sequence[str] | None = None,
    poll_interval_s: float = 0.01,
) -> Stack:
    """A gateway stack; ``sqlite_path`` switches every store to the SQLite backend.

    aiosqlite resolves each call on the caller's own loop, so the stores work
    from the TestClient's loop and from the tests' ``asyncio.run`` alike.
    ``extra_agents`` join the kit's own (the conformance recorder adds its own);
    ``worker_agents`` limits which agents get a worker (all by default).
    """
    agents = {**all_agents(), **{agent.name: agent for agent in extra_agents}}
    queue: InMemoryRunQueue
    log: InMemoryEventLog | SqliteEventLog
    threads: InMemoryThreadStore
    interrupts: InMemoryInterruptStore
    artifacts: ArtifactStore
    db = None
    if sqlite_path is None:
        queue = InMemoryRunQueue(clock=time.time)
        log = InMemoryEventLog()
        threads = InMemoryThreadStore()
        interrupts = InMemoryInterruptStore()
        artifacts = InMemoryArtifactStore()
        outbox: InMemoryOutboxStore = InMemoryOutboxStore()
    else:
        db = asyncio.run(open_state_db(sqlite_path))
        queue = SqliteRunQueue(db)
        log = SqliteEventLog(db)
        threads = SqliteThreadStore(db)
        interrupts = SqliteInterruptStore(db)
        artifacts = SqliteArtifactStore(db, sqlite_path.parent)
        outbox = SqliteOutboxStore(db)
    checkpointer = InMemorySaver()
    # Exactly the `vdreamer dev` wiring: every event write records its outbox entries.
    writer = OutboxEventSink(log, outbox, queue, OUTBOX_SETTINGS)
    backend = GatewayBackend(
        agents=agents,
        queue=queue,
        threads=threads,
        event_reader=log,
        event_writer=writer,
        hub=log.hub,
        interrupts=interrupts,
        artifacts=artifacts,
        checkpointer=checkpointer,
    )

    lifespan = None
    if workers:

        @contextlib.asynccontextmanager
        async def lifespan() -> AsyncIterator[None]:
            async def sink_factory(run_id: str) -> OutboxEventSink:
                return writer

            stop = asyncio.Event()
            tasks = [
                asyncio.create_task(
                    Worker(
                        AgentRunner(agent, checkpointer=checkpointer),
                        queue,
                        agent_name=agent.name,
                        claimant=f"w-{agent.name}",
                        sink_factory=sink_factory,
                        poll_interval_s=poll_interval_s,
                        interrupt_store=interrupts,
                        artifact_store=artifacts,
                    ).run(stop)
                )
                for agent in agents.values()
                if worker_agents is None or agent.name in worker_agents
            ]
            try:
                yield
            finally:
                stop.set()
                await asyncio.wait_for(asyncio.gather(*tasks), 5.0)

    app = create_app(
        backend,
        GatewaySettings(
            tokens=(TOKEN, PREVIOUS_TOKEN),
            heartbeat_s=heartbeat_s,
            strict_scopes=STRICT_SCOPES,
            service_tokens=((SERVICE, SERVICE_TOKEN),),
            callback_allowlist=NetworkAllowlist.parse(CALLBACK_ALLOWLIST),
        ),
        lifespan=lifespan,
    )
    return Stack(
        app, backend, queue, log, threads, interrupts, artifacts, checkpointer, db, outbox
    )


# -- helpers --------------------------------------------------------------------------------


def assert_error(response: Any, status: int, code: str) -> dict[str, Any]:
    """The one error body (§2.5) with the expected status and stable code."""
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"}, body
    assert body["error"]["code"] == code, body
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    assert set(body["error"]) <= {"code", "message", "details"}
    return dict(body["error"])


def new_thread(client: TestClient, agent: str = "echo", **kw: Any) -> str:
    response = client.post(f"/api/v1/agents/{agent}/threads", headers=headers(**kw))
    assert response.status_code == 201, response.text
    return str(response.json()["thread_id"])


def start_run(client: TestClient, thread_id: str, body: dict[str, Any], **kw: Any) -> str:
    response = client.post(f"/api/v1/threads/{thread_id}/runs", json=body, headers=headers(**kw))
    assert response.status_code == 202, response.text
    return str(response.json()["run_id"])


def wait_for_status(
    client: TestClient, run_id: str, status: str, timeout: float = 5.0, **kw: Any
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        body = client.get(f"/api/v1/runs/{run_id}", headers=headers(**kw)).json()
        if body.get("status") == status:
            return dict(body)
        if time.monotonic() > deadline:
            raise AssertionError(f"run {run_id} never reached {status!r}: {body}")
        time.sleep(0.02)


@dataclass
class SseFrame:
    id: str | None
    data: dict[str, Any] | None
    comment: str | None = None


def read_sse(client: TestClient, url: str, **kw: Any) -> tuple[Any, list[SseFrame]]:
    """Read one SSE response to its end (the server closes after a terminal status)."""
    request_headers = headers(**kw)
    with client.stream("GET", url, headers=request_headers) as response:
        frames: list[SseFrame] = []
        if response.status_code != 200:
            response.read()
            return response, frames
        event_id: str | None = None
        data: list[str] = []
        for line in response.iter_lines():
            if line == "":
                if data:
                    frames.append(SseFrame(event_id, json.loads("\n".join(data))))
                event_id, data = None, []
            elif line.startswith(":"):
                frames.append(SseFrame(None, None, comment=line[1:].strip()))
            elif line.startswith("id:"):
                event_id = line[3:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        return response, frames


def events_of(frames: list[SseFrame]) -> list[dict[str, Any]]:
    return [frame.data for frame in frames if frame.data is not None]
