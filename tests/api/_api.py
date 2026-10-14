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
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import aiosqlite
import pydantic
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from vdreamer import BaseAgent, BuildContext, Mode, emit_progress, require_approval
from vdreamer.artifacts import InMemoryArtifactStore
from vdreamer.eventlog import InMemoryEventLog
from vdreamer.gateway import GatewayBackend, GatewaySettings, create_app
from vdreamer.interrupts import InMemoryInterruptStore
from vdreamer.queue import InMemoryRunQueue
from vdreamer.runtime import AgentRunner
from vdreamer.sqlite_state import (
    SqliteEventLog,
    SqliteInterruptStore,
    SqliteRunQueue,
    SqliteThreadStore,
    open_state_db,
)
from vdreamer.threads import InMemoryThreadStore
from vdreamer.worker import Worker

TOKEN = "current-internal-token-0123456789"
PREVIOUS_TOKEN = "previous-internal-token-012345678"


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


def all_agents() -> dict[str, BaseAgent]:
    agents: list[BaseAgent] = [EchoAgent(), FormAgent(), BatchAgent(), SecretAgent(), GateAgent()]
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
    artifacts: InMemoryArtifactStore
    checkpointer: InMemorySaver
    db: aiosqlite.Connection | None = None

    def close(self) -> None:
        if self.db is not None:
            asyncio.run(self.db.close())


def make_stack(
    *, workers: bool, heartbeat_s: float = 0.2, sqlite_path: Path | None = None
) -> Stack:
    """A gateway stack; ``sqlite_path`` switches every store to the SQLite backend.

    aiosqlite resolves each call on the caller's own loop, so the stores work
    from the TestClient's loop and from the tests' ``asyncio.run`` alike.
    """
    agents = all_agents()
    queue: InMemoryRunQueue
    log: InMemoryEventLog | SqliteEventLog
    threads: InMemoryThreadStore
    interrupts: InMemoryInterruptStore
    db = None
    if sqlite_path is None:
        queue = InMemoryRunQueue(clock=time.time)
        log = InMemoryEventLog()
        threads = InMemoryThreadStore()
        interrupts = InMemoryInterruptStore()
    else:
        db = asyncio.run(open_state_db(sqlite_path))
        queue = SqliteRunQueue(db)
        log = SqliteEventLog(db)
        threads = SqliteThreadStore(db)
        interrupts = SqliteInterruptStore(db)
    artifacts = InMemoryArtifactStore()
    checkpointer = InMemorySaver()
    backend = GatewayBackend(
        agents=agents,
        queue=queue,
        threads=threads,
        event_reader=log,
        event_writer=log,
        hub=log.hub,
        interrupts=interrupts,
        artifacts=artifacts,
        checkpointer=checkpointer,
    )

    lifespan = None
    if workers:

        @contextlib.asynccontextmanager
        async def lifespan() -> AsyncIterator[None]:
            async def sink_factory(run_id: str) -> InMemoryEventLog | SqliteEventLog:
                return log

            stop = asyncio.Event()
            tasks = [
                asyncio.create_task(
                    Worker(
                        AgentRunner(agent, checkpointer=checkpointer),
                        queue,
                        agent_name=agent.name,
                        claimant=f"w-{agent.name}",
                        sink_factory=sink_factory,
                        poll_interval_s=0.01,
                        interrupt_store=interrupts,
                    ).run(stop)
                )
                for agent in agents.values()
            ]
            try:
                yield
            finally:
                stop.set()
                await asyncio.wait_for(asyncio.gather(*tasks), 5.0)

    app = create_app(
        backend,
        GatewaySettings(tokens=(TOKEN, PREVIOUS_TOKEN), heartbeat_s=heartbeat_s),
        lifespan=lifespan,
    )
    return Stack(app, backend, queue, log, threads, interrupts, artifacts, checkpointer, db)


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
