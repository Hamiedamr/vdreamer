"""Shared scaffolding for the WP2 stream-driver tests (PRD §2.4, A31, A43).

Build-blocks (agents over tiny ``StateGraph`` graphs on ``InMemorySaver`` —
never a live Postgres) plus :func:`run_graph_all_drivers`, the helper the
Milestone-2 tests and the WP3 golden-event suite build on: run the *same*
agent through *both* drivers and compare envelope sequences.
"""

from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from vdreamer.agents import BaseAgent, BuildContext, Mode
from vdreamer.context import (
    ArtifactKind,
    VDreamerContext,
    emit_artifact,
    emit_citation,
    emit_log,
    emit_progress,
)
from vdreamer.eventlog import EventEnvelope, InMemoryEventSink
from vdreamer.runtime import AgentRunner
from vdreamer.streaming import STREAM_DRIVER_ENV_VAR

#: Built by concatenation so the literal tag never appears in source.
THINK_CLOSE = "<" + "/" + "think" + ">"


class ChatState(TypedDict):
    """One messages channel plus a flag, enough for every test graph."""

    messages: Any
    flag: str


def _fake_model(*chunks: AIMessageChunk) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter(list(chunks)))


def _text_chunk(text: str, message_id: str, **extra: Any) -> AIMessageChunk:
    return AIMessageChunk(content=text, id=message_id, **extra)  # type: ignore[arg-type]


class BaseTestAgent(BaseAgent):
    """Minimal agent defaults; subclasses set ``name`` and ``build``."""

    mode = Mode.CHAT
    expose_thinking = True


class ChattyAgent(BaseTestAgent):
    """Streams tokens, emits every custom record kind, then answers."""

    name = "chatty"

    def build(self, ctx: BuildContext) -> Any:
        def node(state: ChatState) -> dict[str, Any]:
            emit_progress("parsing", 10.0)
            emit_log("hello from the node", level="info")
            emit_artifact("verdict", ArtifactKind.JSON, {"ok": True}, renderer="verdict.v1")
            model = _fake_model(
                _text_chunk("Hel", "m1"),
                _text_chunk("lo ", "m2"),
                _text_chunk("world", "m3"),
            )
            answer = model.invoke(state["messages"])
            # Bind the citation to the message the model actually produced —
            # the driver's id (the v2 and v3 protocols assign different message
            # ids to the same model call), like UC3 binds chunk citations.
            emit_citation(
                str(answer.id),
                {"uri": "doc://a", "title": "Doc A"},
                locator={"page": 3},
                snippet="quote",
            )
            get_stream_writer()({"untyped": ["custom", "data"]})
            return {"messages": [answer]}

        graph = StateGraph(ChatState)
        graph.add_node("chat", node)
        graph.add_edge(START, "chat")
        graph.add_edge("chat", END)
        return graph


class ToolAgent(BaseTestAgent):
    """Model asks for a tool (with a redactable arg), a node answers it."""

    name = "toolbox"

    def build(self, ctx: BuildContext) -> Any:
        def call_tool(state: ChatState) -> dict[str, Any]:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "run_query",
                                "args": {"query": "SELECT 1", "api_key": "hunter2"},
                                "id": "call_1",
                            }
                        ],
                        id="m_tool",
                    )
                ]
            }

        def tool_result(state: ChatState) -> dict[str, Any]:
            return {
                "messages": [
                    ToolMessage(content="rows: 1", tool_call_id="call_1", id="tm_1")
                ]
            }

        graph = StateGraph(ChatState)
        graph.add_node("call", call_tool)
        graph.add_node("tool", tool_result)
        graph.add_edge(START, "call")
        graph.add_edge("call", "tool")
        graph.add_edge("tool", END)
        return graph


class BareArgsToolAgent(BaseTestAgent):
    """The LangGraph quickstart's tool node: a governed tool run with bare args.

    ``tool.invoke(tool_call["args"])`` hands governance no id, so it mints one;
    the stream must still show one call, under the model's id.
    """

    name = "bare-args"

    def build(self, ctx: BuildContext) -> Any:
        from vdreamer.tools import tool as vdreamer_tool

        @vdreamer_tool(redact=["secret"])
        def lookup(city: str, secret: str = "") -> str:
            """Look a city up."""
            return f"{city}: 10M"

        (governed,) = ctx.tools.govern([lookup])

        def call(state: ChatState) -> dict[str, Any]:
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "lookup",
                                "args": {"city": "Cairo", "secret": "hunter2"},
                                "id": "call_model_1",
                            }
                        ],
                        id="m_call",
                    )
                ]
            }

        def tool_node(state: ChatState) -> dict[str, Any]:
            tool_call = state["messages"][-1].tool_calls[0]
            observation = governed.invoke(tool_call["args"])
            return {
                "messages": [
                    ToolMessage(content=observation, tool_call_id=tool_call["id"], id="tm_bare")
                ]
            }

        graph = StateGraph(ChatState)
        graph.add_node("call", call)
        graph.add_node("tool", tool_node)
        graph.add_edge(START, "call")
        graph.add_edge("call", "tool")
        graph.add_edge("tool", END)
        return graph


class ApprovalAgent(BaseTestAgent):
    """Streams one message, then pauses on ``require_approval``."""

    name = "gatekeeper"

    def build(self, ctx: BuildContext) -> Any:
        from vdreamer.context import require_approval

        def chat(state: ChatState) -> dict[str, Any]:
            model = _fake_model(_text_chunk("Draft", "m1"), _text_chunk(" answer", "m1"))
            return {"messages": [model.invoke(state["messages"])]}

        def gate(state: ChatState) -> dict[str, Any]:
            decision = require_approval(
                "sql_review",
                payload={"sql": "UPDATE t SET x = 1"},
                actions=["approve", "reject", "edit"],
                approver_role="dba",
                scope="db:write",
                ttl_hours=2,
            )
            return {"flag": str(decision)}

        graph = StateGraph(ChatState)
        graph.add_node("chat", chat)
        graph.add_node("gate", gate)
        graph.add_edge(START, "chat")
        graph.add_edge("chat", "gate")
        graph.add_edge("gate", END)
        return graph


class RawInterruptAgent(BaseTestAgent):
    """A raw ``interrupt(value)`` with a non-dict value (§4.5 generic)."""

    name = "rawgate"

    def build(self, ctx: BuildContext) -> Any:
        from langgraph.types import interrupt

        def gate(state: ChatState) -> dict[str, Any]:
            return {"flag": str(interrupt("just answer this"))}

        graph = StateGraph(ChatState)
        graph.add_node("gate", gate)
        graph.add_edge(START, "gate")
        graph.add_edge("gate", END)
        return graph


class BoomyAgent(BaseTestAgent):
    """Fails mid-run (run.status failed, exception re-raised)."""

    name = "boomy"

    def build(self, ctx: BuildContext) -> Any:
        def boom(state: ChatState) -> dict[str, Any]:
            raise ValueError("boom from streaming node")

        graph = StateGraph(ChatState)
        graph.add_node("boom", boom)
        graph.add_edge(START, "boom")
        graph.add_edge("boom", END)
        return graph


class SubgraphAgent(BaseTestAgent):
    """A compiled subgraph node that streams — ``ns`` preservation test."""

    name = "nested"

    def build(self, ctx: BuildContext) -> Any:
        def inner(state: ChatState) -> dict[str, Any]:
            emit_progress("inside subgraph", 50.0)
            model = _fake_model(_text_chunk("sub", "m_sub"))
            return {"messages": [model.invoke(state["messages"])]}

        inner_graph = StateGraph(ChatState)
        inner_graph.add_node("inner", inner)
        inner_graph.add_edge(START, "inner")
        inner_graph.add_edge("inner", END)
        inner_compiled = inner_graph.compile()

        def outer(state: ChatState) -> dict[str, Any]:
            return {"flag": "outer"}

        graph = StateGraph(ChatState)
        graph.add_node("child", inner_compiled)
        graph.add_node("outer", outer)
        graph.add_edge(START, "child")
        graph.add_edge("child", "outer")
        graph.add_edge("outer", END)
        return graph


@contextlib.contextmanager
def driver_env(driver: str) -> Any:
    """Temporarily force :data:`STREAM_DRIVER_ENV_VAR` to ``driver``."""
    old = os.environ.get(STREAM_DRIVER_ENV_VAR)
    os.environ[STREAM_DRIVER_ENV_VAR] = driver
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(STREAM_DRIVER_ENV_VAR, None)
        else:
            os.environ[STREAM_DRIVER_ENV_VAR] = old


@dataclass
class StreamedRun:
    """What one streamed run produced."""

    driver: str
    events: tuple[EventEnvelope, ...]
    result: Any


async def run_streamed(
    agent: BaseAgent,
    input: Any,
    *,
    driver: str,
    context: VDreamerContext | None = None,
    config: dict[str, Any] | None = None,
) -> StreamedRun:
    """Run ``agent`` on a fresh ``InMemorySaver`` runner under one driver."""
    if config is None:
        config = {"configurable": {"thread_id": f"t-{driver}"}}
    runner = AgentRunner(agent, checkpointer=InMemorySaver())
    sink = InMemoryEventSink()
    with driver_env(driver):
        result = await runner.arun(input, config=config, context=context, sink=sink)
    return StreamedRun(driver=driver, events=sink.events, result=result)


async def run_streamed_ctx(agent: BaseAgent, driver: str, input: Any | None = None) -> StreamedRun:
    """``run_streamed`` with a fresh bound context — the common test shape."""
    return await run_streamed(
        agent,
        input_for(agent) if input is None else input,
        driver=driver,
        context=make_context(agent),
    )


async def run_graph_all_drivers(
    agent: BaseAgent,
    input: Any,
    *,
    context: VDreamerContext | None = None,
) -> dict[str, StreamedRun]:
    """Run the *same* agent through both drivers (the WP3 golden-suite hook).

    Each driver gets a fresh checkpointer and a fresh sink; the returned map
    is keyed by driver name so tests compare envelope sequences.
    """
    return {
        driver: await run_streamed(agent, input, driver=driver, context=context)
        for driver in ("v2", "v3")
    }


def normalized(envelopes: Any) -> list[Any]:
    """Envelope sequence → comparable rows (drops per-run allocations).

    ``seq`` and ``ts`` are per-run allocations; ``expires_at`` (TTL math) and
    ``duration_ms`` (a tool's run time) are wall-clock; generated ids
    (``art_``/``cit_``/``int_``/``msg_``/
    ``lc_run--``/chunk ids like ``m1``) differ run to run — the v2 and v3
    chat-model protocols assign *different* ids to the same model call — so
    each distinct id maps to its first-appearance ordinal. Everything else —
    ``type``, ``ns``, ``payload`` — must be identical across drivers for the
    same graph execution.
    """
    id_map: dict[str, str] = {}

    def mask(value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                # Wall-clock measurements, like seq/ts: a slower driver (or a
                # loaded CI runner) times the same tool call differently.
                if key in ("expires_at", "duration_ms"):
                    continue
                # ``message_id`` is a provider-assigned id (any format — the
                # v2 and v3 protocols assign different ids to the same call),
                # so it maps by first appearance, not by pattern.
                if key == "message_id":
                    out[key] = id_map.setdefault(str(item), f"<id{len(id_map)}>")
                else:
                    out[key] = mask(item)
            return out
        if isinstance(value, list):
            return [mask(item) for item in value]
        if isinstance(value, tuple):
            return tuple(mask(item) for item in value)
        if isinstance(value, str):
            # Generated ids first (art_/cit_/…/chunk ids/lc_run--…), then any
            # remaining uuids (subgraph namespaces embed a run-unique uuid).
            if _ID_RE.match(value):
                return id_map.setdefault(value, f"<id{len(id_map)}>")
            return _UUID_RE.sub("<uuid>", value)
        return value

    return [
        (envelope.type.value, mask(tuple(envelope.ns)), mask(envelope.payload))
        for envelope in envelopes
    ]


_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_ID_RE = re.compile(
    r"^(?:art_|cit_|int_|msg_|lc_run--)[0-9A-Za-z-]{8,}$|^[0-9a-f]{32}$|^m\d+$"
)


def input_for(agent: BaseAgent, text: str = "hello") -> dict[str, Any]:
    """A one-human-message input for the chat-state test graphs."""
    return {"messages": [{"role": "user", "content": text}]}


def make_context(agent: BaseAgent) -> VDreamerContext:
    """A run identity for one streamed test run (run/thread ids fixed for parity)."""
    return VDreamerContext(
        run_id="run_stream", thread_id="t_stream", agent_name=agent.name
    )
