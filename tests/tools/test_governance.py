"""The governed tool pipeline (PRD §2.7; Milestone 5 exit).

Both entry points — ``VDreamerToolMiddleware`` under ``create_agent`` and the
``ctx.tools.govern`` proxy in a plain ``StateGraph`` — run the same seven
steps, verified here through a real streamed run (both stream drivers).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from _tools_kit import CreateAgentAgent, big_json, calling, drive, make_ctx, tool_messages
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.types import Command

import vdreamer
from vdreamer import BaseAgent, BuildContext
from vdreamer.artifacts import InMemoryArtifactStore, bind_artifact_store
from vdreamer.context import bind_context, current_context
from vdreamer.governance import (
    GovernedTool,
    GovernedToolError,
    ToolKit,
    authorize,
    governed_call,
    normalize,
    tool_source,
)
from vdreamer.runtime import AgentRunner
from vdreamer.tools import ApprovalSpec, Policy


@vdreamer.tool(redact=["query"])
def lookup(query: str, password: str) -> str:
    """Look something up; returns a large JSON document."""
    return big_json(500)


@pytest.mark.smoke  # floor/latest pins (A41): the governed pipeline on create_agent
@pytest.mark.parametrize("driver", ["v2", "v3"])
async def test_exit_redacted_call_then_result_with_an_overflow_artifact(driver: str) -> None:
    """The M5 exit: redacted ``tool.call`` → ``tool.result`` + overflow artifact."""
    agent = CreateAgentAgent([lookup], [("lookup", {"query": "SELECT 1", "password": "pw"})])
    run = await drive(agent, driver=driver)
    assert run.error is None

    types = run.types()
    assert types.index("tool.call") < types.index("artifact.created") < types.index("tool.result")
    call = run.one("tool.call")
    assert call == {
        "tool_call_id": "call_1",
        "name": "lookup",
        "source": "custom",
        "args": {"query": "***", "password": "***"},  # policy key + A33 default
    }
    result = run.one("tool.result")
    (artifact_id,) = result["artifact_ids"]
    assert result["status"] == "ok" and result["duration_ms"] >= 0
    assert artifact_id in result["summary"]
    created = run.one("artifact.created")
    assert created["artifact_id"] == artifact_id
    assert created["kind"] == "json" and created["mime_type"] == "application/json"
    assert created["view_url"] == f"/api/v1/artifacts/{artifact_id}/raw"
    assert created["metadata"] == {"tool": "lookup", "tool_call_id": "call_1", "source": "custom"}

    # The model saw the truncated body; the store kept the whole result.
    (message,) = tool_messages(run.result)
    assert len(message.content) < 8200 and artifact_id in message.content
    record = await run.store.get(artifact_id)
    assert record is not None and record.size_bytes == len(big_json(500))
    assert await run.store.read_bytes(record) == big_json(500).encode()
    # Governed calls are emitted once: the raw ToolMessage maps to nothing.
    assert types.count("tool.call") == 1 and types.count("tool.result") == 1
    assert "pw" not in json.dumps([dict(e.payload) for e in run.events])


class ProxyAgent(BaseAgent):
    """A plain ``StateGraph`` + ``ToolNode`` over ``ctx.tools.govern`` proxies."""

    name = "proxy"

    def __init__(self, tools: list[Any], calls: list[tuple[str, dict[str, Any]]]) -> None:
        self._tools = tools
        self._calls = calls

    def build(self, ctx: BuildContext) -> Any:
        model = calling(*self._calls)

        def think(state: MessagesState) -> dict[str, Any]:
            return {"messages": [model.invoke(state["messages"])]}

        def route(state: MessagesState) -> str:
            last = state["messages"][-1]
            return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

        graph = StateGraph(MessagesState)
        graph.add_node("think", think)
        graph.add_node("tools", ToolNode(ctx.tools.govern(self._tools)))
        graph.add_edge(START, "think")
        graph.add_conditional_edges("think", route)
        graph.add_edge("tools", "think")
        return graph


@pytest.mark.parametrize("driver", ["v2", "v3"])
async def test_the_proxy_path_governs_a_plain_tool_graph(driver: str) -> None:
    @lc_tool
    def shout(text: str, api_key: str) -> str:
        """Upper-case the text."""
        return text.upper()

    agent = ProxyAgent([shout], [("shout", {"text": "hi", "api_key": "k"})])
    agent.tool_policies = {"sh*": Policy(redact=("text",))}
    run = await drive(agent, driver=driver)
    assert run.error is None
    assert run.one("tool.call")["args"] == {"text": "***", "api_key": "***"}
    assert run.one("tool.result")["summary"] == "HI"
    assert run.one("tool.result")["artifact_ids"] == []


def test_the_proxy_keeps_name_description_and_schema() -> None:
    governed = ToolKit().govern([lookup])[0]
    assert isinstance(governed, GovernedTool)
    assert (governed.name, governed.description) == (lookup.name, lookup.description)
    assert governed.args == lookup.args
    schema = governed.tool_call_schema.model_json_schema()
    assert schema == lookup.tool_call_schema.model_json_schema()
    assert ToolKit().govern([governed])[0] is governed  # never twice


async def test_a_direct_invoke_returns_the_plain_output() -> None:
    def dbl(text: str) -> str:
        """Double."""
        return text * 2

    governed = ToolKit().govern([dbl])[0]  # a plain function becomes a tool
    assert await governed.ainvoke({"text": "ab"}) == "abab"


@pytest.mark.parametrize(
    ("policy", "roles", "allowed"),
    [
        (Policy(), (), True),
        (Policy(requires_roles=("netops", "admin")), ("admin",), True),
        (Policy(requires_roles=("netops",)), ("viewer",), False),
        (Policy(scope="db:write:prod"), ("db:*",), True),
        (Policy(scope="db:write:prod"), ("db:read:*",), False),
    ],
)
def test_authorize(policy: Policy, roles: tuple[str, ...], allowed: bool) -> None:
    assert (authorize(policy, roles) is None) is allowed


def _contained_agent(fn: Any, args: dict[str, Any], **policy: Any) -> CreateAgentAgent:
    return CreateAgentAgent([vdreamer.tool(**policy)(fn)], [(fn.__name__, args)])


async def test_errors_are_contained_with_stable_codes() -> None:
    def boom(x: int) -> str:
        """Raise."""
        raise RuntimeError("kaput")

    async def slow(x: int) -> str:
        """Sleep."""
        await asyncio.sleep(5)
        return "late"

    def gated(x: int) -> str:
        """Needs a role."""
        return "ok"

    cases = [
        (_contained_agent(boom, {"x": 1}), "tool_error", "kaput"),
        (_contained_agent(slow, {"x": 1}, timeout_s=0.1), "timeout", "within 0.1s"),
        (_contained_agent(gated, {"x": 1}, requires_roles=["netops"]), "denied", "netops"),
        (_contained_agent(gated, {"x": "not-an-int"}), "invalid_args", "x:"),
    ]
    for agent, code, fragment in cases:
        run = await drive(agent)
        assert run.error is None, code  # the run continues
        result = run.one("tool.result")
        assert result["status"] == "error"
        assert result["error"]["code"] == code
        assert fragment in result["error"]["message"]
        (message,) = tool_messages(run.result)
        assert message.status == "error" and f"Error ({code})" in message.text
        assert run.result["messages"][-1].content == "done"


async def test_invalid_args_messages_never_echo_the_values() -> None:
    def typed(count: int) -> str:
        """Typed."""
        return "ok"

    run = await drive(_contained_agent(typed, {"count": "s3cret-value"}))
    assert "s3cret-value" not in run.one("tool.result")["error"]["message"]


async def test_error_messages_mask_secrets_the_exception_echoes() -> None:
    def login(user: str, token: str) -> str:
        """Log in."""
        raise RuntimeError(f"login failed for {user} with {token}; db password=hunter2")

    run = await drive(_contained_agent(login, {"user": "ana", "token": "tok-123456"}))
    result = run.one("tool.result")
    for text in (result["error"]["message"], result["summary"]):
        assert "tok-123456" not in text and "hunter2" not in text
        assert "login failed for ana with ***" in text and "password=***" in text
    (message,) = tool_messages(run.result)
    assert "hunter2" not in message.text and "tok-123456" not in message.text


def test_redact_text() -> None:
    from vdreamer import redact_text

    assert redact_text("password=hunter2, ok") == "password=***, ok"
    assert redact_text("Authorization: Bearer abc.def") == "Authorization: Bearer ***"
    assert redact_text('api_key="a b c" done') == "api_key=*** done"
    assert redact_text("echo s3cret!", {"pin": "s3cret"}, ["pin"]) == "echo ***!"
    assert redact_text("the tokenizer = fine") == "the tokenizer = fine"
    assert redact_text("short ab", {"secret": "ab"}) == "short ab"  # too short to search


async def test_fail_run_fails_the_run() -> None:
    def boom(x: int) -> str:
        """Raise."""
        raise RuntimeError("kaput")

    run = await drive(_contained_agent(boom, {"x": 1}, fail_run=True))
    assert isinstance(run.error, GovernedToolError) and run.error.code == "tool_error"
    assert run.one("tool.result")["error"]["code"] == "tool_error"
    assert run.of("run.status")[-1]["status"] == "failed"


async def test_side_effect_tools_get_a_per_call_idempotency_key() -> None:
    seen: list[tuple[str | None, str | None]] = []

    @vdreamer.tool(side_effects=True)
    def charge(amount: int, runtime: ToolRuntime) -> str:
        """Charge once."""
        seen.append((runtime.context.idempotency_key, current_context().idempotency_key))
        return "charged"

    @vdreamer.tool
    def peek(runtime: ToolRuntime) -> str:
        """No side effects."""
        seen.append((runtime.context.idempotency_key, current_context().idempotency_key))
        return "ok"

    agent = CreateAgentAgent([charge, peek], [("charge", {"amount": 5}), ("peek", {})])
    run = await drive(agent)
    assert run.error is None
    assert sorted(seen) == sorted([("run_1:call_1", "run_1:call_1"), ("run_1", "run_1")])
    # Injected args never reach the event.
    calls = {c["name"]: c["args"] for c in run.of("tool.call")}
    assert calls == {"charge": {"amount": 5}, "peek": {}}


class ApprovalAgent(BaseAgent):
    name = "approve"

    def build(self, ctx: BuildContext) -> Any:
        @vdreamer.tool(approval=ApprovalSpec(kind="tool_approval", approver_role="ops"))
        def drop_table(table: str, token: str) -> str:
            """Drop it."""
            return f"dropped {table}"

        return create_agent(
            model=calling(("drop_table", {"table": "t", "token": "tk"})),
            tools=[drop_table],
            **ctx.agent_kwargs(),
        )


@pytest.mark.parametrize(("decision", "status"), [("approve", "ok"), ("reject", "error")])
async def test_the_approval_gate_interrupts_then_resumes(decision: str, status: str) -> None:
    agent = ApprovalAgent()
    runner = AgentRunner(agent, checkpointer=InMemorySaver())
    first = await drive(agent, runner=runner)
    requested = first.one("interrupt.requested")
    assert requested["kind"] == "tool_approval" and requested["approver_role"] == "ops"
    assert requested["actions"] == ["approve", "reject"]
    assert requested["payload"]["args"] == {"table": "t", "token": "***"}
    assert first.of("run.status")[-1]["status"] == "awaiting_approval"
    assert first.of("tool.result") == []

    resumed = await drive(
        agent, runner=runner, input=Command(resume={"action": decision, "text": "no"})
    )
    result = resumed.one("tool.result")
    assert result["status"] == status
    if status == "error":
        assert result["error"]["code"] == "denied" and "no" in result["error"]["message"]
    else:
        assert result["summary"] == "dropped t"
    assert resumed.of("run.status")[-1]["status"] == "completed"


async def test_an_ungoverned_tool_falls_back_to_the_default_mapping() -> None:
    @lc_tool
    def raw(secret: str, note: str) -> str:
        """Ungoverned."""
        return "raw-ok"

    class RawAgent(ProxyAgent):
        def build(self, ctx: BuildContext) -> Any:
            graph = super().build(ctx)
            graph.nodes["tools"].runnable = ToolNode([raw])  # bypass governance
            return graph

    run = await drive(RawAgent([raw], [("raw", {"secret": "s", "note": "n"})]))
    assert run.types().count("tool.call") == 1
    assert run.one("tool.call")["args"] == {"secret": "***", "note": "n"}  # A33 defaults only
    assert run.one("tool.result")["summary"] == "raw-ok"


def test_tool_source() -> None:
    @lc_tool
    def mine(x: int) -> int:
        """Mine."""
        return x

    from langchain_core.tools import StructuredTool

    assert tool_source(lookup) == "custom"
    assert tool_source(mine) == "custom"
    mcp_tool = mine.model_copy(update={"metadata": {"vdreamer_source": "mcp:netinv"}})
    assert tool_source(mcp_tool) == "mcp:netinv"
    ecosystem = StructuredTool.from_function(json.dumps, name="dumps", description="d")
    assert tool_source(ecosystem) == "custom"  # json is the stdlib, not LangChain


async def test_structured_content_and_media_blocks_overflow_to_artifacts() -> None:
    store = InMemoryArtifactStore()
    ctx = make_ctx()
    call = governed_call(
        lookup, {"name": "lookup", "args": {}, "id": "c9"}, policy=Policy(max_result_chars=50)
    )
    message = ToolMessage(
        content=[
            {"type": "text", "text": "y" * 80},
            {"type": "image", "base64": "aGVsbG8=", "mime_type": "image/png"},
        ],
        tool_call_id="c9",
        artifact={"structured_content": {"rows": list(range(100))}},
    )
    with bind_context(ctx), bind_artifact_store(store):
        normalized, ids = await normalize(message, call)
    assert len(ids) == 3
    text_block, media_block = normalized.content
    assert ids[0] in text_block["text"] and len(text_block["text"]) < 200
    assert ids[1] in media_block["text"]
    assert normalized.artifact == {"artifact_id": ids[2]}
    media = await store.get(ids[1])
    assert media is not None and media.mime_type == "image/png"
    assert await store.read_bytes(media) == b"hello"
    kinds = {record["kind"] for record in ctx.pending if record["type"] == "artifact"}
    assert kinds == {"file", "json"}


async def test_governance_also_runs_on_the_sync_path() -> None:
    @vdreamer.tool(redact=["q"])
    def find(q: str) -> str:
        """Find."""
        return "found"

    governed = ToolKit().govern([find])[0]
    out = governed.invoke({"type": "tool_call", "name": "find", "args": {"q": "x"}, "id": "c1"})
    assert isinstance(out, ToolMessage) and out.content == "found"
    slow = ToolKit().govern([vdreamer.tool(timeout_s=0.05)(_sleepy)])[0]
    timed_out = slow.invoke({"type": "tool_call", "name": "_sleepy", "args": {}, "id": "c2"})
    assert timed_out.status == "error" and "timeout" in timed_out.text


def _sleepy() -> str:
    """Sleep a bit."""
    import time

    time.sleep(0.5)
    return "late"


async def test_middleware_and_proxy_never_double_govern() -> None:
    governed = ToolKit().govern([lookup])
    agent = CreateAgentAgent(governed, [("lookup", {"query": "q", "password": "p"})])
    run = await drive(agent)
    assert run.types().count("tool.call") == 1 and run.types().count("tool.result") == 1
    assert len(run.of("artifact.created")) == 1

