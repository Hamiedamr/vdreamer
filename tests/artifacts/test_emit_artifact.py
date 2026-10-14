"""``emit_artifact`` / ``aemit_artifact`` / ``return Artifact(...)`` inside real runs (§3.4, D7).

Runs are driven as the worker drives them (a store bound around ``arun``), so
the D7 order is observable: the bytes and the row exist before the stream
carries ``artifact.created``.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pytest
from _tools_kit import CreateAgentAgent, drive, tool_messages
from langchain_core.tools import tool as lc_tool
from langgraph.graph import END, START, StateGraph

import vdreamer
from vdreamer import Artifact, ArtifactKind, BaseAgent, BuildContext, aemit_artifact, emit_artifact
from vdreamer.artifacts import ArtifactRecord, InMemoryArtifactStore


class S(TypedDict, total=False):
    text: str
    ids: list[str]


class _Agent(BaseAgent):
    name = "emitter"

    def __init__(self, node: Any) -> None:
        self._node = node

    def build(self, ctx: BuildContext) -> Any:
        graph = StateGraph(S)
        graph.add_node("n", self._node)
        graph.add_edge(START, "n")
        graph.add_edge("n", END)
        return graph


class OrderedStore(InMemoryArtifactStore):
    """Records when each write happened relative to the run's buffered records."""

    def __init__(self) -> None:
        super().__init__()
        self.log: list[str] = []

    async def insert(self, record: ArtifactRecord) -> None:
        from vdreamer.context import current_context

        assert self.objects.get_bytes(record.object_key) is not None, "bytes before the row"
        seen = {r.get("artifact_id") for r in current_context().pending}
        assert record.artifact_id not in seen, "the row before the event"
        self.log.append(record.artifact_id)
        await super().insert(record)


def _sync_node(state: S) -> dict[str, Any]:
    ref = emit_artifact("rows", ArtifactKind.TABLE, [{"a": 1}, {"a": 2}])
    return {"ids": [ref.artifact_id]}


async def _async_node(state: S) -> dict[str, Any]:
    first = await aemit_artifact("v", ArtifactKind.JSON, {"ok": True}, renderer="verdict.v1")
    # The sync helper from async code: blocks this loop on a helper thread, still works.
    second = emit_artifact("note", ArtifactKind.FILE, b"hi", mime_type="text/plain")
    return {"ids": [first.artifact_id, second.artifact_id]}


@pytest.mark.smoke
@pytest.mark.parametrize("node", [_sync_node, _async_node], ids=["sync-node", "async-node"])
@pytest.mark.parametrize("driver", ["v2", "v3"])
async def test_emit_follows_the_d7_order(node: Any, driver: str) -> None:
    store = OrderedStore()
    run = await drive(_Agent(node), driver=driver, input={"text": "x"}, store=store)
    assert run.error is None
    ids = run.result["ids"]
    assert store.log == ids
    created = run.of("artifact.created")
    assert [c["artifact_id"] for c in created] == ids
    for payload in created:
        record = await store.get(payload["artifact_id"])
        assert record is not None and record.run_id == "run_1" and record.thread_id == "t1"
        assert payload["view_url"] == f"/api/v1/artifacts/{record.artifact_id}/raw"
        assert payload["size_bytes"] == record.size_bytes and payload["preview"] == record.preview
        assert record.object_key.startswith(f"artifacts/t1/run_1/{record.artifact_id}.")


async def test_bad_data_fails_before_anything_is_stored() -> None:
    def node(state: S) -> dict[str, Any]:
        emit_artifact("bad", ArtifactKind.IMAGE, b"GIF89a")
        return {}

    run = await drive(_Agent(node), input={"text": "x"})
    assert isinstance(run.error, TypeError)
    assert run.store.objects.keys() == [] and run.of("artifact.created") == []  # type: ignore[attr-defined]


@lc_tool
def chart(n: int) -> Artifact:
    """Make a chart table."""
    return Artifact(kind=ArtifactKind.TABLE, data=[{"x": i} for i in range(n)], name="series")


@vdreamer.tool()
async def two(n: int) -> list[Artifact]:
    """Two artifacts."""
    return [
        Artifact(kind=ArtifactKind.JSON, data={"n": n}, name="a"),
        Artifact(kind=ArtifactKind.HTML, data="<title>B</title>", name="b"),
    ]


@lc_tool(response_format="content_and_artifact")
def both(n: int) -> tuple[str, Artifact]:
    """Text for the model, an artifact for the user."""
    return f"made {n}", Artifact(kind=ArtifactKind.JSON, data={"n": n}, name="side")


@pytest.mark.smoke
async def test_governed_tools_may_return_artifacts() -> None:
    calls = [("chart", {"n": 3}), ("two", {"n": 1}), ("both", {"n": 2})]
    agent = CreateAgentAgent([chart, two, both], calls)
    run = await drive(agent)
    assert run.error is None
    results = {r["tool_call_id"]: r for r in run.of("tool.result")}
    assert all(r["status"] == "ok" for r in results.values())
    assert [len(results[f"call_{i}"]["artifact_ids"]) for i in (1, 2, 3)] == [1, 2, 1]
    created = {c["artifact_id"]: c for c in run.of("artifact.created")}
    (series,) = results["call_1"]["artifact_ids"]
    assert created[series]["kind"] == "table" and created[series]["preview"]["row_count"] == 3
    messages = {m.tool_call_id: m for m in tool_messages(run.result)}
    assert series in messages["call_1"].text and "table" in messages["call_1"].text
    assert messages["call_1"].artifact == {"artifact_ids": [series]}
    assert messages["call_3"].text == "made 2"  # content_and_artifact keeps the model's text
    assert messages["call_3"].artifact == {"artifact_ids": results["call_3"]["artifact_ids"]}


def test_emit_outside_a_run_raises() -> None:
    with pytest.raises(RuntimeError, match="no active vdreamer run"):
        emit_artifact("x", ArtifactKind.JSON, {"a": 1})


class ProxyAgent(BaseAgent):
    """A hand-built graph over ``ctx.tools.govern`` (the proxy path)."""

    name = "proxy"

    def build(self, ctx: BuildContext) -> Any:
        from langchain_core.messages import AIMessage
        from langgraph.graph import MessagesState
        from langgraph.prebuilt import ToolNode

        def plan(state: MessagesState) -> dict[str, Any]:
            call = {"name": "chart", "args": {"n": 2}, "id": "call_p", "type": "tool_call"}
            return {"messages": [AIMessage("", tool_calls=[call])]}

        graph = StateGraph(MessagesState)
        graph.add_node("plan", plan)
        graph.add_node("tools", ToolNode(ctx.tools.govern([chart])))
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "tools")
        graph.add_edge("tools", END)
        return graph


async def test_the_govern_proxy_stores_returned_artifacts() -> None:
    run = await drive(ProxyAgent())
    assert run.error is None
    result = run.one("tool.result")
    (artifact_id,) = result["artifact_ids"]
    assert run.one("artifact.created")["artifact_id"] == artifact_id
    (message,) = tool_messages(run.result)
    assert artifact_id in message.text
