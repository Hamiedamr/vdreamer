"""``AgentRunner.arun(sink=...)`` — the WP2 runner integration.

Public surface is additive only: hooks, exception semantics, and the return
value match the M1 non-streaming path; every envelope is written to the sink
with a per-run ``SeqCounter``.
"""

from __future__ import annotations

from typing import Any

import _kit
import pytest

from vdreamer.eventlog import InMemoryEventSink, SeqCounter
from vdreamer.events import PROTOCOL_VERSION
from vdreamer.runtime import AgentRunner


class _HookAgent(_kit.BaseTestAgent):
    name = "hooky"

    def __init__(self) -> None:
        self.starts: list[Any] = []
        self.ends: list[tuple[Any, str]] = []

    def build(self, ctx: Any) -> Any:
        def reply(state: _kit.ChatState) -> dict[str, Any]:
            if state.get("flag") == "boom":
                raise ValueError("sink boom")
            return {"flag": "done"}

        graph = _kit.StateGraph(_kit.ChatState)
        graph.add_node("reply", reply)
        graph.add_edge(_kit.START, "reply")
        graph.add_edge("reply", _kit.END)
        return graph

    async def on_run_start(self, run: Any) -> None:
        self.starts.append(run)

    async def on_run_end(self, run: Any, status: str) -> None:
        self.ends.append((run, status))


async def test_sink_run_matches_plain_run_result_and_fires_hooks() -> None:
    agent = _HookAgent()
    ctx = _kit.make_context(agent)
    config = {"configurable": {"thread_id": "t-both"}}

    plain_runner = AgentRunner(agent, checkpointer=_kit.InMemorySaver())
    plain = await plain_runner.arun(_kit.input_for(agent), config=config, context=ctx)

    sink = InMemoryEventSink()
    streamed_runner = AgentRunner(agent, checkpointer=_kit.InMemorySaver())
    streamed = await streamed_runner.arun(
        _kit.input_for(agent), config=config, context=ctx, sink=sink
    )

    assert streamed == plain, "the sink path returns the same final state"
    assert agent.starts[0].status == "running"
    assert [status for _, status in agent.ends] == ["completed", "completed"]
    # The streamed run itself produced the protocol sequence.
    statuses = [e.payload["status"] for e in sink.events if e.type.value == "run.status"]
    assert statuses == ["running", "completed"]


async def test_sink_envelopes_are_seq_monotonic_per_run() -> None:
    agent = _kit.ChattyAgent()
    ctx = _kit.make_context(agent)
    sink = InMemoryEventSink()
    runner = AgentRunner(agent, checkpointer=_kit.InMemorySaver())
    await runner.arun(
        _kit.input_for(agent),
        config={"configurable": {"thread_id": "t-seq"}},
        context=ctx,
        sink=sink,
    )

    assert len(sink.events) >= 2
    seqs = [envelope.seq for envelope in sink.events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs), "no duplicate seq within a run"
    assert seqs[0] == 1
    for envelope in sink.events:
        assert envelope.v == PROTOCOL_VERSION
        assert envelope.agent == agent.name
        assert envelope.run_id == ctx.run_id
        assert envelope.thread_id == ctx.thread_id


async def test_two_runs_count_seq_independently() -> None:
    """``SeqCounter`` is per-run: each run starts back at 1 (§3.3)."""
    agent = _kit.ChattyAgent()
    sink = InMemoryEventSink()
    runner = AgentRunner(agent, checkpointer=_kit.InMemorySaver())
    for index in range(2):
        ctx = _kit.make_context(agent).model_copy(update={"run_id": f"run_{index}"})
        await runner.arun(
            _kit.input_for(agent),
            config={"configurable": {"thread_id": f"t-{index}"}},
            context=ctx,
            sink=sink,
        )
    first_run = [e for e in sink.events if e.run_id == "run_0"]
    second_run = [e for e in sink.events if e.run_id == "run_1"]
    assert first_run[0].seq == 1
    assert second_run[0].seq == 1


async def test_seq_counter_allocates_from_one_per_run() -> None:
    counter = SeqCounter()
    assert [counter.next_seq("r1") for _ in range(3)] == [1, 2, 3]
    assert counter.next_seq("r2") == 1
    assert counter.next_seq("r1") == 4


async def test_sink_run_without_context_still_envelopes() -> None:
    """A sink run without a bound context generates ids (M1 semantics)."""
    agent = _HookAgent()
    sink = InMemoryEventSink()
    runner = AgentRunner(agent, checkpointer=_kit.InMemorySaver())
    result = await runner.arun(
        _kit.input_for(agent), config={"configurable": {"thread_id": "t-noctx"}}, sink=sink
    )
    assert result is not None
    assert sink.events
    assert sink.events[0].agent == agent.name
    assert sink.events[0].thread_id == "t-noctx", "config thread_id keys the envelopes"
    assert sink.events[0].run_id.startswith("run_"), "run_id is generated"


async def test_failed_sink_run_fires_both_hooks_with_failed() -> None:
    agent = _HookAgent()
    sink = InMemoryEventSink()
    runner = AgentRunner(agent, checkpointer=_kit.InMemorySaver())
    with pytest.raises(ValueError, match="sink boom"):
        await runner.arun(
            {"flag": "boom"},
            config={"configurable": {"thread_id": "t-boom"}},
            context=_kit.make_context(agent),
            sink=sink,
        )
    assert [status for _, status in agent.ends] == ["failed"]
    statuses = [e.payload["status"] for e in sink.events if e.type.value == "run.status"]
    assert statuses == ["running", "failed"]
