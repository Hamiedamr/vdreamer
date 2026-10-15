"""Smoke contract tests — run against the floor and latest 1.x pins (A41).

``pytest -m smoke`` is what the floor/latest CI jobs execute: build the agent
through the public contract, compile the graph, and run one node through a
checkpointer. The trivial in-repo agent stands in for the reference agents
UC1–UC3 until Milestone 12.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from vdreamer.agents import BuildContext
from vdreamer.discovery import load_agent
from vdreamer.validate import ensure_compiled


@pytest.mark.smoke
async def test_smoke_agent_builds_compiles_runs_through_checkpointer() -> None:
    agent = load_agent("pinger:agent")
    graph = agent.build(BuildContext())
    compiled = ensure_compiled(graph, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t-smoke"}}
    result = await compiled.ainvoke({"message": "hello"}, config=config)
    assert {key: result[key] for key in ("message", "reply")} == {
        "message": "hello",
        "reply": "pong: hello",
    }
    assert [m.text for m in result["messages"]] == ["pong: hello"]  # what a client sees
    # Persisted through the checkpointer: state is recoverable for the thread.
    state = await compiled.aget_state(config)
    assert state.values["reply"] == "pong: hello"
