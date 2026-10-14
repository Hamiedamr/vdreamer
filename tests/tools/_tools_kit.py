"""Shared fixtures for the Milestone 5 tool tests: fake tool-calling models and agents."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from _kit import driver_env
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from vdreamer import BaseAgent, BuildContext
from vdreamer.artifacts import InMemoryArtifactStore, bind_artifact_store
from vdreamer.context import VDreamerContext
from vdreamer.eventlog import InMemoryEventSink
from vdreamer.events import EventEnvelope
from vdreamer.runtime import AgentRunner


class FakeToolCallingModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel plus bind_tools (the venv fakes lack it)."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def calling(*calls: tuple[str, dict[str, Any]], final: str = "done") -> FakeToolCallingModel:
    """A model that asks for ``calls`` (ids call_1..n) in one turn, then answers."""
    return FakeToolCallingModel(
        responses=[
            AIMessage(
                "",
                id="ai_1",
                tool_calls=[
                    {"name": name, "args": args, "id": f"call_{index}", "type": "tool_call"}
                    for index, (name, args) in enumerate(calls, start=1)
                ],
            ),
            AIMessage(final, id="ai_2"),
        ]
    )


class CreateAgentAgent(BaseAgent):
    """``create_agent`` wired through ``ctx.agent_kwargs()`` (the middleware path)."""

    name = "governed"

    def __init__(self, tools: Sequence[Any], calls: Sequence[tuple[str, dict[str, Any]]]) -> None:
        self._tools = list(tools)
        self._calls = list(calls)

    def build(self, ctx: BuildContext) -> Any:
        return create_agent(model=calling(*self._calls), tools=self._tools, **ctx.agent_kwargs())


@dataclass
class Run:
    events: list[EventEnvelope]
    result: Any
    store: InMemoryArtifactStore
    error: BaseException | None = None

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [dict(event.payload) for event in self.events if event.type.value == kind]

    def types(self) -> list[str]:
        return [event.type.value for event in self.events]

    def one(self, kind: str) -> dict[str, Any]:
        (payload,) = self.of(kind)
        return payload


def make_ctx(
    *, roles: Sequence[str] = (), run_id: str = "run_1", thread_id: str = "t1"
) -> VDreamerContext:
    return VDreamerContext(
        run_id=run_id,
        thread_id=thread_id,
        agent_name="governed",
        user_id="alice",
        roles=tuple(roles),
        idempotency_key=run_id,
    )


async def drive(
    agent: BaseAgent,
    *,
    driver: str = "v2",
    roles: Sequence[str] = (),
    input: Any = None,
    runner: AgentRunner | None = None,
    store: InMemoryArtifactStore | None = None,
    thread_id: str = "t1",
    run_id: str = "run_1",
) -> Run:
    """One streamed run with an artifact store bound, as the worker binds it."""
    runner = runner if runner is not None else AgentRunner(agent, checkpointer=InMemorySaver())
    store = store if store is not None else InMemoryArtifactStore()
    sink = InMemoryEventSink()
    payload = input if input is not None else {"messages": [HumanMessage("go")]}
    error: BaseException | None = None
    result = None
    with driver_env(driver), bind_artifact_store(store):
        try:
            result = await runner.arun(
                payload,
                config={"configurable": {"thread_id": thread_id}},
                context=make_ctx(roles=roles, run_id=run_id, thread_id=thread_id),
                sink=sink,
            )
        except Exception as exc:  # fail_run surfaces here
            error = exc
    return Run(list(sink.events), result, store, error)


def tool_messages(result: Any) -> list[Any]:
    from langchain_core.messages import ToolMessage

    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


def big_json(n: int) -> str:
    return json.dumps({"rows": [{"i": i, "v": "x" * 20} for i in range(n)]})
