"""A models.yaml alias's thinking mode, end to end through both stream drivers.

The model layer turns ``reasoning_content`` and ``<think>`` tags into
standard ``reasoning`` blocks (§2.4), so the v2 and v3 drivers emit the
same ``thinking.delta`` / ``token.delta`` / ``message.completed`` events
(A31), and ``expose_thinking = False`` removes the thinking from all three.
"""

from __future__ import annotations

from typing import Any

import pytest
from _stub_llm import Reply, StubServer
from _tools_kit import drive
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from vdreamer import BaseAgent, BuildContext
from vdreamer.models import AliasMap, build_chat_model, parse_models
from vdreamer.runtime import AgentRunner


class Thinker(BaseAgent):
    name = "thinker"
    models = ["brain"]

    def __init__(self, expose: bool = True) -> None:
        self.expose_thinking = expose

    def build(self, ctx: BuildContext) -> Any:
        model = ctx.models["brain"]

        async def answer(state: MessagesState) -> dict[str, Any]:
            return {"messages": [await model.ainvoke(state["messages"])]}

        graph = StateGraph(MessagesState)
        graph.add_node("answer", answer)
        graph.add_edge(START, "answer")
        return graph


def _runner(stub: StubServer, thinking: str, expose: bool = True) -> AgentRunner:
    config = parse_models(
        {"aliases": {"brain": {"provider": stub.kind, "base_url": stub.base_url,
                               "model": stub.models[0], "thinking": thinking}}}
    ).aliases["brain"]
    agent = Thinker(expose)
    models = AliasMap("model", "agent 'thinker'", {"brain": build_chat_model(config)})  # type: ignore[arg-type]
    return AgentRunner(agent, checkpointer=InMemorySaver(), models=models)


CASES = [
    ("vllm", "reasoning_content", Reply(content="It is 4.", reasoning="two and two")),
    ("ollama", "think_tags", Reply(content="It is 4.", reasoning="two and two",
                                   inline_think=True, chunk=3)),
    ("ollama", "reasoning_content", Reply(content="It is 4.", reasoning="two and two")),
]


@pytest.mark.smoke
@pytest.mark.parametrize("driver", ["v2", "v3"])
@pytest.mark.parametrize(("kind", "thinking", "reply"), CASES)
async def test_thinking_reaches_the_protocol(
    request: pytest.FixtureRequest, driver: str, kind: str, thinking: str, reply: Reply
) -> None:
    stub: StubServer = request.getfixturevalue(kind)
    stub.script[:] = [reply]
    runner = _runner(stub, thinking)
    run = await drive(runner.agent, driver=driver, runner=runner)
    assert run.error is None, run.error
    assert "".join(e["text"] for e in run.of("thinking.delta")) == "two and two"
    assert "".join(e["text"] for e in run.of("token.delta")) == "It is 4."
    completed = run.one("message.completed")
    assert completed["text"] == "It is 4." and completed["thinking"] == "two and two"
    assert completed["usage"]["input_tokens"] == 7  # the server's usage, as a plain dict
    final = run.result["messages"][-1]
    assert isinstance(final, AIMessage)
    assert [b["type"] for b in final.content_blocks if b["type"] in ("reasoning", "text")] == [
        "reasoning",
        "text",
    ]


@pytest.mark.parametrize("driver", ["v2", "v3"])
async def test_expose_thinking_false_hides_it(ollama: StubServer, driver: str) -> None:
    ollama.script[:] = [Reply(content="It is 4.", reasoning="secret", inline_think=True)]
    runner = _runner(ollama, "think_tags", expose=False)
    run = await drive(runner.agent, driver=driver, runner=runner)
    assert run.of("thinking.delta") == []
    assert "".join(e["text"] for e in run.of("token.delta")) == "It is 4."
    assert run.one("message.completed").get("thinking") is None
