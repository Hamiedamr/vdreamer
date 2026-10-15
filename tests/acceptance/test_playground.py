"""The playground agents (``examples/playground``) end to end through ``vdreamer run``.

The first-contact agents and the LangChain/LangGraph docs examples, served by
a real gateway, with the chat model scripted at the boundary (``_uc_kit``).
What each test pins is what a developer trying vdreamer would notice first:
streamed answers, one ``tool.call`` per model tool call (the quickstart's
bare-args tool node included), approvals answered inline, contained tool
failures with no secret in the message.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from _uc_kit import ScriptedModel, scripted, serve, vdreamer_run
from langchain_core.messages import AIMessage
from playground import broken, calculator, hello, hitl_tools, workflows


class RoutingModel(ScriptedModel):
    """``with_structured_output(..., method="function_calling")`` on the script."""

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        from langchain_core.output_parsers.openai_tools import PydanticToolsParser

        return self | PydanticToolsParser(tools=[schema], first_tool_only=True)


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _arithmetic() -> ScriptedModel:
    return scripted(
        _call("add", {"a": 3, "b": 4}, "call_add"),
        _call("multiply", {"a": 7, "b": 10}, "call_mul"),
        "The result is 70.",
    )


MODELS: dict[str, Callable[[], ScriptedModel]] = {
    "calculator": _arithmetic,
    "calculator-functional": _arithmetic,
    "router": lambda: RoutingModel(
        responses=[_call("Route", {"step": "joke"}, "call_route"), AIMessage("Why cats? Mice!")]
    ),
    "joke-chain": lambda: scripted(
        "Cats are fluffy.", "Cats are purr-fectly fluffy.", "Cats are dogs in disguise."
    ),
    "hitl-tools": lambda: scripted(
        _call("write_file", {"path": "notes.txt", "content": "hello"}, "call_write"),
        "Saved notes.txt.",
    ),
}

AGENTS = [
    hello.agent,
    broken.agent,
    calculator.graph_agent,
    calculator.functional_agent,
    workflows.chain_agent,
    workflows.router_agent,
    hitl_tools.agent,
]


def _backends(make: Callable[[], ScriptedModel]) -> Any:
    @asynccontextmanager
    async def backends() -> AsyncIterator[dict[str, Any]]:
        yield {"models": {"chat": make()}}

    return backends


@pytest.fixture(scope="module")
def url() -> Iterator[str]:
    from _api import make_stack

    stack = make_stack(
        workers=True,
        extra_agents=AGENTS,
        worker_agents=[agent.name for agent in AGENTS],
        poll_interval_s=0.05,
        runner_options={name: _backends(make) for name, make in MODELS.items()},
    )
    try:
        with serve(stack) as base:
            yield base
    finally:
        stack.close()


def test_hello_streams_a_tool_a_table_and_an_approval(url: str, monkeypatch: Any) -> None:
    result = vdreamer_run(
        url, monkeypatch, "approve", agent="hello", message="hi", once=True, color=False
    )
    assert result.code == 0, result.out
    assert hello.GREETING in result.out
    assert "[progress] thinking 20%" in result.out
    assert '-> lookup_city({"city": "Cairo", "secret": "***"})' in result.out
    assert "[artifact] table 'cities'" in result.out
    assert "Decision: approve by alice" in result.out


def test_broken_contains_tool_failures_without_leaking(url: str, monkeypatch: Any) -> None:
    result = vdreamer_run(
        url, monkeypatch, agent="broken", input_json='{"message": "go"}', color=False
    )
    assert result.code == 2, result.out  # the node error fails the run
    assert "hunter2" not in result.out
    assert "password=***" in result.out
    assert "Error (timeout)" in result.out
    assert "result truncated to 100 of 5000 characters" in result.out
    assert "[run failed]" in result.out and "node_error: node blew up" in result.out


@pytest.mark.parametrize("agent", ["calculator", "calculator-functional"])
def test_the_quickstart_calculator_shows_each_call_once(
    url: str, monkeypatch: Any, agent: str
) -> None:
    result = vdreamer_run(
        url, monkeypatch, agent=agent, message="Add 3 and 4, times 10", once=True, color=False
    )
    assert result.code == 0, result.out
    assert result.out.count("-> add(") == 1, result.out
    assert result.out.count("-> multiply(") == 1, result.out
    assert "<- ok 7" in result.out and "<- ok 70" in result.out
    assert "The result is 70." in result.out


def test_the_router_routes_with_structured_output(url: str, monkeypatch: Any) -> None:
    # A BOTH agent with a one-string-field schema: --message lands in that field.
    result = vdreamer_run(
        url, monkeypatch, agent="router", message="a joke about cats", once=True, color=False
    )
    assert result.code == 0, result.out
    assert "Why cats? Mice!" in result.out


def test_the_prompt_chain_improves_a_joke_without_a_punchline(url: str, monkeypatch: Any) -> None:
    result = vdreamer_run(
        url, monkeypatch, agent="joke-chain", input_json='{"topic": "cats"}', color=False
    )
    assert result.code == 0, result.out
    assert "Cats are dogs in disguise." in result.out  # the gate failed → improve → polish


def test_the_hitl_middleware_example_reviews_the_write(url: str, monkeypatch: Any) -> None:
    hitl_tools.FILES.clear()
    result = vdreamer_run(
        url,
        monkeypatch,
        "approve",
        agent="hitl-tools",
        message="save hello",
        once=True,
        color=False,
    )
    assert result.code == 0, result.out
    assert "[interrupt tool_review]" in result.out
    assert "Saved notes.txt." in result.out
    assert hitl_tools.FILES == {"notes.txt": "hello"}
