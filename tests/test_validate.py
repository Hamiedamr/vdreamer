"""``vdreamer validate`` contract checks."""

from typing import Any, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from vdreamer.agents import BaseAgent, BuildContext, Mode
from vdreamer.cli import main
from vdreamer.discovery import AgentLoadError, load_agent
from vdreamer.validate import validate_agent


class _State(TypedDict):
    message: str


def _graph() -> StateGraph:
    graph = StateGraph(_State)
    graph.add_node("echo", lambda state: {"message": state["message"]})
    graph.add_edge(START, "echo")
    graph.add_edge("echo", END)
    return graph


def _failures(checks: Any) -> list[tuple[str, str]]:
    return [(check.name, check.detail) for check in checks if not check.ok]


def test_smoke_agent_passes_validation(capsys: pytest.CaptureFixture[str]) -> None:
    agent = load_agent("pinger:agent")
    assert agent.name == "pinger"
    assert _failures(validate_agent(agent)) == []
    # CLI level: exit 0 and a PASS line
    assert main(["validate", "pinger:agent"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_headless_agent_requires_input_schema() -> None:
    class BadAgent(BaseAgent):
        name = "bad"
        mode = Mode.HEADLESS
        input_schema = None
        models = ()
        resources = ()
        access_roles = ()

        def build(self, ctx: BuildContext) -> Any:
            return _graph()

    checks = validate_agent(BadAgent())
    assert any(check.name == "input_schema" and not check.ok for check in checks)


def test_chat_agent_without_schema_is_valid() -> None:
    class ChatAgent(BaseAgent):
        name = "chat-only"
        mode = Mode.CHAT
        input_schema = None

        def build(self, ctx: BuildContext) -> Any:
            return _graph()

    assert _failures(validate_agent(ChatAgent())) == []


def test_bad_name_fails() -> None:
    class BadAgent(BaseAgent):
        name = "Not A Slug"
        mode = Mode.CHAT

        def build(self, ctx: BuildContext) -> Any:
            return _graph()

    assert _failures(validate_agent(BadAgent()))[0][0] == "name"


def test_unimplemented_build_fails() -> None:
    class StubbornAgent(BaseAgent):
        name = "stubborn"
        mode = Mode.CHAT

        def build(self, ctx: BuildContext) -> Any:
            raise NotImplementedError

    assert _failures(validate_agent(StubbornAgent()))[0][0] == "graph.build"


def test_unknown_module_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", "no.such.module:agent"]) == 1
    assert "cannot import module" in capsys.readouterr().err


def test_spec_without_attribute_rejected() -> None:
    with pytest.raises(AgentLoadError):
        load_agent("pinger")  # missing ':agent'
