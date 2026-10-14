"""``vdreamer validate``'s Milestone 5 checks: no ungoverned tools, approval routes wired."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from _tools_kit import CreateAgentAgent, calling
from langchain_core.tools import tool as lc_tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode
from vdreamer.tools import ApprovalSpec, Policy
from vdreamer.validate import run_validate, validate_agent


@lc_tool
def plain(x: int) -> int:
    """Plain."""
    return x


def _graph(tools: list[Any]) -> StateGraph:
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "tools")
    return graph


class RawToolNode(BaseAgent):
    name = "raw"

    def build(self, ctx: BuildContext) -> Any:
        return _graph([plain])


class GovernedToolNode(BaseAgent):
    name = "governed-node"

    def build(self, ctx: BuildContext) -> Any:
        return _graph(ctx.tools.govern([plain]))


def _check(agent: BaseAgent, name: str) -> tuple[bool, str]:
    (check,) = [c for c in validate_agent(agent) if c.name == name]
    return check.ok, check.detail


def test_a_raw_tool_node_is_flagged() -> None:
    ok, detail = _check(RawToolNode(), "tools.governed")
    assert not ok and "['plain']" in detail and "ctx.tools.govern" in detail


def test_governed_proxies_and_the_middleware_pass() -> None:
    assert _check(GovernedToolNode(), "tools.governed") == (True, "1 tool(s)")
    agent = CreateAgentAgent([plain], [("plain", {"x": 1})])
    assert _check(agent, "tools.governed") == (True, "1 tool(s)")


def test_create_agent_without_the_middleware_is_flagged() -> None:
    from langchain.agents import create_agent

    class Bare(BaseAgent):
        name = "bare"

        def build(self, ctx: BuildContext) -> Any:
            return create_agent(model=calling(("plain", {"x": 1})), tools=[plain])

    assert _check(Bare(), "tools.governed")[0] is False


def test_headless_approval_gates_need_an_approver_role() -> None:
    @vdreamer.tool(approval=ApprovalSpec(kind="tool_approval"))
    def risky(x: int) -> int:
        """Risky."""
        return x

    class Headless(GovernedToolNode):
        name = "headless"
        mode = Mode.HEADLESS

        def build(self, ctx: BuildContext) -> Any:
            return _graph(ctx.tools.govern([risky]))

    ok, detail = _check(Headless(), "approval.routes")
    assert not ok and "risky" in detail and "A16" in detail

    class Chat(Headless):
        name = "chat"
        mode = Mode.CHAT

    assert _check(Chat(), "approval.routes") == (True, "risky")

    class Fixed(Headless):
        name = "fixed"
        tool_policies = {
            "risky": Policy(approval=ApprovalSpec(kind="tool_approval", approver_role="ops"))
        }

    assert _check(Fixed(), "approval.routes")[0] is True


class McpAgent(BaseAgent):
    name = "mcp-validate"

    def build(self, ctx: BuildContext) -> Any:
        return _graph(ctx.tools.mcp("fx") or ctx.tools.govern([plain]))


def test_validate_resolves_mcp_servers_from_resources_without_contacting_them(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    module = tmp_path / "mcp_validate_agent.py"
    module.write_text(
        "from test_tool_validate import McpAgent\nagent = McpAgent()\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    resources = tmp_path / "resources.yaml"
    resources.write_text(
        f"fx: {{type: mcp, command: {sys.executable!r}, args: ['server.py']}}\n",
        encoding="utf-8",
    )
    assert run_validate("mcp_validate_agent:agent") == 1  # no resources → unknown server
    assert "no MCP servers are configured" in capsys.readouterr().out
    assert run_validate("mcp_validate_agent:agent", resources=resources) == 0

    resources.write_text("fx: {type: mcp, command: /usr/bin/npx}\n", encoding="utf-8")
    assert run_validate("mcp_validate_agent:agent", resources=resources) == 1
    assert "fetches packages" in capsys.readouterr().err
