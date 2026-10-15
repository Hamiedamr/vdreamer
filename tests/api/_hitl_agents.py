"""Milestone 8 agents: one per interrupt source (§4.5, A44), plus the policy cases.

- ``flow`` — an ``ApprovalGate`` (``require_approval``) with ``edit_schema``,
  ``feedback → replan`` and a chat ``reject`` route;
- ``review`` — ``create_agent`` + ``HumanInTheLoopMiddleware`` (``tool_review``),
  with ``interrupt_policies`` attaching a scope and an approver role;
- ``ask`` — a raw ``interrupt(value)`` (``generic``);
- ``strict`` — a ``require_approval`` under a four-eyes scope (A32);
- ``sitecheck`` — a ``HEADLESS`` agent whose ``reject`` ends the run (§4.3).

The MCP elicitation source is ``_api.ElicitAgent`` (``elicit``).
"""

from __future__ import annotations

from typing import Any, TypedDict

import pydantic
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from vdreamer import ApprovalGate, BaseAgent, BuildContext, InterruptPolicy, Mode, require_approval

#: Every call the ``review`` agent's tool really made (args as executed).
SENT: list[dict[str, Any]] = []


class SqlEdit(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    sql: str = pydantic.Field(min_length=1)
    limit: int = pydantic.Field(default=100, ge=1, le=1000)


class FlowState(TypedDict, total=False):
    question: str
    sql: str
    limit: int
    plans: int
    approval: Any
    result: str


class FlowAgent(BaseAgent):
    """Chat: ``prepare → gate → {approve|edit: execute, reject: on_reject, feedback: replan}``."""

    name = "flow"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> Any:
        def prepare(state: FlowState) -> dict[str, Any]:
            plans = state.get("plans", 0) + 1
            sql = state.get("sql") or f"SELECT * FROM {state.get('question', 't')}"
            return {"sql": sql, "plans": plans}

        def execute(state: FlowState) -> dict[str, Any]:
            return {"result": f"ran {state['sql']} limit {state.get('limit', 100)}"}

        def on_reject(state: FlowState) -> dict[str, Any]:
            return {"result": "not run"}

        def replan(state: FlowState) -> dict[str, Any]:
            return {"sql": f"{state['sql']} /* {state['approval']['text']} */"}

        return ApprovalGate(
            FlowState,
            kind="sql_review",
            payload=lambda state: {"sql": state["sql"], "plans": state["plans"]},
            prepare=prepare,
            execute=execute,
            on_reject=on_reject,
            replan=replan,
            edit_schema=SqlEdit,
            decision_key="approval",
            ttl_hours=1,
        ).build()


class _FakeToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    SENT.append({"to": to, "body": body})
    return f"sent to {to}"


class ReviewAgent(BaseAgent):
    """Chat: ``create_agent`` whose ``send_email`` calls wait for a ``tool_review``."""

    name = "review"
    mode = Mode.CHAT
    interrupt_policies = {
        "send_*": InterruptPolicy(scope="mail:send"),
        "tool_review": InterruptPolicy(approver_role="mail_reviewer", ttl_hours=2),
    }

    def build(self, ctx: BuildContext) -> Any:
        model = _FakeToolCallingModel(
            responses=[
                AIMessage(
                    "",
                    id="ai_call",
                    tool_calls=[
                        {
                            "name": "send_email",
                            "args": {"to": "ops@example.internal", "body": "draft"},
                            "id": "call_mail",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage("mailed", id="ai_done"),
            ]
        )
        kwargs = ctx.agent_kwargs()
        hitl = HumanInTheLoopMiddleware(
            interrupt_on={"send_email": {"allowed_decisions": ["approve", "edit", "reject"]}}
        )
        kwargs["middleware"] = [hitl, *kwargs["middleware"]]
        return create_agent(model=model, tools=[send_email], **kwargs)


class AskState(TypedDict, total=False):
    question: str
    answer: Any


class AskAgent(BaseAgent):
    """Chat: a raw ``interrupt("…")`` — the ``generic`` source."""

    name = "ask"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> Any:
        def ask(state: AskState) -> dict[str, Any]:
            return {"answer": interrupt(f"which region for {state.get('question', '?')}?")}

        graph = StateGraph(AskState)
        graph.add_node("ask", ask)
        graph.add_edge(START, "ask")
        graph.add_edge("ask", END)
        return graph


class DecisionState(TypedDict, total=False):
    text: str
    decision: Any


class StrictAgent(BaseAgent):
    """Chat: a ``db:write:prod`` approval — four-eyes under ``HITL_STRICT_SCOPES``."""

    name = "strict"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> Any:
        def gate(state: DecisionState) -> dict[str, Any]:
            decision = require_approval(
                "sql_write",
                payload={"sql": state.get("text", "")},
                actions=["approve", "reject"],
                approver_role="dba",
                scope="db:write:prod",
            )
            return {"decision": decision}

        graph = StateGraph(DecisionState)
        graph.add_node("gate", gate)
        graph.add_edge(START, "gate")
        graph.add_edge("gate", END)
        return graph


class SiteInput(pydantic.BaseModel):
    site_id: str


class SiteState(TypedDict, total=False):
    site_id: str
    decision: Any


class SiteCheckAgent(BaseAgent):
    """Headless: a role-gated verdict approval (UC2's shape); ``reject`` → ``rejected``."""

    name = "sitecheck"
    mode = Mode.HEADLESS
    input_schema = SiteInput

    def build(self, ctx: BuildContext) -> Any:
        def gate(state: SiteState) -> dict[str, Any]:
            decision = require_approval(
                "site_verdict",
                payload={"site_id": state.get("site_id")},
                actions=["approve", "reject"],
                approver_role="site_approver",
            )
            return {"decision": decision}

        graph = StateGraph(SiteState)
        graph.add_node("gate", gate)
        graph.add_edge(START, "gate")
        graph.add_edge("gate", END)
        return graph


def hitl_agents() -> list[BaseAgent]:
    return [FlowAgent(), ReviewAgent(), AskAgent(), StrictAgent(), SiteCheckAgent()]
