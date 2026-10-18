"""Lesson 27, part 2: reviewing a ReAct agent's *tool calls* before they run.

Two ways, side by side:

- ``issue_credit`` — LangChain's ``HumanInTheLoopMiddleware``: the reviewer sees
  the proposed call and may approve it, **edit its arguments**, or reject it
  with a message the model reads;
- ``cancel_order`` — a vdreamer tool policy, ``approval=ApprovalSpec(...)``:
  a governance gate with approve/reject and an approver role.

::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t27-tool-review \
        --models config/models.ollama.yaml
    uv run vdreamer run t27-tool-review --roles supervisor \
        --message "A1002 was late, give EUR 30 credit"
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

import vdreamer
from vdreamer import ApprovalSpec, BaseAgent, BuildContext, Mode

CREDITS: dict[str, float] = {}
CANCELLED: set[str] = set()


@vdreamer.tool
def issue_credit(order_id: str, amount_eur: float) -> str:
    """Give the customer a goodwill credit for an order."""
    CREDITS[order_id] = amount_eur
    return f"Credited EUR {amount_eur:.2f} for {order_id}."


@vdreamer.tool(approval=ApprovalSpec(kind="cancel_review", approver_role="supervisor"))
def cancel_order(order_id: str) -> str:
    """Cancel an order that has not shipped yet."""
    CANCELLED.add(order_id)
    return f"Cancelled {order_id}."


class ToolReviewAgent(BaseAgent):
    name = "t27-tool-review"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> Any:
        kwargs = ctx.agent_kwargs()
        review = HumanInTheLoopMiddleware(
            interrupt_on={"issue_credit": {"allowed_decisions": ["approve", "edit", "reject"]}},
            description_prefix="Credit pending review",
        )
        return create_agent(
            ctx.models["chat"],
            tools=[issue_credit, cancel_order],
            system_prompt="You handle goodwill credits and cancellations for an online shop.",
            middleware=[*kwargs.pop("middleware"), review],
            **kwargs,
        )


agent = ToolReviewAgent()
