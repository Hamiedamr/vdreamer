"""Lesson 9: deciding which tools the model may see and how often it may use them.

A support agent with customer tools and staff-only tools. Three controls work
together:

- ``staff_tools_for_staff_only`` (middleware) hides staff tools from the model
  unless the caller has the ``staff`` role — the model cannot choose what it
  cannot see;
- ``requires_roles`` on those tools (vdreamer governance) refuses them at
  execution time anyway — defense in depth;
- ``ToolCallLimitMiddleware`` allows at most one refund per run.

::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t09-support-desk \
        --models config/models.ollama.yaml
    uv run vdreamer run t09-support-desk --roles staff --message "Refund order A1001"
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    ToolCallLimitMiddleware,
    wrap_model_call,
)

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode, VDreamerContext

ORDERS = {
    "A1001": {"status": "delivered", "total_eur": 149.0, "refunded": False},
    "A1002": {"status": "processing", "total_eur": 89.0, "refunded": False},
}
STAFF_ROLE = "staff"


@vdreamer.tool
def get_order_status(order_id: str) -> str:
    """Look up the status of an order by its id, for example "A1001"."""
    order = ORDERS.get(order_id.strip().upper())
    return f"Order {order_id}: {order['status']}" if order else f"No order {order_id!r}."


@vdreamer.tool(requires_roles=[STAFF_ROLE])
def get_order_details(order_id: str) -> dict[str, Any]:
    """Staff only: the full internal record of an order, including totals."""
    order = ORDERS.get(order_id.strip().upper())
    return {"order_id": order_id, **order} if order else {"error": "no such order"}


@vdreamer.tool(requires_roles=[STAFF_ROLE], side_effects=True)
def issue_refund(order_id: str, reason: str) -> str:
    """Staff only: refund an order in full. Changes money: use only when asked to refund."""
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        return f"No order {order_id!r}."
    order["refunded"] = True
    return f"Refunded EUR {order['total_eur']:.2f} for {order_id} ({reason})."


STAFF_TOOLS = {"get_order_details", "issue_refund"}


@wrap_model_call
async def staff_tools_for_staff_only(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse:
    """Offer staff tools to the model only when the caller has the staff role."""
    context = request.runtime.context
    roles = context.roles if isinstance(context, VDreamerContext) else ()
    if STAFF_ROLE not in roles:
        visible = [t for t in request.tools if getattr(t, "name", None) not in STAFF_TOOLS]
        request = request.override(tools=visible)
    return await handler(request)


SYSTEM_PROMPT = (
    "You are a support agent for an online shop. Use the tools you have; if a request needs "
    "a tool you do not have, say that a staff member must handle it."
)


class SupportDeskAgent(BaseAgent):
    name = "t09-support-desk"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> Any:
        kwargs = ctx.agent_kwargs()
        middleware = [
            *kwargs.pop("middleware"),  # vdreamer's governance stays first (outermost)
            staff_tools_for_staff_only,
            ToolCallLimitMiddleware(tool_name="issue_refund", run_limit=1),
        ]
        return create_agent(
            ctx.models["chat"],
            tools=[get_order_status, get_order_details, issue_refund],
            system_prompt=SYSTEM_PROMPT,
            middleware=middleware,
            **kwargs,
        )


agent = SupportDeskAgent()
