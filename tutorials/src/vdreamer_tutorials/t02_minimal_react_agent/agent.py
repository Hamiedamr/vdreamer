"""Lesson 2: a minimal ReAct agent — one model, one tool, ``create_agent``.

The model comes from the ``chat`` alias in ``models.yaml``::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t02-order-status \
        --models config/models.ollama.yaml
    uv run vdreamer run t02-order-status --message "Where is my order A1001?"
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode

# A stand-in for the shop's order system.
ORDERS = {
    "A1001": "shipped on 2026-10-01, arriving 2026-10-04",
    "A1002": "processing: the warehouse is packing it",
    "A1003": "delivered on 2026-09-28",
}


@vdreamer.tool
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an order by its id, for example "A1001"."""
    key = order_id.strip().upper()
    status = ORDERS.get(key)
    if status is None:
        return f"No order with id {order_id!r} was found."
    return f"Order {key}: {status}"


SYSTEM_PROMPT = (
    "You are the order-support assistant of an online shop. "
    "Use the get_order_status tool to answer questions about orders. "
    "If the customer gives no order id, ask for it. Never invent an order status."
)


class OrderStatusAgent(BaseAgent):
    name = "t02-order-status"
    mode = Mode.CHAT
    models = ("chat",)  # an alias from models.yaml, available as ctx.models["chat"]

    def build(self, ctx: BuildContext) -> Any:
        return create_agent(
            ctx.models["chat"],
            tools=[get_order_status],
            system_prompt=SYSTEM_PROMPT,
            **ctx.agent_kwargs(),  # checkpointer, store, run context, tool governance
        )


agent = OrderStatusAgent()
