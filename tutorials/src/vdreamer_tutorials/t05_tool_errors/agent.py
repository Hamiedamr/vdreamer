"""Lesson 5: hardened tools — validated inputs, contained errors, bounded outputs.

Three order-desk tools, each showing how vdreamer's governance handles one kind
of trouble: arguments that fail validation, errors the tool raises on purpose
or by accident, a call that hangs, and a result too large for the model::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t05-order-desk \
        --models config/models.ollama.yaml
    uv run vdreamer run t05-order-desk --message "Order A1001, email ada@example.com"
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode

ORDERS = {
    "A1001": {"email": "ada@example.com", "status": "shipped", "tracking_id": "TRK-1001"},
    "A1002": {"email": "ada@example.com", "status": "processing", "tracking_id": None},
    "A1003": {"email": "lin@example.com", "status": "delivered", "tracking_id": "TRK-1003"},
}
PARCELS = {"TRK-1001": "in transit, at the Frankfurt hub", "TRK-1003": "delivered to a neighbour"}
#: Parcels whose carrier lookup hangs (a stand-in for a slow external API).
SLOW_PARCELS = {"TRK-SLOW"}
#: Flip to False to simulate the order database being down.
ORDERS_DB_ONLINE = True


# -- 1. validated input, deliberate errors, redaction ------------------------------------


class OrderLookup(BaseModel):
    """The arguments of ``get_order``, validated before the tool runs."""

    order_id: str = Field(
        pattern=r"^A\d{4}$", description='The order id: "A" and four digits, e.g. "A1001".'
    )
    email: str = Field(
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        description="The email address the order was placed with.",
    )


@vdreamer.tool(args_schema=OrderLookup, redact=["email"], timeout_s=5)
def get_order(order_id: str, email: str) -> dict[str, Any]:
    """Look up one order. The customer must give the order id and their email address."""
    if not ORDERS_DB_ONLINE:
        # An unexpected failure: governance contains it and masks the secret.
        raise RuntimeError("orders-db login failed: password=hunter2")
    order = ORDERS.get(order_id)
    if order is None or order["email"] != email.lower():
        # An expected, explainable failure: the message is for the model.
        raise ToolException(f"No order {order_id} was found for that email address.")
    return {"order_id": order_id, "status": order["status"], "tracking_id": order["tracking_id"]}


# -- 2. a call that may hang ------------------------------------------------------------


@vdreamer.tool(timeout_s=1.0)
async def track_parcel(tracking_id: str) -> str:
    """Ask the carrier where a parcel is, by tracking id (from get_order)."""
    if tracking_id in SLOW_PARCELS:
        await asyncio.sleep(30)  # the carrier never answers in time
    return PARCELS.get(tracking_id, f"The carrier does not know {tracking_id}.")


# -- 3. a result that can be very large -----------------------------------------------------


@vdreamer.tool(max_result_chars=400, redact=["email"])
def order_history(email: str) -> str:
    """List every past order event for a customer, newest first."""
    events = [f"2026-09-{day:02d}: order event #{day} for this customer"
              for day in range(30, 0, -1)]
    return "\n".join(events)


SYSTEM_PROMPT = (
    "You are an order-desk assistant. To look up an order you need its id and the "
    "customer's email address: ask for whatever is missing. If a tool reports an error, "
    "explain it plainly and suggest what the customer can do; never make up data."
)


class OrderDeskAgent(BaseAgent):
    name = "t05-order-desk"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> Any:
        return create_agent(
            ctx.models["chat"],
            tools=[get_order, track_parcel, order_history],
            system_prompt=SYSTEM_PROMPT,
            **ctx.agent_kwargs(),
        )


agent = OrderDeskAgent()
