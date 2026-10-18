"""The agent-facing contract: tools built over an ``OrderRepository``."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, ToolException

import vdreamer

from .domain import OrderRepository, can_cancel
from .settings import HelpdeskSettings


def make_tools(orders: OrderRepository, settings: HelpdeskSettings) -> list[BaseTool]:
    def _get(order_id: str) -> Any:
        order = orders.get(order_id)
        if order is None:
            raise ToolException(f"No order {order_id.strip().upper()} exists.")
        return order

    @vdreamer.tool
    def get_order(order_id: str) -> dict[str, Any]:
        """Get one order by id, for example "A1001"."""
        return dict(_get(order_id).model_dump(mode="json"))

    @vdreamer.tool(redact=["email"])
    def list_orders(email: str) -> list[dict[str, Any]]:
        """List a customer's most recent orders by their email address."""
        recent = orders.for_customer(email)[: settings.max_orders_listed]
        return [{"order_id": o.order_id, "status": o.status} for o in recent]

    @vdreamer.tool(name="can_cancel")
    def can_cancel_tool(order_id: str) -> str:
        """Check whether an order can still be cancelled."""
        order = _get(order_id)
        verdict = "can" if can_cancel(order) else "cannot"
        return f"Order {order.order_id} {verdict} be cancelled (status: {order.status})."

    return [get_order, list_orders, can_cancel_tool]
