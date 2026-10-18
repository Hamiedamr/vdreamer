"""Business types and rules. Pure Python: no LangChain, no vdreamer, no I/O."""

from __future__ import annotations

from datetime import date
from typing import Literal, Protocol

from pydantic import BaseModel

OrderStatus = Literal["processing", "shipped", "delivered", "cancelled"]


class Order(BaseModel):
    order_id: str
    customer_email: str
    status: OrderStatus
    total_eur: float
    placed_on: date


class OrderRepository(Protocol):
    def get(self, order_id: str) -> Order | None: ...

    def for_customer(self, email: str) -> list[Order]: ...


class InMemoryOrders:
    """An ``OrderRepository`` over a list (tests, demos)."""

    def __init__(self, orders: list[Order]) -> None:
        self._orders = {order.order_id: order for order in orders}

    def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id.strip().upper())

    def for_customer(self, email: str) -> list[Order]:
        wanted = email.strip().lower()
        found = [o for o in self._orders.values() if o.customer_email == wanted]
        return sorted(found, key=lambda o: o.placed_on, reverse=True)


def can_cancel(order: Order) -> bool:
    """The rule: only orders that have not shipped can be cancelled."""
    return order.status == "processing"


def demo_orders() -> InMemoryOrders:
    return InMemoryOrders([
        Order(order_id="A1001", customer_email="ada@example.com", status="shipped",
              total_eur=149.0, placed_on=date(2026, 9, 28)),
        Order(order_id="A1002", customer_email="ada@example.com", status="processing",
              total_eur=89.0, placed_on=date(2026, 10, 2)),
        Order(order_id="A1003", customer_email="lin@example.com", status="delivered",
              total_eur=24.5, placed_on=date(2026, 9, 1)),
    ])
