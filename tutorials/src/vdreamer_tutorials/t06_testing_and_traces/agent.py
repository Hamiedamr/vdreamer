"""Lesson 6: an agent built to be tested — returns eligibility with an injectable clock.

The tool's answer depends on today's date, the classic source of flaky tests.
The agent takes the clock as a constructor argument, so tests pin the date;
the registered instance uses the real one::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t06-returns \
        --models config/models.ollama.yaml
    uv run vdreamer run t06-returns --message "Can I still return order A1003?"
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import BaseTool

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode

#: Delivery date per order; None = not delivered yet.
DELIVERED: dict[str, date | None] = {
    "A1001": date(2026, 10, 4),
    "A1002": None,
    "A1003": date(2026, 9, 1),
}
RETURN_WINDOW_DAYS = 30


def return_decision(order_id: str, today: date) -> str:
    """The business rule, as a plain function: easy to test exhaustively."""
    key = order_id.strip().upper()
    if key not in DELIVERED:
        return f"No order with id {key!r} was found."
    delivered = DELIVERED[key]
    if delivered is None or delivered > today:  # a future date: bad data or clock skew
        return f"Order {key} has not been delivered yet, so it cannot be returned yet."
    days = (today - delivered).days
    window = RETURN_WINDOW_DAYS
    if days <= window:
        return f"Order {key} can be returned: delivered {days} days ago, within {window}."
    return f"Order {key} cannot be returned: delivered {days} days ago, past {window}."


def make_check_return(today: Callable[[], date]) -> BaseTool:
    """Build the tool around a clock, so tests can choose what "today" is."""

    @vdreamer.tool
    def check_return(order_id: str) -> str:
        """Check whether an order can still be returned (30 days from delivery)."""
        return return_decision(order_id, today())

    return check_return


SYSTEM_PROMPT = (
    "You help customers with returns. Use check_return to decide whether an order can be "
    "returned and repeat its reason. Ask for the order id if it is missing."
)


class ReturnsAgent(BaseAgent):
    name = "t06-returns"
    mode = Mode.CHAT
    models = ("chat",)

    def __init__(self, today: Callable[[], date] = date.today) -> None:
        self.today = today

    def build(self, ctx: BuildContext) -> Any:
        return create_agent(
            ctx.models["chat"],
            tools=[make_check_return(self.today)],
            system_prompt=SYSTEM_PROMPT,
            **ctx.agent_kwargs(),
        )


agent = ReturnsAgent()
