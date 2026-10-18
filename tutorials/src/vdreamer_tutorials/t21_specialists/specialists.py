"""Three specialists, each with one area of responsibility, its own tools and prompt.

``build_specialist(spec, model, **agent_kwargs)`` makes a ``create_agent`` graph
from a ``Specialist`` description. Lesson 21 serves each one alone; lessons
22–24 combine the same specialists into multi-agent systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, ToolException

import vdreamer

ORDERS = {
    "A1001": {"status": "shipped", "items": ["TENT-2P"], "total_eur": 149.0},
    "A1002": {"status": "processing", "items": ["BAG-0C"], "total_eur": 89.0},
}
CATALOG = {
    "TENT-2P": {"name": "Trail tent, 2-person", "price_eur": 149.0, "stock": 0},
    "TENT-4P": {"name": "Family tent, 4-person", "price_eur": 289.0, "stock": 3},
    "BAG-0C": {"name": "Sleeping bag, comfort 0 °C", "price_eur": 89.0, "stock": 12},
}
POLICIES = {
    "returns": "Unused items can be returned within 30 days of delivery.",
    "shipping": "Orders over EUR 100 ship free; standard delivery takes 2-4 working days.",
    "warranty": "Tents and boots carry a 2-year warranty against manufacturing defects.",
}

# -- tools, grouped by owner ----------------------------------------------------------------


@vdreamer.tool
def get_order(order_id: str) -> dict[str, Any]:
    """Get an order's status, items and total by its id, for example "A1001"."""
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        raise ToolException(f"No order {order_id}.")
    return {"order_id": order_id.upper(), **order}


@vdreamer.tool
def search_catalog(query: str) -> list[dict[str, Any]]:
    """Find products by keyword; returns SKU, name, price and stock."""
    words = query.lower().split()
    return [{"sku": sku, **item} for sku, item in CATALOG.items()
            if all(word in item["name"].lower() for word in words)]


@vdreamer.tool
def get_policy(topic: str) -> str:
    """Quote the shop's policy on returns, shipping or warranty."""
    policy = POLICIES.get(topic.strip().lower())
    if policy is None:
        raise ToolException(f"No policy on {topic!r}; topics: {', '.join(POLICIES)}.")
    return policy


# -- the specialists ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Specialist:
    name: str
    responsibility: str  # one sentence: what it does, used by supervisors (lesson 22)
    prompt: str
    tools: tuple[BaseTool, ...]


ORDERS_SPECIALIST = Specialist(
    name="orders",
    responsibility="Answers questions about existing orders: status, contents and totals.",
    prompt="You handle questions about existing orders only. Look orders up with get_order. "
           "If asked about anything else, say it is outside your area.",
    tools=(get_order,),
)
PRODUCTS_SPECIALIST = Specialist(
    name="products",
    responsibility="Answers questions about products: what we sell, prices and stock.",
    prompt="You handle product questions only: find products with search_catalog and report "
           "prices and stock. If asked about anything else, say it is outside your area.",
    tools=(search_catalog,),
)
POLICIES_SPECIALIST = Specialist(
    name="policies",
    responsibility="Quotes the shop's returns, shipping and warranty policies.",
    prompt="You quote shop policies only, always with get_policy, never from memory. If "
           "asked about anything else, say it is outside your area.",
    tools=(get_policy,),
)
SPECIALISTS = (ORDERS_SPECIALIST, PRODUCTS_SPECIALIST, POLICIES_SPECIALIST)


def build_specialist(spec: Specialist, model: BaseChatModel, **agent_kwargs: Any) -> Any:
    """A ``create_agent`` graph for one specialist.

    Served alone, pass ``**ctx.agent_kwargs()``. Nested inside another agent
    (lessons 22–24), pass just the governance middleware: the specialist then
    keeps no state of its own between calls.
    """
    return create_agent(model, tools=list(spec.tools), system_prompt=spec.prompt,
                        name=spec.name, **agent_kwargs)
