"""Lesson 7: one agent, four tools — the model chooses, chains and combines them.

A shop assistant that can search the catalog, check stock, look up an order and
quote store policy. Questions like "is the 2-person tent in stock?" need two
tools in sequence (search for the SKU, then check its stock)::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t07-shop-assistant \
        --models config/models.ollama.yaml
    uv run vdreamer run t07-shop-assistant --message "Is the 2-person tent in stock?"
"""

from __future__ import annotations

from typing import Any, Literal

from langchain.agents import create_agent

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode

CATALOG: list[dict[str, Any]] = [
    {"sku": "TENT-2P", "name": "Trail tent, 2-person", "price_eur": 149.0,
     "tags": ["tent", "camping"]},
    {"sku": "TENT-4P", "name": "Family tent, 4-person", "price_eur": 289.0,
     "tags": ["tent", "camping", "family"]},
    {"sku": "BAG-0C", "name": "Sleeping bag, comfort 0 °C", "price_eur": 89.0,
     "tags": ["sleeping bag", "camping"]},
    {"sku": "LAMP-USB", "name": "Rechargeable camping lamp", "price_eur": 24.5,
     "tags": ["lamp", "light", "camping"]},
    {"sku": "BOOT-W", "name": "Waterproof hiking boots", "price_eur": 129.0,
     "tags": ["boots", "hiking", "shoes"]},
]
STOCK = {"TENT-2P": 0, "TENT-4P": 3, "BAG-0C": 12, "LAMP-USB": 40, "BOOT-W": 5}
RESTOCK = {"TENT-2P": "2026-10-20"}
ORDERS = {
    "A1001": "shipped on 2026-10-01, arriving 2026-10-04",
    "A1002": "processing: the warehouse is packing it",
}
POLICIES = {
    "returns": "Unused items can be returned within 30 days of delivery for a full refund.",
    "shipping": "Standard delivery takes 2-4 working days; orders over EUR 100 ship free.",
    "warranty": "All tents and boots carry a 2-year warranty against manufacturing defects.",
}


@vdreamer.tool(parse_docstring=True)
def search_products(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Search the product catalog by keywords. Returns SKU, name and price per match.

    Use this first when the customer names a product rather than a SKU.

    Args:
        query: Keywords, for example "tent" or "sleeping bag".
        max_results: How many matches to return at most.
    """
    words = query.lower().split()
    matches = [
        {key: item[key] for key in ("sku", "name", "price_eur")}
        for item in CATALOG
        if all(word in (item["name"].lower() + " " + " ".join(item["tags"])) for word in words)
    ]
    return matches[:max_results]


@vdreamer.tool(parse_docstring=True)
def check_stock(sku: str) -> str:
    """Check how many units of a product are in stock, by its SKU (from search_products).

    Args:
        sku: The product SKU, for example "TENT-4P".
    """
    key = sku.strip().upper()
    if key not in STOCK:
        return f"Unknown SKU {key!r}. Use search_products to find the right SKU."
    units = STOCK[key]
    if units:
        return f"{key}: {units} in stock."
    restock = RESTOCK.get(key)
    return f"{key}: out of stock" + (f", expected back on {restock}." if restock else ".")


@vdreamer.tool(parse_docstring=True)
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an existing order by its id.

    Args:
        order_id: The order id, for example "A1001".
    """
    key = order_id.strip().upper()
    status = ORDERS.get(key)
    return f"Order {key}: {status}" if status else f"No order {key!r} found."


@vdreamer.tool(parse_docstring=True)
def get_policy(topic: Literal["returns", "shipping", "warranty"]) -> str:
    """Quote the shop's official policy on one topic. Never paraphrase policy from memory.

    Args:
        topic: Which policy: returns, shipping or warranty.
    """
    return POLICIES[topic]


TOOLS = [search_products, check_stock, get_order_status, get_policy]

SYSTEM_PROMPT = """You are the shop assistant of an outdoor-gear store.
Tools:
- search_products: find products and their SKUs from a description.
- check_stock: stock for one SKU. If you only have a product name, search first.
- get_order_status: the status of an existing order.
- get_policy: returns, shipping and warranty rules. Quote it; do not rely on memory.
Use as few tool calls as needed. Independent lookups can be requested together.
Never invent products, prices, stock levels or policies."""


class ShopAssistantAgent(BaseAgent):
    name = "t07-shop-assistant"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> Any:
        return create_agent(
            ctx.models["chat"],
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
            **ctx.agent_kwargs(),
        )


agent = ShopAssistantAgent()
