"""Lesson 4: a custom tool — from a Python function to the schema the model sees.

``estimate_shipping`` prices a parcel. Its name, docstring and type hints become
the JSON schema the model reads; ``parse_docstring=True`` moves each argument's
description from the docstring into that schema::

    cd tutorials
    uv run vdreamer dev --registry vdreamer.yaml --agent t04-shipping-quote \
        --models config/models.ollama.yaml
    uv run vdreamer run t04-shipping-quote --message "How much to send 2.5 kg to Spain, fast?"
"""

from __future__ import annotations

from typing import Any, Literal

from langchain.agents import create_agent

import vdreamer
from vdreamer import BaseAgent, BuildContext, Mode

Destination = Literal["domestic", "eu", "international"]

#: Per destination: (base price in EUR, price per kg in EUR, standard delivery days).
RATES: dict[str, tuple[float, float, int]] = {
    "domestic": (4.50, 1.20, 2),
    "eu": (9.00, 2.50, 4),
    "international": (19.00, 6.00, 8),
}
EXPRESS_SURCHARGE = 1.8  # express costs 80% more and takes half the days


@vdreamer.tool(parse_docstring=True)
def estimate_shipping(
    weight_kg: float, destination: Destination, express: bool = False
) -> dict[str, Any]:
    """Estimate the price and delivery time of shipping one parcel.

    Args:
        weight_kg: Parcel weight in kilograms, for example 2.5.
        destination: "domestic" (same country), "eu" (another EU country) or "international".
        express: Use the express service: faster, but more expensive.
    """
    base, per_kg, days = RATES[destination]
    cost = base + per_kg * weight_kg
    if express:
        cost *= EXPRESS_SURCHARGE
        days = max(1, days // 2)
    return {
        "destination": destination,
        "weight_kg": weight_kg,
        "express": express,
        "cost_eur": round(cost, 2),
        "delivery_days": days,
    }


SYSTEM_PROMPT = (
    "You quote shipping prices for an online shop. Always use the estimate_shipping tool; "
    "never guess a price. Countries in the EU other than the shop's own are 'eu'. "
    "State the price in euros and the delivery time in days."
)


class ShippingQuoteAgent(BaseAgent):
    name = "t04-shipping-quote"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> Any:
        return create_agent(
            ctx.models["chat"],
            tools=[estimate_shipping],
            system_prompt=SYSTEM_PROMPT,
            **ctx.agent_kwargs(),
        )


agent = ShippingQuoteAgent()
