"""The tool layer: a factory that turns any ``Inventory`` into two agent tools.

The tools are thin adapters. They validate input, call the domain, and
translate its results and errors into the tool contract: a JSON object on
success, a ``ToolException`` with an actionable message on failure.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

import vdreamer

from .inventory import InsufficientStock, Inventory

SKU = Field(pattern=r"^[A-Z0-9]+-[A-Z0-9]+$", description='A product SKU, e.g. "TENT-4P".')


class CheckStockInput(BaseModel):
    sku: str = SKU


class ReserveStockInput(BaseModel):
    sku: str = SKU
    units: int = Field(gt=0, le=10, description="How many units to reserve (1 to 10).")


def make_inventory_tools(inventory: Inventory) -> list[BaseTool]:
    """``check_stock`` (read-only) and ``reserve_stock`` (side effect) over ``inventory``."""

    @vdreamer.tool(args_schema=CheckStockInput, timeout_s=10)
    def check_stock(sku: str) -> dict[str, Any]:
        """Get the available and reserved units of one product. Read-only."""
        level = inventory.level(sku)
        if level is None:
            raise ToolException(f"Unknown SKU {sku}. Check the SKU with the customer.")
        return level.model_dump()

    @vdreamer.tool(args_schema=ReserveStockInput, side_effects=True, timeout_s=10)
    def reserve_stock(sku: str, units: int) -> dict[str, Any]:
        """Reserve units of a product for the customer. Changes stock: call it only when
        the customer explicitly asks to reserve, and only once per request."""
        key = vdreamer.current_context().idempotency_key
        assert key is not None  # side_effects=True: governance always sets one
        try:
            reservation = inventory.reserve(sku, units, idempotency_key=key)
        except KeyError:
            raise ToolException(f"Unknown SKU {sku}. Check the SKU with the customer.") from None
        except InsufficientStock as exc:
            raise ToolException(f"Cannot reserve: {exc}. Offer fewer units.") from None
        return reservation.model_dump()

    return [check_stock, reserve_stock]
