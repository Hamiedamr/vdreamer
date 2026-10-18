"""The typed contracts: what comes in (a ticket) and what must come out (a triage)."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ORDER_ID = re.compile(r"^A\d{4}$")


class TicketInput(BaseModel):
    """The run input: validated by the gateway (422 if invalid) before anything runs."""

    subject: str = Field(min_length=3, max_length=200)
    body: str = Field(min_length=1, max_length=5000)
    customer_email: str | None = None


class TicketTriage(BaseModel):
    """Triage of one customer-support ticket."""

    category: Literal["order_status", "returns", "billing", "product_question", "complaint",
                      "other"]
    priority: Literal["low", "normal", "high", "urgent"] = Field(
        description="urgent only for safety issues or money taken twice"
    )
    order_ids: list[str] = Field(
        default_factory=list, description='Order ids mentioned in the ticket, like "A1001".'
    )
    summary: str = Field(max_length=200, description="One sentence for the support team.")
    needs_human: bool = Field(description="True if an agent must not answer this alone.")

    @field_validator("order_ids")
    @classmethod
    def _order_ids(cls, ids: list[str]) -> list[str]:
        bad = [i for i in ids if not ORDER_ID.match(i)]
        if bad:
            raise ValueError(f'order ids look like "A1001", got {bad}')
        return ids


#: What the agent falls back to when the model cannot produce a valid triage.
UNCLASSIFIED = TicketTriage(category="other", priority="normal",
                            summary="Could not be classified automatically.", needs_human=True)
