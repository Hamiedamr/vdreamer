"""Prompt text lives in one place, built from settings, so it can be reviewed and tested."""

from __future__ import annotations

from .settings import HelpdeskSettings


def system_prompt(settings: HelpdeskSettings) -> str:
    return (
        f"You are the order helpdesk of {settings.shop_name}. "
        "Look orders up with get_order or list_orders; check can_cancel before promising "
        "a cancellation. Never invent order data."
    )
