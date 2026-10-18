"""Settings: one immutable object, built once and passed down (lesson 12 loads it from env)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HelpdeskSettings:
    shop_name: str = "Trailhead Outfitters"
    max_orders_listed: int = 5
