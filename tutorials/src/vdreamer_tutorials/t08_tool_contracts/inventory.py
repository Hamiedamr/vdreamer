"""The domain: stock levels and reservations, with no agent or LangChain code.

``Inventory`` is the interface the tools depend on. ``InMemoryInventory`` is
the implementation the tutorial ships; a real one would talk to a warehouse
system. Because the tools only know the protocol, swapping implementations (or
using a fake in tests) changes nothing else.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class StockLevel(BaseModel):
    """What ``check_stock`` returns: a stable, typed result."""

    sku: str
    available_units: int
    reserved_units: int


class Reservation(BaseModel):
    """What ``reserve_stock`` returns."""

    reservation_id: str
    sku: str
    units: int


class InsufficientStock(Exception):
    """Fewer units are available than requested."""


class Inventory(Protocol):
    """The operations the inventory tools need, and nothing more."""

    def level(self, sku: str) -> StockLevel | None: ...

    def reserve(self, sku: str, units: int, idempotency_key: str) -> Reservation: ...


class InMemoryInventory:
    """An ``Inventory`` in a dictionary. ``reserve`` is idempotent per key."""

    def __init__(self, units: dict[str, int]) -> None:
        self._units = dict(units)
        self._reserved: dict[str, int] = {sku: 0 for sku in units}
        self._by_key: dict[str, Reservation] = {}

    def level(self, sku: str) -> StockLevel | None:
        if sku not in self._units:
            return None
        return StockLevel(sku=sku, available_units=self._units[sku] - self._reserved[sku],
                          reserved_units=self._reserved[sku])

    def reserve(self, sku: str, units: int, idempotency_key: str) -> Reservation:
        if idempotency_key in self._by_key:  # a retry of a call that already happened
            return self._by_key[idempotency_key]
        level = self.level(sku)
        if level is None:
            raise KeyError(sku)
        if units > level.available_units:
            raise InsufficientStock(f"only {level.available_units} units of {sku} available")
        self._reserved[sku] += units
        reservation = Reservation(reservation_id=f"R-{len(self._by_key) + 1:04d}", sku=sku,
                                  units=units)
        self._by_key[idempotency_key] = reservation
        return reservation
