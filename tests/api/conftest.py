"""``tests/api`` fixtures over the ``_api`` kit (see its docstring for the two stacks).

Every stack fixture is parametrized over the store backend, so each REST
contract test runs against the in-memory stores and the SQLite ones.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from _api import Stack, make_stack
from fastapi.testclient import TestClient

STORES = ("memory", "sqlite")


def _stack(request: pytest.FixtureRequest, tmp_path: Path, *, workers: bool) -> Iterator[Stack]:
    path = tmp_path / "state.db" if request.param == "sqlite" else None
    built = make_stack(workers=workers, sqlite_path=path)
    try:
        yield built
    finally:
        built.close()


@pytest.fixture(params=STORES)
def stack(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Stack]:
    yield from _stack(request, tmp_path, workers=False)


@pytest.fixture
def client(stack: Stack) -> Iterator[TestClient]:
    with TestClient(stack.app) as test_client:
        yield test_client


@pytest.fixture(params=STORES)
def live_stack(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Stack]:
    yield from _stack(request, tmp_path, workers=True)


@pytest.fixture
def live(live_stack: Stack) -> Iterator[TestClient]:
    with TestClient(live_stack.app) as test_client:
        yield test_client
