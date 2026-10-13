"""``tests/api`` fixtures over the ``_api`` kit (see its docstring for the two stacks)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from _api import Stack, make_stack
from fastapi.testclient import TestClient


@pytest.fixture
def stack() -> Stack:
    return make_stack(workers=False)


@pytest.fixture
def client(stack: Stack) -> Iterator[TestClient]:
    with TestClient(stack.app) as test_client:
        yield test_client


@pytest.fixture
def live_stack() -> Stack:
    return make_stack(workers=True)


@pytest.fixture
def live(live_stack: Stack) -> Iterator[TestClient]:
    with TestClient(live_stack.app) as test_client:
        yield test_client
