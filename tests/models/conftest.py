"""Models/resources setup: the streaming and tool kits on the path, stub servers."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent.parent
for kit in ("streaming", "tools"):
    path = str(TESTS / kit)
    if path not in sys.path:
        sys.path.insert(0, path)

from _stub_llm import Reply, StubServer, start_stub  # noqa: E402


@pytest.fixture
def vllm() -> Iterator[StubServer]:
    stub = start_stub("vllm", ["served-chat"], [Reply()])
    yield stub
    stub.stop()


@pytest.fixture
def ollama() -> Iterator[StubServer]:
    stub = start_stub("ollama", ["local-chat"], [Reply()])
    yield stub
    stub.stop()
