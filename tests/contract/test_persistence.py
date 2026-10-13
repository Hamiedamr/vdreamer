"""Persistence tests — durability defaults (A42, A6) and the saver-factory seam.

Everything here runs offline: no Postgres server, no sockets. The real
``_load_saver_class`` import is exercised directly (psycopg is installed in
the dev venv; the missing-extra path is simulated by intercepting the import),
and the ``open_checkpointer`` composition is proven against a fake saver
class — calling the real one would open a network connection and violate the
offline/zero-egress spirit (PRD §5.2).
"""

from __future__ import annotations

import builtins
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from vdreamer.agents import BaseAgent
from vdreamer.persistence import (
    DEFAULT_DURABILITY,
    MissingPostgresDependencyError,
    PersistenceConfigError,
    _load_saver_class,
    dsn_from_env,
    durability_for,
    open_checkpointer,
)

#: RFC 2606 ``.invalid`` — unresolvable, so a stray connection attempt could
#: never reach a real database.
DSN = "postgresql://vdreamer@db.invalid:5432/vdreamer"


class _Stub:
    """Minimal agent-shaped object: ``name`` plus a ``durability`` attribute."""

    def __init__(self, name: str = "stub", durability: Any = None) -> None:
        self.name = name
        self.durability = durability


def test_durability_defaults_to_sync_when_attribute_missing() -> None:
    assert durability_for(object()) == DEFAULT_DURABILITY

    class Bare:
        name = "bare"

    assert durability_for(Bare()) == "sync"


def test_durability_passthrough_sync_and_async() -> None:
    assert durability_for(_Stub("s", "sync")) == "sync"
    assert durability_for(_Stub("a", "async")) == "async"


def test_durability_invalid_value_raises_with_agent_name() -> None:
    with pytest.raises(ValueError, match="durability must be 'sync' or 'async'") as exc_info:
        durability_for(_Stub("greedy", "always"))
    message = str(exc_info.value)
    assert "'greedy'" in message
    assert "'always'" in message


def test_durability_invalid_without_name_falls_back_to_class_name() -> None:
    class NoName:
        durability = "instant"

    with pytest.raises(ValueError, match="'NoName'"):
        durability_for(NoName())


def test_durability_for_baseagent_shaped_classes() -> None:
    class DefaultAgent(BaseAgent):
        name = "default-agent"

    class AsyncAgent(BaseAgent):
        name = "async-agent"
        durability = "async"

    assert durability_for(DefaultAgent()) == "sync"
    assert durability_for(AsyncAgent()) == "async"


def test_dsn_from_env_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VDREAMER_DB_DSN", DSN)
    assert dsn_from_env("VDREAMER_DB_DSN") == DSN


def test_dsn_from_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VDREAMER_DB_DSN", raising=False)
    with pytest.raises(PersistenceConfigError, match="VDREAMER_DB_DSN") as exc_info:
        dsn_from_env("VDREAMER_DB_DSN")
    assert "A23" in str(exc_info.value)


def test_load_saver_class_is_the_upstream_async_postgres_saver() -> None:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    saver_cls = _load_saver_class()
    assert saver_cls is AsyncPostgresSaver


def test_load_saver_class_wraps_missing_psycopg(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real ImportError path: a missing saver module becomes a friendly error."""
    real_import = builtins.__import__

    def _no_postgres(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langgraph.checkpoint.postgres.aio":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_postgres)
    with pytest.raises(MissingPostgresDependencyError) as exc_info:
        _load_saver_class()

    message = str(exc_info.value)
    assert "vdreamer[postgres]" in message
    assert "pip" in message
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


async def test_open_checkpointer_runs_setup_once_before_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Composition contract: setup() runs exactly once, before the saver is yielded."""
    events: list[str] = []

    class _FakeSaver:
        @classmethod
        @asynccontextmanager
        async def from_conn_string(cls, conn_string: str) -> AsyncGenerator[Any]:
            events.append(f"from_conn_string:{conn_string}")
            yield cls()

        async def setup(self) -> None:
            events.append("setup")

    monkeypatch.setattr("vdreamer.persistence._load_saver_class", lambda: _FakeSaver)
    async with open_checkpointer(DSN) as saver:
        # setup ran exactly once and before the yield:
        assert events == [f"from_conn_string:{DSN}", "setup"]
        assert isinstance(saver, _FakeSaver)
    # Nothing extra happens on close:
    assert events == [f"from_conn_string:{DSN}", "setup"]


async def test_open_checkpointer_propagates_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the saver class is unavailable, the error propagates unchanged (A42 seam)."""

    def _unavailable() -> type[object]:
        raise MissingPostgresDependencyError("saver class unavailable in this test")

    monkeypatch.setattr("vdreamer.persistence._load_saver_class", _unavailable)
    with pytest.raises(
        MissingPostgresDependencyError, match="saver class unavailable in this test"
    ):
        async with open_checkpointer(DSN) as _saver:
            pytest.fail("open_checkpointer must not yield a saver when the class is unavailable")
