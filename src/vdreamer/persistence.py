"""Postgres-backed persistence: the checkpointer and store factories, durability policy.

vdreamer uses LangGraph's ``AsyncPostgresSaver`` and never reimplements
checkpointing. Durability is ``"sync"`` by default because crash recovery
needs every step checkpointed; agents may opt into ``"async"`` per agent.
Connection strings come from the environment, never YAML.

:func:`open_store` opens LangGraph's ``AsyncPostgresStore`` (the long-term
memory store) on the same DSN, runs its migrations, and yields it — workers
hand it to ``build()`` as ``ctx.store`` (``agent_kwargs()["store"]``) and
``vdreamer migrate`` applies its tables.

``vdreamer dev`` runs on SQLite instead (``AsyncSqliteSaver`` /
``AsyncSqliteStore`` on the dev file); that mode is not this module's concern.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, cast

if TYPE_CHECKING:
    # Typing-only: importing langgraph.checkpoint.postgres at module level
    # would hard-require psycopg (the [postgres] extra) in every process.
    # The runtime imports live in _load_saver_class() / _load_store_class().
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore

#: Per-superstep invocation durability.
Durability: TypeAlias = Literal["sync", "async"]

#: "sync" by default: crash recovery needs every step checkpointed.
DEFAULT_DURABILITY: Final[Durability] = "sync"

#: The Postgres DSN environment variable: connection strings come from
#: the environment, never YAML and never the command line. ``vdreamer worker``
#: and ``vdreamer migrate`` resolve it here, at startup, via :func:`dsn_from_env`.
PG_DSN_ENV_VAR: Final[str] = "VDREAMER_PG_DSN"


class PersistenceConfigError(RuntimeError):
    """Persistence configuration is missing or unusable."""


class MissingPostgresDependencyError(RuntimeError):
    """The psycopg dependency ([postgres] extra) is not installed."""


def durability_for(agent: Any) -> Durability:
    """Return the agent's invocation durability, defaulting to ``"sync"``.

    ``BaseAgent`` declares ``durability``; the defensive ``getattr`` treats
    agents without the attribute (agents written before it existed, test
    doubles) as the default instead of failing. Invalid values raise here —
    at registration — rather than surfacing as a silent durability change
    mid-run, when crash recovery would depend on it.
    """
    value = getattr(agent, "durability", DEFAULT_DURABILITY)
    if value not in ("sync", "async"):
        name = getattr(agent, "name", type(agent).__name__)
        raise ValueError(
            f"agent {name!r} durability must be 'sync' or 'async', got {value!r}"
        )
    return cast(Durability, value)


def dsn_from_env(name: str) -> str:
    """Resolve a Postgres DSN from an environment variable.

    Connection strings never appear in YAML: config references the variable
    name and the worker resolves it here, at startup, where a missing value
    fails loudly before any run is accepted.
    """
    try:
        return os.environ[name]
    except KeyError:
        raise PersistenceConfigError(
            f"environment variable {name} is not set "
            "(connection strings come from env, never YAML)"
        ) from None


def _load_saver_class() -> type[AsyncPostgresSaver]:
    """Import ``AsyncPostgresSaver`` lazily — the ``[postgres]`` extra seam.

    Importing ``langgraph.checkpoint.postgres`` requires psycopg (its
    ``_internal``/``_ainternal`` modules import psycopg at module scope), so
    the import stays inside this function: environments without the extra
    fail with a friendly error at the point of use instead of an
    ``ImportError`` at ``vdreamer.persistence`` import time. This is also the
    seam tests monkeypatch to fake the saver class offline.
    """
    try:
        # ModuleNotFoundError is an ImportError; both mean "no psycopg".
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise MissingPostgresDependencyError(
            "psycopg is required for the vdreamer[postgres] extra: "
            "pip install 'vdreamer[postgres]'"
        ) from exc
    return AsyncPostgresSaver


@asynccontextmanager
async def open_checkpointer(dsn: str) -> AsyncGenerator[AsyncPostgresSaver]:
    """Open the worker's checkpointer: connect, migrate, then yield.

    A worker holds this open for its lifetime — one saver per process, shared
    by every agent and run via ``compile(checkpointer=...)`` /
    ``create_agent(checkpointer=...)``. ``setup()`` (tables + migrations) runs
    once per database before the saver is yielded, so no run ever sees a
    half-migrated checkpointer.

    Per-run invocation durability — the ``"sync"``/``"async"`` passed to
    ``ainvoke`` — comes from ``durability_for(agent)``, threaded by the
    runtime; the checkpointer instance is the same either way (is about
    invocation durability, not a different saver).
    """
    saver_cls = _load_saver_class()
    async with saver_cls.from_conn_string(dsn) as saver:
        await saver.setup()
        yield saver


def _load_store_class() -> type[AsyncPostgresStore]:
    """Import ``AsyncPostgresStore`` lazily (psycopg, the ``[postgres]`` extra)."""
    try:
        from langgraph.store.postgres.aio import AsyncPostgresStore
    except ImportError as exc:
        raise MissingPostgresDependencyError(
            "psycopg is required for the vdreamer[postgres] extra: "
            "pip install 'vdreamer[postgres]'"
        ) from exc
    return AsyncPostgresStore


@asynccontextmanager
async def open_store(dsn: str) -> AsyncGenerator[AsyncPostgresStore]:
    """Open the worker's long-term memory store: connect, migrate, then yield.

    One store per worker process, like the checkpointer; ``build()`` receives
    it as ``ctx.store``. vdreamer never reimplements it (LangGraph's own
    ``AsyncPostgresStore``, no vector index — retrieval is agent code).
    """
    store_cls = _load_store_class()
    async with store_cls.from_conn_string(dsn) as store:
        await store.setup()
        yield store
