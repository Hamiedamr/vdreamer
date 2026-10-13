"""The ``vdreamer worker`` and ``vdreamer dev`` process entrypoints (M3 WP3).

``worker`` is the PRD §2.1 Worker row: one process serves one agent, claims
runs from the Postgres queue (A7 — Postgres is the only gateway↔worker
transport), drives them through the agent's runner, and writes the protocol
events to the ``run_events`` log. ``serve`` (the gateway) is Milestone 4;
``vdreamer run`` (the terminal client) is Milestone 11.

``dev`` is the D11 single-process mode: one process, in-memory queue, the
agent's checkpoints on SQLite (a local ``--db`` file) instead of Postgres —
the smoke path for the whole enqueue → claim → drive → settle lifecycle
without any infrastructure. Hot reload (PRD §2.3's "single-process hot
reload") is future work; ``dev`` builds the agent once, like the worker.

Both resolve the agent through :func:`vdreamer.discovery.resolve_agent` (a
``pkg.module:agent`` spec, or a bare name via ``--registry``/``--agents-dir``,
D2) and take connection strings from the environment only (A23).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from vdreamer.agents import BaseAgent
from vdreamer.discovery import AgentLoadError, resolve_agent
from vdreamer.eventlog import EventSink, InMemoryEventSink, PostgresEventLog
from vdreamer.persistence import (
    PG_DSN_ENV_VAR,
    MissingPostgresDependencyError,
    PersistenceConfigError,
    dsn_from_env,
    open_checkpointer,
)
from vdreamer.queue import InMemoryRunQueue, PostgresRunQueue, RunState
from vdreamer.runtime import AgentRunner
from vdreamer.worker import Worker


def _default_claimant() -> str:
    """The lease-claim stamp when ``--claimant`` is omitted."""
    return f"worker-{socket.gethostname()}-{os.getpid()}"


def _resolve_cli_agent(
    command: str,
    spec: str,
    registry: str | None,
    agents_dir: str | None,
) -> BaseAgent | None:
    """Resolve ``--agent`` for a process entrypoint; ``None`` (message printed) on failure."""
    try:
        return resolve_agent(spec, registry=registry, agents_dir=agents_dir)
    except AgentLoadError as exc:  # RegistryError subclasses AgentLoadError
        print(f"{command}: {exc}", file=sys.stderr)
    except Exception as exc:  # agent code raised while importing; surface it, don't crash
        print(f"{command}: cannot load {spec!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return None


def _install_sigint_stop(stop: asyncio.Event) -> bool:
    """Make Ctrl-C set ``stop`` instead of raising KeyboardInterrupt.

    POSIX only: the proactor loop on Windows has no ``add_signal_handler``;
    there the KeyboardInterrupt fallback in the callers stops the process and
    the in-flight row recovers through the lease sweep (§3.2).
    """
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, stop.set)
    except NotImplementedError:
        return False
    return True


# ---------------------------------------------------------------------------
# vdreamer worker
# ---------------------------------------------------------------------------


async def _worker_main(agent: BaseAgent, dsn: str, claimant: str) -> int:
    """The worker process body: everything inside the checkpointer's lifetime."""
    queue = PostgresRunQueue(dsn)
    await queue.setup()  # boot-time guard; `vdreamer migrate` is the explicit path

    stop = asyncio.Event()
    posix_sigint = _install_sigint_stop(stop)
    try:
        async with open_checkpointer(dsn) as saver:
            runner = AgentRunner(agent, checkpointer=saver)

            async def sink_factory(run_id: str) -> EventSink:
                return await PostgresEventLog.connect(dsn)

            worker = Worker(
                runner,
                queue,
                agent_name=agent.name,
                claimant=claimant,
                checkpointer=saver,
                sink_factory=sink_factory,
            )
            print(f"vdreamer worker: agent {agent.name!r} (mode {agent.mode.value})")
            print(f"vdreamer worker: claimant {claimant} — Postgres queue + run_events")
            if not posix_sigint:
                print("vdreamer worker: no SIGINT handler here (Windows) — Ctrl-C stops hard")
            print("vdreamer worker: running — Ctrl-C to stop")
            await worker.run(stop)
    except KeyboardInterrupt:
        print("vdreamer worker: interrupted — the in-flight row recovers via the lease sweep")
        return 130
    print("vdreamer worker: stopped")
    return 0


def run_worker(
    *,
    agent: str,
    registry: str | None = None,
    agents_dir: str | None = None,
    claimant: str | None = None,
) -> int:
    """CLI entry point for ``vdreamer worker`` (PRD §2.1)."""
    resolved = _resolve_cli_agent("worker", agent, registry, agents_dir)
    if resolved is None:
        return 1
    try:
        dsn = dsn_from_env(PG_DSN_ENV_VAR)
    except PersistenceConfigError as exc:
        print(f"worker: {exc}", file=sys.stderr)
        return 1
    try:
        return asyncio.run(_worker_main(resolved, dsn, claimant or _default_claimant()))
    except MissingPostgresDependencyError as exc:
        print(f"worker: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# vdreamer dev (D11)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _open_dev_checkpointer(db: str) -> AsyncIterator[AsyncSqliteSaver]:
    """The D11 checkpointer: the agent's checkpoints on SQLite at ``db``.

    ``AsyncSqliteSaver`` (``langgraph-checkpoint-sqlite``, a base runtime
    dependency — the ``dev`` entrypoint is core CLI) holds the file open for
    the dev process's lifetime and runs its own migrations (``setup()``),
    like the worker's Postgres saver. Run state survives a dev restart: a
    fresh saver re-opened on the same file reads the previous runs' final
    state back.
    """
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        await saver.setup()  # the saver's own migrations, like the worker's
        yield saver


async def _dev_main(
    agent: BaseAgent,
    db: str,
    input_json: str | None,
    thread_id: str | None,
) -> int:
    """The dev process body: one runner, in-memory queue, until Ctrl-C or one run."""
    sinks: dict[str, InMemoryEventSink] = {}

    async def sink_factory(run_id: str) -> EventSink:
        sink = InMemoryEventSink()
        sinks[run_id] = sink
        return sink

    async with _open_dev_checkpointer(db) as checkpointer:
        runner = AgentRunner(agent, checkpointer=checkpointer)
        queue = InMemoryRunQueue()
        worker = Worker(
            runner,
            queue,
            agent_name=agent.name,
            claimant="dev",
            checkpointer=checkpointer,
            sink_factory=sink_factory,
            poll_interval_s=0.05,
        )
        print(f"vdreamer dev: agent {agent.name!r} (mode {agent.mode.value})")
        print(f"vdreamer dev: queue=in-memory, events=in-memory, checkpoints=sqlite:{db}")
        if input_json is None:
            stop = asyncio.Event()
            _install_sigint_stop(stop)
            try:
                print("vdreamer dev: running — Ctrl-C to stop")
                await worker.run(stop)
            except KeyboardInterrupt:
                print("vdreamer dev: interrupted")
                return 130
            print("vdreamer dev: stopped")
            return 0

        # The smoke: enqueue one run from --input, drive it synchronously, report.
        try:
            payload: Any = json.loads(input_json)
        except json.JSONDecodeError as exc:
            print(f"dev: --input is not valid JSON: {exc}", file=sys.stderr)
            return 1
        record = await queue.enqueue(
            agent.name, payload, thread_id=thread_id, run_id=None
        )
        print(f"vdreamer dev: queued {record.id} (thread {record.thread_id})")
        await worker.run_once()
        final = await queue.get(record.id)
        assert final is not None  # enqueue created it; nothing removes rows
        events = len(sinks[record.id].events)
        print(
            f"vdreamer dev: run {record.id} -> {final.status.value} "
            f"(attempt {final.attempt}, {events} events)"
        )
        return 0 if final.status is RunState.COMPLETED else 1


def run_dev(
    *,
    agent: str,
    registry: str | None = None,
    agents_dir: str | None = None,
    db: str,
    input_json: str | None = None,
    thread_id: str | None = None,
) -> int:
    """CLI entry point for ``vdreamer dev`` (D11): one process, no infrastructure."""
    resolved = _resolve_cli_agent("dev", agent, registry, agents_dir)
    if resolved is None:
        return 1
    try:
        return asyncio.run(_dev_main(resolved, db, input_json, thread_id))
    except MissingPostgresDependencyError as exc:  # pragma: no cover - dev never touches psycopg
        print(f"dev: {exc}", file=sys.stderr)
        return 1
