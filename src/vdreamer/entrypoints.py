"""The ``serve``, ``worker`` and ``dev`` process entrypoints (M3 WP3, M4).

``serve`` is the PRD §2.1 Gateway row (Milestone 4): a stateless FastAPI app
(D20) over Postgres — queue, threads, ``run_events`` (+ ``LISTEN`` for live
SSE), interrupts, and the checkpointer for thread history — serving every
agent named by ``--agent``/``--registry``/``--agents-dir``. Uvicorn binds the
internal interface only (A14). ``vdreamer run`` (the terminal client) is
Milestone 11.

``worker`` is the §2.1 Worker row: one process serves one agent, claims
runs from the Postgres queue (A7 — Postgres is the only gateway↔worker
transport), drives them through the agent's runner, writes the protocol
events to the ``run_events`` log, and records interrupts for the gateway's
inbox. ``--metrics-port`` exposes its Prometheus metrics (§5.4).

``dev`` is the D11 single-process mode: one process, the agent's
checkpoints on SQLite (a local ``--db`` file) and everything else in memory.
With ``--input`` it smokes one run through enqueue → claim → drive → settle;
without it, it serves the full gateway *and* runs the worker loop in the same
process, so the REST/SSE contract works locally with no infrastructure. Hot
reload (PRD §2.3's "single-process hot reload") is future work; ``dev``
builds the agent once, like the worker.

Both resolve the agent through :func:`vdreamer.discovery.resolve_agent` (a
``pkg.module:agent`` spec, or a bare name via ``--registry``/``--agents-dir``,
D2) and take connection strings from the environment only (A23).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import signal
import socket
import sys
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from vdreamer.agents import BaseAgent
from vdreamer.artifacts import InMemoryArtifactStore, UnavailableArtifactStore
from vdreamer.discovery import (
    AgentLoadError,
    discover_agents_dir,
    load_agent,
    load_registry,
    resolve_agent,
)
from vdreamer.eventlog import (
    EventHub,
    EventSink,
    InMemoryEventLog,
    InMemoryEventSink,
    PostgresEventListener,
    PostgresEventLog,
    PostgresEventStore,
)
from vdreamer.interrupts import InMemoryInterruptStore, PostgresInterruptStore
from vdreamer.persistence import (
    PG_DSN_ENV_VAR,
    MissingPostgresDependencyError,
    PersistenceConfigError,
    dsn_from_env,
    open_checkpointer,
)
from vdreamer.queue import InMemoryRunQueue, PostgresRunQueue, RunState
from vdreamer.runtime import AgentRunner
from vdreamer.threads import InMemoryThreadStore, PostgresThreadStore
from vdreamer.worker import Worker

if TYPE_CHECKING:
    from fastapi import FastAPI


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


def _start_metrics_server(command: str, host: str, port: int) -> None:
    """Expose :data:`vdreamer.metrics.REGISTRY` over HTTP (pull-only, §5.4)."""
    from prometheus_client import start_http_server

    from vdreamer import metrics
    from vdreamer.gateway.settings import validate_bind_host

    validate_bind_host(host)
    start_http_server(port, addr=host, registry=metrics.REGISTRY)
    print(f"vdreamer {command}: metrics on http://{host}:{port}/metrics")


async def _worker_main(
    agent: BaseAgent,
    dsn: str,
    claimant: str,
    metrics_host: str,
    metrics_port: int | None,
) -> int:
    """The worker process body: everything inside the checkpointer's lifetime."""
    queue = PostgresRunQueue(dsn)
    await queue.setup()  # boot-time guard; `vdreamer migrate` is the explicit path
    interrupts = PostgresInterruptStore(dsn)
    await interrupts.setup()
    if metrics_port is not None:
        _start_metrics_server("worker", metrics_host, metrics_port)

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
                interrupt_store=interrupts,
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
    metrics_host: str = "127.0.0.1",
    metrics_port: int | None = None,
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
        return asyncio.run(
            _worker_main(
                resolved, dsn, claimant or _default_claimant(), metrics_host, metrics_port
            )
        )
    except MissingPostgresDependencyError as exc:
        print(f"worker: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # GatewayConfigError from a wildcard --metrics-host, mostly
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


async def _dev_gateway(
    agent: BaseAgent, checkpointer: Any, host: str, port: int, tokens: tuple[str, ...]
) -> int:
    """``vdreamer dev`` without ``--input``: gateway + worker loop in one process (D11)."""
    from vdreamer.gateway import GatewayBackend, GatewaySettings, create_app

    queue = InMemoryRunQueue(clock=time.time)  # wall clock: the API shows timestamps
    log = InMemoryEventLog()
    interrupts = InMemoryInterruptStore()
    runner = AgentRunner(agent, checkpointer=checkpointer)

    async def sink_factory(run_id: str) -> EventSink:
        return log  # one shared log: the gateway reads what the worker writes

    worker = Worker(
        runner,
        queue,
        agent_name=agent.name,
        claimant="dev",
        checkpointer=checkpointer,
        sink_factory=sink_factory,
        poll_interval_s=0.05,
        interrupt_store=interrupts,
    )
    backend = GatewayBackend(
        agents={agent.name: agent},
        queue=queue,
        threads=InMemoryThreadStore(),
        event_reader=log,
        event_writer=log,
        hub=log.hub,
        interrupts=interrupts,
        artifacts=InMemoryArtifactStore(),
        checkpointer=checkpointer,
    )

    @asynccontextmanager
    async def lifespan() -> AsyncIterator[None]:
        stop = asyncio.Event()
        loop_task = asyncio.create_task(worker.run(stop), name="vdreamer-dev-worker")
        try:
            yield
        finally:
            stop.set()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(loop_task, 5.0)

    app = create_app(backend, GatewaySettings(tokens=tokens), lifespan=lifespan)
    print(f"vdreamer dev: gateway on http://{host}:{port}/api/v1 (docs: /api/v1/docs)")
    await _uvicorn_serve(app, host, port)
    print("vdreamer dev: stopped")
    return 0


async def _dev_main(
    agent: BaseAgent,
    db: str,
    input_json: str | None,
    thread_id: str | None,
    host: str = "127.0.0.1",
    port: int = 8000,
    tokens: tuple[str, ...] = (),
) -> int:
    """The dev process body: the gateway + worker until Ctrl-C, or one smoke run."""
    sinks: dict[str, InMemoryEventSink] = {}

    async def sink_factory(run_id: str) -> EventSink:
        sink = InMemoryEventSink()
        sinks[run_id] = sink
        return sink

    async with _open_dev_checkpointer(db) as checkpointer:
        if input_json is None:
            print(f"vdreamer dev: agent {agent.name!r} (mode {agent.mode.value})")
            print(f"vdreamer dev: queue/events/threads in memory, checkpoints=sqlite:{db}")
            return await _dev_gateway(agent, checkpointer, host, port, tokens)
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
    host: str = "127.0.0.1",
    port: int = 8000,
) -> int:
    """CLI entry point for ``vdreamer dev`` (D11): one process, no infrastructure."""
    resolved = _resolve_cli_agent("dev", agent, registry, agents_dir)
    if resolved is None:
        return 1
    tokens: tuple[str, ...] = ()
    if input_json is None:
        from vdreamer.gateway.settings import (
            TOKEN_ENV_VAR,
            GatewayConfigError,
            tokens_from_env,
            validate_bind_host,
        )

        try:
            validate_bind_host(host)
            tokens = tokens_from_env()
        except GatewayConfigError as exc:
            if os.environ.get(TOKEN_ENV_VAR, "").strip():
                print(f"dev: {exc}", file=sys.stderr)
                return 1
            # Dev convenience: no token configured → a fresh one for this process.
            tokens = (secrets.token_urlsafe(24),)
            print(f"vdreamer dev: {TOKEN_ENV_VAR} unset — using a one-off token: {tokens[0]}")
    try:
        return asyncio.run(_dev_main(resolved, db, input_json, thread_id, host, port, tokens))
    except MissingPostgresDependencyError as exc:  # pragma: no cover - dev never touches psycopg
        print(f"dev: {exc}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# uvicorn
# ---------------------------------------------------------------------------


async def _uvicorn_serve(app: FastAPI, host: str, port: int) -> None:
    """Serve ``app`` with uvicorn until it is told to stop (Ctrl-C / SIGTERM).

    No ``Server`` header, no proxy-header trust (the Web Backend is the only
    caller and identity comes from its explicit headers, D13).
    """
    import uvicorn

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        lifespan="on",
        server_header=False,
        proxy_headers=False,
        log_level="info",
    )
    await uvicorn.Server(config).serve()


# ---------------------------------------------------------------------------
# vdreamer serve (the gateway, M4)
# ---------------------------------------------------------------------------


class ServeAgentsError(Exception):
    """``serve`` was not told which agents to serve, or two share a name."""


def resolve_serve_agents(
    specs: Sequence[str],
    *,
    registry: str | None = None,
    agents_dir: str | None = None,
) -> dict[str, BaseAgent]:
    """The agents one gateway serves (D2), keyed by ``name``.

    ``--agent`` (repeatable) picks agents explicitly — specs, or names resolved
    against ``--registry``/``--agents-dir``. Without ``--agent``, every agent of
    the registry and of the agents dir is served.
    """
    loaded: list[BaseAgent] = []
    if specs:
        loaded = [resolve_agent(spec, registry=registry, agents_dir=agents_dir) for spec in specs]
    else:
        if registry is not None:
            loaded.extend(load_agent(spec) for spec in load_registry(registry).values())
        if agents_dir is not None:
            loaded.extend(discover_agents_dir(agents_dir).values())
    if not loaded:
        raise ServeAgentsError("serve needs --agent, --registry, or --agents-dir (D2)")
    agents: dict[str, BaseAgent] = {}
    for agent in loaded:
        existing = agents.get(agent.name)
        if existing is not None and existing is not agent:
            raise ServeAgentsError(f"two agents are named {agent.name!r}")
        agents[agent.name] = agent
    return agents


async def _serve_main(
    agents: dict[str, BaseAgent], dsn: str, tokens: tuple[str, ...], host: str, port: int
) -> int:
    """The gateway process body: Postgres stores, ``LISTEN`` loop, uvicorn (§2.1)."""
    from vdreamer.gateway import GatewayBackend, GatewaySettings, create_app

    queue = PostgresRunQueue(dsn)
    threads = PostgresThreadStore(dsn)
    interrupts = PostgresInterruptStore(dsn)
    # Boot-time guards, like the worker's; `vdreamer migrate` is the explicit path.
    await queue.setup()
    await threads.setup()
    await interrupts.setup()
    log = await PostgresEventLog.connect(dsn)
    try:
        await log.setup()
    finally:
        await log.close()

    hub = EventHub()
    listener = PostgresEventListener(dsn, hub)
    store = PostgresEventStore(dsn)

    async with open_checkpointer(dsn) as saver:
        backend = GatewayBackend(
            agents=agents,
            queue=queue,
            threads=threads,
            event_reader=store,
            event_writer=store,
            hub=hub,
            interrupts=interrupts,
            artifacts=UnavailableArtifactStore(),  # the MinIO-backed store is Milestone 6
            checkpointer=saver,
            probes={"postgres": queue.probe, "event_listener": listener.probe},
        )

        @asynccontextmanager
        async def lifespan() -> AsyncIterator[None]:
            listener.start()
            try:
                yield
            finally:
                await listener.stop()

        app = create_app(backend, GatewaySettings(tokens=tokens), lifespan=lifespan)
        names = ", ".join(sorted(agents))
        print(f"vdreamer serve: agents {names}")
        print(f"vdreamer serve: http://{host}:{port}/api/v1 (docs: /api/v1/docs)")
        await _uvicorn_serve(app, host, port)
    print("vdreamer serve: stopped")
    return 0


def run_serve(
    *,
    agents: Sequence[str] = (),
    registry: str | None = None,
    agents_dir: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> int:
    """CLI entry point for ``vdreamer serve`` (PRD §2.1, §2.5)."""
    from vdreamer.gateway.settings import GatewayConfigError, tokens_from_env, validate_bind_host

    try:
        validate_bind_host(host)
        tokens = tokens_from_env()
    except GatewayConfigError as exc:
        print(f"serve: {exc}", file=sys.stderr)
        return 1
    try:
        served = resolve_serve_agents(agents, registry=registry, agents_dir=agents_dir)
    except (AgentLoadError, ServeAgentsError) as exc:
        print(f"serve: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # agent code raised while importing; surface it, don't crash
        print(f"serve: cannot load agents: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    try:
        dsn = dsn_from_env(PG_DSN_ENV_VAR)
    except PersistenceConfigError as exc:
        print(f"serve: {exc}", file=sys.stderr)
        return 1
    try:
        return asyncio.run(_serve_main(served, dsn, tokens, host, port))
    except MissingPostgresDependencyError as exc:
        print(f"serve: {exc}", file=sys.stderr)
        return 1
