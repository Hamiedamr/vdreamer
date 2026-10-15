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

``dev`` is the D11 single-process mode on one local SQLite ``--db`` file.
With ``--input`` it smokes one run through enqueue → claim → drive → settle
(checkpoints on SQLite, the rest in memory); without it, it serves the full
gateway *and* runs the worker loop in the same process, with runs, events,
threads, interrupts, and checkpoints all persisted to that file
(``vdreamer.sqlite_state``), so the REST/SSE contract works locally with no
infrastructure and survives a restart. The testing phase runs on this mode;
the Postgres ``serve``/``worker`` topology is the later migration. Hot
reload (PRD §2.3's "single-process hot reload") is future work; ``dev``
builds the agent once, like the worker.

Both resolve the agent through :func:`vdreamer.discovery.resolve_agent` (a
``pkg.module:agent`` spec, or a bare name via ``--registry``/``--agents-dir``,
D2) and take connection strings from the environment only (A23).

``worker`` and ``dev`` (Milestone 5) read ``resources.yaml`` (``--resources``
or ``$VDREAMER_RESOURCES``) and start its ``type: mcp`` servers at boot
(``vdreamer.mcp``, §2.7): their tools reach the agent through
``ctx.tools.mcp(...)``, and under ``dev`` every ``required`` server is a
``/readyz`` probe.

Milestone 7 adds ``models.yaml`` (``--models`` or ``$VDREAMER_MODELS``) and
the rest of ``resources.yaml``: at boot, ``worker`` and ``dev`` build the
agent's declared model aliases, load its fastembed models once, probe the
chat endpoints, and open its resource clients (``vdreamer.backends``),
which ``build()`` receives as ``ctx.models`` / ``ctx.resources``. Any
failure — an unknown alias, an endpoint down, a missing model file or
``*_env`` variable — stops the process before it claims a run.

Milestone 9 adds the headless entry: ``serve`` and ``dev`` read
``VDREAMER_SERVICE_TOKENS`` (A16) and ``CALLBACK_ALLOWLIST`` (A26) for
``POST /agents/{name}/trigger``, and run the outbox dispatcher in the gateway
process. Every event writer — the gateway's and each worker's — is wrapped in
``OutboxEventSink``, so the callback and notification entries are recorded
with the events (``vdreamer.outbox``).
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
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from vdreamer.adapters import ResourceUnavailable
from vdreamer.agents import BaseAgent
from vdreamer.artifacts import (
    ArtifactStore,
    InMemoryArtifactStore,
    PostgresArtifactStore,
    UnavailableArtifactStore,
)
from vdreamer.backends import (
    AgentBackends,
    BackendConfig,
    load_backend_config,
    open_agent_backends,
)
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
    InMemoryEventSink,
    PostgresEventListener,
    PostgresEventLog,
    PostgresEventStore,
)
from vdreamer.events import EventType
from vdreamer.interrupts import (
    InterruptSweeper,
    PostgresAdvisoryLeader,
    PostgresInterruptStore,
)
from vdreamer.mcp import MCPRegistry, MCPServerConfig
from vdreamer.models import ModelsError, ModelUnavailable
from vdreamer.network import NetworkAllowlist
from vdreamer.objects import MINIO_ENDPOINT_ENV, MinioConfigError, MinioObjectStore
from vdreamer.outbox import (
    OutboxConfigError,
    OutboxEventSink,
    OutboxSettings,
    PostgresOutboxStore,
    build_dispatcher,
)
from vdreamer.persistence import (
    PG_DSN_ENV_VAR,
    MissingPostgresDependencyError,
    PersistenceConfigError,
    dsn_from_env,
    open_checkpointer,
    open_store,
)
from vdreamer.queue import InMemoryRunQueue, PostgresRunQueue, RunState
from vdreamer.resources import ResourcesError
from vdreamer.runtime import AgentRunner
from vdreamer.sqlite_state import SqliteState, open_sqlite_state
from vdreamer.threads import PostgresThreadStore
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
    """Resolve ``--agent`` for a process entrypoint; ``None`` (message printed) on failure.

    An agent that fails the static contract checks is refused here, so a
    process never serves what ``vdreamer validate`` rejects.
    """
    try:
        agent = resolve_agent(spec, registry=registry, agents_dir=agents_dir)
    except AgentLoadError as exc:  # RegistryError subclasses AgentLoadError
        print(f"{command}: {exc}", file=sys.stderr)
        return None
    except Exception as exc:  # agent code raised while importing; surface it, don't crash
        print(f"{command}: cannot load {spec!r}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    problem = contract_error(agent)
    if problem is not None:
        print(f"{command}: {problem}", file=sys.stderr)
        return None
    return agent


def contract_error(agent: BaseAgent) -> str | None:
    """Why ``agent`` fails the static contract checks, or ``None`` when it passes."""
    from vdreamer.validate import static_checks

    failed = [check for check in static_checks(agent) if not check.ok]
    if not failed:
        return None
    detail = "; ".join(f"{check.name}: {check.detail}" for check in failed)
    label = getattr(agent, "name", "") or type(agent).__name__
    return (
        f"agent {label!r} fails the contract ({detail}); "
        "run `vdreamer validate` for the full report"
    )


def _logger(command: str) -> Callable[[str], None]:
    def log(line: str) -> None:
        print(f"vdreamer {command}: {line}")

    return log


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
# models.yaml + resources.yaml (§2.7, §3.6, §3.7)
# ---------------------------------------------------------------------------


def _load_config(
    command: str, resources: str | None, models: str | None
) -> tuple[list[MCPServerConfig], BackendConfig] | None:
    """The MCP servers and the backend config; ``None`` (printed) on a bad file."""
    from vdreamer.mcp import mcp_servers_from_resources

    allowlist = NetworkAllowlist.from_env()
    try:
        backends = load_backend_config(models, resources, allowlist=allowlist)
        mcp = mcp_servers_from_resources(backends.resources, allowlist=allowlist)
    except (ResourcesError, ModelsError) as exc:  # MCPConfigError subclasses ResourcesError
        print(f"{command}: {exc}", file=sys.stderr)
        return None
    return mcp, backends


#: What stops a process at boot when the agent's backends are not ready.
BACKEND_ERRORS = (ModelsError, ResourcesError, ModelUnavailable, ResourceUnavailable)


# ---------------------------------------------------------------------------
# Headless trigger, outbox and callbacks (M9, A16, A26)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadlessConfig:
    """What the gateway reads from the environment for ``/trigger``: the upstream
    service tokens, ``CALLBACK_ALLOWLIST``, and the outbox's channels."""

    outbox: OutboxSettings = field(default_factory=OutboxSettings)
    service_tokens: tuple[tuple[str, str], ...] = ()
    callback_allowlist: NetworkAllowlist | None = None

    def summary(self) -> str:
        parts = [
            "callbacks on"
            if self.callback_allowlist is not None
            else "callbacks off (CALLBACK_ALLOWLIST unset)"
        ]
        if self.outbox.webhook_url is not None:
            parts.append("webhook")
        if self.outbox.smtp is not None:
            parts.append("email")
        if self.service_tokens:
            parts.append(f"{len(self.service_tokens)} service token(s)")
        return ", ".join(parts)


def _outbox_settings(command: str) -> OutboxSettings | None:
    """The outbox's environment; ``None`` (printed) when it is incomplete or unsafe."""
    try:
        return OutboxSettings.from_env()
    except OutboxConfigError as exc:
        print(f"{command}: {exc}", file=sys.stderr)
        return None


def _headless_config(command: str, tokens: tuple[str, ...]) -> HeadlessConfig | None:
    """The gateway's M9 environment, checked against ``tokens``; ``None`` (printed) when wrong."""
    from vdreamer.gateway import GatewaySettings
    from vdreamer.gateway.settings import (
        GatewayConfigError,
        callback_allowlist_from_env,
        service_tokens_from_env,
    )

    outbox = _outbox_settings(command)
    if outbox is None:
        return None
    try:
        service_tokens = service_tokens_from_env()
        allowlist = callback_allowlist_from_env()
        GatewaySettings(tokens=tokens, service_tokens=service_tokens, callback_allowlist=allowlist)
    except GatewayConfigError as exc:
        print(f"{command}: {exc}", file=sys.stderr)
        return None
    return HeadlessConfig(outbox, service_tokens, allowlist)


@asynccontextmanager
async def _open_mcp(
    command: str, configs: Sequence[MCPServerConfig]
) -> AsyncIterator[MCPRegistry | None]:
    """Start the MCP servers for the process lifetime (none configured -> ``None``)."""
    if not configs:
        yield None
        return
    from vdreamer.mcp import open_mcp_registry

    async with open_mcp_registry(configs, allowlist=NetworkAllowlist.from_env()) as registry:
        for server in registry.servers.values():
            state = "up" if server.up else f"DOWN ({server.last_error})"
            role = "required" if server.config.required else "optional"
            tools = len(server.tools())
            print(f"vdreamer {command}: MCP {server.name} [{role}] {state}, {tools} tool(s)")
        yield registry


# ---------------------------------------------------------------------------
# Artifact storage (serve / worker)
# ---------------------------------------------------------------------------


async def _open_artifact_store(command: str, dsn: str) -> tuple[ArtifactStore, Any]:
    """``(store, probe)``: Postgres rows + MinIO bytes, or the unavailable store.

    MinIO comes from ``VDREAMER_MINIO_*`` (A23). Without it, artifact routes
    answer ``404`` and artifact writes fail — said once at boot.
    """
    objects = MinioObjectStore.from_env()  # MinioConfigError propagates: half-configured
    if objects is None:
        print(f"vdreamer {command}: {MINIO_ENDPOINT_ENV} unset — no artifact storage")
        return UnavailableArtifactStore(), None
    store = PostgresArtifactStore(dsn, objects)
    await store.setup()

    async def probe() -> None:
        await asyncio.to_thread(objects.probe)

    print(f"vdreamer {command}: artifacts in MinIO bucket {objects.bucket!r}")
    return store, probe


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
    mcp_configs: Sequence[MCPServerConfig] = (),
    backend_config: BackendConfig | None = None,
    outbox_settings: OutboxSettings | None = None,
) -> int:
    """The worker process body: everything inside the checkpointer's lifetime."""
    queue = PostgresRunQueue(dsn)
    await queue.setup()  # boot-time guard; `vdreamer migrate` is the explicit path
    interrupts = PostgresInterruptStore(dsn)
    await interrupts.setup()
    outbox = PostgresOutboxStore(dsn)
    await outbox.setup()
    artifacts, _ = await _open_artifact_store("worker", dsn)
    if metrics_port is not None:
        _start_metrics_server("worker", metrics_host, metrics_port)

    stop = asyncio.Event()
    posix_sigint = _install_sigint_stop(stop)
    try:
        async with (
            open_agent_backends(
                agent, backend_config or BackendConfig(), log=_logger("worker")
            ) as backends,
            open_checkpointer(dsn) as saver,
            open_store(dsn) as store,
            _open_mcp("worker", mcp_configs) as mcp,
        ):
            runner = AgentRunner(
                agent,
                checkpointer=saver,
                store=store,
                mcp=mcp,
                models=backends.models,
                resources=backends.resources,
            )

            async def sink_factory(run_id: str) -> EventSink:
                return OutboxEventSink(
                    await PostgresEventLog.connect(dsn),
                    outbox,
                    queue,
                    outbox_settings or OutboxSettings(),
                )

            worker = Worker(
                runner,
                queue,
                agent_name=agent.name,
                claimant=claimant,
                checkpointer=saver,
                sink_factory=sink_factory,
                interrupt_store=interrupts,
                artifact_store=artifacts,
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
    except BACKEND_ERRORS as exc:
        print(f"worker: {exc}", file=sys.stderr)
        return 1
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
    resources: str | None = None,
    models: str | None = None,
) -> int:
    """CLI entry point for ``vdreamer worker`` (PRD §2.1)."""
    if metrics_port is not None:
        from vdreamer.gateway.settings import GatewayConfigError, validate_bind_host

        try:  # refuse a wildcard bind before any store is opened (A14, §5.1)
            validate_bind_host(metrics_host)
        except GatewayConfigError as exc:
            print(f"worker: {exc}", file=sys.stderr)
            return 1
    resolved = _resolve_cli_agent("worker", agent, registry, agents_dir)
    if resolved is None:
        return 1
    loaded = _load_config("worker", resources, models)
    if loaded is None:
        return 1
    mcp_configs, backend_config = loaded
    outbox_settings = _outbox_settings("worker")
    if outbox_settings is None:
        return 1
    try:
        dsn = dsn_from_env(PG_DSN_ENV_VAR)
    except PersistenceConfigError as exc:
        print(f"worker: {exc}", file=sys.stderr)
        return 1
    try:
        return asyncio.run(
            _worker_main(
                resolved,
                dsn,
                claimant or _default_claimant(),
                metrics_host,
                metrics_port,
                mcp_configs,
                backend_config,
                outbox_settings,
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


@asynccontextmanager
async def _open_dev_store(db: str) -> AsyncIterator[Any]:
    """The dev long-term memory store (§2.4): ``AsyncSqliteStore`` on the same file."""
    from langgraph.store.sqlite.aio import AsyncSqliteStore

    Path(db).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteStore.from_conn_string(db) as store:
        await store.setup()
        yield store


async def _dev_gateway(
    agent: BaseAgent,
    checkpointer: Any,
    state: SqliteState,
    host: str,
    port: int,
    tokens: tuple[str, ...],
    mcp: MCPRegistry | None = None,
    backends: AgentBackends | None = None,
    headless: HeadlessConfig | None = None,
    store: Any = None,
) -> int:
    """``vdreamer dev`` without ``--input``: gateway + worker loop in one process (D11).

    Runs, events, threads, interrupts, outbox, and artifact metadata live in
    ``state`` — the same SQLite file as the checkpoints (artifact bytes beside
    it) — so a restart resumes where the last process stopped (the testing
    phase runs on SQLite, not Postgres). Required MCP servers are ``/readyz``
    probes.
    """
    from vdreamer.gateway import GatewayBackend, GatewaySettings, create_app
    from vdreamer.gateway.settings import strict_scopes_from_env

    headless = headless or HeadlessConfig()
    queue = state.queue
    interrupts = state.interrupts
    log = OutboxEventSink(state.events, state.outbox, queue, headless.outbox)
    runner = AgentRunner(
        agent,
        checkpointer=checkpointer,
        store=store,
        mcp=mcp,
        models=backends.models if backends is not None else None,
        resources=backends.resources if backends is not None else None,
    )

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
        artifact_store=state.artifacts,
    )
    backend = GatewayBackend(
        agents={agent.name: agent},
        queue=queue,
        threads=state.threads,
        event_reader=state.events,
        event_writer=log,
        hub=state.events.hub,
        interrupts=interrupts,
        artifacts=state.artifacts,
        checkpointer=checkpointer,
        probes=mcp.probes() if mcp is not None else {},
    )

    # §4.4: one process, so the sweeper is always the leader.
    sweeper = InterruptSweeper(interrupts, queue, log)
    dispatcher = build_dispatcher(state.outbox, headless.outbox, headless.callback_allowlist)

    @asynccontextmanager
    async def lifespan() -> AsyncIterator[None]:
        stop = asyncio.Event()
        loop_task = asyncio.create_task(worker.run(stop), name="vdreamer-dev-worker")
        sweep_task = asyncio.create_task(sweeper.run(stop), name="vdreamer-dev-sweeper")
        outbox_task = asyncio.create_task(dispatcher.run(stop), name="vdreamer-dev-outbox")
        try:
            yield
        finally:
            stop.set()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.gather(loop_task, sweep_task, outbox_task), 5.0)

    settings = GatewaySettings(
        tokens=tokens,
        strict_scopes=strict_scopes_from_env(),
        service_tokens=headless.service_tokens,
        callback_allowlist=headless.callback_allowlist,
    )
    app = create_app(backend, settings, lifespan=lifespan)
    print(f"vdreamer dev: trigger/outbox: {headless.summary()}")
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
    mcp_configs: Sequence[MCPServerConfig] = (),
    backend_config: BackendConfig | None = None,
    headless: HeadlessConfig | None = None,
) -> int:
    """The dev process body: the gateway + worker until Ctrl-C, or one smoke run."""
    sinks: dict[str, InMemoryEventSink] = {}

    async def sink_factory(run_id: str) -> EventSink:
        sink = InMemoryEventSink()
        sinks[run_id] = sink
        return sink

    async with (
        open_agent_backends(
            agent, backend_config or BackendConfig(), log=_logger("dev")
        ) as backends,
        _open_dev_checkpointer(db) as checkpointer,
        _open_dev_store(db) as store,
        _open_mcp("dev", mcp_configs) as mcp,
    ):
        if input_json is None:
            print(f"vdreamer dev: agent {agent.name!r} (mode {agent.mode.value})")
            print(
                "vdreamer dev: runs/events/threads/interrupts/artifacts/checkpoints="
                f"sqlite:{db}"
            )
            async with open_sqlite_state(db) as state:
                expired = await asyncio.to_thread(state.artifacts.objects.sweep)
                root = state.artifacts.objects.root
                print(
                    f"vdreamer dev: objects (artifacts/, inputs/, extracted/) under {root}"
                    + (f" — removed {expired} past retention" if expired else "")
                )
                return await _dev_gateway(
                    agent, checkpointer, state, host, port, tokens, mcp, backends, headless,
                    store,
                )
        runner = AgentRunner(
            agent,
            checkpointer=checkpointer,
            store=store,
            mcp=mcp,
            models=backends.models,
            resources=backends.resources,
        )
        queue = InMemoryRunQueue()
        worker = Worker(
            runner,
            queue,
            agent_name=agent.name,
            claimant="dev",
            checkpointer=checkpointer,
            sink_factory=sink_factory,
            poll_interval_s=0.05,
            artifact_store=InMemoryArtifactStore(),
        )
        print(f"vdreamer dev: agent {agent.name!r} (mode {agent.mode.value})")
        print(f"vdreamer dev: queue=in-memory, events=in-memory, checkpoints=sqlite:{db}")

        # The smoke: enqueue one run from --input, drive it synchronously, report.
        try:
            payload: Any = json.loads(input_json)
        except json.JSONDecodeError as exc:
            print(f"dev: --input is not valid JSON: {exc}", file=sys.stderr)
            return 1
        problem = input_problem(agent, payload)
        if problem is not None:  # the gateway's 422 invalid_input, before anything runs
            print(f"dev: {problem}", file=sys.stderr)
            return 1
        if agent.input_schema is not None:
            payload = agent.input_schema.model_validate(payload).model_dump(mode="json")
        record = await queue.enqueue(
            agent.name, payload, thread_id=thread_id, run_id=None
        )
        print(f"vdreamer dev: queued {record.id} (thread {record.thread_id})")
        await worker.run_once()
        final = await queue.get(record.id)
        assert final is not None  # enqueue created it; nothing removes rows
        events = sinks[record.id].events
        _print_events(events)
        print(
            f"vdreamer dev: run {record.id} -> {final.status.value} "
            f"(attempt {final.attempt}, {len(events)} events)"
        )
        reason = _failure_reason(events)
        if final.status is not RunState.COMPLETED and reason:
            print(f"vdreamer dev: {reason}", file=sys.stderr)
        return 0 if final.status is RunState.COMPLETED else 1


def input_problem(agent: BaseAgent, payload: Any) -> str | None:
    """Why ``payload`` fails ``agent.input_schema`` (as the gateway's ``422``), or ``None``."""
    import pydantic

    schema = agent.input_schema
    if schema is None:
        return None
    try:
        schema.model_validate(payload)
    except pydantic.ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '(root)'}: {error['msg']}"
            for error in exc.errors(include_url=False)
        )
        return f"--input does not match the input_schema of agent {agent.name!r}: {errors}"
    return None


def _print_events(events: Sequence[Any]) -> None:
    """Render the run's events as ``vdreamer run`` would (plain, no color)."""
    from vdreamer.client import Printer, Style

    printer = Printer(sys.stdout, Style(False))
    for envelope in events:
        printer.event(envelope.model_dump(mode="json"))


def _failure_reason(events: Sequence[Any]) -> str | None:
    """The final ``run.status`` error, ``code: message``, if the run ended with one."""
    for envelope in reversed(events):
        if envelope.type is not EventType.RUN_STATUS:
            continue
        error = envelope.payload.get("error")
        if isinstance(error, dict):
            return f"{error.get('code')}: {error.get('message')}"
        return None
    return None


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
    resources: str | None = None,
    models: str | None = None,
) -> int:
    """CLI entry point for ``vdreamer dev`` (D11): one process, no infrastructure."""
    resolved = _resolve_cli_agent("dev", agent, registry, agents_dir)
    if resolved is None:
        return 1
    loaded = _load_config("dev", resources, models)
    if loaded is None:
        return 1
    mcp_configs, backend_config = loaded
    tokens: tuple[str, ...] = ()
    headless: HeadlessConfig | None = None
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
        headless = _headless_config("dev", tokens)
        if headless is None:
            return 1
    try:
        return asyncio.run(
            _dev_main(
                resolved,
                db,
                input_json,
                thread_id,
                host,
                port,
                tokens,
                mcp_configs,
                backend_config,
                headless,
            )
        )
    except MissingPostgresDependencyError as exc:  # pragma: no cover - dev never touches psycopg
        print(f"dev: {exc}", file=sys.stderr)
        return 1
    except BACKEND_ERRORS as exc:
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
        problem = contract_error(agent)
        if problem is not None:
            raise ServeAgentsError(problem)
        existing = agents.get(agent.name)
        if existing is not None and existing is not agent:
            raise ServeAgentsError(f"two agents are named {agent.name!r}")
        agents[agent.name] = agent
    return agents


async def _serve_main(
    agents: dict[str, BaseAgent],
    dsn: str,
    tokens: tuple[str, ...],
    host: str,
    port: int,
    headless: HeadlessConfig | None = None,
) -> int:
    """The gateway process body: Postgres stores, ``LISTEN`` loop, uvicorn (§2.1)."""
    from vdreamer.gateway import GatewayBackend, GatewaySettings, create_app
    from vdreamer.gateway.settings import strict_scopes_from_env

    headless = headless or HeadlessConfig()
    queue = PostgresRunQueue(dsn)
    threads = PostgresThreadStore(dsn)
    interrupts = PostgresInterruptStore(dsn)
    outbox = PostgresOutboxStore(dsn)
    # Boot-time guards, like the worker's; `vdreamer migrate` is the explicit path.
    await queue.setup()
    await threads.setup()
    await interrupts.setup()
    await outbox.setup()
    log = await PostgresEventLog.connect(dsn)
    try:
        await log.setup()
    finally:
        await log.close()
    artifacts, artifacts_probe = await _open_artifact_store("serve", dsn)
    probes: dict[str, Any] = {}
    if artifacts_probe is not None:
        probes["minio"] = artifacts_probe

    hub = EventHub()
    listener = PostgresEventListener(dsn, hub)
    store = PostgresEventStore(dsn)
    writer = OutboxEventSink(store, outbox, queue, headless.outbox)

    async with open_checkpointer(dsn) as saver:
        backend = GatewayBackend(
            agents=agents,
            queue=queue,
            threads=threads,
            event_reader=store,
            event_writer=writer,
            hub=hub,
            interrupts=interrupts,
            artifacts=artifacts,
            checkpointer=saver,
            probes={"postgres": queue.probe, "event_listener": listener.probe, **probes},
        )

        # §4.4: every replica runs a sweeper; the advisory lock elects the one that sweeps.
        sweeper = InterruptSweeper(
            interrupts, queue, writer, leader=PostgresAdvisoryLeader(dsn)
        )
        # Leases make any number of replicas safe: each delivers what it claims.
        dispatcher = build_dispatcher(outbox, headless.outbox, headless.callback_allowlist)

        @asynccontextmanager
        async def lifespan() -> AsyncIterator[None]:
            listener.start()
            stop = asyncio.Event()
            sweep_task = asyncio.create_task(sweeper.run(stop), name="vdreamer-sweeper")
            outbox_task = asyncio.create_task(dispatcher.run(stop), name="vdreamer-outbox")
            try:
                yield
            finally:
                stop.set()
                with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                    await asyncio.wait_for(asyncio.gather(sweep_task, outbox_task), 5.0)
                await listener.stop()

        settings = GatewaySettings(
            tokens=tokens,
            strict_scopes=strict_scopes_from_env(),
            service_tokens=headless.service_tokens,
            callback_allowlist=headless.callback_allowlist,
        )
        app = create_app(backend, settings, lifespan=lifespan)
        names = ", ".join(sorted(agents))
        print(f"vdreamer serve: agents {names}")
        print(f"vdreamer serve: trigger/outbox: {headless.summary()}")
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
    headless = _headless_config("serve", tokens)
    if headless is None:
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
        return asyncio.run(_serve_main(served, dsn, tokens, host, port, headless))
    except (MissingPostgresDependencyError, MinioConfigError) as exc:
        print(f"serve: {exc}", file=sys.stderr)
        return 1
