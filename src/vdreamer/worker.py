"""The worker loop (PRD §2.1, §3.2; Milestone 3 WP2).

One :class:`Worker` serves one agent: it claims ``queued`` runs from a
:class:`~vdreamer.queue.RunQueue`, drives each through the agent's
:class:`~vdreamer.runtime.AgentRunner`, and moves the row through the §3.2
lifecycle. Postgres is the only gateway↔worker transport (A7) — the worker
meets the gateway on the queue rows and the ``run_events`` log, never over a
direct API.

The three §3.2 control paths, and where each hooks in:

- **Cancel** — the gateway (M4) calls :meth:`Worker.request_cancel`; a locally
  active run gets :meth:`RunControl.request_drain`, and the run drains
  cooperatively at the next superstep boundary: the drain check lives in the
  per-run sink wrapper's ``write`` — the only hook between driver envelopes
  (``runtime.AgentRunner._execute_streaming`` writes every envelope there in
  arrival order) — and raises :class:`RunDrained` out of the streaming loop.
  The stream aborts without a terminal envelope (a consumer-side raise closes
  the driver generator at its yield; the driver's ``except Exception`` never
  sees it), the last checkpoint stays resumable, and the run is finished
  ``cancelled`` (PRD §3.2: "cancel calls ``RunControl.request_drain()`` and
  the run → ``cancelled``").
- **SIGTERM** — :meth:`Worker.run` installs a POSIX main-thread handler (on
  Windows ``SIGTERM`` is never delivered, so the install is skipped there and
  :meth:`Worker.drain_all` is the portable entry point): every active run is
  drained and **requeued without consuming an attempt** via
  :meth:`~vdreamer.queue.RunQueue.release`, which gives back the claim's
  ``attempt`` increment.
- **Recovery** — an expired lease is the queue's business
  (:meth:`~vdreamer.queue.RunQueue.requeue_expired`); the worker runs it once
  at startup (``recover_on_start``) and its heartbeat task drains a run whose
  lease the queue no longer honors (``lease_lost``), marking nothing — the
  row belongs to whoever holds it now. Side effects re-executed on resume
  deduplicate through ``ctx.idempotency_key`` (A6).

Interrupts (:meth:`~vdreamer.context.require_approval`, A44) need no worker
machinery of their own: the stream driver detects the pending interrupts and
ends the run on ``run.status`` ``awaiting_approval`` (``streaming._RunMapper.
finish``; the v2 driver reads them from ``aget_state(config).interrupts``) —
the worker reads that terminal status back from the sink and calls
``queue.suspend``: approvals free the worker (§3.2), and a resolution
enqueues a resume (the gateway plants the ``Command(resume=...)`` input on
the row, M4/M5).

Milestone 4 (the gateway contract) adds four things the worker owns:

- **Input selection** (:meth:`Worker._graph_input`): the gateway's interrupt
  resolution plants a resume map ``{interrupt_id: value}`` on the row and
  requeues it; the worker turns the entries the checkpoint still awaits into
  ``Command(resume=...)``. A row whose thread's latest checkpoint was written by
  *this* run (``RUN_ID_CONFIGURABLE_KEY`` in the checkpoint metadata) with
  nodes still to run — a recovered or released run — resumes from that
  checkpoint with ``None``; anything else drives the row's original input.
- **Gateway cancel** — ``POST /runs/{id}/cancel`` marks the row (the gateway
  is another process); the heartbeat notices, reads the row back, and drains
  the run with reason ``"cancelled"``: the worker still authors the final
  ``cancelled`` envelope, but leaves the already-final row alone.
- **Interrupt records** — every ``interrupt.requested`` envelope also becomes
  an :class:`~vdreamer.interrupts.InterruptRecord` (idempotent) for the
  gateway's inbox, when an ``interrupt_store`` is wired.
- **Identity and metrics** — the row's initiator (``user_id``/``roles``) is
  bound into the run's ``VDreamerContext``; each settled attempt is observed in
  ``vdreamer_run_duration_seconds``.

Node failures map through the driver's failure path: a node raising
:class:`NodeTimeoutError` / :class:`NodeError` (PRD §2.4 — developer-owned
``add_node(timeout=…, error_handler=…)``) reaches the worker as any other
exception, with the driver's ``run.status`` ``failed`` envelope —
``error {code, message}`` — already in the sink. Only when the failure
happened outside the driver (driver selection, a lifecycle hook) does the
worker write the failed envelope itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from vdreamer import metrics
from vdreamer.agents import runs_headless
from vdreamer.artifacts import ArtifactStore, bind_artifact_store
from vdreamer.context import VDreamerContext
from vdreamer.eventlog import EventSink, InMemoryEventSink
from vdreamer.events import ErrorInfo, EventEnvelope, EventType, RunStatusPayload
from vdreamer.interrupts import InterruptStore, apply_interrupt_policy, record_from_event
from vdreamer.queue import (
    LEASE_INTERVAL_S,
    InvalidTransition,
    RunQueue,
    RunRecord,
    RunState,
)
from vdreamer.runtime import RUN_ID_CONFIGURABLE_KEY, AgentRunner, is_reserved_configurable_key
from vdreamer.tools import InterruptPolicy

_log = logging.getLogger(__name__)


class NodeError(Exception):
    """A graph node failed (PRD §2.4: ``NodeError`` maps to ``run.status.error``).

    Raised by agent code (a node body or its ``add_node(error_handler=…)``,
    both developer-owned). ``node`` names the failing node when known; the
    original failure rides ``__cause__`` (``raise NodeError(...) from exc``).
    The worker ends the run ``failed`` with the driver's ``run.status`` error
    block — ``str(self)`` is the event's ``message``.
    """

    def __init__(self, message: str, *, node: str | None = None) -> None:
        super().__init__(message)
        self.node = node


class NodeTimeoutError(Exception):
    """A node exceeded its ``add_node(timeout=…)`` budget (PRD §2.4).

    Like :class:`NodeError` it maps to ``run.status.error``; it additionally
    carries the ``timeout_s`` budget that was exceeded.
    """

    def __init__(self, message: str, *, node: str | None = None, timeout_s: float) -> None:
        super().__init__(message)
        self.node = node
        self.timeout_s = timeout_s


class RunDrained(Exception):
    """Raised out of the streaming loop when a drain was requested (WP2).

    Carries the :meth:`RunControl.request_drain` ``reason`` that won —
    ``"cancel"`` finishes the run ``cancelled``; ``"sigterm"`` requeues it
    without consuming an attempt; anything else (``"lease_lost"``) leaves the
    row to whoever holds it now.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"run drained: {reason}")
        self.reason = reason


class RunControl:
    """Per-active-run drain control (PRD §3.2: "cancel calls
    ``RunControl.request_drain()``").

    One instance per locally active run, held by the :class:`Worker` in
    ``Worker._active``. ``request_drain`` is idempotent — the first reason
    wins (a ``cancel`` arriving after a ``lease_lost`` cannot resurrect a run
    the queue has already taken away, and vice versa). The drain itself is
    cooperative: the worker's sink wrapper raises :class:`RunDrained` at the
    next envelope write, which is the next superstep boundary (§2.5:
    "cooperative cancel at next superstep"), leaving the last checkpoint
    resumable.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None

    def request_drain(self, reason: str = "cancel") -> None:
        """Request a cooperative drain of the run; idempotent, first reason wins."""
        if self._reason is None:
            self._reason = reason
        self._event.set()

    @property
    def drain_requested(self) -> bool:
        """Whether a drain has been requested."""
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        """The winning drain reason, or ``None`` before the first request."""
        return self._reason

    async def wait(self) -> None:
        """Block until a drain is requested."""
        await self._event.wait()

    async def wait_for_drain_requested(self) -> str:
        """Block until a drain is requested and return the winning reason."""
        await self._event.wait()
        return self._reason if self._reason is not None else "cancel"


class _RunSink(EventSink):
    """Per-run sink wrapper: the drain check, terminal tracking, seq handoff.

    The worker's only hook between driver envelopes is the sink write —
    ``runtime.AgentRunner._execute_streaming`` writes each envelope here in
    arrival order — so ``write`` checks the run's :class:`RunControl` first
    and raises :class:`RunDrained` *before* persisting, cancelling the graph
    execution at the next superstep boundary. It also remembers the last
    ``run.status`` payload (the terminal status is the driver's — the runner
    returns normally for both ``completed`` and ``awaiting_approval``) and the
    highest ``seq``, so the worker can tell the two apart and continue the
    run's ``seq`` when it emits an envelope of its own.
    """

    def __init__(
        self,
        inner: EventSink,
        control: RunControl,
        on_interrupt: Callable[[EventEnvelope], Awaitable[None]] | None = None,
        interrupt_policies: Mapping[str, InterruptPolicy] | None = None,
    ) -> None:
        self._inner = inner
        self._control = control
        self._on_interrupt = on_interrupt
        self._interrupt_policies = interrupt_policies or {}
        self.last_seq = 0
        self.last_status: str | None = None

    def next_seq(self) -> int:
        """The next free ``seq`` for this run (continues the driver's count)."""
        self.last_seq += 1
        return self.last_seq

    async def write(self, envelope: EventEnvelope) -> None:
        if self._control.drain_requested:
            raise RunDrained(self._control.reason or "cancel")
        self.last_seq = max(self.last_seq, envelope.seq)
        if envelope.type is EventType.RUN_STATUS:
            self.last_status = str(envelope.payload.get("status"))
        elif envelope.type is EventType.INTERRUPT_REQUESTED:
            if self._interrupt_policies:
                # §4.5: the agent's interrupt_policies attach approver_role /
                # scope / TTL to non-require_approval sources — before the
                # record and the event, so both carry the same policy.
                envelope = envelope.model_copy(
                    update={
                        "payload": apply_interrupt_policy(
                            envelope.payload, self._interrupt_policies
                        )
                    }
                )
            if self._on_interrupt is not None:
                # The record first, then the event: a client reacting to
                # ``interrupt.requested`` must find the record in the inbox (§4.2).
                await self._on_interrupt(envelope)
        await self._inner.write(envelope)

    async def write_terminal(self, envelope: EventEnvelope) -> None:
        """Persist a worker-authored terminal envelope, past the drain check.

        The run is already ending when the worker authors an envelope (a
        cancel drain's final status, or a failure the driver never saw) — the
        envelope must land even though a drain is pending.
        """
        self.last_seq = max(self.last_seq, envelope.seq)
        if envelope.type is EventType.RUN_STATUS:
            self.last_status = str(envelope.payload.get("status"))
        await self._inner.write(envelope)

    async def close(self) -> None:
        await self._inner.close()


def _error_code(error: BaseException) -> str:
    """The stable ``run.status.error`` code for a worker-authored envelope."""
    if isinstance(error, NodeTimeoutError):
        return "node_timeout"
    return "node_error"


class Worker:
    """The worker loop: claim queued runs, drive them, settle the row (§3.2).

    One worker serves one agent (one :class:`~vdreamer.runtime.AgentRunner`);
    the CLI's ``vdreamer worker`` entrypoint (WP3) runs one process per agent,
    picked with ``--agent``. Claims go through :meth:`~vdreamer.queue.RunQueue.claim` — which
    stamps the lease and increments ``attempt`` — and every settled row is
    either terminal, ``awaiting_approval`` (no lease — approvals free the
    worker), or back in the queue (a SIGTERM drain gives the attempt back via
    ``release``).
    """

    def __init__(
        self,
        runner: AgentRunner,
        queue: RunQueue,
        *,
        agent_name: str,
        claimant: str | None = None,
        checkpointer: Any = None,
        sink_factory: Callable[[str], Awaitable[EventSink]] | None = None,
        lease_interval_s: float = LEASE_INTERVAL_S,
        poll_interval_s: float = 0.1,
        recover_on_start: bool = True,
        interrupt_store: InterruptStore | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        """Wire the loop.

        Args:
            runner: the agent's runner (already carrying the checkpointer the
                graphs compile with).
            queue: the run queue to claim from and settle into.
            agent_name: the agent this worker serves — the ``agent`` field of
                worker-authored envelopes and the run context's ``agent_name``.
            claimant: the ``claimed_by`` stamp; a fresh id when omitted.
            checkpointer: the worker-opened checkpointer (WP3 wires
                ``persistence.open_checkpointer`` here and hands it to the
                runners it builds; the runner already carries its own today).
            sink_factory: builds the per-run event sink from the ``run_id``;
                an in-memory discard sink when omitted (the wrapper still
                observes the envelopes — the terminal status is read from
                them, not from a return value).
            lease_interval_s: the claim's lease window (§3.2); the heartbeat
                fires at half of it.
            poll_interval_s: the idle sleep between claims.
            recover_on_start: run the §3.2 recovery sweep
                (``queue.requeue_expired``) once on the worker's first
                iteration. The constructor is synchronous — the sweep is an
                await — so ``recover_on_start`` schedules it for the first
                :meth:`run_once` / :meth:`run`; call :meth:`recover` directly
                to run (or re-run) it explicitly.
            interrupt_store: where ``interrupt.requested`` envelopes become
                §4.2 records for the gateway's inbox; ``None`` keeps none.
            artifact_store: bound around every run, so the artifacts a run
                writes (governed-tool overflow, §2.7 step 5) are kept for the
                gateway's artifact routes; ``None`` keeps only the events.
        """
        self._runner = runner
        self._queue = queue
        self._agent_name = agent_name
        self._claimant = claimant if claimant is not None else f"worker-{uuid.uuid4().hex[:12]}"
        self._checkpointer = checkpointer
        self._sink_factory = sink_factory
        self._lease_interval_s = lease_interval_s
        self._poll_interval_s = poll_interval_s
        self._recover_on_start = recover_on_start
        self._interrupt_store = interrupt_store
        self._artifact_store = artifact_store
        self._recovered = False
        #: The locally active runs' drain controls (``run_id → RunControl``);
        #: popped when the run settles. ``request_cancel`` and the SIGTERM
        #: path reach the live run through this.
        self._active: dict[str, RunControl] = {}
        self._sigterm_received = False
        self._sigterm_installed = False
        self._prev_sigterm_handler: Any = None

    # -- loop ---------------------------------------------------------------

    async def run_once(self) -> bool:
        """Claim and drive at most one run.

        Returns ``True`` when a run was claimed and handled (any outcome —
        completed, suspended, cancelled, released, failed — counts; the loop
        must survive every failure), ``False`` when the queue had nothing to
        claim (after sleeping ``poll_interval_s``).
        """
        if self._recover_on_start and not self._recovered:
            await self.recover()
        record = await self._queue.claim(self._claimant, agent=self._agent_name)
        if record is None:
            await asyncio.sleep(self._poll_interval_s)
            return False
        return await self._drive(record)

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Claim-and-drive until ``stop`` is set or a SIGTERM drain landed.

        ``stop`` stops the loop without touching the in-flight run — the
        current :meth:`run_once` completes it normally first. A SIGTERM (see
        :meth:`_install_sigterm`) instead drains every active run with reason
        ``"sigterm"`` — the current :meth:`run_once` releases the run back to
        the queue, attempt unconsumed — and the loop then exits. A run blocked
        inside a node delays both (node timeouts are the developer-owned
        answer, PRD §2.4).
        """
        self._install_sigterm()
        try:
            while not self._sigterm_received and (stop is None or not stop.is_set()):
                await self.run_once()
        finally:
            self._uninstall_sigterm()

    # -- cancel / shutdown ---------------------------------------------------

    async def request_cancel(self, run_id: str) -> bool:
        """The worker half of ``POST /runs/{id}/cancel`` (§2.5; M4 wires the route).

        A locally active run is drained cooperatively: ``True`` and the
        :meth:`run_once` driving it finishes the row ``cancelled``. A run not
        active here belongs to another worker (or is still queued) — fall back
        to :meth:`~vdreamer.queue.RunQueue.cancel`, which marks the row; the
        owning worker's next heartbeat then fails, reads the row back, and
        drains with ``"cancelled"`` (see :meth:`_heartbeat_loop`). ``False``
        when the run is neither active here nor cancellable (terminal or
        unknown).
        """
        control = self._active.get(run_id)
        if control is not None:
            control.request_drain("cancel")
            return True
        return await self._queue.cancel(run_id) is not None

    def drain_all(self, reason: str = "sigterm") -> None:
        """Request a drain on every locally active run.

        The SIGTERM path's body (the signal handler calls this via
        ``loop.call_soon_threadsafe``); a ``"sigterm"`` reason also marks the
        worker shutting down, so a running :meth:`run` exits after the
        in-flight drain completes.
        """
        for control in list(self._active.values()):
            control.request_drain(reason)
        if reason == "sigterm":
            self._sigterm_received = True

    # -- recovery ------------------------------------------------------------

    async def recover(self) -> int:
        """Run the §3.2 recovery sweep once: every expired lease requeued.

        Requeued runs resume from their checkpoint on the next claim (the
        in-flight node re-executes; side effects deduplicate through
        ``ctx.idempotency_key``, A6). Returns how many runs were handled.
        """
        handled = await self._queue.requeue_expired()
        self._recovered = True
        if handled:
            _log.info("recovery sweep requeued %d run(s)", handled)
        return handled

    # -- internals -----------------------------------------------------------

    async def _drive(self, record: RunRecord) -> bool:
        """One claimed run, end to end: heartbeat, stream, settle the row."""
        control = RunControl()
        self._active[record.id] = control
        started = time.monotonic()
        on_interrupt: Callable[[EventEnvelope], Awaitable[None]] | None = None
        if self._interrupt_store is not None:

            async def on_interrupt(envelope: EventEnvelope) -> None:
                await self._record_interrupt(record, envelope)

        policies = getattr(self._runner.agent, "interrupt_policies", None)
        sink = _RunSink(
            await self._open_sink(record.id),
            control,
            on_interrupt=on_interrupt,
            interrupt_policies=policies if isinstance(policies, Mapping) else None,
        )
        heartbeat = asyncio.create_task(self._heartbeat_loop(record.id, control))
        try:
            if control.drain_requested:  # a drain landed before the run started
                raise RunDrained(control.reason or "cancel")
            config = self._config_for(record)
            with bind_artifact_store(self._artifact_store):
                await self._runner.arun(
                    await self._graph_input(record, config),
                    config=config,
                    context=self._context_for(record),
                    sink=sink,
                )
        except RunDrained as drained:
            await self._finish_drained(record, sink, drained.reason)
        except Exception as error:  # NodeTimeoutError/NodeError ride this too
            await self._fail(record, sink, error)
        else:
            await self._finish_terminal(record, sink)
        finally:
            self._active.pop(record.id, None)
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            await sink.close()
            metrics.RUN_DURATION.labels(
                agent=self._agent_name, status=sink.last_status or "unknown"
            ).observe(time.monotonic() - started)
        return True

    def _config_for(self, record: RunRecord) -> RunnableConfig:
        """The run's LangGraph config: the client's ``config`` plus the addressing.

        The client's ``configurable`` entries (validated by the gateway, §2.5)
        are kept except the reserved keys; ``thread_id`` keys the checkpointer
        and :data:`~vdreamer.runtime.RUN_ID_CONFIGURABLE_KEY` stamps every
        checkpoint with this run (see :meth:`_graph_input`).
        """
        client = dict(record.config or {})
        configurable = {
            str(key): value
            for key, value in dict(client.get("configurable") or {}).items()
            if not is_reserved_configurable_key(str(key))
        }
        configurable["thread_id"] = record.thread_id or record.id
        configurable[RUN_ID_CONFIGURABLE_KEY] = record.id
        config: RunnableConfig = {"configurable": configurable}
        recursion_limit = client.get("recursion_limit")
        if isinstance(recursion_limit, int) and not isinstance(recursion_limit, bool):
            config["recursion_limit"] = recursion_limit
        return config

    async def _graph_input(self, record: RunRecord, config: RunnableConfig) -> Any:
        """What this attempt feeds the graph (§3.2, §4.3).

        In order:

        1. the row's resume map, narrowed to the interrupts the checkpoint
           still awaits → ``Command(resume={interrupt_id: value})`` (an
           approval resume — "resolution enqueues a resume", §3.2);
        2. a checkpoint written by this very run with nodes left to run → the
           ``None`` input, i.e. resume from the checkpoint (a recovered or
           released attempt: the in-flight node re-executes, A6; a pending
           interrupt with no resolution yet simply re-suspends);
        3. otherwise the row's original input (a fresh run).

        Without a checkpointer there is nothing to resume, and a state read
        that fails falls back to the input — the M3 behaviour.
        """
        if not self._runner.has_checkpointer:
            return record.input
        try:
            state = await self._runner.aget_state(config)
        except Exception as error:  # pragma: no cover - defensive: unreadable state
            _log.warning("run %s: state read failed (%s); driving the input", record.id, error)
            return record.input
        pending = {str(getattr(item, "id", "")) for item in (state.interrupts or ())}
        if isinstance(record.resume, Mapping):
            matched = {key: value for key, value in record.resume.items() if key in pending}
            if matched:
                return Command(resume=matched)
        metadata = state.metadata or {}
        if state.next and metadata.get(RUN_ID_CONFIGURABLE_KEY) == record.id:
            return None
        return record.input

    async def _record_interrupt(self, record: RunRecord, envelope: EventEnvelope) -> None:
        """Persist the §4.2 record for one ``interrupt.requested`` envelope."""
        assert self._interrupt_store is not None  # wired only with a store
        await self._interrupt_store.create(
            record_from_event(envelope, initiator_id=record.user_id)
        )

    async def _heartbeat_loop(self, run_id: str, control: RunControl) -> None:
        """Extend the lease every half window; a lost lease drains the run.

        ``heartbeat`` returning ``False`` (or raising — a queue outage must
        not run blind) means the row is gone or no longer ours. A row the
        gateway cancelled (``POST /runs/{id}/cancel``, M4) drains with
        ``"cancelled"`` — the worker still owes the run's final event; any
        other loss (the recovery sweep took it) drains with ``lease_lost`` and
        the worker marks nothing.
        """
        interval = self._lease_interval_s / 2
        while True:
            await asyncio.sleep(interval)
            try:
                alive = await self._queue.heartbeat(run_id)
            except Exception as error:  # any queue failure ends the lease
                _log.warning("run %s: heartbeat failed (%s) — draining", run_id, error)
                alive = False
            if not alive:
                control.request_drain(await self._lost_lease_reason(run_id))
                return

    async def _lost_lease_reason(self, run_id: str) -> str:
        """``"cancelled"`` when the row was cancelled under us, else ``"lease_lost"``."""
        try:
            current = await self._queue.get(run_id)
        except Exception:  # the queue is unreachable: nothing to learn
            return "lease_lost"
        if current is not None and current.status is RunState.CANCELLED:
            return "cancelled"
        return "lease_lost"

    async def _finish_terminal(self, record: RunRecord, sink: _RunSink) -> None:
        """Settle a run whose stream ended normally (the driver's terminal status)."""
        status = sink.last_status
        if status == "awaiting_approval":
            # Approvals free the worker (§3.2): no lease while the interrupt
            # waits; a resolution requeues the row (the gateway plants the
            # Command(resume=…) input, M4/M5).
            try:
                await self._queue.suspend(record.id)
            except InvalidTransition:  # raced an external cancel/finish
                _log.warning("run %s: suspend raced an external transition", record.id)
        elif status == "completed":
            await self._finish_row(record, RunState.COMPLETED)
        else:
            # Defensive: a normal return without the driver's terminal status
            # (only reachable for a sink that dropped envelopes) — fail the row
            # so it cannot linger ``running`` past its lease.
            await self._fail(record, sink, RuntimeError("stream ended without a terminal status"))

    async def _finish_drained(self, record: RunRecord, sink: _RunSink, reason: str) -> None:
        """Settle a run the cooperative drain took out of its streaming loop."""
        if reason == "cancel":
            # §3.2: the run moves to cancelled, its last checkpoint resumable.
            # The stream aborted without a terminal envelope, so the worker
            # authors the final ``run.status`` the SSE log should end on.
            await sink.write_terminal(self._status_envelope(record, sink, "cancelled"))
            await self._finish_row(record, RunState.CANCELLED)
        elif reason == "cancelled":
            # The gateway cancelled the row (another process, M4): the row is
            # already final, but the event log still needs its last word.
            await sink.write_terminal(self._status_envelope(record, sink, "cancelled"))
        elif reason == "sigterm":
            # §3.2: drained and requeued WITHOUT consuming an attempt —
            # release() gives back the claim's attempt increment. Not
            # terminal, so no final envelope: the next attempt's driver
            # emits a fresh ``running`` status.
            try:
                await self._queue.release(record.id)
            except InvalidTransition:  # raced an external cancel/finish
                _log.warning("run %s: release raced an external transition", record.id)
        else:
            # ``lease_lost`` (and any other reason): the row may already be
            # cancelled, expired, or re-claimed elsewhere — mark nothing.
            _log.info("run %s: drained (%s); row left to its holder", record.id, reason)

    async def _fail(self, record: RunRecord, sink: _RunSink, error: BaseException) -> None:
        """Settle a failed run, emitting the failed envelope when the driver did not.

        The driver's failure path (``streaming._RunMapper.finish``) emits the
        ``run.status`` ``failed`` envelope — ``error {code, message}`` —
        before re-raising, so a node's :class:`NodeTimeoutError`/
        :class:`NodeError` (§2.4) is already in the log with the generic
        ``node_error`` code. When the failure happened outside the driver
        (driver selection, a lifecycle hook) no envelope exists and the
        worker authors it here, with the stable per-type code.
        """
        if sink.last_status != "failed":
            await sink.write_terminal(
                self._status_envelope(
                    record,
                    sink,
                    "failed",
                    error=ErrorInfo(
                        code=_error_code(error),
                        message=str(error) or type(error).__name__,
                    ),
                )
            )
        await self._finish_row(record, RunState.FAILED)

    async def _finish_row(self, record: RunRecord, status: RunState) -> None:
        """Move the row to a terminal status, tolerating an external race."""
        try:
            await self._queue.finish(record.id, status)
        except InvalidTransition:
            # Raced an external terminal transition (e.g. queue.cancel while a
            # drain was in flight): the row is already final, which is the
            # same outcome.
            _log.warning("run %s: finish(%s) raced an external transition", record.id, status)

    def _status_envelope(
        self,
        record: RunRecord,
        sink: _RunSink,
        status: str,
        error: ErrorInfo | None = None,
    ) -> EventEnvelope:
        """A worker-authored ``run.status`` envelope, continuing the run's ``seq``."""
        return EventEnvelope(
            type=EventType.RUN_STATUS,
            run_id=record.id,
            thread_id=record.thread_id or record.id,
            agent=self._agent_name,
            seq=sink.next_seq(),
            payload=RunStatusPayload(
                status=status, attempt=record.attempt, error=error
            ).event_payload(),
        )

    def _context_for(self, record: RunRecord) -> VDreamerContext:
        """The run identity bound for the implicit helpers (PRD §2.2).

        ``idempotency_key`` is the ``run_id`` itself — stable across recovery
        attempts (PRD §3.2/A6: an expired lease requeues the run and the
        in-flight node re-executes from its checkpoint, so the side effects
        that re-run must dedupe on a key that does not vary with ``attempt``;
        an attempt-varying key could never make a re-executed effect
        exactly-once). The run IS the effect's identity (§3.1: one run = one
        execution = one user turn).

        Within one run, call-site disambiguation — a graph calling the same
        side-effecting tool twice — is the tool author's concern: they may
        namespace the key with ``runtime.tool_call_id``. Cross-attempt
        stability is the core's guarantee. ``attempt`` remains a separate
        :class:`~vdreamer.context.VDreamerContext` field; only the key is
        pinned to the run.
        """
        return VDreamerContext(
            run_id=record.id,
            thread_id=record.thread_id or record.id,
            agent_name=self._agent_name,
            # The initiator (X-User-Id / X-User-Roles, D13) — §2.7: tools see
            # the run's user and roles through ``runtime.context``.
            user_id=record.user_id,
            roles=tuple(record.roles),
            attempt=record.attempt,
            # A6: stable per run — one run = one execution = one user turn.
            idempotency_key=record.id,
            headless=runs_headless(self._runner.agent, record.headless),
        )

    async def _open_sink(self, run_id: str) -> EventSink:
        """The run's event sink — the factory's, or an in-memory discard one."""
        if self._sink_factory is None:
            return InMemoryEventSink()
        return await self._sink_factory(run_id)

    # -- SIGTERM (POSIX main thread only) ------------------------------------

    def _install_sigterm(self) -> None:
        """SIGTERM → drain every active run, then stop (PRD §3.2).

        POSIX main-thread only: ``signal.signal`` raises off the main thread,
        and on Windows ``SIGTERM`` exists but the OS never delivers it — so
        the install is skipped there (documented; the drained path stays
        reachable portably via :meth:`drain_all`). Any install failure is
        swallowed: a worker without SIGTERM handling still stops on ``stop``.
        """
        if sys.platform == "win32" or threading.current_thread() is not threading.main_thread():
            return
        loop = asyncio.get_running_loop()

        def _handler(signum: int, frame: Any) -> None:
            loop.call_soon_threadsafe(self.drain_all, "sigterm")

        try:
            self._prev_sigterm_handler = signal.signal(signal.SIGTERM, _handler)
        except (ValueError, OSError):  # pragma: no cover - defensive: off-main-thread race
            self._prev_sigterm_handler = None
        else:
            self._sigterm_installed = True

    def _uninstall_sigterm(self) -> None:
        """Restore the handler :meth:`_install_sigterm` replaced, if it did."""
        if not self._sigterm_installed:
            return
        self._sigterm_installed = False
        with contextlib.suppress(ValueError, OSError, TypeError):
            signal.signal(signal.SIGTERM, self._prev_sigterm_handler)
        self._prev_sigterm_handler = None
