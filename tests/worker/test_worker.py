"""The worker loop (Milestone 3 WP2): claims, drains, interrupts, recovery.

Everything runs in-process on ``InMemoryRunQueue`` + ``InMemorySaver`` — no
Postgres, no sockets (the zero-egress doctor path must stay clean). The
mid-run control tests use a gated graph: the node blocks on a test event
after emitting a marker envelope, so the test observes the run mid-flight,
fires its control, and the *next* envelope write is where the drain check
deterministically fires.

PRD §3.2 coverage per test:
- claim → drive → complete (the happy path),
- approvals free the worker; a resume re-drives from the checkpoint,
- cancel drains to ``cancelled``, leaving a resumable checkpoint,
- SIGTERM drains and requeues without consuming an attempt (``release``),
- a lost lease drains and marks nothing,
- ``NodeTimeoutError``/``NodeError`` end the run ``failed`` with the
  ``run.status.error`` block (§2.4),
- startup recovery requeues expired leases.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from dataclasses import dataclass, replace
from typing import Any

import _kit
import pytest
from langgraph.config import get_stream_writer
from langgraph.types import Command

from vdreamer import (
    LEASE_INTERVAL_S,
    InMemoryEventSink,
    InMemoryRunQueue,
    NodeError,
    NodeTimeoutError,
    RunControl,
    RunState,
    Worker,
)
from vdreamer.context import VDreamerContext, current_context
from vdreamer.eventlog import EventEnvelope, EventSink
from vdreamer.runtime import AgentRunner

# ---------------------------------------------------------------------------
# Test agents
# ---------------------------------------------------------------------------


class GatedAgent(_kit.BaseTestAgent):
    """``first`` completes → ``second`` emits a marker, parks on ``gate``, ticks.

    The two events make mid-run controls deterministic. The test observes the
    ``second-started`` marker envelope, fires its control, then opens
    ``gate``: the node's *next* envelope write (the first ``ticking`` record)
    is where the worker's drain check fires — before that record is ever
    persisted. While ``release`` stays closed the node never returns, so the
    in-flight superstep cannot complete: the last checkpoint holds exactly
    ``first``'s writes. Opening both events (as the context-wiring test does)
    lets the node run straight through.
    """

    name = "gated"

    def __init__(self, gate: asyncio.Event, release: asyncio.Event) -> None:
        self._gate = gate
        self._release = release
        self.seen_context: VDreamerContext | None = None

    def build(self, ctx: Any) -> Any:
        gate, release = self._gate, self._release

        def first(state: _kit.ChatState) -> dict[str, Any]:
            self.seen_context = current_context()  # the worker's run identity
            get_stream_writer()({"type": "progress", "label": "first-done"})
            return {"flag": "first"}

        async def second(state: _kit.ChatState) -> dict[str, Any]:
            writer = get_stream_writer()
            writer({"type": "progress", "label": "second-started"})
            await gate.wait()
            while not release.is_set():
                writer({"type": "progress", "label": "ticking"})
                await asyncio.sleep(0.005)
            writer({"type": "progress", "label": "second-done"})
            return {"flag": "second"}

        graph = _kit.StateGraph(_kit.ChatState)
        graph.add_node("first", first)
        graph.add_node("second", second)
        graph.add_edge(_kit.START, "first")
        graph.add_edge("first", "second")
        graph.add_edge("second", _kit.END)
        return graph


class _BoomAgent(_kit.BaseTestAgent):
    """A node raising a given exception — the worker's failure paths."""

    name = "boomish"

    def __init__(self, error: Exception) -> None:
        self._error = error

    def build(self, ctx: Any) -> Any:
        error = self._error

        def boom(state: _kit.ChatState) -> dict[str, Any]:
            raise error

        graph = _kit.StateGraph(_kit.ChatState)
        graph.add_node("boom", boom)
        graph.add_edge(_kit.START, "boom")
        graph.add_edge("boom", _kit.END)
        return graph


# ---------------------------------------------------------------------------
# Queue doubles and the harness
# ---------------------------------------------------------------------------


class _LeaseLostQueue(InMemoryRunQueue):
    """A queue that stopped honoring our lease after a few beats.

    The first ``_GRACE_BEATS`` heartbeats still succeed (the lease is merely
    extended), so a slow CI machine can never lose the run before the test has
    observed its marker envelope; the next beat reports the lease gone.
    """

    _GRACE_BEATS = 8

    def __init__(self) -> None:
        super().__init__()
        self.beats = 0

    async def heartbeat(self, run_id: str) -> bool:
        self.beats += 1
        return self.beats <= self._GRACE_BEATS


class _ResumableQueue(InMemoryRunQueue):
    """Stand-in for the M4 resolve endpoint: plants the resume input on the row.

    PRD §4.3: the gateway builds ``Command(resume=...)`` from the resolution
    and requeues the run. WP1's queue has no input-update surface yet, so the
    test supplies it here — the worker itself just drives the row's input.
    """

    async def update_input(self, run_id: str, input: Any) -> None:
        async with self._lock:
            self._runs[run_id] = replace(self._runs[run_id], input=input)


class _ObservedSink(InMemoryEventSink):
    """Fires ``seen`` once an envelope matches ``trigger`` (mid-run sync point)."""

    def __init__(self, trigger: Any = None) -> None:
        super().__init__()
        self.seen = asyncio.Event()
        self._trigger = trigger

    async def write(self, envelope: EventEnvelope) -> None:
        await super().write(envelope)
        if self._trigger is not None and not self.seen.is_set() and self._trigger(envelope):
            self.seen.set()


class _Clock:
    """Deterministic clock: stands still until ``advance`` moves it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class _Harness:
    """One worker wired against a fresh queue, runner, and per-run sinks."""

    worker: Worker
    queue: InMemoryRunQueue
    runner: AgentRunner
    sinks: dict[str, _ObservedSink]


def make_worker(
    agent: Any,
    queue: InMemoryRunQueue | None = None,
    *,
    trigger: Any = None,
    **worker_kwargs: Any,
) -> _Harness:
    """Build the harness; the worker captures every run's sink for asserting."""
    queue = queue if queue is not None else InMemoryRunQueue()
    runner = AgentRunner(agent, checkpointer=_kit.InMemorySaver())
    sinks: dict[str, _ObservedSink] = {}

    async def factory(run_id: str) -> EventSink:
        sink = _ObservedSink(trigger)
        sinks[run_id] = sink
        return sink

    worker_kwargs.setdefault("claimant", "w-test")
    worker_kwargs.setdefault("poll_interval_s", 0.01)
    worker = Worker(runner, queue, agent_name=agent.name, sink_factory=factory, **worker_kwargs)
    return _Harness(worker=worker, queue=queue, runner=runner, sinks=sinks)


def _progress(label: str) -> Any:
    """A sink trigger matching one ``progress`` record's label."""

    def match(envelope: EventEnvelope) -> bool:
        return envelope.type.value == "progress" and envelope.payload.get("label") == label

    return match


async def _observed(harness: _Harness, run_id: str) -> _ObservedSink:
    """The run's sink once the worker opened it (a few loop turns at most)."""
    while run_id not in harness.sinks:
        await asyncio.sleep(0)
    return harness.sinks[run_id]


def _status_payloads(sink: _ObservedSink) -> list[dict[str, Any]]:
    """Every ``run.status`` payload the sink captured, in order."""
    return [envelope.payload for envelope in sink.events if envelope.type.value == "run.status"]


def _status_envelopes(sink: _ObservedSink) -> list[EventEnvelope]:
    return [envelope for envelope in sink.events if envelope.type.value == "run.status"]


def _labels(sink: _ObservedSink) -> list[Any]:
    return [
        envelope.payload.get("label")
        for envelope in sink.events
        if envelope.type.value == "progress"
    ]


# ---------------------------------------------------------------------------
# Happy path, context wiring, approvals
# ---------------------------------------------------------------------------


async def test_claim_drive_complete() -> None:
    agent = _kit.ChattyAgent()
    harness = make_worker(agent)
    record = await harness.queue.enqueue(
        agent.name, _kit.input_for(agent), thread_id="t1", run_id=None
    )
    assert await harness.worker.run_once() is True
    final = await harness.queue.get(record.id)
    assert final is not None
    assert final.status is RunState.COMPLETED
    assert final.attempt == 1
    assert final.claimed_by is None
    assert final.lease_expires_at is None
    sink = await _observed(harness, record.id)
    assert [payload["status"] for payload in _status_payloads(sink)] == ["running", "completed"]


async def test_worker_binds_the_run_context_and_idempotency_key() -> None:
    gate = asyncio.Event()
    gate.set()  # both pre-set: the gated node runs straight through
    release = asyncio.Event()
    release.set()
    agent = GatedAgent(gate, release)
    harness = make_worker(agent)
    record = await harness.queue.enqueue(
        agent.name, _kit.input_for(agent), thread_id="t-ctx", run_id=None
    )
    assert await harness.worker.run_once() is True
    final = await harness.queue.get(record.id)
    assert final is not None and final.status is RunState.COMPLETED
    ctx = agent.seen_context
    assert ctx is not None, "the node saw the worker's VDreamerContext"
    assert ctx.run_id == record.id
    assert ctx.thread_id == "t-ctx"
    assert ctx.agent_name == agent.name
    assert ctx.attempt == 1
    # A6: the key is the run id itself — stable across recovery attempts so a
    # re-executed side effect can dedupe on it (not per-claim, not per-attempt).
    assert ctx.idempotency_key == record.id


async def test_approval_suspends_then_resume_requeues_with_the_resume_input() -> None:
    agent = _kit.ApprovalAgent()
    harness = make_worker(agent, _ResumableQueue())
    record = await harness.queue.enqueue(
        agent.name, _kit.input_for(agent), thread_id="t1", run_id=None
    )

    # First drive: the graph interrupts, the driver ends on awaiting_approval,
    # the worker suspends the row — approvals free the worker (§3.2).
    assert await harness.worker.run_once() is True
    suspended = await harness.queue.get(record.id)
    assert suspended is not None
    assert suspended.status is RunState.AWAITING_APPROVAL
    assert suspended.lease_expires_at is None
    sink = await _observed(harness, record.id)
    assert [payload["status"] for payload in _status_payloads(sink)] == [
        "running",
        "awaiting_approval",
    ]
    assert any(envelope.type.value == "interrupt.requested" for envelope in sink.events)
    # The thread is still occupied, but nothing is claimed or leased.
    assert await harness.queue.claim("w2") is None

    # Resolution (the gateway's job, M4): plant Command(resume=...) on the
    # row, then resume → queued → the next claim re-drives from the checkpoint.
    await harness.queue.update_input(record.id, Command(resume={"action": "approve"}))
    await harness.queue.resume(record.id)
    assert await harness.worker.run_once() is True
    final = await harness.queue.get(record.id)
    assert final is not None
    assert final.status is RunState.COMPLETED
    assert final.attempt == 2  # the resume consumed a fresh claim
    snapshot = await harness.runner.compiled.aget_state({"configurable": {"thread_id": "t1"}})
    assert snapshot.values["flag"] == str({"action": "approve"}), "resume value reached the gate"


# ---------------------------------------------------------------------------
# Cancel: drain → cancelled, resumable checkpoint
# ---------------------------------------------------------------------------


async def test_cancel_drains_to_cancelled_and_leaves_a_resumable_checkpoint() -> None:
    gate = asyncio.Event()
    release = asyncio.Event()  # stays closed: the node never returns once draining
    agent = GatedAgent(gate, release)
    harness = make_worker(agent, trigger=_progress("second-started"))
    record = await harness.queue.enqueue(
        agent.name, _kit.input_for(agent), thread_id="t-cancel", run_id=None
    )
    task = asyncio.create_task(harness.worker.run_once())
    sink = await _observed(harness, record.id)
    await asyncio.wait_for(sink.seen.wait(), 5.0)  # mid-run: first done, second gated

    assert await harness.worker.request_cancel(record.id) is True
    gate.set()
    assert await asyncio.wait_for(task, 5.0) is True

    final = await harness.queue.get(record.id)
    assert final is not None
    assert final.status is RunState.CANCELLED
    assert final.claimed_by is None
    assert final.lease_expires_at is None

    # The stream aborted before any terminal driver envelope; the worker
    # authored the final cancelled status the SSE log should end on, and the
    # first post-gate record never landed (the drain fired at its write).
    assert [payload["status"] for payload in _status_payloads(sink)] == ["running", "cancelled"]
    assert _labels(sink) == ["first-done", "second-started"]

    # The last checkpoint survives the drain: the in-flight superstep never
    # completed, so exactly the first node's writes are there — and the thread
    # accepts a fresh invocation that completes.
    snapshot = await harness.runner.compiled.aget_state(
        {"configurable": {"thread_id": "t-cancel"}}
    )
    assert snapshot.values["flag"] == "first"
    release.set()
    resume = await harness.queue.enqueue(
        agent.name, Command(resume={"action": "approve"}), thread_id="t-cancel", run_id=None
    )
    assert await harness.worker.run_once() is True
    final = await harness.queue.get(resume.id)
    assert final is not None and final.status is RunState.COMPLETED
    snapshot = await harness.runner.compiled.aget_state(
        {"configurable": {"thread_id": "t-cancel"}}
    )
    assert snapshot.values["flag"] == "second"


async def test_request_cancel_falls_back_to_the_queue_for_non_active_runs() -> None:
    agent = _kit.ChattyAgent()
    harness = make_worker(agent)
    queued = await harness.queue.enqueue(
        agent.name, _kit.input_for(agent), thread_id="t-q", run_id=None
    )
    # Not active locally (never claimed): falls back to queue.cancel → marked.
    assert await harness.worker.request_cancel(queued.id) is True
    final = await harness.queue.get(queued.id)
    assert final is not None and final.status is RunState.CANCELLED
    # Terminal runs are neither active here nor cancellable.
    assert await harness.worker.request_cancel(queued.id) is False
    assert await harness.worker.request_cancel("run_missing") is False


async def test_drain_requested_before_the_run_starts_writes_no_driver_events() -> None:
    """A drain landing inside the sink-open window aborts before any envelope."""
    gate = asyncio.Event()
    release = asyncio.Event()
    agent = GatedAgent(gate, release)
    queue = InMemoryRunQueue()
    runner = AgentRunner(agent, checkpointer=_kit.InMemorySaver())
    sinks: dict[str, _ObservedSink] = {}
    opened = asyncio.Event()

    async def factory(run_id: str) -> EventSink:
        sink = _ObservedSink(None)
        sinks[run_id] = sink
        await opened.wait()  # a window for a drain to land before the run starts
        return sink

    worker = Worker(
        runner, queue, agent_name=agent.name, sink_factory=factory, poll_interval_s=0.01
    )
    record = await queue.enqueue(
        agent.name, _kit.input_for(agent), thread_id="t-early", run_id=None
    )
    task = asyncio.create_task(worker.run_once())
    while record.id not in sinks:
        await asyncio.sleep(0)  # the factory ran and is parked on `opened`
    assert await worker.request_cancel(record.id) is True
    opened.set()
    gate.set()
    await asyncio.wait_for(task, 5.0)

    final = await queue.get(record.id)
    assert final is not None and final.status is RunState.CANCELLED
    # The driver wrote nothing (not even the initial ``running`` status); the
    # worker's own cancelled status is the run's entire event log.
    assert [payload["status"] for payload in _status_payloads(sinks[record.id])] == ["cancelled"]


# ---------------------------------------------------------------------------
# SIGTERM: drain and requeue without consuming an attempt
# ---------------------------------------------------------------------------


async def test_sigterm_drain_requeues_without_consuming_an_attempt() -> None:
    gate = asyncio.Event()
    release = asyncio.Event()  # stays closed: the node never returns once draining
    agent = GatedAgent(gate, release)
    harness = make_worker(agent, trigger=_progress("second-started"))
    record = await harness.queue.enqueue(
        agent.name, _kit.input_for(agent), thread_id="t-term", run_id=None
    )
    task = asyncio.create_task(harness.worker.run_once())
    sink = await _observed(harness, record.id)
    await asyncio.wait_for(sink.seen.wait(), 5.0)

    harness.worker.drain_all("sigterm")
    gate.set()
    assert await asyncio.wait_for(task, 5.0) is True

    final = await harness.queue.get(record.id)
    assert final is not None
    assert final.status is RunState.QUEUED
    assert final.attempt == 0, "the claim's attempt increment was given back"
    assert final.claimed_by is None
    assert final.lease_expires_at is None
    # No terminal envelope: the run is not over — the next attempt re-drives.
    assert [payload["status"] for payload in _status_payloads(sink)] == ["running"]
    assert _labels(sink) == ["first-done", "second-started"]
    # A later claim re-increments to the same attempt number.
    reclaimed = await harness.queue.claim("w2")
    assert reclaimed is not None and reclaimed.attempt == 1

    # The sigterm drain also marks shutdown: a running run() exits at once.
    await asyncio.wait_for(harness.worker.run(), 2.0)


# ---------------------------------------------------------------------------
# Lost leases and node errors
# ---------------------------------------------------------------------------


async def test_lease_lost_drains_and_marks_nothing() -> None:
    gate = asyncio.Event()
    release = asyncio.Event()  # stays closed: the node keeps ticking, never returns
    agent = GatedAgent(gate, release)
    harness = make_worker(
        agent, _LeaseLostQueue(), trigger=_progress("second-started"), lease_interval_s=0.1
    )
    record = await harness.queue.enqueue(
        agent.name, _kit.input_for(agent), thread_id="t-lease", run_id=None
    )
    task = asyncio.create_task(harness.worker.run_once())
    sink = await _observed(harness, record.id)
    await asyncio.wait_for(sink.seen.wait(), 5.0)  # mid-run, before the first heartbeat

    gate.set()  # the node ticks; the heartbeat tick (False) drains with lease_lost
    assert await asyncio.wait_for(task, 5.0) is True

    # The worker marked nothing: the row is still running under its (stale)
    # lease — whoever holds it now (the recovery sweep, another worker, the
    # canceller) owns its fate.
    final = await harness.queue.get(record.id)
    assert final is not None
    assert final.status is RunState.RUNNING
    assert final.claimed_by == "w-test"
    # The stream aborted without any terminal envelope, and the node's final
    # record never landed (a tick's write is where the drain fired).
    assert [payload["status"] for payload in _status_payloads(sink)] == ["running"]
    labels = _labels(sink)
    assert labels[:2] == ["first-done", "second-started"]
    assert "second-done" not in labels
    assert set(labels[2:]) == {"ticking"}


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (NodeError("sql exploded", node="execute"), "sql exploded"),
        (NodeTimeoutError("sql timed out", node="execute", timeout_s=1.5), "sql timed out"),
    ],
)
async def test_node_errors_map_to_run_status_error(
    error: Exception, expected_message: str
) -> None:
    agent = _BoomAgent(error)
    harness = make_worker(agent)
    record = await harness.queue.enqueue(
        agent.name, _kit.input_for(agent), thread_id="t-fail", run_id=None
    )
    assert await harness.worker.run_once() is True
    final = await harness.queue.get(record.id)
    assert final is not None and final.status is RunState.FAILED
    sink = await _observed(harness, record.id)
    statuses = _status_payloads(sink)
    assert statuses[-1]["status"] == "failed"
    assert statuses[-1]["attempt"] == 1
    assert statuses[-1]["error"]["code"] == "node_error"
    assert expected_message in statuses[-1]["error"]["message"]


async def test_failure_before_the_driver_emits_the_worker_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure outside the driver (bad VDREAMER_STREAM_DRIVER): the worker
    authors the failed envelope itself — the §3.3 error block, seq continuing."""
    agent = _kit.ChattyAgent()
    harness = make_worker(agent)
    record = await harness.queue.enqueue(
        agent.name, _kit.input_for(agent), thread_id="t-pre", run_id=None
    )
    monkeypatch.setenv("VDREAMER_STREAM_DRIVER", "bogus")
    assert await harness.worker.run_once() is True
    final = await harness.queue.get(record.id)
    assert final is not None and final.status is RunState.FAILED
    sink = await _observed(harness, record.id)
    envelopes = _status_envelopes(sink)
    assert [envelope.payload["status"] for envelope in envelopes] == ["failed"]
    assert envelopes[0].payload["error"]["code"] == "node_error"
    assert "bogus" in envelopes[0].payload["error"]["message"]
    assert envelopes[0].seq == 1  # the worker continues the run's seq from 1


def test_node_error_shapes() -> None:
    error = NodeError("boom", node="n1")
    cause = ValueError("root")
    try:
        try:
            raise cause
        except ValueError as exc:
            raise error from exc
    except NodeError as caught:
        assert caught is error
        assert caught.node == "n1"
        assert caught.__cause__ is cause
    timeout = NodeTimeoutError("slow", node="n2", timeout_s=2.5)
    assert timeout.node == "n2"
    assert timeout.timeout_s == 2.5


# ---------------------------------------------------------------------------
# RunControl and startup recovery
# ---------------------------------------------------------------------------


async def test_run_control_is_idempotent_and_observable() -> None:
    control = RunControl()
    assert control.drain_requested is False
    assert control.reason is None
    control.request_drain("cancel")
    control.request_drain("sigterm")  # idempotent: the first reason wins
    assert control.drain_requested is True
    assert control.reason == "cancel"
    assert await asyncio.wait_for(control.wait_for_drain_requested(), 1.0) == "cancel"
    await asyncio.wait_for(control.wait(), 1.0)  # an already-set event doesn't block


async def test_run_control_wait_blocks_until_requested() -> None:
    control = RunControl()
    task = asyncio.create_task(control.wait_for_drain_requested())
    await asyncio.sleep(0)
    assert not task.done()
    control.request_drain("sigterm")
    assert await asyncio.wait_for(task, 1.0) == "sigterm"


async def test_recover_on_start_requeues_expired_leases() -> None:
    clock = _Clock()
    queue = InMemoryRunQueue(clock)
    agent = _kit.ChattyAgent()
    record = await queue.enqueue(agent.name, _kit.input_for(agent), thread_id="t-rec", run_id=None)
    claimed = await queue.claim("w1")
    assert claimed is not None and claimed.attempt == 1
    clock.advance(LEASE_INTERVAL_S + 1.0)  # the lease lapses

    # Without recovery the expired row is invisible to the claimer: it stays
    # ``running`` until a sweep handles it.
    stale = make_worker(_kit.ChattyAgent(), queue, recover_on_start=False)
    assert await stale.worker.run_once() is False
    still = await queue.get(record.id)
    assert still is not None and still.status is RunState.RUNNING

    # With recover_on_start the first iteration requeues it, then claims it.
    harness = make_worker(_kit.ChattyAgent(), queue)
    assert await harness.worker.run_once() is True
    final = await queue.get(record.id)
    assert final is not None
    assert final.status is RunState.COMPLETED
    assert final.attempt == 2  # requeued unchanged (1), the re-claim → 2


async def test_run_exits_when_stop_is_set() -> None:
    harness = make_worker(_kit.ChattyAgent())
    stop = asyncio.Event()
    asyncio.get_running_loop().call_later(0.05, stop.set)
    await asyncio.wait_for(harness.worker.run(stop), 2.0)


# ---------------------------------------------------------------------------
# SIGTERM handler install (POSIX main thread only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="Windows never delivers SIGTERM")
async def test_sigterm_handler_is_installed_and_restored_on_posix() -> None:
    harness = make_worker(_kit.ChattyAgent())
    previous = signal.getsignal(signal.SIGTERM)
    stop = asyncio.Event()
    stop.set()
    await harness.worker.run(stop)  # installs and uninstalls around the no-op loop
    assert signal.getsignal(signal.SIGTERM) is previous
    # The install itself swaps the handler (main thread, POSIX):
    harness.worker._install_sigterm()
    try:
        assert signal.getsignal(signal.SIGTERM) is not previous
    finally:
        harness.worker._uninstall_sigterm()
    assert signal.getsignal(signal.SIGTERM) is previous


@pytest.mark.skipif(sys.platform != "win32", reason="POSIX installs the handler")
async def test_sigterm_install_noops_on_windows() -> None:
    harness = make_worker(_kit.ChattyAgent())
    previous = signal.getsignal(signal.SIGTERM)
    harness.worker._install_sigterm()
    assert harness.worker._sigterm_installed is False
    assert signal.getsignal(signal.SIGTERM) is previous
