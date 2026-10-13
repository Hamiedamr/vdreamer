"""The Milestone-3 exit test (PRD §3.2, A6): kill a worker mid-run; the run
resumes and the side effect runs exactly once.

The scenario, entirely in-process and deterministic (no sleeps beyond the
worker's own poll ticks, an injectable queue clock, event-gated sync points):

1. Worker 1 claims a run whose ``charge`` node performs a side effect through
   a *persisted ledger* — the stand-in for a real tool's idempotency store —
   then parks mid-node (never returns), exactly like an in-flight node at
   kill time.
2. The kill: worker 1 drains with reason ``lease_lost`` and marks nothing
   (``Worker._finish_drained``) — the row keeps its claim and its lease,
   which then lapses on the injectable clock. This is the hard-crash shape:
   no release, no settle, no checkpoint of the in-flight superstep.
3. The §3.2 recovery sweep (``requeue_expired``) requeues the run (attempt
   stays 1; it increments on claim), and worker 2 — a fresh process
   equivalent: fresh agent instance and fresh ``AgentRunner`` over the SAME
   checkpoint store, like two processes on one Postgres — claims it
   (attempt 2) and re-executes the side-effecting node.
4. The re-executed node checks the ledger, skips the side effect, and
   returns the recorded result. The run completes.

The exactly-once recipe (A6), and what the test pins:

- ``ctx.idempotency_key`` is the ``run_id`` itself — stable across recovery
  attempts. The ledger dedupes on ``ctx.idempotency_key`` ITSELF, the PRD's
  letter (§3.2: "side effects use ``ctx.idempotency_key``"): the run is the
  effect's identity (§3.1: one run = one execution = one user turn), so an
  attempt-varying key could never dedupe the re-executed node. Within one
  run, call-site disambiguation — a graph calling the same side-effecting
  tool twice — is the tool author's concern: they may namespace the key with
  ``tool_call_id``. Cross-attempt stability is the core's guarantee, and
  ``attempt`` remains a separate ``VDreamerContext`` field.
- The pre-kill checkpoint survives: the ``first`` node's attempt-1 append to
  the additive ``ops`` channel is still in the state the resumed run built
  on. The re-claim re-drives the row's input (langgraph starts a fresh
  superstep on the same thread), so completed nodes re-execute too — the
  ledger, not node skipping, is what keeps the *side effect* exactly-once;
  planting a literal ``Command(resume=...)``/``None`` resume on requeued
  rows is the gateway's M4 planting job (same mechanism as approval
  resumes).
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, TypedDict

import _kit
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from vdreamer import (
    LEASE_INTERVAL_S,
    InMemoryEventSink,
    InMemoryRunQueue,
    RunState,
    Worker,
)
from vdreamer.context import current_context
from vdreamer.eventlog import EventEnvelope, EventSink
from vdreamer.runtime import AgentRunner


class EffectState(TypedDict):
    """``ChatState`` plus an additive channel — the checkpoint-carry probe."""

    messages: Any
    flag: str
    ops: Annotated[list[str], operator.add]


class _EffectLedger:
    """The persisted side-effect store: one entry per idempotency key.

    The ledger keys on ``ctx.idempotency_key`` ITSELF — no tool-side
    stable-key derivation. ``recorded`` is the ledger the node checks on
    re-execution (skip the effect, return the recorded result); ``effects``
    is the assertion target — the keys whose *external* effect actually
    fired; ``executions`` counts node executions, expected to exceed effects
    on recovery; ``keys_seen`` records the context key on every execution so
    the test can pin it identical across attempts.
    """

    def __init__(self) -> None:
        self.recorded: dict[str, str] = {}
        self.executions: dict[str, int] = {}
        self.keys_seen: list[str] = []
        self.effects: list[str] = []

    def charge_card(self, idempotency_key: str) -> str:
        """The side-effecting tool: charge once per idempotency key, return the recorded result."""
        self.keys_seen.append(idempotency_key)
        self.executions[idempotency_key] = self.executions.get(idempotency_key, 0) + 1
        if idempotency_key in self.recorded:
            return self.recorded[idempotency_key]  # already charged by an earlier attempt
        result = f"charged-by-{idempotency_key}"
        self.recorded[idempotency_key] = result
        self.effects.append(idempotency_key)
        return result


class SideEffectAgent(_kit.BaseTestAgent):
    """``first`` checkpoints → ``charge`` runs the tool, then parks mid-node.

    ``first`` appends to the additive ``ops`` channel: whatever it checkpointed
    before the kill must still be in the state the resumed run builds on.
    ``charge`` performs the ledger-protected side effect, emits its marker,
    and then parks on ``gate`` — the worker's drain can only fire at a later
    envelope write (WP2's gated-node pattern), so the node is left unfinished:
    its superstep never commits and the re-claim re-executes it.
    """

    name = "effectful"

    def __init__(
        self,
        ledger: _EffectLedger,
        gate: asyncio.Event,
        release: asyncio.Event,
    ) -> None:
        self._ledger = ledger
        self._gate = gate
        self._release = release

    def build(self, ctx: Any) -> Any:
        ledger, gate, release = self._ledger, self._gate, self._release

        async def first(state: EffectState) -> dict[str, Any]:
            attempt = current_context().attempt
            get_stream_writer()({"type": "progress", "label": "first-done"})
            return {"flag": "first", "ops": [f"first@{attempt}"]}

        async def charge(state: EffectState) -> dict[str, Any]:
            run = current_context()
            assert run.idempotency_key is not None, (
                "the worker binds a non-None idempotency key (A6)"
            )
            result = ledger.charge_card(run.idempotency_key)
            get_stream_writer()({"type": "progress", "label": "effect-done"})
            await gate.wait()
            while not release.is_set():
                get_stream_writer()({"type": "progress", "label": "ticking"})
                await asyncio.sleep(0.005)
            get_stream_writer()({"type": "progress", "label": "charge-done"})
            return {"flag": result, "ops": [f"charge@{run.attempt}"]}

        graph = StateGraph(EffectState)
        graph.add_node("first", first)
        graph.add_node("charge", charge)
        graph.add_edge(START, "first")
        graph.add_edge("first", "charge")
        graph.add_edge("charge", END)
        return graph


class _Clock:
    """Deterministic clock: stands still until ``advance`` moves it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _ObservedSink(InMemoryEventSink):
    """Fires ``seen`` once the trigger envelope lands (mid-run sync point)."""

    def __init__(self, trigger: Any = None) -> None:
        super().__init__()
        self.seen = asyncio.Event()
        self._trigger = trigger

    async def write(self, envelope: EventEnvelope) -> None:
        await super().write(envelope)
        if self._trigger is not None and not self.seen.is_set() and self._trigger(envelope):
            self.seen.set()


def _progress(label: str) -> Any:
    """A sink trigger matching one ``progress`` record's label."""

    def match(envelope: EventEnvelope) -> bool:
        return envelope.type.value == "progress" and envelope.payload.get("label") == label

    return match


async def test_kill_midrun_resumes_and_the_side_effect_runs_exactly_once() -> None:
    clock = _Clock()
    queue = InMemoryRunQueue(clock)
    checkpointer = _kit.InMemorySaver()  # the one "Postgres" both workers share
    ledger = _EffectLedger()

    # -- worker 1: claims and parks mid-node, right after the side effect ----
    gate1, release1 = asyncio.Event(), asyncio.Event()  # both stay closed
    agent1 = SideEffectAgent(ledger, gate1, release1)
    runner1 = AgentRunner(agent1, checkpointer=checkpointer)
    sinks1: dict[str, _ObservedSink] = {}

    async def factory1(run_id: str) -> EventSink:
        sink = _ObservedSink(trigger=_progress("effect-done"))
        sinks1[run_id] = sink
        return sink

    worker1 = Worker(
        runner1,
        queue,
        agent_name=agent1.name,
        claimant="w1",
        sink_factory=factory1,
        poll_interval_s=0.01,
    )
    record = await queue.enqueue(
        agent1.name, _kit.input_for(agent1), thread_id="t-recover", run_id=None
    )
    task = asyncio.create_task(worker1.run_once())
    while record.id not in sinks1:
        await asyncio.sleep(0)
    await asyncio.wait_for(sinks1[record.id].seen.wait(), 5.0)
    assert ledger.effects == [record.id], "the effect fired under the first claim"
    assert ledger.keys_seen == [record.id], (
        "the tool saw the context key — the run id itself, stable across attempts"
    )

    # -- the kill: no release, no settle — the row keeps claim and lease -----
    worker1.drain_all("lease_lost")
    gate1.set()  # the node's next write is where the drain check fires
    assert await asyncio.wait_for(task, 5.0) is True

    row = await queue.get(record.id)
    assert row is not None
    assert row.status is RunState.RUNNING, "the killed worker marked nothing"
    assert row.attempt == 1
    assert row.claimed_by == "w1"
    assert row.lease_expires_at is not None
    # The drain aborted the stream before the node finished: the marker landed,
    # the tick and the node's return never did.
    sink1 = sinks1[record.id]
    assert [
        envelope.payload.get("label")
        for envelope in sink1.events
        if envelope.type.value == "progress"
    ] == ["first-done", "effect-done"]

    # -- the lease lapses; the recovery sweep requeues (attempt unchanged) ---
    clock.advance(LEASE_INTERVAL_S + 1.0)
    assert await queue.requeue_expired() == 1
    row = await queue.get(record.id)
    assert row is not None
    assert row.status is RunState.QUEUED
    assert row.attempt == 1
    assert row.lease_expires_at is None

    # -- worker 2: fresh process-equivalent over the same checkpoint store ---
    gate2, release2 = asyncio.Event(), asyncio.Event()
    gate2.set()  # worker 2's node runs straight through
    release2.set()
    agent2 = SideEffectAgent(ledger, gate2, release2)
    runner2 = AgentRunner(agent2, checkpointer=checkpointer)
    sinks2: dict[str, InMemoryEventSink] = {}

    async def factory2(run_id: str) -> EventSink:
        sink = InMemoryEventSink()
        sinks2[run_id] = sink
        return sink

    worker2 = Worker(
        runner2,
        queue,
        agent_name=agent2.name,
        claimant="w2",
        sink_factory=factory2,
        poll_interval_s=0.01,
        recover_on_start=False,  # the sweep already ran above
    )
    assert await worker2.run_once() is True

    # -- the run completed, from the checkpoint, with the effect exactly once
    final = await queue.get(record.id)
    assert final is not None
    assert final.status is RunState.COMPLETED
    assert final.attempt == 2, "the recovery claim consumed attempt 2"

    assert ledger.effects == [record.id], "the side effect ran EXACTLY ONCE (A6)"
    assert ledger.executions[record.id] == 2, "the side-effecting node re-executed on resume"
    assert ledger.keys_seen == [record.id, record.id], (
        "the context key is IDENTICAL across both attempts (A6: the key is the run id)"
    )
    assert ledger.recorded == {record.id: f"charged-by-{record.id}"}, (
        "the ledger recorded the effect-time key — ctx.idempotency_key itself"
    )

    snapshot = await runner2.compiled.aget_state({"configurable": {"thread_id": "t-recover"}})
    assert snapshot.values["flag"] == ledger.recorded[record.id], (
        "the re-executed node returned the ledger's recorded result"
    )
    assert snapshot.values["ops"] == ["first@1", "first@2", "charge@2"], (
        "the pre-kill checkpointed state survived the kill and carried into the resume"
    )

    sink2 = sinks2[record.id]
    assert [
        envelope.payload.get("label")
        for envelope in sink2.events
        if envelope.type.value == "progress"
    ] == ["first-done", "effect-done", "charge-done"]
    assert [
        envelope.payload.get("status")
        for envelope in sink2.events
        if envelope.type.value == "run.status"
    ] == ["running", "completed"]
