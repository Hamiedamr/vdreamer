"""The SSE transport (PRD §2.5, §3.3, A3, A22).

One stream serves one run: replay every envelope with ``seq >
Last-Event-ID``, then tail live until the run ends. Framing per §3.3::

    id: {run_id}:{seq}
    data: {envelope JSON}

with ``: heartbeat`` comment frames every 15 s of silence (§3.3) and the
response headers ``Cache-Control: no-cache``, ``X-Accel-Buffering: no`` (A22).
No ``event:`` field: clients consume with fetch (D14), read the type from the
envelope, and ignore unknown types (§6.1).

The loop never trusts a wake-up for content — the log is the source of truth
(A3). It subscribes to the hub *before* the first read, re-reads after every
wake-up and every heartbeat (so a lost ``NOTIFY`` costs at most one heartbeat
of latency), and ends when it has sent a terminal ``run.status`` or the run row
is terminal with nothing left to send. ``awaiting_approval`` is not an end:
the same run resumes after a resolution (§3.2), so the stream stays open.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Final

from vdreamer import metrics
from vdreamer.eventlog import EventHub, EventReader
from vdreamer.events import SSE_BUFFERING_HEADER, SSE_BUFFERING_VALUE, EventEnvelope, EventType
from vdreamer.queue import TERMINAL_STATES, RunRecord, RunState

#: SSE response headers (A22; ``X-Protocol-Version`` is added by the middleware).
SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    SSE_BUFFERING_HEADER: SSE_BUFFERING_VALUE,
}

SSE_MEDIA_TYPE: Final[str] = "text/event-stream"

#: The comment frame sent after ``heartbeat_s`` of silence (§3.3).
HEARTBEAT_FRAME: Final[str] = ": heartbeat\n\n"

_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(state.value for state in TERMINAL_STATES)


def format_event(envelope: EventEnvelope) -> str:
    """One envelope as an SSE frame: ``id: {run_id}:{seq}`` + one ``data:`` line."""
    data = json.dumps(envelope.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False)
    return f"id: {envelope.sse_id}\ndata: {data}\n\n"


def is_terminal_event(envelope: EventEnvelope) -> bool:
    """A ``run.status`` that ends the run (``awaiting_approval`` does not)."""
    return (
        envelope.type is EventType.RUN_STATUS
        and str(envelope.payload.get("status")) in _TERMINAL_STATUSES
    )


class LastEventIdError(ValueError):
    """A ``Last-Event-ID`` that cannot be resumed from."""


def parse_last_event_id(value: str | None, run_id: str) -> int:
    """The ``seq`` to resume after: ``"{run_id}:{seq}"`` (§3.3) or a bare ``seq``.

    ``None``/empty → 0 (replay from the start).

    Raises:
        LastEventIdError: malformed, or it names another run.
    """
    if value is None or not value.strip():
        return 0
    raw = value.strip()
    named_run, sep, seq_text = raw.rpartition(":")
    if not sep:
        named_run, seq_text = run_id, raw
    if named_run != run_id:
        raise LastEventIdError(f"Last-Event-ID {raw!r} belongs to another run")
    try:
        seq = int(seq_text)
    except ValueError:
        raise LastEventIdError(f"Last-Event-ID {raw!r} is not '{{run_id}}:{{seq}}'") from None
    if seq < 0:
        raise LastEventIdError(f"Last-Event-ID {raw!r} has a negative seq")
    return seq


async def run_event_stream(
    *,
    run: RunRecord,
    after_seq: int,
    reader: EventReader,
    hub: EventHub,
    get_run: Callable[[str], Awaitable[RunRecord | None]],
    heartbeat_s: float,
) -> AsyncIterator[str]:
    """Yield SSE frames for ``run``: replay ``seq > after_seq``, then tail live."""
    run_id = run.id
    thread_id = run.thread_id or run.id
    last = after_seq
    metrics.SSE_CONNECTIONS.inc()
    try:
        with hub.subscribe(run_id) as subscription:
            while True:
                drained = False
                while not drained:  # page through everything available
                    batch = await reader.read(
                        run_id, last, agent=run.agent, thread_id=thread_id
                    )
                    for envelope in batch:
                        last = envelope.seq
                        yield format_event(envelope)
                        if is_terminal_event(envelope):
                            return
                    drained = not batch
                current = await get_run(run_id)
                if current is None or current.status in TERMINAL_STATES:
                    # Terminal row, nothing more in the log (e.g. a sweep that
                    # wrote no event): one last read closes the race with a
                    # writer that committed between our read and the row read.
                    for envelope in await reader.read(
                        run_id, last, agent=run.agent, thread_id=thread_id
                    ):
                        last = envelope.seq
                        yield format_event(envelope)
                    return
                if not await subscription.wait(heartbeat_s):
                    yield HEARTBEAT_FRAME
    finally:
        metrics.SSE_CONNECTIONS.dec()


def choose_thread_run(runs: list[RunRecord]) -> RunRecord | None:
    """The run ``GET /threads/{id}/events`` joins (§2.5): the active one, else the latest.

    "Active" prefers ``running``/``awaiting_approval`` over ``queued``: with
    ``wait_for_idle`` a thread can hold one of each, and the one in flight is
    what a reattaching client is looking at.
    """
    for status in (RunState.RUNNING, RunState.AWAITING_APPROVAL, RunState.QUEUED):
        for run in runs:
            if run.status is status:
                return run
    return runs[-1] if runs else None
