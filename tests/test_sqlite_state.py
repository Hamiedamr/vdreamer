"""The SQLite dev stores: everything ``vdreamer dev`` keeps survives a restart.

The REST contract suite (``tests/api``) already runs against these stores;
this module covers what only a second process can see — reloading the file.
"""

from __future__ import annotations

import decimal
from pathlib import Path

from vdreamer.events import EventEnvelope, EventType
from vdreamer.interrupts import InterruptRecord, InterruptStatus
from vdreamer.queue import RunState
from vdreamer.sqlite_state import open_sqlite_state


def _log(run_id: str, message: str) -> EventEnvelope:
    return EventEnvelope(
        type=EventType.LOG,
        run_id=run_id,
        thread_id="t1",
        agent="a",
        seq=0,
        payload={"level": "info", "message": message},
    )


async def test_runs_threads_interrupts_and_events_survive_a_restart(tmp_path: Path) -> None:
    db = tmp_path / "dev.db"
    async with open_sqlite_state(db) as state:
        thread = await state.threads.create("a", "alice", metadata={"k": "v"})
        gone = await state.threads.create("a", "alice")
        assert await state.threads.soft_delete(gone.id)
        run = await state.queue.enqueue(
            "a", {"q": 1}, thread_id=thread.id, run_id=None, user_id="alice", roles=["r"]
        )
        assert await state.queue.claim("dev", agent="a") is not None
        await state.queue.suspend(run.id)
        await state.interrupts.create(
            InterruptRecord(
                interrupt_id="i1",
                run_id=run.id,
                thread_id=thread.id,
                agent="a",
                kind="approval",
                actions=("approve", "reject"),
                payload={"cost": decimal.Decimal("1.5")},
                initiator_id="alice",
                created_at=1.0,
            )
        )
        await state.events.write(_log(run.id, "one"))
        await state.events.write(_log(run.id, "two"))

    async with open_sqlite_state(db) as state:
        assert await state.threads.get(gone.id) is None
        reloaded_thread = await state.threads.get(thread.id)
        assert reloaded_thread is not None and reloaded_thread.metadata == {"k": "v"}
        assert [t.id for t in await state.threads.list("alice")] == [thread.id]

        reloaded = await state.queue.get(run.id)
        assert reloaded is not None
        assert reloaded.status is RunState.AWAITING_APPROVAL
        assert (reloaded.input, reloaded.roles, reloaded.attempt) == ({"q": 1}, ("r",), 1)

        (pending,) = await state.interrupts.list(run_id=run.id)
        assert pending.actions == ("approve", "reject")
        assert pending.payload == {"cost": "1.5"}  # stringified, never a failed write
        resolved = await state.interrupts.resolve(
            "i1", expected_version=1, resolved_by="alice", resolution={"action": "approve"}
        )
        await state.queue.resume(run.id, resume={"i1": resolved.resolution})

        # The log owns seq across processes: the next write continues at 3.
        await state.events.write(_log(run.id, "three"))
        events = await state.events.read(run.id)
        assert [(e.seq, e.payload["message"]) for e in events] == [
            (1, "one"),
            (2, "two"),
            (3, "three"),
        ]
        assert [e.seq for e in await state.events.read(run.id, after_seq=1, limit=1)] == [2]

    async with open_sqlite_state(db) as state:
        record = await state.interrupts.get("i1")
        assert record is not None and record.status is InterruptStatus.RESOLVED
        requeued = await state.queue.get(run.id)
        assert requeued is not None and requeued.status is RunState.QUEUED
        assert requeued.resume == {"i1": {"action": "approve"}}


async def test_a_run_left_running_is_recovered_on_the_next_boot(tmp_path: Path) -> None:
    """One process owns the file: a ``running`` row at boot lost its worker."""
    db = tmp_path / "dev.db"
    async with open_sqlite_state(db) as state:
        thread = await state.threads.create("a", "alice")
        run = await state.queue.enqueue("a", {}, thread_id=thread.id, run_id=None)
        assert await state.queue.claim("dev") is not None  # then the process dies

    async with open_sqlite_state(db) as state:
        assert await state.queue.requeue_expired() == 1  # the worker's boot sweep
        recovered = await state.queue.get(run.id)
        assert recovered is not None and recovered.status is RunState.QUEUED
        claimed = await state.queue.claim("dev")
        assert claimed is not None and claimed.attempt == 2

    async with open_sqlite_state(db) as state:
        current = await state.queue.get(run.id)
        assert current is not None and current.status is RunState.RUNNING


async def test_artifacts_survive_a_restart_with_their_bytes_on_disk(tmp_path: Path) -> None:
    """Rows in the ``.db``, bytes at ``artifacts/{thread}/{run}/{id}.{ext}`` (D7, D11)."""
    from vdreamer.artifacts import ArtifactRecord

    db = tmp_path / "dev.db"
    record = ArtifactRecord(
        artifact_id="art_1",
        run_id="run_1",
        thread_id="t1",
        kind="json",
        name="result",
        mime_type="application/json",
        size_bytes=2,
        preview={"body": {}},
    )
    async with open_sqlite_state(db) as state:
        await state.artifacts.put(record, b"{}")
        assert state.artifacts.object_path(record) == tmp_path / "artifacts/t1/run_1/art_1.json"

    async with open_sqlite_state(db) as state:
        reloaded = await state.artifacts.get("art_1")
        assert reloaded == record
        assert await state.artifacts.read_bytes(record) == b"{}"
        await state.artifacts.put(record, None)  # the lifecycle deleted the object (A9)
        assert await state.artifacts.read_bytes(record) is None
        assert await state.artifacts.get("missing") is None


def test_artifact_paths_refuse_unsafe_segments(tmp_path: Path) -> None:
    import pytest

    from vdreamer.artifacts import ArtifactRecord
    from vdreamer.sqlite_state import SqliteArtifactStore

    store = SqliteArtifactStore(None, tmp_path)  # type: ignore[arg-type]
    bad = ArtifactRecord(
        artifact_id="art_1", run_id="..", thread_id="t1", kind="file", name="x", mime_type="x/y"
    )
    with pytest.raises(ValueError, match="unsafe"):
        store.object_path(bad)
