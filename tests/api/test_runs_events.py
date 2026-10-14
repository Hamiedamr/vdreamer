"""Runs (§2.5, §3.2) and their SSE streams (§3.3, A3, A22)."""

from __future__ import annotations

import asyncio

import pytest
from _api import (
    Stack,
    assert_error,
    events_of,
    headers,
    new_thread,
    read_sse,
    start_run,
    wait_for_status,
)
from fastapi.testclient import TestClient

from vdreamer.eventlog import EventHub, InMemoryEventLog
from vdreamer.events import EventEnvelope, EventType
from vdreamer.gateway.sse import (
    HEARTBEAT_FRAME,
    LastEventIdError,
    choose_thread_run,
    format_event,
    parse_last_event_id,
    run_event_stream,
)
from vdreamer.queue import InMemoryRunQueue, RunState

# -- create / get / conflicts ------------------------------------------------------------


def test_create_run_is_202_and_stores_identity_and_config(
    client: TestClient, stack: Stack
) -> None:
    thread_id = new_thread(client)
    response = client.post(
        f"/api/v1/threads/{thread_id}/runs",
        json={
            "input": {"text": "hi"},
            "config": {"configurable": {"model": "x"}, "recursion_limit": 9},
        },
        headers=headers(roles=("analyst",)),
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued" and body["thread_id"] == thread_id
    assert response.headers["Location"] == f"/api/v1/runs/{body['run_id']}"

    record = asyncio.run(stack.queue.get(body["run_id"]))
    assert record is not None
    assert record.user_id == "alice" and record.roles == ("analyst",)
    assert record.config == {"configurable": {"model": "x"}, "recursion_limit": 9}

    got = client.get(f"/api/v1/runs/{body['run_id']}", headers=headers()).json()
    assert got["status"] == "queued" and got["agent"] == "echo" and got["attempt"] == 0


def test_second_run_on_a_busy_thread_is_409_run_conflict(client: TestClient) -> None:
    thread_id = new_thread(client)
    first = start_run(client, thread_id, {"input": {"text": "a"}})
    error = assert_error(
        client.post(f"/api/v1/threads/{thread_id}/runs", json={}, headers=headers()),
        409,
        "run_conflict",
    )
    assert error["details"] == {"active_run_id": first, "reason": "thread_has_active_run"}
    # wait_for_idle queues behind running only — a queued run ahead still conflicts.
    assert_error(
        client.post(
            f"/api/v1/threads/{thread_id}/runs?wait_for_idle=true", json={}, headers=headers()
        ),
        409,
        "run_conflict",
    )


def test_wait_for_idle_queues_behind_a_running_run(client: TestClient, stack: Stack) -> None:
    thread_id = new_thread(client)
    first = start_run(client, thread_id, {"input": {"text": "a"}})
    claimed = asyncio.run(stack.queue.claim("w1"))
    assert claimed is not None and claimed.id == first
    response = client.post(
        f"/api/v1/threads/{thread_id}/runs?wait_for_idle=true", json={}, headers=headers()
    )
    assert response.status_code == 202
    assert response.json()["status"] == "queued"


def test_create_run_validation(client: TestClient) -> None:
    form = new_thread(client, agent="form")
    error = assert_error(
        client.post(
            f"/api/v1/threads/{form}/runs", json={"input": {"n": "x"}}, headers=headers()
        ),
        422,
        "invalid_input",
    )
    locs = sorted(tuple(item["loc"]) for item in error["details"]["errors"])
    assert locs == [("n",), ("text",)]
    assert (
        client.post(
            f"/api/v1/threads/{form}/runs", json={"input": {"text": "ok"}}, headers=headers()
        ).status_code
        == 202
    )

    echo = new_thread(client)
    error = assert_error(
        client.post(
            f"/api/v1/threads/{echo}/runs",
            json={"config": {"configurable": {"thread_id": "t", "__pregel_x": 1}}},
            headers=headers(),
        ),
        422,
        "invalid_request",
    )
    assert error["details"]["reserved_keys"] == ["__pregel_x", "thread_id"]
    assert_error(
        client.post(f"/api/v1/threads/{echo}/runs", json={"extra": 1}, headers=headers()),
        422,
        "invalid_request",
    )
    assert_error(
        client.post("/api/v1/threads/ghost/runs", json={}, headers=headers()),
        404,
        "thread_not_found",
    )


def test_create_run_rechecks_access_roles(client: TestClient) -> None:
    thread_id = new_thread(client, agent="secret", roles=("netops",))
    assert_error(
        client.post(f"/api/v1/threads/{thread_id}/runs", json={}, headers=headers()),
        403,
        "forbidden_role",
    )


def test_runs_are_owner_scoped(client: TestClient) -> None:
    run_id = start_run(client, new_thread(client), {"input": {}})
    assert_error(client.get(f"/api/v1/runs/{run_id}", headers=headers(user="bob")), 404,
                 "run_not_found")
    assert_error(client.get("/api/v1/runs/run_ghost", headers=headers()), 404, "run_not_found")
    assert_error(
        client.post(f"/api/v1/runs/{run_id}/cancel", headers=headers(user="bob")),
        404,
        "run_not_found",
    )


# -- cancel ------------------------------------------------------------------------------


def test_cancel_a_queued_run_writes_the_final_event(client: TestClient, stack: Stack) -> None:
    thread_id = new_thread(client)
    run_id = start_run(client, thread_id, {"input": {}})
    response = client.post(f"/api/v1/runs/{run_id}/cancel", headers=headers())
    assert response.status_code == 202
    assert response.json()["status"] == "cancelled"
    # No worker held the run, so the gateway authored its last word.
    (event,) = asyncio.run(stack.log.read(run_id))
    assert event.type is EventType.RUN_STATUS and event.payload["status"] == "cancelled"
    assert_error(
        client.post(f"/api/v1/runs/{run_id}/cancel", headers=headers()), 409, "run_not_active"
    )
    # The thread is free again.
    assert start_run(client, thread_id, {"input": {}})


def test_cancel_a_running_run_leaves_the_event_to_its_worker(
    client: TestClient, stack: Stack
) -> None:
    run_id = start_run(client, new_thread(client), {"input": {}})
    asyncio.run(stack.queue.claim("w1"))
    response = client.post(f"/api/v1/runs/{run_id}/cancel", headers=headers())
    assert response.status_code == 202 and response.json()["status"] == "cancelled"
    assert asyncio.run(stack.log.read(run_id)) == []  # the worker drains and writes it


# -- SSE ---------------------------------------------------------------------------------


@pytest.mark.smoke  # floor/latest pins (A41): FastAPI + LangGraph together
def test_run_events_replay_then_close_on_terminal_status(live: TestClient) -> None:
    thread_id = new_thread(live)
    run_id = start_run(live, thread_id, {"input": {"text": "hi"}})
    wait_for_status(live, run_id, "completed")
    response, frames = read_sse(live, f"/api/v1/runs/{run_id}/events")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["Cache-Control"] == "no-cache"
    assert response.headers["X-Accel-Buffering"] == "no"
    events = events_of(frames)
    assert [e["type"] for e in events] == ["run.status", "progress", "run.status"]
    assert [e["payload"].get("status") for e in events if e["type"] == "run.status"] == [
        "running",
        "completed",
    ]
    for frame, event in zip([f for f in frames if f.data], events, strict=True):
        assert frame.id == f"{run_id}:{event['seq']}"  # SSE id = {run_id}:{seq}
        assert event["v"] == "1.0" and event["thread_id"] == thread_id
        assert event["agent"] == "echo"
    assert [e["seq"] for e in events] == [1, 2, 3]


def test_last_event_id_resumes_after_seq(live: TestClient) -> None:
    run_id = start_run(live, new_thread(live), {"input": {"text": "hi"}})
    wait_for_status(live, run_id, "completed")
    _, frames = read_sse(
        live, f"/api/v1/runs/{run_id}/events", **{"Last-Event-ID": f"{run_id}:2"}
    )
    assert [e["seq"] for e in events_of(frames)] == [3]
    _, frames = read_sse(live, f"/api/v1/runs/{run_id}/events", **{"Last-Event-ID": "1"})
    assert [e["seq"] for e in events_of(frames)] == [2, 3]


def test_last_event_id_for_another_run_is_400(live: TestClient) -> None:
    run_id = start_run(live, new_thread(live), {"input": {}})
    response, _ = read_sse(
        live, f"/api/v1/runs/{run_id}/events", **{"Last-Event-ID": "run_other:3"}
    )
    assert_error(response, 400, "invalid_last_event_id")


def test_thread_events_join_the_latest_run(live: TestClient) -> None:
    thread_id = new_thread(live)
    response, _ = read_sse(live, f"/api/v1/threads/{thread_id}/events")
    assert_error(response, 404, "run_not_found")

    first = start_run(live, thread_id, {"input": {"text": "one"}})
    wait_for_status(live, first, "completed")
    second = start_run(live, thread_id, {"input": {"text": "two"}})
    wait_for_status(live, second, "completed")
    # A Last-Event-ID from the older run replays the joined (latest) run in full.
    _, frames = read_sse(
        live, f"/api/v1/threads/{thread_id}/events", **{"Last-Event-ID": f"{first}:3"}
    )
    events = events_of(frames)
    assert {e["run_id"] for e in events} == {second}
    assert [e["seq"] for e in events] == [1, 2, 3]


def test_run_events_owner_scoped(live: TestClient) -> None:
    run_id = start_run(live, new_thread(live), {"input": {}})
    response, _ = read_sse(live, f"/api/v1/runs/{run_id}/events", user="bob")
    assert_error(response, 404, "run_not_found")


def test_parse_last_event_id() -> None:
    assert parse_last_event_id(None, "r") == 0
    assert parse_last_event_id("", "r") == 0
    assert parse_last_event_id("r:7", "r") == 7
    assert parse_last_event_id("7", "r") == 7
    assert parse_last_event_id("run:with:colons:4", "run:with:colons") == 4
    for bad in ("other:1", "r:x", "r:-1"):
        with pytest.raises(LastEventIdError):
            parse_last_event_id(bad, "r")


def test_choose_thread_run_prefers_the_run_in_flight() -> None:
    async def scenario() -> None:
        queue = InMemoryRunQueue()
        done = await queue.enqueue("a", {}, thread_id="t", run_id="r1")
        await queue.claim("w")
        await queue.finish(done.id, RunState.COMPLETED)
        latest = await queue.enqueue("a", {}, thread_id="t", run_id="r2")
        assert choose_thread_run(await queue.list_for_thread("t")) == latest
        await queue.claim("w")
        await queue.enqueue("a", {}, thread_id="t", run_id="r3", wait_for_idle=True)
        chosen = choose_thread_run(await queue.list_for_thread("t"))
        assert chosen is not None and chosen.id == "r2"  # running beats queued
        assert choose_thread_run([]) is None

    asyncio.run(scenario())


def _status(run_id: str, status: str) -> EventEnvelope:
    return EventEnvelope(
        type=EventType.RUN_STATUS,
        run_id=run_id,
        thread_id="t",
        agent="a",
        seq=0,
        payload={"status": status, "attempt": 1},
    )


async def test_stream_tails_live_with_heartbeats_and_stays_open_on_approval() -> None:
    queue = InMemoryRunQueue()
    log = InMemoryEventLog(EventHub())
    run = await queue.enqueue("a", {}, thread_id="t", run_id="r1")
    await queue.claim("w")
    frames: list[str] = []

    async def consume() -> None:
        async for frame in run_event_stream(
            run=run,
            after_seq=0,
            reader=log,
            hub=log.hub,
            get_run=queue.get,
            heartbeat_s=0.02,
        ):
            frames.append(frame)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.07)  # silence → heartbeats
    assert HEARTBEAT_FRAME in frames
    await log.write(_status("r1", "running"))
    await log.write(_status("r1", "awaiting_approval"))  # not an end: the run resumes
    await asyncio.sleep(0.05)
    assert not task.done()
    await log.write(_status("r1", "completed"))
    await asyncio.wait_for(task, 1.0)
    data = [frame for frame in frames if frame.startswith("id:")]
    assert [frame.split("\n", 1)[0] for frame in data] == ["id: r1:1", "id: r1:2", "id: r1:3"]
    assert log.hub.subscriber_count == 0  # the subscription was released


async def test_stream_ends_when_the_row_is_terminal_without_an_event() -> None:
    queue = InMemoryRunQueue()
    log = InMemoryEventLog()
    run = await queue.enqueue("a", {}, thread_id="t", run_id="r1")
    await queue.cancel(run.id)  # e.g. a sweep that wrote nothing
    frames = [
        frame
        async for frame in run_event_stream(
            run=run, after_seq=0, reader=log, hub=log.hub, get_run=queue.get, heartbeat_s=5.0
        )
    ]
    assert frames == []


def test_format_event_is_one_data_line() -> None:
    envelope = _status("r1", "running")
    envelope.seq = 4
    envelope.payload["note"] = "multi\nline"
    frame = format_event(envelope)
    lines = frame.split("\n")
    assert lines[0] == "id: r1:4"
    assert lines[1].startswith("data: {") and lines[2:] == ["", ""]
