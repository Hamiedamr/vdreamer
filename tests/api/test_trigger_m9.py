"""Milestone 9 exit: ``POST /agents/{name}/trigger`` and the outbox it feeds (D15, A18, A26).

The trigger contract (``Idempotency-Key``, ``input_schema``, ``thread_id``,
``callback`` against ``CALLBACK_ALLOWLIST``), service tokens (A16), and end to
end: a triggered run's ``completed|failed|interrupt`` reach the outbox once per
event and are delivered — signed, at least once — to the allowlisted callback.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from _api import (
    CALLBACK_ALLOWLIST,
    OUTBOX_SETTINGS,
    SERVICE_TOKEN,
    Stack,
    assert_error,
    headers,
    new_thread,
    wait_for_status,
)
from fastapi.testclient import TestClient

from vdreamer.network import NetworkAllowlist
from vdreamer.outbox import (
    CHANNEL_CALLBACK,
    CHANNEL_WEBHOOK,
    EVENT_ID_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    HttpDeliverer,
    OutboxDispatcher,
    OutboxStatus,
    event_id_for,
    verify_signature,
)

CALLBACK = "https://ops.internal/hooks/runs"


def _trigger(
    client: TestClient,
    agent: str,
    body: dict[str, Any],
    key: str | None = "key-1",
    **kw: Any,
) -> httpx.Response:
    extra = {"Idempotency-Key": key} if key is not None else {}
    return client.post(
        f"/api/v1/agents/{agent}/trigger", json=body, headers=headers(**kw, **extra)
    )


def _service_headers(**extra: str) -> dict[str, str]:
    return headers(user=None, token=SERVICE_TOKEN, **extra)


# -- the trigger contract -----------------------------------------------------------------


def test_trigger_requires_an_idempotency_key(client: TestClient) -> None:
    response = _trigger(client, "batch", {"input": {"text": "x"}}, key=None)
    assert_error(response, 400, "missing_idempotency_key")
    blank = _trigger(client, "batch", {"input": {"text": "x"}}, key="  ")
    assert_error(blank, 400, "missing_idempotency_key")
    long = _trigger(client, "batch", {"input": {"text": "x"}}, key="k" * 256)
    assert_error(long, 422, "invalid_request")


def test_replayed_key_returns_the_original_run(client: TestClient, stack: Stack) -> None:
    first = _trigger(client, "batch", {"input": {"text": "hello", "n": 2}})
    assert first.status_code == 202, first.text
    created = first.json()
    assert created["status"] == "queued"
    assert first.headers["Location"] == f"/api/v1/runs/{created['run_id']}"
    assert "Idempotent-Replayed" not in first.headers

    # A18: the retry — even with a different body — is the first run, not a second one.
    again = _trigger(client, "batch", {"input": {"text": "other"}})
    assert again.status_code == 200, again.text
    assert again.json() == created
    assert again.headers["Idempotent-Replayed"] == "true"

    threads = client.get("/api/v1/threads", headers=headers()).json()["threads"]
    assert [t["thread_id"] for t in threads] == [created["thread_id"]]
    run = asyncio.run(stack.queue.get(created["run_id"]))
    assert run is not None and run.headless is True
    assert run.input == {"text": "hello", "n": 2}  # validated against input_schema


def test_the_key_is_scoped_to_caller_and_agent(client: TestClient) -> None:
    alice = _trigger(client, "batch", {"input": {"text": "a"}}).json()
    bob = _trigger(client, "batch", {"input": {"text": "b"}}, user="bob").json()
    form = _trigger(client, "form", {"input": {"text": "f"}}).json()
    assert len({alice["run_id"], bob["run_id"], form["run_id"]}) == 3


def test_trigger_validates_agent_mode_and_input(client: TestClient) -> None:
    assert_error(_trigger(client, "echo", {"input": {"text": "x"}}), 400, "mode_not_supported")
    assert_error(_trigger(client, "nope", {"input": {}}), 404, "agent_not_found")
    bad = _trigger(client, "batch", {"input": {"n": "many"}})
    error = assert_error(bad, 422, "invalid_input")
    assert {tuple(e["loc"]) for e in error["details"]["errors"]} == {("text",), ("n",)}
    reserved = _trigger(
        client, "batch",
        {"input": {"text": "x"}, "config": {"configurable": {"thread_id": "t"}}},
    )
    assert_error(reserved, 422, "invalid_request")
    # Nothing was created by any refused request: the key is still free.
    assert _trigger(client, "batch", {"input": {"text": "x"}}).status_code == 202


def test_trigger_continues_an_owned_thread(client: TestClient) -> None:
    thread_id = new_thread(client, agent="form")
    response = _trigger(client, "form", {"input": {"text": "x"}, "thread_id": thread_id})
    assert response.status_code == 202, response.text
    assert response.json()["thread_id"] == thread_id

    # Another user's thread, or another agent's, is not found.
    foreign = _trigger(
        client, "form", {"input": {"text": "x"}, "thread_id": thread_id}, key="k2", user="bob"
    )
    assert_error(foreign, 404, "thread_not_found")
    other = new_thread(client, agent="echo")
    wrong = _trigger(client, "form", {"input": {"text": "x"}, "thread_id": other}, key="k3")
    assert_error(wrong, 404, "thread_not_found")

    # One active run per thread (D12): a second key on the busy thread conflicts.
    busy = _trigger(client, "form", {"input": {"text": "y"}, "thread_id": thread_id}, key="k4")
    assert_error(busy, 409, "run_conflict")


@pytest.mark.parametrize(
    "callback",
    [
        "https://evil.example.com/hook",  # off the host globs, not resolving into the networks
        "https://callbacks.invalid/hook",  # does not resolve at all (RFC 2606)
        "https://10.30.0.1/hook",  # an IP outside the allowed network
        "ftp://ops.internal/hook",  # not http(s)
        "https://user:pw@ops.internal/hook",  # credentials in the URL (A23)
    ],
)
def test_callbacks_must_be_on_the_allowlist(client: TestClient, callback: str) -> None:
    response = _trigger(client, "batch", {"input": {"text": "x"}, "callback": callback})
    assert_error(response, 422, "callback_not_allowed")


def test_allowlisted_callbacks_are_stored_on_the_run(client: TestClient, stack: Stack) -> None:
    for key, callback in (("a", CALLBACK), ("b", "http://10.20.1.5:8080/cb")):
        response = _trigger(
            client, "batch", {"input": {"text": "x"}, "callback": callback}, key=key
        )
        assert response.status_code == 202, response.text
        run = asyncio.run(stack.queue.get(response.json()["run_id"]))
        assert run is not None and run.callback == callback


# -- service tokens (A16) -------------------------------------------------------------------


def test_a_service_token_triggers_and_reads_its_own_runs(client: TestClient) -> None:
    response = client.post(
        "/api/v1/agents/sitecheck/trigger",
        json={"input": {"site_id": "S-1"}},
        headers=_service_headers(**{"Idempotency-Key": "mail-42"}),
    )
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    run = client.get(f"/api/v1/runs/{run_id}", headers=_service_headers())
    assert run.status_code == 200, run.text
    # The run belongs to service:<name>; an end user does not see it.
    assert_error(client.get(f"/api/v1/runs/{run_id}", headers=headers()), 404, "run_not_found")
    # Access roles are vouched for by the token: a role-gated agent is open to it.
    assert client.post(
        "/api/v1/agents/sitecheck/trigger",
        json={"input": {"site_id": "S-1"}},
        headers=_service_headers(**{"Idempotency-Key": "mail-42"}),
    ).status_code == 200


def test_a_service_token_opens_nothing_else(client: TestClient) -> None:
    for method, path in (
        ("GET", "/api/v1/threads"),
        ("GET", "/api/v1/agents"),
        ("GET", "/api/v1/interrupts"),
        ("POST", "/api/v1/agents/batch/threads"),
    ):
        response = client.request(method, path, headers=_service_headers())
        assert_error(response, 401, "unauthorized")
    wrong = client.post(
        "/api/v1/agents/batch/trigger",
        json={"input": {"text": "x"}},
        headers=headers(user=None, token="not-a-service-token-0123456789", **{
            "Idempotency-Key": "k"
        }),
    )
    assert_error(wrong, 401, "unauthorized")


# -- end to end: the run's events reach the outbox and the callback --------------------------


def _entries(stack: Stack, run_id: str) -> list[Any]:
    assert stack.outbox is not None
    return asyncio.run(stack.outbox.entries(run_id=run_id))


class _Receiver:
    """An allowlisted callback receiver that dedups on ``event_id`` (as A26 asks)."""

    def __init__(self, fail_first: int = 0) -> None:
        self.requests: list[httpx.Request] = []
        self.fail_first = fail_first
        self.seen: dict[str, dict[str, Any]] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if len(self.requests) <= self.fail_first:
            return httpx.Response(503)
        body = json.loads(request.content)
        self.seen.setdefault(request.headers[EVENT_ID_HEADER], body)
        return httpx.Response(204)


def _dispatcher(stack: Stack, receiver: _Receiver) -> OutboxDispatcher:
    assert stack.outbox is not None
    transport = httpx.MockTransport(receiver.handler)
    allowlist = NetworkAllowlist.parse(CALLBACK_ALLOWLIST)

    def http() -> HttpDeliverer:
        return HttpDeliverer(
            secret=OUTBOX_SETTINGS.secret,
            allowlist=allowlist,
            require_allowlist=True,
            client=httpx.AsyncClient(transport=transport),
        )

    return OutboxDispatcher(
        stack.outbox,
        {CHANNEL_CALLBACK: http(), CHANNEL_WEBHOOK: http()},
        backoff_base_s=0.0,
    )


@pytest.mark.smoke  # floor/latest pins (A41): the M9 exit criterion end to end
def test_a_completed_run_calls_back_once_signed(live: TestClient, live_stack: Stack) -> None:
    response = _trigger(live, "batch", {"input": {"text": "hi"}, "callback": CALLBACK})
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    wait_for_status(live, run_id, "completed")

    (entry,) = _entries(live_stack, run_id)
    assert entry.channel == CHANNEL_CALLBACK and entry.kind == "completed"
    assert entry.event_id == event_id_for(run_id, "completed", "completed")

    receiver = _Receiver(fail_first=1)  # the first try fails: retried, not lost
    dispatcher = _dispatcher(live_stack, receiver)

    async def deliver() -> None:
        assert await dispatcher.deliver_due() == 1
        assert await dispatcher.deliver_due() == 1
        assert await dispatcher.deliver_due() == 0
        await dispatcher.aclose()

    asyncio.run(deliver())
    assert len(receiver.requests) == 2
    request = receiver.requests[-1]
    assert str(request.url) == CALLBACK
    assert verify_signature(
        str(OUTBOX_SETTINGS.secret),
        request.headers[TIMESTAMP_HEADER],
        request.content,
        request.headers[SIGNATURE_HEADER],
    )
    body = receiver.seen[entry.event_id]
    assert body["event"] == "completed" and body["status"] == "completed"
    assert body["run_id"] == run_id and body["agent"] == "batch"
    (delivered,) = _entries(live_stack, run_id)
    assert delivered.status is OutboxStatus.DELIVERED and delivered.attempts == 2

    # Replaying the key re-triggers nothing and records nothing new.
    assert _trigger(live, "batch", {"input": {"text": "hi"}}).status_code == 200
    assert len(_entries(live_stack, run_id)) == 1


def test_a_headless_approval_notifies_then_reject_calls_back_failed(
    live: TestClient, live_stack: Stack
) -> None:
    response = live.post(
        "/api/v1/agents/sitecheck/trigger",
        json={"input": {"site_id": "S-9"}, "callback": CALLBACK},
        headers=_service_headers(**{"Idempotency-Key": "mail-9"}),
    )
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    wait_for_status(live, run_id, "awaiting_approval", token=SERVICE_TOKEN, user=None)

    inbox = live.get("/api/v1/interrupts", headers=headers(user="sam", roles=("site_approver",)))
    (record,) = [r for r in inbox.json()["interrupts"] if r["run_id"] == run_id]
    interrupt_id = record["interrupt_id"]

    entries = _entries(live_stack, run_id)
    assert {(e.channel, e.kind) for e in entries} == {
        (CHANNEL_CALLBACK, "interrupt"),
        (CHANNEL_WEBHOOK, "interrupt"),
    }
    assert len({e.event_id for e in entries}) == 1  # one event, one id on every channel
    block = entries[0].payload["interrupt"]
    assert block["review_url"] == f"https://web.internal/review/{interrupt_id}"
    assert block["approver_role"] == "site_approver"
    assert "payload" not in block  # the link, never the interrupt's payload
    webhook = next(e for e in entries if e.channel == CHANNEL_WEBHOOK)
    assert webhook.target == OUTBOX_SETTINGS.webhook_url

    resolved = live.post(
        f"/api/v1/interrupts/{interrupt_id}/resolve",
        json={"action": "reject", "expected_version": record["version"]},
        headers=headers(user="sam", roles=("site_approver",)),
    )
    assert resolved.status_code == 200, resolved.text
    wait_for_status(live, run_id, "rejected", token=SERVICE_TOKEN, user=None)

    failed = [e for e in _entries(live_stack, run_id) if e.kind == "failed"]
    assert len(failed) == 1
    assert failed[0].channel == CHANNEL_CALLBACK
    assert failed[0].payload["status"] == "rejected"

    receiver = _Receiver()
    dispatcher = _dispatcher(live_stack, receiver)

    async def deliver() -> None:
        await dispatcher.deliver_due()
        await dispatcher.aclose()

    asyncio.run(deliver())
    events = sorted(body["event"] for body in receiver.seen.values())
    assert events == ["failed", "interrupt"]  # the webhook shares the interrupt's event_id
    assert len(receiver.requests) == 3
