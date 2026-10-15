"""Milestone 9: the outbox (§4.4, A26) — config, entries, dedup, delivery, retries, stores.

The exit criterion — "a callback is delivered at least once, with dedup" — is
pinned here at the unit level (a lapsed lease redelivers the same ``event_id``;
re-emitting an event adds nothing) and end to end in ``api/test_trigger_m9.py``.
"""

from __future__ import annotations

import asyncio
import json
import os
import smtplib
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from vdreamer import metrics
from vdreamer.eventlog import InMemoryEventSink
from vdreamer.events import EventEnvelope, EventType
from vdreamer.network import NetworkAllowlist
from vdreamer.outbox import (
    CHANNEL_CALLBACK,
    CHANNEL_EMAIL,
    CHANNEL_WEBHOOK,
    EVENT_HEADER,
    EVENT_ID_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    DeliveryError,
    HttpDeliverer,
    InMemoryOutboxStore,
    OutboxConfigError,
    OutboxDispatcher,
    OutboxEntry,
    OutboxEventSink,
    OutboxSettings,
    OutboxStatus,
    PermanentDeliveryError,
    PostgresOutboxStore,
    SmtpDeliverer,
    SmtpSettings,
    build_dispatcher,
    encode_body,
    entries_for,
    event_id_for,
    render_email,
    review_url,
    sign,
    verify_signature,
)
from vdreamer.queue import RunRecord, RunState

SECRET = "outbox-secret-0123456789"
CALLBACK = "https://ops.internal/cb"
ALLOWLIST = NetworkAllowlist.parse("*.internal")
SMTP = SmtpSettings(host="relay.internal", sender="vd@x.internal", recipients=("a@x", "b@x"))
SETTINGS = OutboxSettings(
    webhook_url="https://hooks.internal/vd",
    secret=SECRET,
    web_base_url="https://web.internal/app/",
    smtp=SMTP,
)


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _run(*, callback: str | None = CALLBACK, headless: bool = True) -> RunRecord:
    return RunRecord(
        id="run_1",
        thread_id="thread_1",
        agent="sitecheck",
        input={},
        status=RunState.RUNNING,
        attempt=1,
        claimed_by="w",
        lease_expires_at=None,
        created_at=0.0,
        updated_at=0.0,
        callback=callback,
        headless=headless,
    )


def _status(status: str, **payload: Any) -> EventEnvelope:
    return EventEnvelope(
        type=EventType.RUN_STATUS,
        run_id="run_1",
        thread_id="thread_1",
        agent="sitecheck",
        seq=9,
        payload={"status": status, **payload},
    )


def _requested(interrupt_id: str = "int_1", approver_role: str | None = "site_approver") -> (
    EventEnvelope
):
    payload: dict[str, Any] = {
        "interrupt_id": interrupt_id,
        "kind": "site_verdict",
        "actions": ["approve", "reject"],
        "payload": {"secret_site_data": 1},
        "expires_at": "2026-10-16T00:00:00+00:00",
    }
    if approver_role is not None:
        payload["approver_role"] = approver_role
    return EventEnvelope(
        type=EventType.INTERRUPT_REQUESTED,
        run_id="run_1",
        thread_id="thread_1",
        agent="sitecheck",
        seq=4,
        payload=payload,
    )


def _entry(
    entry_id: str = "e1", *, channel: str = CHANNEL_CALLBACK, target: str = CALLBACK,
    created_at: float = 1_000.0,
) -> OutboxEntry:
    return OutboxEntry(
        id=f"{entry_id}:{channel}",
        event_id=entry_id,
        run_id="run_1",
        channel=channel,
        target=target,
        kind="completed",
        payload={"event_id": entry_id, "event": "completed"},
        created_at=created_at,
        next_attempt_at=created_at,
    )


# -- configuration ----------------------------------------------------------------------


def test_settings_from_an_empty_environment_have_no_channels() -> None:
    settings = OutboxSettings.from_env({})
    assert settings == OutboxSettings()


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"VDREAMER_OUTBOX_WEBHOOK_URL": "https://hooks.internal/x"}, "HMAC-signed"),
        ({"VDREAMER_OUTBOX_SECRET": "short"}, "at least 16"),
        (
            {
                "VDREAMER_OUTBOX_WEBHOOK_URL": "https://hooks.example.com/x",
                "VDREAMER_OUTBOX_SECRET": SECRET,
                "CALLBACK_ALLOWLIST": "*.internal",
            },
            "not on",
        ),
        (
            {"VDREAMER_OUTBOX_WEBHOOK_URL": "https://u:p@hooks.internal/x",
             "VDREAMER_OUTBOX_SECRET": SECRET},
            "credentials",
        ),
        ({"WEB_BASE_URL": "web.internal"}, "WEB_BASE_URL"),
        ({"VDREAMER_SMTP_TO": "a@x"}, "without VDREAMER_SMTP_HOST"),
        ({"VDREAMER_SMTP_HOST": "relay.internal"}, "needs VDREAMER_SMTP_FROM"),
        (
            {"VDREAMER_SMTP_HOST": "relay.internal", "VDREAMER_SMTP_FROM": "a@x",
             "VDREAMER_SMTP_TO": "b@x", "VDREAMER_SMTP_PORT": "smtp"},
            "port number",
        ),
        (
            {"VDREAMER_SMTP_HOST": "relay.internal", "VDREAMER_SMTP_FROM": "a@x",
             "VDREAMER_SMTP_TO": "b@x", "VDREAMER_SMTP_PASSWORD": "pw"},
            "without VDREAMER_SMTP_USER",
        ),
        (
            {"VDREAMER_SMTP_HOST": "relay.example.com", "VDREAMER_SMTP_FROM": "a@x",
             "VDREAMER_SMTP_TO": "b@x", "NETWORK_ALLOWLIST": "*.internal"},
            "VDREAMER_SMTP_HOST",
        ),
    ],
)
def test_incomplete_or_unsafe_settings_are_refused(env: dict[str, str], message: str) -> None:
    with pytest.raises(OutboxConfigError, match=message):
        OutboxSettings.from_env(env)


def test_full_settings_from_the_environment() -> None:
    settings = OutboxSettings.from_env(
        {
            "VDREAMER_OUTBOX_WEBHOOK_URL": "https://hooks.internal/vd",
            "VDREAMER_OUTBOX_SECRET": SECRET,
            "WEB_BASE_URL": "https://web.internal",
            "CALLBACK_ALLOWLIST": "*.internal",
            "VDREAMER_SMTP_HOST": "relay.internal",
            "VDREAMER_SMTP_PORT": "587",
            "VDREAMER_SMTP_FROM": "vd@x.internal",
            "VDREAMER_SMTP_TO": "a@x, b@x",
            "VDREAMER_SMTP_USER": "vd",
            "VDREAMER_SMTP_PASSWORD": "pw",
            "VDREAMER_SMTP_STARTTLS": "yes",
        }
    )
    assert settings.webhook_url == "https://hooks.internal/vd"
    assert settings.smtp == SmtpSettings(
        host="relay.internal", sender="vd@x.internal", recipients=("a@x", "b@x"), port=587,
        username="vd", password="pw", starttls=True,
    )


# -- entries ------------------------------------------------------------------------------


def test_review_url_is_the_deep_link() -> None:
    assert review_url("https://web.internal/app/", "int 1/x") == (
        "https://web.internal/app/review/int%201%2Fx"
    )


@pytest.mark.parametrize(
    ("status", "kind"),
    [("completed", "completed"), ("failed", "failed"), ("rejected", "failed"),
     ("expired", "failed"), ("cancelled", "failed")],
)
def test_terminal_statuses_owe_one_callback(status: str, kind: str) -> None:
    error = {"code": "boom", "message": "x"} if status == "failed" else None
    envelope = _status(status, **({"error": error} if error else {}))
    (entry,) = entries_for(envelope, _run(), SETTINGS, now=5.0)
    assert (entry.channel, entry.kind, entry.target) == (CHANNEL_CALLBACK, kind, CALLBACK)
    assert entry.event_id == event_id_for("run_1", kind, status)
    assert entry.id == f"{entry.event_id}:callback"
    assert entry.payload["status"] == status and entry.payload["event"] == kind
    assert entry.payload.get("error") == error
    assert entry.created_at == entry.next_attempt_at == 5.0


def test_non_terminal_statuses_and_runs_without_callback_owe_nothing() -> None:
    assert entries_for(_status("running"), _run(), SETTINGS, now=0) == []
    assert entries_for(_status("awaiting_approval"), _run(), SETTINGS, now=0) == []
    assert entries_for(_status("completed"), _run(callback=None), SETTINGS, now=0) == []
    progress = EventEnvelope(
        type=EventType.PROGRESS, run_id="run_1", thread_id="t", agent="a", seq=1,
        payload={"message": "x"},
    )
    assert entries_for(progress, _run(), SETTINGS, now=0) == []


def test_a_headless_interrupt_goes_to_every_channel_with_one_event_id() -> None:
    entries = entries_for(_requested(), _run(), SETTINGS, now=0)
    channels = {entry.channel for entry in entries}
    assert channels == {CHANNEL_CALLBACK, CHANNEL_WEBHOOK, CHANNEL_EMAIL}
    assert len({entry.event_id for entry in entries}) == 1
    email = next(entry for entry in entries if entry.channel == CHANNEL_EMAIL)
    assert email.target == "a@x,b@x"
    block = entries[0].payload["interrupt"]
    assert block == {
        "interrupt_id": "int_1",
        "kind": "site_verdict",
        "approver_role": "site_approver",
        "actions": ["approve", "reject"],
        "expires_at": "2026-10-16T00:00:00+00:00",
        "review_url": "https://web.internal/app/review/int_1",
    }
    assert "secret_site_data" not in json.dumps(entries[0].payload)


def test_who_is_notified_of_an_interrupt() -> None:
    # A chat run's role-gated interrupt notifies; an initiator-only one does not.
    chat = _run(callback=None, headless=False)
    gated = entries_for(_requested(), chat, SETTINGS, now=0)
    assert {entry.channel for entry in gated} == {CHANNEL_WEBHOOK, CHANNEL_EMAIL}
    assert entries_for(_requested(approver_role=None), chat, SETTINGS, now=0) == []
    # No channel configured: only the run's own callback.
    (only,) = entries_for(_requested(), _run(), OutboxSettings(), now=0)
    assert only.channel == CHANNEL_CALLBACK
    assert "review_url" not in only.payload["interrupt"]


class _Runs:
    def __init__(self, run: RunRecord | None) -> None:
        self.run = run
        self.lookups = 0

    async def get(self, run_id: str) -> RunRecord | None:
        self.lookups += 1
        return self.run


async def test_the_sink_records_once_and_always_forwards() -> None:
    inner = InMemoryEventSink()
    store = InMemoryOutboxStore()
    runs = _Runs(_run())
    sink = OutboxEventSink(inner, store, runs, SETTINGS, clock=Clock())
    progress = EventEnvelope(
        type=EventType.PROGRESS, run_id="run_1", thread_id="t", agent="a", seq=1,
        payload={"message": "x"},
    )
    await sink.write(progress)
    assert runs.lookups == 0  # nothing notifiable: no run lookup at all
    # A recovered attempt re-emits the same events: same ids, nothing new (dedup).
    for _ in range(2):
        await sink.write(_requested())
        await sink.write(_status("completed"))
    assert len(await store.entries()) == 4  # 3 interrupt channels + 1 callback
    assert len(inner.events) == 5
    # A run the queue does not know is forwarded, not recorded.
    runs.run = None
    await sink.write(_status("failed"))
    assert len(await store.entries()) == 4 and len(inner.events) == 6


async def test_entries_are_added_before_the_event_is_written() -> None:
    class Failing(InMemoryEventSink):
        async def write(self, envelope: EventEnvelope) -> None:
            raise RuntimeError("log down")

    store = InMemoryOutboxStore()
    sink = OutboxEventSink(Failing(), store, _Runs(_run()), SETTINGS)
    with pytest.raises(RuntimeError):
        await sink.write(_status("completed"))
    assert len(await store.entries()) == 1  # a crash repeats a callback, never drops one


# -- signing ------------------------------------------------------------------------------


def test_signatures_round_trip_and_reject_tampering() -> None:
    body = encode_body({"b": 1, "a": [1, 2]})
    assert body == b'{"a":[1,2],"b":1}'
    signature = sign(SECRET, "1700000000", body)
    assert signature.startswith("sha256=")
    assert verify_signature(SECRET, "1700000000", body, signature)
    assert not verify_signature(SECRET, "1700000001", body, signature)
    assert not verify_signature(SECRET, "1700000000", body + b" ", signature)
    assert not verify_signature("another-secret-0123456", "1700000000", body, signature)


# -- HTTP delivery --------------------------------------------------------------------------


def _http(
    handler: Any, *, secret: str | None = SECRET, allowlist: NetworkAllowlist | None = ALLOWLIST,
    require_allowlist: bool = True,
) -> HttpDeliverer:
    return HttpDeliverer(
        secret=secret,
        allowlist=allowlist,
        require_allowlist=require_allowlist,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        clock=Clock(1_700_000_000.0),
    )


async def test_http_delivery_posts_the_signed_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    deliverer = _http(handler)
    entry = _entry()
    await deliverer.deliver(entry)
    await deliverer.aclose()
    (request,) = seen
    assert request.method == "POST" and str(request.url) == CALLBACK
    assert request.content == encode_body(entry.payload)
    assert request.headers[EVENT_ID_HEADER] == "e1"
    assert request.headers[EVENT_HEADER] == "completed"
    assert request.headers[TIMESTAMP_HEADER] == "1700000000"
    assert verify_signature(SECRET, "1700000000", request.content,
                            request.headers[SIGNATURE_HEADER])


async def test_unsigned_without_a_secret() -> None:
    seen: list[httpx.Request] = []
    deliverer = _http(lambda r: seen.append(r) or httpx.Response(204), secret=None)
    await deliverer.deliver(_entry())
    assert SIGNATURE_HEADER not in seen[0].headers


@pytest.mark.parametrize("status", [301, 404, 500, 503])
async def test_non_success_answers_are_retryable_failures(status: int) -> None:
    deliverer = _http(lambda r: httpx.Response(status, headers={"Location": "https://x.y/"}))
    with pytest.raises(DeliveryError, match=str(status)) as raised:
        await deliverer.deliver(_entry())
    assert not isinstance(raised.value, PermanentDeliveryError)


async def test_transport_errors_are_retryable_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(DeliveryError, match="ConnectError"):
        await _http(handler).deliver(_entry())


async def test_targets_off_the_allowlist_are_permanent_failures() -> None:
    ok = _http(lambda r: httpx.Response(200))
    with pytest.raises(PermanentDeliveryError):
        await ok.deliver(_entry(target="https://evil.example.com/cb"))
    off = _http(lambda r: httpx.Response(200), allowlist=None)
    with pytest.raises(PermanentDeliveryError, match="CALLBACK_ALLOWLIST"):
        await off.deliver(_entry())
    # The operator's webhook does not need an allowlist (it is checked when there is one).
    webhook = _http(lambda r: httpx.Response(200), allowlist=None, require_allowlist=False)
    await webhook.deliver(_entry(channel=CHANNEL_WEBHOOK, target="https://hooks.anywhere/x"))


# -- email delivery -------------------------------------------------------------------------


def test_the_email_carries_the_link_not_the_payload() -> None:
    (entry,) = [
        e for e in entries_for(_requested(), _run(), SETTINGS, now=0) if e.channel == CHANNEL_EMAIL
    ]
    message = render_email(entry, SMTP)
    assert message["To"] == "a@x, b@x" and message["From"] == "vd@x.internal"
    assert message["Subject"] == "[vdreamer] Approval needed: sitecheck"
    assert message["Message-ID"] == f"<{entry.event_id}@vdreamer>"
    text = message.get_content()
    assert "Review: https://web.internal/app/review/int_1" in text
    assert "Approver role: site_approver" in text
    assert "secret_site_data" not in text


class _FakeSmtp:
    instances: list[_FakeSmtp] = []
    fail: Exception | None = None

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.args = (host, port, timeout)
        self.calls: list[str] = []
        _FakeSmtp.instances.append(self)

    def __enter__(self) -> _FakeSmtp:
        return self

    def __exit__(self, *exc: object) -> None:
        self.calls.append("quit")

    def starttls(self, context: Any = None) -> None:
        self.calls.append("starttls")

    def login(self, user: str, password: str) -> None:
        self.calls.append(f"login:{user}")

    def send_message(self, message: Any) -> None:
        if _FakeSmtp.fail is not None:
            raise _FakeSmtp.fail
        self.calls.append(f"send:{message['To']}")


async def test_smtp_delivery_goes_through_the_relay() -> None:
    _FakeSmtp.instances.clear()
    _FakeSmtp.fail = None
    settings = SmtpSettings(
        host="relay.internal", sender="vd@x", recipients=("a@x",), port=587,
        username="vd", password="pw", starttls=True,
    )
    deliverer = SmtpDeliverer(settings, smtp_factory=_FakeSmtp)  # type: ignore[arg-type]
    await deliverer.deliver(_entry(channel=CHANNEL_EMAIL, target="a@x"))
    (smtp,) = _FakeSmtp.instances
    assert smtp.args[:2] == ("relay.internal", 587)
    assert smtp.calls == ["starttls", "login:vd", "send:a@x", "quit"]

    _FakeSmtp.fail = smtplib.SMTPRecipientsRefused({})
    with pytest.raises(DeliveryError, match="SMTPRecipientsRefused"):
        await deliverer.deliver(_entry(channel=CHANNEL_EMAIL, target="a@x"))
    _FakeSmtp.fail = None


# -- the dispatcher: at least once, retries, dead letters, leases ----------------------------


class _Scripted:
    """A deliverer that fails its first ``failures`` calls."""

    def __init__(self, failures: int = 0, error: Exception | None = None) -> None:
        self.failures = failures
        self.error = error or DeliveryError("HTTP 503")
        self.delivered: list[str] = []
        self.calls = 0
        self.closed = False

    async def deliver(self, entry: OutboxEntry) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        self.delivered.append(entry.event_id)

    async def aclose(self) -> None:
        self.closed = True


async def test_a_failed_delivery_is_retried_with_backoff_until_delivered() -> None:
    clock = Clock()
    store = InMemoryOutboxStore(clock=clock)
    await store.add([_entry()])
    deliverer = _Scripted(failures=2)
    dispatcher = OutboxDispatcher(
        store, {CHANNEL_CALLBACK: deliverer}, backoff_base_s=5.0, backoff_max_s=60.0, clock=clock
    )
    assert [dispatcher.retry_delay(n) for n in range(6)] == [5, 10, 20, 40, 60, 60]

    assert await dispatcher.deliver_due() == 1  # attempt 1 fails → retry in 5 s
    entry = await store.get("e1:callback")
    assert entry is not None and entry.attempts == 1 and entry.next_attempt_at == 1_005.0
    assert entry.last_error is not None and "503" in entry.last_error
    assert await dispatcher.deliver_due() == 0  # not due yet
    clock.now = 1_005.0
    assert await dispatcher.deliver_due() == 1  # attempt 2 fails → retry in 10 s
    # Each pass publishes the lag: the oldest pending entry is 5 s old.
    assert metrics.OUTBOX_LAG._value.get() == 5.0  # type: ignore[attr-defined]
    clock.now = 1_015.0
    assert await dispatcher.deliver_due() == 1  # attempt 3 delivers
    done = await store.get("e1:callback")
    assert done is not None and done.status is OutboxStatus.DELIVERED and done.attempts == 3
    assert done.delivered_at == 1_015.0 and done.last_error is None
    assert deliverer.delivered == ["e1"]
    assert await store.lag() == 0.0
    assert metrics.OUTBOX_LAG._value.get() == 0.0  # type: ignore[attr-defined]


async def test_an_entry_is_parked_dead_after_max_attempts() -> None:
    clock = Clock()
    store = InMemoryOutboxStore(clock=clock)
    await store.add([_entry()])
    dispatcher = OutboxDispatcher(
        store, {CHANNEL_CALLBACK: _Scripted(failures=99)}, max_attempts=3, backoff_base_s=0.0,
        clock=clock,
    )
    for _ in range(5):
        await dispatcher.deliver_due()
    (entry,) = await store.entries()
    assert entry.status is OutboxStatus.DEAD and entry.attempts == 3
    assert await store.entries(status=OutboxStatus.PENDING) == []


async def test_a_permanent_failure_is_dead_at_once() -> None:
    store = InMemoryOutboxStore()
    await store.add([_entry()])
    dispatcher = OutboxDispatcher(
        store, {CHANNEL_CALLBACK: _Scripted(failures=1, error=PermanentDeliveryError("off"))}
    )
    await dispatcher.deliver_due()
    (entry,) = await store.entries()
    assert entry.status is OutboxStatus.DEAD and entry.attempts == 1 and entry.last_error == "off"


async def test_a_lapsed_lease_redelivers_the_same_event_id() -> None:
    """At least once: a dispatcher that dies after sending (before settling) is re-done
    once its lease lapses; the receiver sees the same ``event_id`` and dedups."""
    clock = Clock()
    store = InMemoryOutboxStore(clock=clock)
    await store.add([_entry()])
    (claimed,) = await store.claim(channels=[CHANNEL_CALLBACK], limit=10, lease_s=60.0)
    # The first dispatcher sent it and died: nobody completes the claim.
    assert await store.claim(channels=[CHANNEL_CALLBACK], limit=10, lease_s=60.0) == []
    clock.now += 60.0
    deliverer = _Scripted()
    dispatcher = OutboxDispatcher(store, {CHANNEL_CALLBACK: deliverer}, clock=clock)
    assert await dispatcher.deliver_due() == 1
    assert deliverer.delivered == [claimed.event_id]
    # A stale settle from the dead dispatcher changes nothing.
    await store.fail(claimed.id, "late", retry_at=0.0)
    entry = await store.get(claimed.id)
    assert entry is not None and entry.status is OutboxStatus.DELIVERED


async def test_claims_are_oldest_first_per_channel_and_batched() -> None:
    store = InMemoryOutboxStore(clock=Clock(2_000.0))
    await store.add(
        [
            _entry("b", created_at=20.0),
            _entry("a", created_at=10.0),
            _entry("w", channel=CHANNEL_WEBHOOK, created_at=1.0),
            _entry("c", created_at=30.0),
        ]
    )
    assert await store.add([_entry("a", created_at=99.0)]) == 0  # the id exists: skipped
    first = await store.claim(channels=[CHANNEL_CALLBACK], limit=2, lease_s=60.0)
    assert [entry.event_id for entry in first] == ["a", "b"]
    rest = await store.claim(channels=[CHANNEL_CALLBACK], limit=10, lease_s=60.0)
    assert [entry.event_id for entry in rest] == ["c"]
    assert await store.lag() == 2_000.0 - 1.0


async def test_the_run_loop_delivers_and_closes_its_deliverers() -> None:
    store = InMemoryOutboxStore()
    await store.add([_entry("x"), _entry("y")])
    deliverer = _Scripted()
    dispatcher = OutboxDispatcher(store, {CHANNEL_CALLBACK: deliverer}, interval_s=0.01)
    stop = asyncio.Event()
    task = asyncio.create_task(dispatcher.run(stop))
    for _ in range(200):
        if len(deliverer.delivered) == 2:
            break
        await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(task, 2.0)
    assert sorted(deliverer.delivered) == ["x", "y"] and deliverer.closed


def test_build_dispatcher_wires_only_configured_channels() -> None:
    minimal = build_dispatcher(InMemoryOutboxStore(), OutboxSettings(), None)
    assert set(minimal._deliverers) == {CHANNEL_CALLBACK}
    full = build_dispatcher(InMemoryOutboxStore(), SETTINGS, ALLOWLIST)
    assert set(full._deliverers) == {CHANNEL_CALLBACK, CHANNEL_WEBHOOK, CHANNEL_EMAIL}
    asyncio.run(minimal.aclose())
    asyncio.run(full.aclose())


# -- persistence --------------------------------------------------------------------------


async def test_the_sqlite_outbox_survives_a_restart(tmp_path: Path) -> None:
    from vdreamer.sqlite_state import SqliteOutboxStore, open_state_db

    db_path = tmp_path / "state.db"
    db = await open_state_db(db_path)
    try:
        store = SqliteOutboxStore(db)
        await store.add([_entry("a"), _entry("b"), _entry("c")])
        await store.complete("a:callback")
        await store.fail("b:callback", "HTTP 503", retry_at=5.0)
        await store.claim(channels=[CHANNEL_CALLBACK], limit=10, lease_s=3600.0)  # leases c
    finally:
        await db.close()

    db = await open_state_db(db_path)
    try:
        reopened = SqliteOutboxStore(db)
        await reopened.load()
        by_id = {entry.event_id: entry for entry in await reopened.entries()}
        assert by_id["a"].status is OutboxStatus.DELIVERED
        assert by_id["b"].attempts == 1 and by_id["b"].last_error == "HTTP 503"
        assert by_id["b"].payload == {"event_id": "b", "event": "completed"}
        # The previous process's lease is void: c is claimable straight away.
        claimed = await reopened.claim(channels=[CHANNEL_CALLBACK], limit=10, lease_s=60.0)
        assert {entry.event_id for entry in claimed} == {"b", "c"}
    finally:
        await db.close()


def test_postgres_sql_shapes() -> None:
    from vdreamer import outbox

    assert "on conflict (id) do nothing" in outbox._ADD_SQL.lower()
    assert "for update skip locked" in outbox._CLAIM_SQL.lower()
    assert "status = 'pending'" in outbox._COMPLETE_SQL
    assert "status = 'pending'" in outbox._FAIL_SQL
    assert "%s::jsonb" in outbox._ADD_SQL  # psycopg cannot adapt a bare dict


def test_postgres_row_mapping() -> None:
    from datetime import UTC, datetime

    now = datetime(2026, 10, 15, tzinfo=UTC)
    row = (
        "e:callback", "e", "run_1", "callback", CALLBACK, "completed", {"a": 1}, "dead", 3,
        now, None, now, None, "HTTP 500",
    )
    entry = PostgresOutboxStore._row_to_entry(row)
    assert entry.status is OutboxStatus.DEAD and entry.attempts == 3
    assert entry.payload == {"a": 1} and entry.created_at == now.timestamp()
    assert entry.lease_expires_at is None and entry.last_error == "HTTP 500"


@pytest.mark.skipif(
    os.environ.get("VDREAMER_TEST_PG_DSN") is None,
    reason="VDREAMER_TEST_PG_DSN is not set — live Postgres coverage only",
)
async def test_postgres_outbox_live() -> None:
    import time

    store = PostgresOutboxStore(os.environ["VDREAMER_TEST_PG_DSN"])
    await store.setup()
    run_id = f"run_{uuid.uuid4().hex}"
    now = time.time()
    entry = OutboxEntry(
        id=f"{run_id}:callback", event_id=run_id, run_id=run_id, channel=CHANNEL_CALLBACK,
        target=CALLBACK, kind="completed", payload={"run_id": run_id}, created_at=now - 1,
        next_attempt_at=now - 1,
    )
    assert await store.add([entry]) == 1
    assert await store.add([entry]) == 0
    claimed = [e for e in await store.claim(channels=[CHANNEL_CALLBACK], limit=50, lease_s=60)
               if e.run_id == run_id]
    assert len(claimed) == 1 and claimed[0].payload == {"run_id": run_id}
    await store.fail(entry.id, "HTTP 503", retry_at=now - 1)
    await store.complete(entry.id)
    (stored,) = await store.entries(run_id=run_id)
    assert stored.status is OutboxStatus.DELIVERED and stored.attempts == 2
