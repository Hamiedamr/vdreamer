"""The outbox (PRD §4.3, A26): notifications and run callbacks, at least once.

Three things leave the platform, and all three go through one table:

- **run callbacks** — a ``/trigger`` run's ``callback`` URL hears ``completed``,
  ``failed`` or ``interrupt`` (A26). Every terminal status but ``completed``
  (``rejected``, ``expired``, ``cancelled``) is ``failed`` here; the body's
  ``status`` says which;
- **webhook notifications** — a headless or role-gated interrupt POSTs to
  ``VDREAMER_OUTBOX_WEBHOOK_URL``;
- **email notifications** — the same interrupt goes through an SMTP relay.

The notification carries the deep link ``{WEB_BASE_URL}/review/{interrupt_id}``
and never the interrupt's payload: the reviewer opens the link, authenticates,
and reads it through ``GET /interrupts``.

The write side is :class:`OutboxEventSink`, a decorator over an event sink: it
adds the entries *before* the inner write, so a crash can duplicate a
notification but never lose one. Entry ids and ``event_id`` derive from
``(run, kind, interrupt or status)``, so a recovered attempt that re-emits the
event adds nothing the second time. The read side is :class:`OutboxDispatcher`:
it claims due entries with a lease (a dispatcher that dies mid-delivery lets
the lease lapse and another re-claims the entry — no leader election), delivers
them, and retries failures with exponential backoff until ``max_attempts``,
then parks them ``dead``. Delivery is at-least-once; receivers dedup on
``event_id``.

HTTP bodies are signed with ``VDREAMER_OUTBOX_SECRET`` (HMAC-SHA256 over
``"{timestamp}.{body}"``, see :func:`sign`). Callback URLs must be on
``CALLBACK_ALLOWLIST`` both when the run is triggered and again when the entry
is delivered; the operator's own webhook is checked against it when it is set.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import enum
import hashlib
import hmac
import json
import logging
import os
import smtplib
import ssl
import time
import urllib.parse
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from email.message import EmailMessage
from typing import Any, Final, Protocol

import httpx

from vdreamer import metrics
from vdreamer.eventlog import EventSink, json_text
from vdreamer.events import EventEnvelope, EventType
from vdreamer.network import (
    CALLBACK_ALLOWLIST_ENV,
    EndpointError,
    EndpointUnavailable,
    NetworkAllowlist,
    check_url,
    default_port,
)
from vdreamer.queue import TERMINAL_STATES, RunRecord, _load_async_connection_class

_log = logging.getLogger(__name__)

# -- configuration ------------------------------------------------------------------

OUTBOX_WEBHOOK_URL_ENV: Final = "VDREAMER_OUTBOX_WEBHOOK_URL"
OUTBOX_SECRET_ENV: Final = "VDREAMER_OUTBOX_SECRET"
WEB_BASE_URL_ENV: Final = "WEB_BASE_URL"
SMTP_HOST_ENV: Final = "VDREAMER_SMTP_HOST"
SMTP_PORT_ENV: Final = "VDREAMER_SMTP_PORT"
SMTP_FROM_ENV: Final = "VDREAMER_SMTP_FROM"
SMTP_TO_ENV: Final = "VDREAMER_SMTP_TO"
SMTP_USER_ENV: Final = "VDREAMER_SMTP_USER"
SMTP_PASSWORD_ENV: Final = "VDREAMER_SMTP_PASSWORD"
SMTP_STARTTLS_ENV: Final = "VDREAMER_SMTP_STARTTLS"

MIN_SECRET_LENGTH: Final = 16

DEFAULT_SMTP_PORT: Final = 25


class OutboxConfigError(RuntimeError):
    """The outbox's environment configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class SmtpSettings:
    """An SMTP relay and who it mails (A23: the password is env-only)."""

    host: str
    sender: str
    recipients: tuple[str, ...]
    port: int = DEFAULT_SMTP_PORT
    username: str | None = None
    password: str | None = None
    starttls: bool = False


@dataclass(frozen=True)
class OutboxSettings:
    """Which notification channels exist, and how links and signatures are built.

    Callbacks need none of it: they are always available (they still need
    ``CALLBACK_ALLOWLIST``); ``secret`` signs them when it is set.
    """

    webhook_url: str | None = None
    secret: str | None = None
    web_base_url: str | None = None
    smtp: SmtpSettings | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> OutboxSettings:
        env = environ if environ is not None else os.environ

        def get(name: str) -> str | None:
            return env.get(name, "").strip() or None

        webhook_url = get(OUTBOX_WEBHOOK_URL_ENV)
        secret = get(OUTBOX_SECRET_ENV)
        if secret is not None and len(secret) < MIN_SECRET_LENGTH:
            raise OutboxConfigError(
                f"{OUTBOX_SECRET_ENV} must be at least {MIN_SECRET_LENGTH} characters"
            )
        if webhook_url is not None:
            if secret is None:
                raise OutboxConfigError(
                    f"{OUTBOX_WEBHOOK_URL_ENV} is set but {OUTBOX_SECRET_ENV} is not: "
                    "webhooks are HMAC-signed"
                )
            try:
                check_url(
                    OUTBOX_WEBHOOK_URL_ENV,
                    webhook_url,
                    NetworkAllowlist.from_env(env, name=CALLBACK_ALLOWLIST_ENV),
                )
            except EndpointError as exc:
                raise OutboxConfigError(str(exc)) from exc
        web_base_url = get(WEB_BASE_URL_ENV)
        if web_base_url is not None:
            try:
                check_url(WEB_BASE_URL_ENV, web_base_url, None)
            except EndpointError as exc:
                raise OutboxConfigError(str(exc)) from exc
        return cls(
            webhook_url=webhook_url,
            secret=secret,
            web_base_url=web_base_url,
            smtp=_smtp_from_env(env),
        )


def _smtp_from_env(env: Mapping[str, str]) -> SmtpSettings | None:
    def get(name: str) -> str | None:
        return env.get(name, "").strip() or None

    host = get(SMTP_HOST_ENV)
    others = (SMTP_PORT_ENV, SMTP_FROM_ENV, SMTP_TO_ENV, SMTP_USER_ENV, SMTP_PASSWORD_ENV)
    if host is None:
        stray = [name for name in others if get(name) is not None]
        if stray:
            raise OutboxConfigError(f"{', '.join(stray)} set without {SMTP_HOST_ENV}")
        return None
    sender = get(SMTP_FROM_ENV)
    recipients = tuple(
        address.strip() for address in (get(SMTP_TO_ENV) or "").split(",") if address.strip()
    )
    if sender is None or not recipients:
        raise OutboxConfigError(f"{SMTP_HOST_ENV} needs {SMTP_FROM_ENV} and {SMTP_TO_ENV}")
    try:
        port = int(get(SMTP_PORT_ENV) or DEFAULT_SMTP_PORT)
    except ValueError:
        raise OutboxConfigError(f"{SMTP_PORT_ENV} must be a port number") from None
    username = get(SMTP_USER_ENV)
    password = get(SMTP_PASSWORD_ENV)
    if password is not None and username is None:
        raise OutboxConfigError(f"{SMTP_PASSWORD_ENV} is set without {SMTP_USER_ENV}")
    allowlist = NetworkAllowlist.from_env(env)  # the relay is a declared endpoint too (§5.2)
    if allowlist is not None:
        try:
            allowlist.check_config(host)
        except EndpointError as exc:
            raise OutboxConfigError(f"{SMTP_HOST_ENV}: {exc}") from exc
    return SmtpSettings(
        host=host,
        sender=sender,
        recipients=recipients,
        port=port,
        username=username,
        password=password,
        starttls=(get(SMTP_STARTTLS_ENV) or "").lower() in {"1", "true", "yes", "on"},
    )


# -- entries ------------------------------------------------------------------------

CHANNEL_CALLBACK: Final = "callback"
CHANNEL_WEBHOOK: Final = "webhook"
CHANNEL_EMAIL: Final = "email"

KIND_COMPLETED: Final = "completed"
KIND_FAILED: Final = "failed"
KIND_INTERRUPT: Final = "interrupt"

_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(state.value for state in TERMINAL_STATES)


class OutboxStatus(enum.StrEnum):
    PENDING = "pending"
    DELIVERED = "delivered"
    DEAD = "dead"


@dataclass(frozen=True)
class OutboxEntry:
    """One delivery to one channel. Timestamps are epoch seconds (wall clock).

    ``id`` is ``{event_id}:{channel}``; ``event_id`` is what receivers dedup on.
    ``attempts`` counts finished tries; ``lease_expires_at`` is set while a
    dispatcher holds the entry.
    """

    id: str
    event_id: str
    run_id: str
    channel: str
    target: str
    kind: str
    payload: Mapping[str, Any]
    created_at: float
    status: OutboxStatus = OutboxStatus.PENDING
    attempts: int = 0
    next_attempt_at: float = 0.0
    lease_expires_at: float | None = None
    delivered_at: float | None = None
    last_error: str | None = None


def event_id_for(run_id: str, kind: str, discriminator: str) -> str:
    """The stable id of one notifiable event: re-emitting it re-derives the same id."""
    digest = hashlib.sha256(f"{run_id}\0{kind}\0{discriminator}".encode()).hexdigest()
    return f"evt_{digest[:32]}"


def review_url(web_base_url: str, interrupt_id: str) -> str:
    """The deep link into the Web UI's review page (§4.3)."""
    return f"{web_base_url.rstrip('/')}/review/{urllib.parse.quote(interrupt_id, safe='')}"


def _body(event_id: str, kind: str, envelope: EventEnvelope, status: str) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event": kind,
        "ts": envelope.ts,
        "run_id": envelope.run_id,
        "thread_id": envelope.thread_id,
        "agent": envelope.agent,
        "status": status,
    }


def _interrupt_block(data: Mapping[str, Any], settings: OutboxSettings) -> dict[str, Any]:
    interrupt_id = str(data["interrupt_id"])
    block: dict[str, Any] = {
        "interrupt_id": interrupt_id,
        "kind": data.get("kind"),
        "scope": data.get("scope"),
        "approver_role": data.get("approver_role"),
        "actions": list(data.get("actions") or ()),
        "expires_at": data.get("expires_at"),
    }
    block = {key: value for key, value in block.items() if value is not None}
    if settings.web_base_url is not None:
        block["review_url"] = review_url(settings.web_base_url, interrupt_id)
    return block


def _entry(
    event_id: str,
    run_id: str,
    channel: str,
    target: str,
    kind: str,
    body: dict[str, Any],
    now: float,
) -> OutboxEntry:
    return OutboxEntry(
        id=f"{event_id}:{channel}",
        event_id=event_id,
        run_id=run_id,
        channel=channel,
        target=target,
        kind=kind,
        payload=body,
        created_at=now,
        next_attempt_at=now,
    )


def watches(envelope: EventEnvelope) -> bool:
    """Whether ``envelope`` can produce entries — a cheap check before any lookup."""
    if envelope.type is EventType.INTERRUPT_REQUESTED:
        return True
    return (
        envelope.type is EventType.RUN_STATUS
        and str(envelope.payload.get("status")) in _TERMINAL_STATUSES
    )


def entries_for(
    envelope: EventEnvelope, run: RunRecord, settings: OutboxSettings, *, now: float
) -> list[OutboxEntry]:
    """The deliveries ``envelope`` owes: the run's callback, and the notifications."""
    if envelope.type is EventType.RUN_STATUS:
        status = str(envelope.payload.get("status"))
        if run.callback is None or status not in _TERMINAL_STATUSES:
            return []
        kind = KIND_COMPLETED if status == "completed" else KIND_FAILED
        event_id = event_id_for(run.id, kind, status)
        body = _body(event_id, kind, envelope, status)
        error = envelope.payload.get("error")
        if isinstance(error, Mapping):
            body["error"] = dict(error)
        return [_entry(event_id, run.id, CHANNEL_CALLBACK, run.callback, kind, body, now)]

    if envelope.type is EventType.INTERRUPT_REQUESTED:
        data = envelope.payload
        if not data.get("interrupt_id"):
            return []
        event_id = event_id_for(run.id, KIND_INTERRUPT, str(data["interrupt_id"]))
        body = _body(event_id, KIND_INTERRUPT, envelope, "awaiting_approval")
        body["interrupt"] = _interrupt_block(data, settings)
        entries: list[OutboxEntry] = []
        if run.callback is not None:
            entries.append(
                _entry(event_id, run.id, CHANNEL_CALLBACK, run.callback, KIND_INTERRUPT, body, now)
            )
        if run.headless or data.get("approver_role"):
            if settings.webhook_url is not None:
                entries.append(
                    _entry(
                        event_id, run.id, CHANNEL_WEBHOOK, settings.webhook_url, KIND_INTERRUPT,
                        body, now,
                    )
                )
            if settings.smtp is not None:
                entries.append(
                    _entry(
                        event_id, run.id, CHANNEL_EMAIL, ",".join(settings.smtp.recipients),
                        KIND_INTERRUPT, body, now,
                    )
                )
        return entries
    return []


# -- stores -------------------------------------------------------------------------


class OutboxStore(abc.ABC):
    """Where entries wait for delivery."""

    @abc.abstractmethod
    async def add(self, entries: Sequence[OutboxEntry]) -> int:
        """Insert the entries; one whose id exists is skipped. Returns how many were new."""

    @abc.abstractmethod
    async def claim(
        self, *, channels: Sequence[str], limit: int, lease_s: float
    ) -> list[OutboxEntry]:
        """Lease up to ``limit`` due entries of ``channels``, oldest first.

        Due = ``pending``, ``next_attempt_at`` reached, and not leased (or the
        lease lapsed — the dispatcher that held it died).
        """

    @abc.abstractmethod
    async def complete(self, entry_id: str) -> None:
        """The entry was delivered."""

    @abc.abstractmethod
    async def fail(self, entry_id: str, error: str, *, retry_at: float | None) -> None:
        """The attempt failed: retry at ``retry_at``, or park the entry ``dead`` when ``None``."""

    @abc.abstractmethod
    async def lag(self) -> float:
        """Seconds since the oldest ``pending`` entry was created; ``0`` when none."""

    @abc.abstractmethod
    async def get(self, entry_id: str) -> OutboxEntry | None:
        """One entry, or ``None``."""

    @abc.abstractmethod
    async def entries(
        self, *, status: OutboxStatus | None = None, run_id: str | None = None
    ) -> list[OutboxEntry]:
        """Entries oldest first, filtered by ``status`` and ``run_id``."""


class InMemoryOutboxStore(OutboxStore):
    """Dict-backed store — the tests, and the twin the SQLite store persists."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._entries: dict[str, OutboxEntry] = {}
        self._lock = asyncio.Lock()
        self._clock = clock

    async def add(self, entries: Sequence[OutboxEntry]) -> int:
        async with self._lock:
            added = 0
            for entry in entries:
                if entry.id not in self._entries:
                    self._entries[entry.id] = entry
                    added += 1
            return added

    async def claim(
        self, *, channels: Sequence[str], limit: int, lease_s: float
    ) -> list[OutboxEntry]:
        async with self._lock:
            now = self._clock()
            due = sorted(
                (
                    entry
                    for entry in self._entries.values()
                    if entry.status is OutboxStatus.PENDING
                    and entry.channel in channels
                    and entry.next_attempt_at <= now
                    and (entry.lease_expires_at is None or entry.lease_expires_at <= now)
                ),
                key=lambda entry: (entry.created_at, entry.id),
            )[:limit]
            claimed: list[OutboxEntry] = []
            for entry in due:
                leased = replace(entry, lease_expires_at=now + lease_s)
                self._entries[entry.id] = leased
                claimed.append(leased)
            return claimed

    async def complete(self, entry_id: str) -> None:
        async with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None or entry.status is not OutboxStatus.PENDING:
                return
            self._entries[entry_id] = replace(
                entry,
                status=OutboxStatus.DELIVERED,
                attempts=entry.attempts + 1,
                lease_expires_at=None,
                delivered_at=self._clock(),
                last_error=None,
            )

    async def fail(self, entry_id: str, error: str, *, retry_at: float | None) -> None:
        async with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None or entry.status is not OutboxStatus.PENDING:
                return
            self._entries[entry_id] = replace(
                entry,
                status=OutboxStatus.DEAD if retry_at is None else OutboxStatus.PENDING,
                attempts=entry.attempts + 1,
                next_attempt_at=entry.next_attempt_at if retry_at is None else retry_at,
                lease_expires_at=None,
                last_error=error,
            )

    async def lag(self) -> float:
        async with self._lock:
            pending = [
                entry.created_at
                for entry in self._entries.values()
                if entry.status is OutboxStatus.PENDING
            ]
            return max(0.0, self._clock() - min(pending)) if pending else 0.0

    async def get(self, entry_id: str) -> OutboxEntry | None:
        return self._entries.get(entry_id)

    async def entries(
        self, *, status: OutboxStatus | None = None, run_id: str | None = None
    ) -> list[OutboxEntry]:
        return sorted(
            (
                entry
                for entry in self._entries.values()
                if (status is None or entry.status is status)
                and (run_id is None or entry.run_id == run_id)
            ),
            key=lambda entry: (entry.created_at, entry.id),
        )


#: The ``outbox`` table. Idempotent; ``vdreamer migrate`` applies it.
OUTBOX_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS outbox (
    id               text        PRIMARY KEY,
    event_id         text        NOT NULL,
    run_id           text        NOT NULL,
    channel          text        NOT NULL,
    target           text        NOT NULL,
    kind             text        NOT NULL,
    payload          jsonb       NOT NULL DEFAULT '{}'::jsonb,
    status           text        NOT NULL DEFAULT 'pending',
    attempts         int         NOT NULL DEFAULT 0,
    next_attempt_at  timestamptz NOT NULL DEFAULT now(),
    lease_expires_at timestamptz,
    created_at       timestamptz NOT NULL DEFAULT now(),
    delivered_at     timestamptz,
    last_error       text
);

CREATE INDEX IF NOT EXISTS outbox_due_idx ON outbox (status, next_attempt_at, created_at);
CREATE INDEX IF NOT EXISTS outbox_run_idx ON outbox (run_id)
"""

_OUTBOX_COLUMNS: Final[str] = (
    "id, event_id, run_id, channel, target, kind, payload, status, attempts, "
    "next_attempt_at, lease_expires_at, created_at, delivered_at, last_error"
)

_ADD_SQL: Final[str] = """
INSERT INTO outbox (
    id, event_id, run_id, channel, target, kind, payload, created_at, next_attempt_at
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s::jsonb,
    to_timestamp(%s::double precision), to_timestamp(%s::double precision)
)
ON CONFLICT (id) DO NOTHING
"""

#: ``SKIP LOCKED`` lets concurrent dispatchers each take different entries.
_CLAIM_SQL: Final[str] = f"""
UPDATE outbox
SET lease_expires_at = to_timestamp(%s::double precision)
WHERE id IN (
    SELECT id
    FROM outbox
    WHERE status = 'pending'
      AND channel = ANY(%s::text[])
      AND next_attempt_at <= to_timestamp(%s::double precision)
      AND (lease_expires_at IS NULL OR lease_expires_at <= to_timestamp(%s::double precision))
    ORDER BY created_at, id
    LIMIT %s
    FOR UPDATE SKIP LOCKED
)
RETURNING {_OUTBOX_COLUMNS}
"""

_COMPLETE_SQL: Final[str] = """
UPDATE outbox
SET status = 'delivered', attempts = attempts + 1, lease_expires_at = NULL,
    delivered_at = to_timestamp(%s::double precision), last_error = NULL
WHERE id = %s AND status = 'pending'
"""

_FAIL_SQL: Final[str] = """
UPDATE outbox
SET status = CASE WHEN %s::double precision IS NULL THEN 'dead' ELSE 'pending' END,
    attempts = attempts + 1,
    next_attempt_at = COALESCE(to_timestamp(%s::double precision), next_attempt_at),
    lease_expires_at = NULL,
    last_error = %s
WHERE id = %s AND status = 'pending'
"""

_LAG_SQL: Final[str] = """
SELECT COALESCE(EXTRACT(EPOCH FROM (to_timestamp(%s::double precision) - MIN(created_at))), 0)
FROM outbox
WHERE status = 'pending'
"""

_GET_SQL: Final[str] = f"SELECT {_OUTBOX_COLUMNS} FROM outbox WHERE id = %s"

_LIST_SQL: Final[str] = f"""
SELECT {_OUTBOX_COLUMNS}
FROM outbox
WHERE (%s::text IS NULL OR status = %s::text)
  AND (%s::text IS NULL OR run_id = %s::text)
ORDER BY created_at, id
"""


def _ts(value: Any) -> float | None:
    return value.timestamp() if value is not None else None


class PostgresOutboxStore(OutboxStore):
    """The ``outbox`` table — short-lived autocommit connections, like the queue."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    @asynccontextmanager
    async def _conn(self) -> AsyncGenerator[Any]:
        conn_cls = _load_async_connection_class()
        conn = await conn_cls.connect(self._dsn, autocommit=True)
        try:
            yield conn
        finally:
            await conn.close()

    @staticmethod
    def _row_to_entry(row: Any) -> OutboxEntry:
        return OutboxEntry(
            id=str(row[0]),
            event_id=str(row[1]),
            run_id=str(row[2]),
            channel=str(row[3]),
            target=str(row[4]),
            kind=str(row[5]),
            payload=dict(row[6] or {}),
            status=OutboxStatus(str(row[7])),
            attempts=int(row[8]),
            next_attempt_at=row[9].timestamp(),
            lease_expires_at=_ts(row[10]),
            created_at=row[11].timestamp(),
            delivered_at=_ts(row[12]),
            last_error=row[13],
        )

    async def setup(self) -> None:
        """Create the table if missing (``migrate`` reuses :data:`OUTBOX_DDL`)."""
        async with self._conn() as conn:
            await conn.execute(OUTBOX_DDL)

    async def add(self, entries: Sequence[OutboxEntry]) -> int:
        added = 0
        async with self._conn() as conn:
            for entry in entries:
                cur = await conn.execute(
                    _ADD_SQL,
                    (
                        entry.id,
                        entry.event_id,
                        entry.run_id,
                        entry.channel,
                        entry.target,
                        entry.kind,
                        json_text(dict(entry.payload)),
                        entry.created_at,
                        entry.next_attempt_at,
                    ),
                )
                added += max(int(cur.rowcount), 0)
        return added

    async def claim(
        self, *, channels: Sequence[str], limit: int, lease_s: float
    ) -> list[OutboxEntry]:
        now = time.time()
        async with self._conn() as conn:
            cur = await conn.execute(
                _CLAIM_SQL, (now + lease_s, list(channels), now, now, limit)
            )
            rows = await cur.fetchall()
        return sorted(
            (self._row_to_entry(row) for row in rows), key=lambda e: (e.created_at, e.id)
        )

    async def complete(self, entry_id: str) -> None:
        async with self._conn() as conn:
            await conn.execute(_COMPLETE_SQL, (time.time(), entry_id))

    async def fail(self, entry_id: str, error: str, *, retry_at: float | None) -> None:
        async with self._conn() as conn:
            await conn.execute(_FAIL_SQL, (retry_at, retry_at, error, entry_id))

    async def lag(self) -> float:
        async with self._conn() as conn:
            cur = await conn.execute(_LAG_SQL, (time.time(),))
            row = await cur.fetchone()
        return max(0.0, float(row[0])) if row is not None else 0.0

    async def get(self, entry_id: str) -> OutboxEntry | None:
        async with self._conn() as conn:
            cur = await conn.execute(_GET_SQL, (entry_id,))
            row = await cur.fetchone()
        return self._row_to_entry(row) if row is not None else None

    async def entries(
        self, *, status: OutboxStatus | None = None, run_id: str | None = None
    ) -> list[OutboxEntry]:
        status_value = status.value if status is not None else None
        async with self._conn() as conn:
            cur = await conn.execute(_LIST_SQL, (status_value, status_value, run_id, run_id))
            rows = await cur.fetchall()
        return [self._row_to_entry(row) for row in rows]


# -- the write side -----------------------------------------------------------------


class RunLookup(Protocol):
    """What the sink needs from the run queue: a run's row (callback, headless)."""

    async def get(self, run_id: str) -> RunRecord | None: ...


class OutboxEventSink(EventSink):
    """An event sink that records what must leave the platform, then forwards.

    Wrap the worker's per-run sink *and* the gateway's event writer: terminal
    ``run.status`` events come from both (a cancel, a headless reject and an
    interrupt expiry are written by the gateway). The entries are added before
    the inner write, so a failure in between repeats the event, never drops it.
    """

    def __init__(
        self,
        inner: EventSink,
        outbox: OutboxStore,
        runs: RunLookup,
        settings: OutboxSettings,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._inner = inner
        self._outbox = outbox
        self._runs = runs
        self._settings = settings
        self._clock = clock

    async def write(self, envelope: EventEnvelope) -> None:
        if watches(envelope):
            run = await self._runs.get(envelope.run_id)
            if run is not None:
                entries = entries_for(envelope, run, self._settings, now=self._clock())
                if entries:
                    await self._outbox.add(entries)
        await self._inner.write(envelope)

    async def close(self) -> None:
        await self._inner.close()


# -- delivery -----------------------------------------------------------------------

SIGNATURE_HEADER: Final = "X-Vdreamer-Signature"
TIMESTAMP_HEADER: Final = "X-Vdreamer-Timestamp"
EVENT_ID_HEADER: Final = "X-Vdreamer-Event-Id"
EVENT_HEADER: Final = "X-Vdreamer-Event"


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """``sha256=<hex>``: HMAC-SHA256 over ``"{timestamp}.{body}"`` with the shared secret."""
    mac = hmac.new(secret.encode("utf-8"), timestamp.encode("ascii") + b"." + body, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def verify_signature(secret: str, timestamp: str, body: bytes, signature: str) -> bool:
    """Constant-time check of a delivery's signature (receivers should also bound the age
    of ``timestamp``)."""
    return hmac.compare_digest(sign(secret, timestamp, body), signature)


def encode_body(payload: Mapping[str, Any]) -> bytes:
    """The exact bytes a delivery sends (and signs): compact, key-sorted JSON."""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str).encode("utf-8")


class DeliveryError(Exception):
    """A delivery failed; the dispatcher retries it."""


class PermanentDeliveryError(DeliveryError):
    """A delivery that retrying cannot fix (a target off the allowlist): the entry goes ``dead``."""


class Deliverer(Protocol):
    async def deliver(self, entry: OutboxEntry) -> None:
        """Send the entry or raise :class:`DeliveryError`."""
        ...

    async def aclose(self) -> None: ...


DELIVERY_TIMEOUT_S: Final = 10.0


class HttpDeliverer:
    """POSTs an entry's JSON body, signed when a secret is set.

    ``require_allowlist`` is the callback rule (A26): with no allowlist
    nothing is sent. The operator's webhook passes ``False`` and is checked only
    when an allowlist exists. Redirects are not followed and proxy environment
    variables are ignored: the target is the allowlisted one.
    """

    def __init__(
        self,
        *,
        secret: str | None,
        allowlist: NetworkAllowlist | None,
        require_allowlist: bool,
        timeout_s: float = DELIVERY_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret = secret
        self._allowlist = allowlist
        self._require_allowlist = require_allowlist
        self._timeout_s = timeout_s
        self._client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False)
        self._clock = clock

    async def _check_target(self, entry: OutboxEntry) -> None:
        if self._require_allowlist and self._allowlist is None:
            raise PermanentDeliveryError(f"{CALLBACK_ALLOWLIST_ENV} is not set: callbacks are off")
        try:
            parsed = check_url(entry.channel, entry.target, self._allowlist)
        except EndpointError as exc:
            raise PermanentDeliveryError(str(exc)) from exc
        if self._allowlist is not None:
            try:
                await self._allowlist.check_connect(parsed.hostname or "", default_port(parsed))
            except EndpointUnavailable as exc:
                raise DeliveryError(str(exc)) from exc

    async def deliver(self, entry: OutboxEntry) -> None:
        await self._check_target(entry)
        body = encode_body(entry.payload)
        timestamp = str(int(self._clock()))
        headers = {
            "Content-Type": "application/json",
            EVENT_ID_HEADER: entry.event_id,
            EVENT_HEADER: entry.kind,
            TIMESTAMP_HEADER: timestamp,
        }
        if self._secret is not None:
            headers[SIGNATURE_HEADER] = sign(self._secret, timestamp, body)
        try:
            response = await self._client.post(
                entry.target, content=body, headers=headers, timeout=self._timeout_s
            )
        except httpx.HTTPError as exc:
            raise DeliveryError(f"{type(exc).__name__}: {exc}") from exc
        if not response.is_success:
            raise DeliveryError(f"the receiver answered HTTP {response.status_code}")

    async def aclose(self) -> None:
        await self._client.aclose()


def render_email(entry: OutboxEntry, smtp: SmtpSettings) -> EmailMessage:
    """The notification mail: who, what, and the review link — never the payload."""
    interrupt = entry.payload.get("interrupt") or {}
    agent = entry.payload.get("agent", "")
    message = EmailMessage()
    message["From"] = smtp.sender
    message["To"] = ", ".join(smtp.recipients)
    message["Subject"] = f"[vdreamer] Approval needed: {agent}"
    message["Message-ID"] = f"<{entry.event_id}@vdreamer>"
    lines = [
        "An agent run is waiting for a decision.",
        "",
        f"Agent: {agent}",
        f"Run: {entry.run_id}",
        f"Interrupt: {interrupt.get('interrupt_id', '')}",
    ]
    for label, key in (("Approver role", "approver_role"), ("Scope", "scope"),
                       ("Expires", "expires_at")):
        if interrupt.get(key):
            lines.append(f"{label}: {interrupt[key]}")
    if interrupt.get("review_url"):
        lines += ["", f"Review: {interrupt['review_url']}"]
    message.set_content("\n".join(lines) + "\n")
    return message


class SmtpDeliverer:
    """Mails an entry through the SMTP relay (stdlib ``smtplib`` in a worker thread)."""

    def __init__(
        self,
        settings: SmtpSettings,
        *,
        timeout_s: float = DELIVERY_TIMEOUT_S,
        smtp_factory: Callable[..., smtplib.SMTP] = smtplib.SMTP,
    ) -> None:
        self._settings = settings
        self._timeout_s = timeout_s
        self._factory = smtp_factory

    def _send(self, message: EmailMessage) -> None:
        settings = self._settings
        with self._factory(settings.host, settings.port, timeout=self._timeout_s) as smtp:
            if settings.starttls:
                smtp.starttls(context=ssl.create_default_context())
            if settings.username is not None:
                smtp.login(settings.username, settings.password or "")
            smtp.send_message(message)

    async def deliver(self, entry: OutboxEntry) -> None:
        message = render_email(entry, self._settings)
        try:
            await asyncio.to_thread(self._send, message)
        except (smtplib.SMTPException, OSError) as exc:
            raise DeliveryError(f"{type(exc).__name__}: {exc}") from exc

    async def aclose(self) -> None:
        return None


#: Retry schedule: ``BACKOFF_BASE_S * 2**attempts``, capped.
BACKOFF_BASE_S: Final = 5.0
BACKOFF_MAX_S: Final = 3600.0

#: Tries before an entry is parked ``dead`` (about 40 minutes of retries at the default backoff).
MAX_DELIVERY_ATTEMPTS: Final = 10

DISPATCH_INTERVAL_S: Final = 1.0
DISPATCH_BATCH: Final = 20

#: A claimed entry is re-claimable after this long (a dead dispatcher's leftovers).
LEASE_S: Final = 60.0


class OutboxDispatcher:
    """Claims due entries, delivers them, and settles each (at-least-once).

    Any number of dispatchers can run against one store: claims are leases, so
    two never hold an entry at once, and one that dies mid-delivery only delays
    it by ``lease_s``.
    """

    def __init__(
        self,
        store: OutboxStore,
        deliverers: Mapping[str, Deliverer],
        *,
        interval_s: float = DISPATCH_INTERVAL_S,
        batch: int = DISPATCH_BATCH,
        lease_s: float = LEASE_S,
        max_attempts: int = MAX_DELIVERY_ATTEMPTS,
        backoff_base_s: float = BACKOFF_BASE_S,
        backoff_max_s: float = BACKOFF_MAX_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store
        self._deliverers = dict(deliverers)
        self._interval_s = interval_s
        self._batch = batch
        self._lease_s = lease_s
        self._max_attempts = max_attempts
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self._clock = clock

    def retry_delay(self, attempts: int) -> float:
        """Seconds to wait after ``attempts`` finished tries."""
        return min(self._backoff_base_s * float(2 ** min(attempts, 30)), self._backoff_max_s)

    async def _deliver(self, entry: OutboxEntry) -> None:
        try:
            await self._deliverers[entry.channel].deliver(entry)
        except PermanentDeliveryError as exc:
            _log.warning("outbox %s (%s) is undeliverable: %s", entry.id, entry.channel, exc)
            await self._store.fail(entry.id, str(exc), retry_at=None)
        except Exception as exc:  # any failure is a retry, whatever the deliverer raised
            exhausted = entry.attempts + 1 >= self._max_attempts
            retry_at = None if exhausted else self._clock() + self.retry_delay(entry.attempts)
            _log.warning(
                "outbox %s (%s) failed, attempt %d: %s", entry.id, entry.channel,
                entry.attempts + 1, exc,
            )
            await self._store.fail(entry.id, f"{type(exc).__name__}: {exc}", retry_at=retry_at)
        else:
            await self._store.complete(entry.id)

    async def deliver_due(self) -> int:
        """One pass: deliver every due entry (concurrently); returns how many were tried."""
        claimed = await self._store.claim(
            channels=tuple(self._deliverers), limit=self._batch, lease_s=self._lease_s
        )
        await asyncio.gather(*(self._deliver(entry) for entry in claimed))
        metrics.OUTBOX_LAG.set(await self._store.lag())
        return len(claimed)

    async def run(self, stop: asyncio.Event) -> None:
        """Deliver every ``interval_s`` until ``stop`` is set; deliverers close at exit."""
        try:
            while not stop.is_set():
                tried = 0
                try:
                    tried = await self.deliver_due()
                except Exception:  # a failed pass is retried next interval
                    _log.exception("outbox dispatch failed")
                if tried >= self._batch:
                    continue
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), self._interval_s)
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        for deliverer in self._deliverers.values():
            with contextlib.suppress(Exception):
                await deliverer.aclose()


def build_dispatcher(
    store: OutboxStore,
    settings: OutboxSettings,
    callback_allowlist: NetworkAllowlist | None,
    **options: Any,
) -> OutboxDispatcher:
    """The dispatcher for ``settings``: callbacks always, webhook and email when configured."""
    deliverers: dict[str, Deliverer] = {
        CHANNEL_CALLBACK: HttpDeliverer(
            secret=settings.secret, allowlist=callback_allowlist, require_allowlist=True
        )
    }
    if settings.webhook_url is not None:
        deliverers[CHANNEL_WEBHOOK] = HttpDeliverer(
            secret=settings.secret, allowlist=callback_allowlist, require_allowlist=False
        )
    if settings.smtp is not None:
        deliverers[CHANNEL_EMAIL] = SmtpDeliverer(settings.smtp)
    return OutboxDispatcher(store, deliverers, **options)


__all__ = [
    "CHANNEL_CALLBACK",
    "CHANNEL_EMAIL",
    "CHANNEL_WEBHOOK",
    "OUTBOX_DDL",
    "DeliveryError",
    "HttpDeliverer",
    "InMemoryOutboxStore",
    "OutboxConfigError",
    "OutboxDispatcher",
    "OutboxEntry",
    "OutboxEventSink",
    "OutboxSettings",
    "OutboxStatus",
    "OutboxStore",
    "PermanentDeliveryError",
    "PostgresOutboxStore",
    "SmtpDeliverer",
    "SmtpSettings",
    "build_dispatcher",
    "entries_for",
    "event_id_for",
    "review_url",
    "sign",
    "verify_signature",
]
