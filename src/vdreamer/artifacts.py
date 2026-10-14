"""Artifacts: the SDK helpers, the write order, and the stores (PRD §3.4, D6–D8, A8, A9).

Agent code produces artifacts two ways (§3.4):

- ``emit_artifact(name, kind, data, metadata?, renderer?) -> ArtifactRef`` from
  any node or tool (``aemit_artifact`` is the awaitable twin for async code);
- ``return Artifact(kind=..., data=..., name=...)`` from a governed tool — the
  pipeline stores it and the model gets a short reference instead.

Each kind accepts specific data (``vdreamer.artifact_data``; no
auto-detection — anything else is a ``TypeError``). The write order is D7:
bytes to the object store at ``artifacts/{thread_id}/{run_id}/{artifact_id}.{ext}``,
*then* the metadata row, *then* the ``artifact.created`` record into the
run's stream — so an event never names an artifact the gateway cannot serve.

The worker binds its store for each run (:func:`bind_artifact_store`); with
no store bound (unit tests, a runner driven without a worker) the event still
flows but nothing is kept, and the event carries no URLs.

Stores pair a metadata table with an :class:`~vdreamer.objects.ObjectStore`:

- :class:`InMemoryArtifactStore` — tests;
- ``sqlite_state.SqliteArtifactStore`` — ``vdreamer dev``: rows in the dev
  ``.db``, bytes on the local filesystem next to it (D11);
- :class:`PostgresArtifactStore` — ``serve``/``worker``: the ``artifacts``
  table plus MinIO;
- :class:`UnavailableArtifactStore` — ``serve``/``worker`` with no MinIO
  configured: every lookup misses (``404``), every write fails loudly.

A record whose object is gone (the 30-day lifecycle, A9) reads back with
``read_bytes`` → ``None``; the gateway answers ``410 artifact_expired``.
"""

from __future__ import annotations

import abc
import asyncio
import concurrent.futures
import contextlib
import contextvars
import enum
import json
import time
import uuid
from collections.abc import AsyncGenerator, Coroutine, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, TypeVar

import pydantic
from langchain_core.messages.tool import ToolOutputMixin

from vdreamer.objects import (
    ARTIFACTS_PREFIX,
    InMemoryObjectStore,
    ObjectStore,
    check_key,
)

_T = TypeVar("_T")

#: The gateway's API prefix — artifact URLs in events point at its routes.
API_PREFIX: Final[str] = "/api/v1"


def artifact_urls(artifact_id: str) -> tuple[str, str]:
    """``(view_url, download_url)`` — the gateway routes (§3.4)."""
    return (
        f"{API_PREFIX}/artifacts/{artifact_id}/raw",
        f"{API_PREFIX}/artifacts/{artifact_id}/download",
    )


class ArtifactKind(enum.StrEnum):
    """The five artifact kinds (PRD §3.4)."""

    TABLE = "table"
    IMAGE = "image"
    HTML = "html"
    JSON = "json"
    FILE = "file"


class ArtifactRef(pydantic.BaseModel):
    """What ``emit_artifact`` returns: identity and display hints, never bytes."""

    artifact_id: str
    name: str
    kind: ArtifactKind
    mime_type: str | None = None
    size_bytes: int | None = None


@dataclass
class Artifact(ToolOutputMixin):
    """A governed tool's artifact return value (§3.4: ``return Artifact(...)``).

    Returning one (or a list of them) from a tool stores it like
    ``emit_artifact`` and hands the model a short reference naming the
    ``artifact_id`` instead of the data. It is a LangChain ``ToolOutputMixin``,
    so the tool layer passes it through unstringified to vdreamer's pipeline.
    """

    kind: ArtifactKind
    data: Any
    name: str
    metadata: dict[str, Any] | None = None
    renderer: str | None = None
    mime_type: str | None = None
    filename: str | None = None


@dataclass(frozen=True)
class ArtifactRecord:
    """One artifact's metadata (§3.4 ``artifact.created`` fields, minus the URLs).

    ``question_id`` / ``session_id`` in the event are ``run_id`` /
    ``thread_id`` here (§3.1). ``preview`` is the per-kind SSE preview block;
    ``view_url`` / ``download_url`` are derived by the gateway, never stored.
    ``object_key`` is where the bytes live; empty on rows written before it
    was recorded, which derive it from the mime type (:func:`key_of`).
    """

    artifact_id: str
    run_id: str
    thread_id: str
    kind: str
    name: str
    mime_type: str
    size_bytes: int | None = None
    renderer: str | None = None
    preview: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    filename: str | None = None
    created_at: float = 0.0
    object_key: str = ""


def object_key_for(thread_id: str, run_id: str, artifact_id: str, ext: str) -> str:
    """``artifacts/{thread_id}/{run_id}/{artifact_id}.{ext}`` (§3.4, D7)."""
    for part in (thread_id, run_id, artifact_id):
        if "/" in part or part in ("", ".", ".."):
            raise ValueError(f"unsafe artifact key segment {part!r}")
    return check_key(f"{ARTIFACTS_PREFIX}{thread_id}/{run_id}/{artifact_id}.{ext}")


def key_of(record: ArtifactRecord) -> str:
    """The record's object key (derived from the mime type on pre-M6 rows)."""
    if record.object_key:
        return record.object_key
    from vdreamer.artifact_data import extension_for

    ext = extension_for(record.mime_type)
    return object_key_for(record.thread_id, record.run_id, record.artifact_id, ext)


class ArtifactStore(abc.ABC):
    """Metadata rows + an object store for the bytes (A8: gateway-proxied reads)."""

    objects: ObjectStore

    @abc.abstractmethod
    async def get(self, artifact_id: str) -> ArtifactRecord | None:
        """The artifact's metadata, or ``None`` when unknown."""

    @abc.abstractmethod
    async def insert(self, record: ArtifactRecord) -> None:
        """Write the metadata row (after the bytes, D7)."""

    async def put(self, record: ArtifactRecord, data: bytes | None) -> None:
        """Store the bytes, then the row (D7); ``data=None`` models an expired object."""
        key = key_of(record)
        if data is None:
            await self.objects.adelete(key)
        else:
            await self.objects.aput_bytes(key, data, record.mime_type)
        await self.insert(record)

    async def put_file(self, record: ArtifactRecord, path: str | Path) -> None:
        """Store the file at ``path`` as the bytes, then the row (D7)."""
        await self.objects.aput_file(key_of(record), path, record.mime_type)
        await self.insert(record)

    async def read_bytes(self, record: ArtifactRecord) -> bytes | None:
        """The artifact's bytes, or ``None`` when the object is gone (A9 → ``410``)."""
        return await self.objects.aget_bytes(key_of(record))

    async def fetch(self, record: ArtifactRecord, dest: str | Path) -> bool:
        """Copy the bytes to ``dest``; ``False`` when the object is gone."""
        return await self.objects.aget_to_file(key_of(record), dest)

    def local_path(self, record: ArtifactRecord) -> Path | None:
        """A local path to the bytes when the object store has one (dev)."""
        return self.objects.local_path(key_of(record))


class InMemoryArtifactStore(ArtifactStore):
    """Dict-backed rows over an in-memory object store — tests and ``dev --input``."""

    def __init__(self, objects: ObjectStore | None = None) -> None:
        self.objects = objects if objects is not None else InMemoryObjectStore()
        self._records: dict[str, ArtifactRecord] = {}

    async def insert(self, record: ArtifactRecord) -> None:
        self._records[record.artifact_id] = record

    async def get(self, artifact_id: str) -> ArtifactRecord | None:
        return self._records.get(artifact_id)


class _NoObjects(ObjectStore):
    """The object store of :class:`UnavailableArtifactStore`: nothing to read or write."""

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        raise RuntimeError(
            "no object store is configured: set VDREAMER_MINIO_ENDPOINT (+ access/secret keys)"
        )

    def get_bytes(self, key: str) -> bytes | None:
        return None

    def size(self, key: str) -> int | None:
        return None

    def delete(self, key: str) -> None:
        return None


class UnavailableArtifactStore(ArtifactStore):
    """No artifact backend configured: every lookup misses, every write raises."""

    def __init__(self) -> None:
        self.objects = _NoObjects()

    async def get(self, artifact_id: str) -> ArtifactRecord | None:
        return None

    async def insert(self, record: ArtifactRecord) -> None:
        raise RuntimeError("no artifact store is configured (set VDREAMER_MINIO_ENDPOINT)")

    async def put(self, record: ArtifactRecord, data: bytes | None) -> None:
        await self.insert(record)


#: The ``artifacts`` table (§3.4). Idempotent; ``vdreamer migrate`` applies it.
ARTIFACTS_DDL: Final[str] = """
CREATE TABLE IF NOT EXISTS artifacts (
    id          text        PRIMARY KEY,
    run_id      text        NOT NULL,
    thread_id   text        NOT NULL,
    kind        text        NOT NULL,
    name        text        NOT NULL,
    mime_type   text        NOT NULL,
    size_bytes  bigint,
    renderer    text,
    preview     jsonb       NOT NULL DEFAULT '{}'::jsonb,
    metadata    jsonb       NOT NULL DEFAULT '{}'::jsonb,
    filename    text,
    object_key  text        NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS artifacts_run_idx ON artifacts (run_id)
"""

_ARTIFACT_COLUMNS: Final[str] = (
    "id, run_id, thread_id, kind, name, mime_type, size_bytes, renderer, preview, metadata, "
    "filename, object_key, created_at"
)

_INSERT_SQL: Final[str] = """
INSERT INTO artifacts (id, run_id, thread_id, kind, name, mime_type, size_bytes, renderer,
                       preview, metadata, filename, object_key, created_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, to_timestamp(%s))
ON CONFLICT (id) DO NOTHING
"""

_GET_SQL: Final[str] = f"SELECT {_ARTIFACT_COLUMNS} FROM artifacts WHERE id = %s"


class PostgresArtifactStore(ArtifactStore):
    """The ``artifacts`` table + MinIO — short-lived autocommit connections, like threads."""

    def __init__(self, dsn: str, objects: ObjectStore) -> None:
        self._dsn = dsn
        self.objects = objects

    @asynccontextmanager
    async def _conn(self) -> AsyncGenerator[Any]:
        from vdreamer.queue import _load_async_connection_class

        conn = await _load_async_connection_class().connect(self._dsn, autocommit=True)
        try:
            yield conn
        finally:
            await conn.close()

    async def setup(self) -> None:
        """Create the table if missing (``migrate`` reuses :data:`ARTIFACTS_DDL`)."""
        async with self._conn() as conn:
            await conn.execute(ARTIFACTS_DDL)

    async def insert(self, record: ArtifactRecord) -> None:
        async with self._conn() as conn:
            await conn.execute(
                _INSERT_SQL,
                (
                    record.artifact_id,
                    record.run_id,
                    record.thread_id,
                    record.kind,
                    record.name,
                    record.mime_type,
                    record.size_bytes,
                    record.renderer,
                    json.dumps(dict(record.preview), default=str),
                    json.dumps(dict(record.metadata), default=str),
                    record.filename,
                    key_of(record),
                    record.created_at or time.time(),
                ),
            )

    async def get(self, artifact_id: str) -> ArtifactRecord | None:
        async with self._conn() as conn:
            cur = await conn.execute(_GET_SQL, (artifact_id,))
            row = await cur.fetchone()
        if row is None:
            return None
        return ArtifactRecord(
            artifact_id=str(row[0]),
            run_id=str(row[1]),
            thread_id=str(row[2]),
            kind=str(row[3]),
            name=str(row[4]),
            mime_type=str(row[5]),
            size_bytes=int(row[6]) if row[6] is not None else None,
            renderer=row[7],
            preview=dict(row[8] or {}),
            metadata=dict(row[9] or {}),
            filename=row[10],
            object_key=str(row[11]),
            created_at=row[12].timestamp(),
        )


# ---------------------------------------------------------------------------
# The per-run binding and the sync bridge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Binding:
    store: ArtifactStore | None
    loop: asyncio.AbstractEventLoop | None


_BINDING: contextvars.ContextVar[_Binding | None] = contextvars.ContextVar(
    "vdreamer_artifact_store", default=None
)


@contextlib.contextmanager
def bind_artifact_store(store: ArtifactStore | None) -> Iterator[None]:
    """Make ``store`` the run's artifact (and object) store for the block.

    Called by the worker around each run, on its event loop — the loop is
    remembered so synchronous helpers on executor threads can hand their
    writes back to it (:func:`run_blocking`).
    """
    try:
        loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    token = _BINDING.set(_Binding(store, loop))
    try:
        yield
    finally:
        _BINDING.reset(token)


def current_artifact_store() -> ArtifactStore | None:
    """The bound store, or ``None`` (tests, or a runner driven without a worker)."""
    binding = _BINDING.get()
    return binding.store if binding is not None else None


def run_blocking(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` to completion from synchronous code.

    - On an executor thread of the bound run (LangGraph runs sync nodes and
      tools there): hand it to the run's loop and wait.
    - On a thread that is itself running a loop (a sync helper called from
      ``async def`` code): a helper thread with its own loop — this blocks
      the caller's loop for the duration, which is why async code should
      prefer the awaitable helpers.
    - Elsewhere: ``asyncio.run``.
    """
    try:
        asyncio.get_running_loop()
        on_loop = True
    except RuntimeError:
        on_loop = False
    binding = _BINDING.get()
    if not on_loop:
        loop = binding.loop if binding is not None else None
        if loop is not None and loop.is_running() and not loop.is_closed():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        return asyncio.run(coro)
    context = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(context.run, asyncio.run, coro).result()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _event_record(record: ArtifactRecord, *, stored: bool) -> dict[str, Any]:
    """The ``artifact`` custom-stream record the mapper turns into ``artifact.created``."""
    return {
        "type": "artifact",
        "artifact_id": record.artifact_id,
        "name": record.name,
        "kind": record.kind,
        "mime_type": record.mime_type,
        "size_bytes": record.size_bytes,
        "preview": dict(record.preview),
        "metadata": dict(record.metadata),
        "renderer": record.renderer,
        "stored": stored,
    }


def _new_record(
    *,
    kind: str,
    name: str,
    mime_type: str,
    size: int,
    ext: str,
    preview: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    filename: str | None,
    renderer: str | None,
) -> ArtifactRecord:
    from vdreamer.context import current_context

    ctx = current_context()
    artifact_id = f"art_{uuid.uuid4().hex}"
    return ArtifactRecord(
        artifact_id=artifact_id,
        run_id=ctx.run_id,
        thread_id=ctx.thread_id,
        kind=kind,
        name=name,
        mime_type=mime_type,
        size_bytes=size,
        renderer=renderer,
        preview=dict(preview or {}),
        metadata=dict(metadata or {}),
        filename=filename,
        created_at=time.time(),
        object_key=object_key_for(ctx.thread_id, ctx.run_id, artifact_id, ext),
    )


async def _store(record: ArtifactRecord, data: bytes | None, path: Path | None) -> bool:
    """D7 steps 1–2 through the bound store; ``False`` when none is bound."""
    store = current_artifact_store()
    if store is None:
        return False
    if path is not None:
        await store.put_file(record, path)
    else:
        await store.put(record, data if data is not None else b"")
    return True


async def write_artifact(
    *,
    kind: str,
    name: str,
    data: bytes,
    mime_type: str,
    preview: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    filename: str | None = None,
    renderer: str | None = None,
) -> str:
    """Persist pre-encoded bytes for the current run and announce them; returns the id.

    The internal path (the governed tool pipeline's overflow artifacts);
    agent code uses :func:`emit_artifact`. D7 order, like every write.
    """
    from vdreamer.artifact_data import extension_for
    from vdreamer.context import emit_record

    record = _new_record(
        kind=kind,
        name=name,
        mime_type=mime_type,
        size=len(data),
        ext=extension_for(mime_type, filename),
        preview=preview,
        metadata=metadata,
        filename=filename,
        renderer=renderer,
    )
    stored = await _store(record, data, None)
    emit_record(_event_record(record, stored=stored))
    return record.artifact_id


def _check_kind(kind: Any) -> ArtifactKind:
    if not isinstance(kind, ArtifactKind):
        raise TypeError(
            f"kind must be an ArtifactKind member (table/image/html/json/file), got {kind!r}"
        )
    return kind


def _ref(record: ArtifactRecord) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=record.artifact_id,
        name=record.name,
        kind=ArtifactKind(record.kind),
        mime_type=record.mime_type,
        size_bytes=record.size_bytes,
    )


def _encode(
    name: str,
    kind: ArtifactKind,
    data: Any,
    metadata: Mapping[str, Any] | None,
    renderer: str | None,
    mime_type: str | None,
    filename: str | None,
) -> tuple[ArtifactRecord, Any]:
    """Encode ``data`` (CPU work, on the caller's thread) and mint its record."""
    from vdreamer.artifact_data import encode

    encoded = encode(
        kind.value, data, name=name, metadata=metadata, mime_type=mime_type, filename=filename
    )
    try:
        record = _new_record(
            kind=kind.value,
            name=name,
            mime_type=encoded.mime_type,
            size=encoded.size,
            ext=encoded.ext,
            preview=encoded.preview,
            metadata=encoded.metadata,
            filename=encoded.filename,
            renderer=renderer,
        )
    except BaseException:
        encoded.cleanup()
        raise
    return record, encoded


async def _persist(record: ArtifactRecord, encoded: Any) -> bool:
    """D7 steps 1–2, then drop the spooled file."""
    try:
        return await _store(record, encoded.data, encoded.path)
    finally:
        encoded.cleanup()


async def aemit_artifact(
    name: str,
    kind: ArtifactKind,
    data: Any,
    metadata: dict[str, Any] | None = None,
    renderer: str | None = None,
    *,
    mime_type: str | None = None,
    filename: str | None = None,
) -> ArtifactRef:
    """The awaitable :func:`emit_artifact` — encodes on a thread, never blocks the loop."""
    from vdreamer.context import current_context, emit_record

    _check_kind(kind)
    current_context()  # fail fast outside a run, before any encoding work
    record, encoded = await asyncio.to_thread(
        _encode, name, kind, data, metadata, renderer, mime_type, filename
    )
    stored = await _persist(record, encoded)
    emit_record(_event_record(record, stored=stored))
    return _ref(record)


def emit_artifact(
    name: str,
    kind: ArtifactKind,
    data: Any,
    metadata: dict[str, Any] | None = None,
    renderer: str | None = None,
    *,
    mime_type: str | None = None,
    filename: str | None = None,
) -> ArtifactRef:
    """Store an artifact for the current run and announce it (PRD §3.4).

    ``kind`` must be an :class:`ArtifactKind` member and ``data`` what that
    kind accepts (see ``vdreamer.artifact_data``) — anything else is a
    ``TypeError``, never auto-detected. ``renderer`` is an optional UI hint
    (e.g. ``verdict.v1``); ``mime_type``/``filename`` describe ``file``
    artifacts. The write follows D7 (bytes → metadata row → event) and the
    returned :class:`ArtifactRef` is what agent code hands to the model or
    keeps in state.

    Works in any node or tool. Called from ``async def`` code it blocks that
    loop while it encodes and writes; prefer ``await aemit_artifact(...)`` there.
    """
    from vdreamer.context import current_context, emit_record

    _check_kind(kind)
    current_context()
    record, encoded = _encode(name, kind, data, metadata, renderer, mime_type, filename)
    if current_artifact_store() is None:
        encoded.cleanup()
        stored = False
    else:
        stored = run_blocking(_persist(record, encoded))
    emit_record(_event_record(record, stored=stored))
    return _ref(record)


async def store_returned(artifact: Artifact) -> ArtifactRef:
    """Store a tool's returned :class:`Artifact` (the governed pipeline's step 5)."""
    return await aemit_artifact(
        artifact.name,
        _check_kind(artifact.kind),
        artifact.data,
        artifact.metadata,
        artifact.renderer,
        mime_type=artifact.mime_type,
        filename=artifact.filename,
    )
