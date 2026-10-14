"""The artifact read side the gateway proxies (PRD §2.5, §3.4, A8, A9).

Milestone 4 lands the *routes* — ``GET /artifacts/{id}``, ``/preview``,
``/raw``, ``/download`` — and the store contract they read through. The write
side is Milestone 6 (D6–D8): ``emit_artifact`` writing bytes to MinIO
(``artifacts/{thread_id}/{run_id}/{artifact_id}.{ext}``), then the metadata
row, then the ``artifact.created`` event; Parquet tables with paged
``/preview?offset&limit&sort``; CSV/XLSX conversion on ``/download``; HTML
``artifact://`` rewriting.

Milestone 5 adds the minimal write side the governed tool pipeline needs for
oversized results (§2.7 step 5): :meth:`ArtifactStore.put` and
:func:`write_artifact`, which keeps the D7 order — bytes and metadata through
the store, *then* the ``artifact.created`` record into the run's stream. The
worker binds its store for each run (:func:`bind_artifact_store`); outside a
bound store the event still flows, but nothing is kept to serve.

- :class:`InMemoryArtifactStore` holds records and bytes for tests;
- ``vdreamer dev`` uses ``sqlite_state.SqliteArtifactStore`` (metadata rows in
  the dev ``.db``, bytes on the local filesystem, D11);
- :class:`UnavailableArtifactStore` is what ``vdreamer serve`` wires until the
  M6 MinIO writer: it knows no artifacts, so every artifact route answers
  ``404`` instead of pretending.

A record whose bytes are gone (the 30-day lifecycle, A9) reads back with
``read_bytes`` → ``None``; the gateway answers ``410 artifact_expired``.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import contextvars
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

#: The gateway's API prefix — artifact URLs in events point at its routes.
API_PREFIX: Final[str] = "/api/v1"


def artifact_urls(artifact_id: str) -> tuple[str, str]:
    """``(view_url, download_url)`` — the gateway routes (§3.4)."""
    return (
        f"{API_PREFIX}/artifacts/{artifact_id}/raw",
        f"{API_PREFIX}/artifacts/{artifact_id}/download",
    )


@dataclass(frozen=True)
class ArtifactRecord:
    """One artifact's metadata (§3.4 ``artifact.created`` fields, minus the URLs).

    ``question_id`` / ``session_id`` in the event are ``run_id`` /
    ``thread_id`` here (§3.1). ``preview`` is the per-kind SSE preview block;
    ``view_url`` / ``download_url`` are derived by the gateway, never stored.
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


class ArtifactStore(abc.ABC):
    """Metadata + bytes for the gateway's artifact proxy (A8: gateway-proxied)."""

    @abc.abstractmethod
    async def get(self, artifact_id: str) -> ArtifactRecord | None:
        """The artifact's metadata, or ``None`` when unknown."""

    @abc.abstractmethod
    async def read_bytes(self, record: ArtifactRecord) -> bytes | None:
        """The artifact's bytes, or ``None`` when the object is gone (A9 → ``410``)."""

    @abc.abstractmethod
    async def put(self, record: ArtifactRecord, data: bytes | None) -> None:
        """Store the bytes, then the record (D7); ``data=None`` models an expired object."""


class InMemoryArtifactStore(ArtifactStore):
    """Dict-backed store — tests and ``vdreamer dev`` until the M6 writer lands."""

    def __init__(self) -> None:
        self._records: dict[str, ArtifactRecord] = {}
        self._bytes: dict[str, bytes] = {}
        self._lock = asyncio.Lock()

    async def put(self, record: ArtifactRecord, data: bytes | None) -> None:
        async with self._lock:
            self._records[record.artifact_id] = record
            if data is None:
                self._bytes.pop(record.artifact_id, None)
            else:
                self._bytes[record.artifact_id] = data

    async def get(self, artifact_id: str) -> ArtifactRecord | None:
        async with self._lock:
            return self._records.get(artifact_id)

    async def read_bytes(self, record: ArtifactRecord) -> bytes | None:
        async with self._lock:
            return self._bytes.get(record.artifact_id)


class UnavailableArtifactStore(ArtifactStore):
    """No artifact backend yet (``vdreamer serve`` before M6): every lookup misses."""

    async def get(self, artifact_id: str) -> ArtifactRecord | None:
        return None

    async def read_bytes(self, record: ArtifactRecord) -> bytes | None:
        return None

    async def put(self, record: ArtifactRecord, data: bytes | None) -> None:
        raise RuntimeError("no artifact store is configured (the MinIO writer lands in M6)")


#: The run's artifact store, bound by the worker around each run.
_STORE: contextvars.ContextVar[ArtifactStore | None] = contextvars.ContextVar(
    "vdreamer_artifact_store", default=None
)


@contextlib.contextmanager
def bind_artifact_store(store: ArtifactStore | None) -> Iterator[None]:
    """Make ``store`` the target of :func:`write_artifact` for the block."""
    token = _STORE.set(store)
    try:
        yield
    finally:
        _STORE.reset(token)


def current_artifact_store() -> ArtifactStore | None:
    """The bound store, or ``None`` (tests, or a runner driven without a worker)."""
    return _STORE.get()


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
    """Persist one artifact for the current run and announce it; returns its id.

    D7 order: the store writes the bytes and then the metadata row, and only
    then is the ``artifact.created`` record emitted into the run (the stream
    mapper turns it into the event, URLs included). Must run inside a vdreamer
    run (``context.current_context``).
    """
    from vdreamer.context import current_context, emit_record

    ctx = current_context()
    artifact_id = f"art_{uuid.uuid4().hex}"
    record = ArtifactRecord(
        artifact_id=artifact_id,
        run_id=ctx.run_id,
        thread_id=ctx.thread_id,
        kind=kind,
        name=name,
        mime_type=mime_type,
        size_bytes=len(data),
        renderer=renderer,
        preview=dict(preview or {}),
        metadata=dict(metadata or {}),
        filename=filename,
        created_at=time.time(),
    )
    store = current_artifact_store()
    if store is not None:
        await store.put(record, data)
    emit_record(
        {
            "type": "artifact",
            "artifact_id": artifact_id,
            "name": name,
            "kind": kind,
            "mime_type": mime_type,
            "size_bytes": len(data),
            "preview": dict(record.preview),
            "metadata": dict(record.metadata),
            "renderer": renderer,
            "stored": store is not None,
        }
    )
    return artifact_id
