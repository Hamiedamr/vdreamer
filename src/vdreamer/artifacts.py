"""The artifact read side the gateway proxies (PRD §2.5, §3.4, A8, A9).

Milestone 4 lands the *routes* — ``GET /artifacts/{id}``, ``/preview``,
``/raw``, ``/download`` — and the store contract they read through. The write
side is Milestone 6 (D6–D8): ``emit_artifact`` writing bytes to MinIO
(``artifacts/{thread_id}/{run_id}/{artifact_id}.{ext}``), then the metadata
row, then the ``artifact.created`` event; Parquet tables with paged
``/preview?offset&limit&sort``; CSV/XLSX conversion on ``/download``; HTML
``artifact://`` rewriting.

Until then:

- :class:`InMemoryArtifactStore` holds records and bytes for tests and
  ``vdreamer dev`` (``put`` is the seam M6's writer replaces);
- :class:`UnavailableArtifactStore` is what ``vdreamer serve`` wires: it knows
  no artifacts, so every artifact route answers ``404`` instead of pretending.

A record whose bytes are gone (the 30-day lifecycle, A9) reads back with
``read_bytes`` → ``None``; the gateway answers ``410 artifact_expired``.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


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


class InMemoryArtifactStore(ArtifactStore):
    """Dict-backed store — tests and ``vdreamer dev`` until the M6 writer lands."""

    def __init__(self) -> None:
        self._records: dict[str, ArtifactRecord] = {}
        self._bytes: dict[str, bytes] = {}
        self._lock = asyncio.Lock()

    async def put(self, record: ArtifactRecord, data: bytes | None) -> None:
        """Store ``record`` and its bytes; ``data=None`` models an expired object."""
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
