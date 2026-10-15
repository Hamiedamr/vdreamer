"""UC3 domain logic: page-offset chunking, embeddings, the Qdrant collection, hybrid search.

Retrieval is agent code, never core (PRD §7.2): vdreamer hands the agents a
native ``AsyncQdrantClient`` (``ctx.resources["vectors"]``) and native
fastembed objects (``ctx.models["dense" | "sparse" | "rerank"]``), and this
module does the rest with their own APIs.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: The Qdrant collection both UC3 agents share.
COLLECTION = "pdf_research"

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150

#: Candidates each retriever (dense, sparse) contributes before RRF fusion.
PREFETCH = 24

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
_NAMESPACE = uuid.UUID("6f1c7a8e-3d2b-4c55-9b7e-2f6d4a1e0c93")


@dataclass(frozen=True)
class Chunk:
    document: str
    page: int
    start: int
    end: int
    text: str
    language: str

    @property
    def point_id(self) -> str:
        """Deterministic: re-ingesting a document overwrites its points (retries are safe)."""
        return str(uuid.uuid5(_NAMESPACE, f"{self.document}\0{self.page}\0{self.start}"))

    def payload(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "page": self.page,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "language": self.language,
        }


def language_of(text: str) -> str:
    """``ar`` when Arabic letters dominate the letters, else ``en``."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "en"
    arabic = sum(1 for ch in letters if _ARABIC.match(ch))
    return "ar" if arabic * 2 >= len(letters) else "en"


def chunk_pages(document: str, pages: Sequence[str]) -> list[Chunk]:
    """Windows of ``CHUNK_CHARS`` with overlap, cut at whitespace, carrying page + offsets."""
    chunks: list[Chunk] = []
    for number, page in enumerate(pages, start=1):
        text = page.strip()
        position = 0
        while position < len(text):
            end = min(len(text), position + CHUNK_CHARS)
            if end < len(text):
                cut = text.rfind(" ", position + CHUNK_CHARS // 2, end)
                end = cut if cut > position else end
            piece = text[position:end].strip()
            if piece:
                chunks.append(
                    Chunk(document, number, position, end, piece, language_of(piece))
                )
            if end >= len(text):
                break
            position = max(end - CHUNK_OVERLAP, position + 1)
    return chunks


def dense_vectors(model: Any, texts: Sequence[str]) -> list[list[float]]:
    return [[float(x) for x in vector] for vector in model.embed(list(texts))]


def dense_query(model: Any, text: str) -> list[float]:
    embed = getattr(model, "query_embed", None) or model.embed
    return [float(x) for x in next(iter(embed([text])))]


def sparse_vectors(model: Any, texts: Sequence[str]) -> list[tuple[list[int], list[float]]]:
    return [
        ([int(i) for i in item.indices], [float(v) for v in item.values])
        for item in model.embed(list(texts))
    ]


def sparse_query(model: Any, text: str) -> tuple[list[int], list[float]]:
    embed = getattr(model, "query_embed", None) or model.embed
    item = next(iter(embed([text])))
    return [int(i) for i in item.indices], [float(v) for v in item.values]


async def ensure_collection(client: Any, size: int) -> None:
    """The collection with a named dense vector and a named sparse vector (hybrid search)."""
    from qdrant_client import models

    if await client.collection_exists(COLLECTION):
        return
    await client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(size=size, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )


async def upsert_chunks(client: Any, dense: Any, sparse: Any, chunks: Sequence[Chunk]) -> int:
    from qdrant_client import models

    if not chunks:
        return 0
    texts = [chunk.text for chunk in chunks]
    dense_list = dense_vectors(dense, texts)
    sparse_list = sparse_vectors(sparse, texts)
    await ensure_collection(client, len(dense_list[0]))
    points = [
        models.PointStruct(
            id=chunk.point_id,
            vector={
                "dense": vector,
                "sparse": models.SparseVector(indices=indices, values=values),
            },
            payload=chunk.payload(),
        )
        for chunk, vector, (indices, values) in zip(chunks, dense_list, sparse_list, strict=True)
    ]
    await client.upsert(COLLECTION, points=points, wait=True)
    return len(points)


async def hybrid_search(
    client: Any,
    dense: Any,
    sparse: Any,
    rerank: Any,
    query: str,
    *,
    limit: int = 5,
    documents: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Dense + sparse prefetch, RRF fusion in Qdrant, then the cross-encoder reranks."""
    from qdrant_client import models

    if not await client.collection_exists(COLLECTION):
        return []
    indices, values = sparse_query(sparse, query)
    wanted = list(documents or [])
    where = (
        models.Filter(must=[models.FieldCondition(
            key="document", match=models.MatchAny(any=wanted)
        )])
        if wanted
        else None
    )
    response = await client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_query(dense, query), using="dense", limit=PREFETCH,
                            filter=where),
            models.Prefetch(query=models.SparseVector(indices=indices, values=values),
                            using="sparse", limit=PREFETCH, filter=where),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=PREFETCH,
        with_payload=True,
    )
    candidates = [dict(point.payload or {}) | {"fused": float(point.score)}
                  for point in response.points]
    if not candidates:
        return []
    scores = [float(s) for s in rerank.rerank(query, [c["text"] for c in candidates])]
    for candidate, score in zip(candidates, scores, strict=True):
        candidate["score"] = score
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:limit]


def cite_markers(text: str) -> set[int]:
    """The ``[n]`` markers an answer uses."""
    return {int(n) for n in re.findall(r"\[(\d{1,3})\]", text)}


def snippet(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def as_sources(results: Sequence[Mapping[str, Any]], first: int = 1) -> list[dict[str, Any]]:
    """Numbered sources the model sees and cites with ``[n]``."""
    return [
        {"n": first + i, "document": r["document"], "page": r["page"], "start": r["start"],
         "end": r["end"], "language": r.get("language"), "score": round(float(r["score"]), 4),
         "text": snippet(str(r["text"]), 600)}
        for i, r in enumerate(results)
    ]
