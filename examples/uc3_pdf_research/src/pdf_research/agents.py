"""UC3 — Multi-PDF Research & Comparison (Chat + Headless, EN + AR). PRD §7.3.

Two agents over one Qdrant collection (``ctx.resources["vectors"]``):

- **``pdf-ingest``** (headless, ``/trigger``): each document → pages (PDFs are
  parsed by ``pypdf`` *in the sandbox*; ``.txt``/``.md`` are split on form
  feeds) → page-offset chunks → ``dense`` + ``sparse`` fastembed vectors →
  upsert. A summary ``table`` artifact lists what was ingested.
- **``pdf-research``** (chat, ``create_agent``): the model calls
  ``search_documents`` (dense + sparse prefetch, RRF fusion in Qdrant, then the
  ``rerank`` cross-encoder) and ``compare_documents`` (a per-document
  comparison ``table``), answers with ``[n]`` markers, and a middleware turns
  every marker into a ``citation`` bound to the answer's ``message_id``. With a
  reasoning model, its reasoning streams as ``thinking.delta``.

Public vdreamer API only (``import vdreamer``).
"""

from __future__ import annotations

import base64
import json
import zlib
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, TypedDict

import pydantic
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

import vdreamer
from pdf_research.retrieval import (
    as_sources,
    chunk_pages,
    cite_markers,
    hybrid_search,
    language_of,
    snippet,
    upsert_chunks,
)
from vdreamer import ArtifactKind, BaseAgent, BuildContext, FileRef, Mode

#: One sandbox run returns at most this much compressed page text (stdout is capped).
BATCH_BYTES = 40_000

#: The sandboxed parser: pages from ``start`` until the batch is full, zlib + base64 JSON.
PDF_PARSER = r"""
import base64, json, sys, zlib
from pypdf import PdfReader

start = json.load(open("range.json"))["start"]
reader = PdfReader("document.pdf")
total = len(reader.pages)
texts, size = [], 0
for index in range(start, total):
    text = reader.pages[index].extract_text() or ""
    encoded = len(zlib.compress(text.encode("utf-8")))
    if texts and size + encoded > LIMIT:
        break
    texts.append(text)
    size += encoded
payload = json.dumps({"total": total, "start": start, "texts": texts}).encode("utf-8")
sys.stdout.write(base64.b64encode(zlib.compress(payload)).decode("ascii"))
""".replace("LIMIT", str(BATCH_BYTES))


class IngestInput(pydantic.BaseModel):
    """What ``/trigger`` sends: uploaded documents under ``inputs/``."""

    documents: list[FileRef] = pydantic.Field(min_length=1, max_length=50)


class IngestState(TypedDict, total=False):
    documents: list[dict[str, Any]]
    results: list[dict[str, Any]]
    summary_id: str


def document_name(ref: FileRef) -> str:
    return PurePosixPath(ref.object_key).name


async def pdf_pages(ref: FileRef) -> list[str]:
    """Every page's text, parsed in the sandbox in batches."""
    sandbox = vdreamer.current_context().sandbox
    pages: list[str] = []
    total: int | None = None
    while total is None or len(pages) < total:
        run = await sandbox.arun(
            PDF_PARSER,
            inputs={"document.pdf": ref, "range.json": json.dumps({"start": len(pages)})},
            timeout_s=120,
        )
        if not run.ok:
            raise RuntimeError(f"the PDF parser failed: {run.stderr.strip()[-300:]}")
        data = json.loads(zlib.decompress(base64.b64decode(run.stdout.strip())))
        total = int(data["total"])
        if not data["texts"] and len(pages) < total:
            raise RuntimeError(f"page {len(pages) + 1} alone exceeds the batch limit")
        pages.extend(str(text) for text in data["texts"])
    return pages


class PdfIngestAgent(BaseAgent):
    """Headless: documents → pages → chunks → dense + sparse vectors → Qdrant."""

    name = "pdf-ingest"
    mode = Mode.HEADLESS
    input_schema = IngestInput
    models = ["dense", "sparse"]
    resources = ["vectors"]

    def build(self, ctx: BuildContext) -> Any:
        dense, sparse = ctx.models["dense"], ctx.models["sparse"]
        client = ctx.resources["vectors"]

        async def ingest(state: IngestState) -> dict[str, Any]:
            files = vdreamer.current_context().files
            results: list[dict[str, Any]] = []
            documents = [FileRef(**doc) for doc in state["documents"]]
            for index, ref in enumerate(documents):
                name = document_name(ref)
                vdreamer.emit_progress(f"ingesting {name}", 100.0 * index / len(documents))
                row: dict[str, Any] = {"document": name, "pages": 0, "chunks": 0,
                                       "language": None, "status": "ok", "error": None}
                try:
                    if name.lower().endswith(".pdf"):
                        pages = await pdf_pages(ref)
                    else:
                        pages = (await files.aread_bytes(ref)).decode("utf-8").split("\f")
                    chunks = chunk_pages(name, pages)
                    row.update(pages=len(pages), chunks=await upsert_chunks(
                        client, dense, sparse, chunks
                    ), language=language_of(" ".join(pages)[:5000]))
                except vdreamer.SandboxUnavailable as exc:
                    row.update(status="skipped", error=f"PDFs are parsed in the sandbox: {exc}")
                except Exception as exc:  # one bad document does not sink the batch
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                if row["error"]:
                    vdreamer.emit_log(f"{name}: {row['error']}", level="warning")
                results.append(row)
            summary = await vdreamer.aemit_artifact("ingested documents", ArtifactKind.TABLE,
                                                    results)
            vdreamer.emit_progress("done", 100.0)
            if not any(row["status"] == "ok" for row in results):
                raise RuntimeError("no document could be ingested")
            return {"results": results, "summary_id": summary.artifact_id}

        graph = StateGraph(IngestState)
        graph.add_node("ingest", ingest)
        graph.add_edge(START, "ingest")
        graph.add_edge("ingest", END)
        return graph


SYSTEM_PROMPT = """You are a research assistant over the user's ingested documents.
Always call search_documents before answering; call compare_documents when the user
compares documents. Answer only from the returned sources and cite each claim with its
source number in brackets, like [1] or [2]. Answer in the language of the question
(Arabic or English). If the sources do not answer the question, say so."""

#: Tools whose results carry numbered sources.
SOURCE_TOOLS = ("search_documents", "compare_documents")


def _turn_sources(messages: Sequence[Any]) -> dict[int, dict[str, Any]]:
    """The numbered sources the tools returned since the last user message."""
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            start = index
    sources: dict[int, dict[str, Any]] = {}
    for message in messages[start:]:
        if isinstance(message, ToolMessage) and message.name in SOURCE_TOOLS:
            try:
                data = json.loads(str(message.content))
            except (json.JSONDecodeError, TypeError):
                continue
            for source in data.get("sources") or []:
                sources[int(source["n"])] = source
    return sources


class CitationMiddleware(AgentMiddleware):  # type: ignore[type-arg]
    """After the answer: one ``citation`` per ``[n]`` it uses, bound to its message_id."""

    def after_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = list(state["messages"])
        answer = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if answer is None or answer.id is None:
            return None
        sources = _turn_sources(messages)
        used = sorted(n for n in cite_markers(str(answer.content)) if n in sources)
        for n in used:
            source = sources[n]
            vdreamer.emit_citation(
                str(answer.id),
                {"uri": f"doc://{source['document']}", "title": source["document"]},
                locator={"page": source["page"], "start": source["start"], "end": source["end"],
                         "marker": n},
                snippet=snippet(str(source["text"])),
                score=float(source["score"]),
            )
        return None


class PdfResearchAgent(BaseAgent):
    """Chat (``create_agent``): hybrid search + rerank → cited answer, comparison tables."""

    name = "pdf-research"
    mode = Mode.CHAT
    models = ["reasoner", "dense", "sparse", "rerank"]
    resources = ["vectors"]
    expose_thinking = True

    def build(self, ctx: BuildContext) -> Any:
        dense, sparse, rerank = ctx.models["dense"], ctx.models["sparse"], ctx.models["rerank"]
        client = ctx.resources["vectors"]
        numbering: dict[str, int] = {}  # run_id → last source number handed out

        def numbered(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
            run_id = vdreamer.current_context().run_id
            first = numbering.get(run_id, 0) + 1
            sources = as_sources(results, first=first)
            numbering[run_id] = first + len(sources) - 1
            return sources

        @tool
        async def search_documents(query: str, documents: list[str] | None = None) -> str:
            """Hybrid search over the ingested documents (optionally only the named ones).

            Returns numbered sources (document, page, text) to cite as [n].
            """
            results = await hybrid_search(
                client, dense, sparse, rerank, query, limit=5, documents=documents
            )
            return json.dumps({"query": query, "sources": numbered(results)}, ensure_ascii=False)

        @tool
        async def compare_documents(question: str, documents: list[str]) -> str:
            """Compare how several documents treat one question: the best passages of each,
            side by side in a table artifact. Returns numbered sources to cite as [n]."""
            rows: list[dict[str, Any]] = []
            found: list[dict[str, Any]] = []
            for name in documents:
                best = await hybrid_search(
                    client, dense, sparse, rerank, question, limit=2, documents=[name]
                )
                found.extend(best)
                for hit in best:
                    rows.append({"document": name, "page": hit["page"],
                                 "score": round(float(hit["score"]), 4),
                                 "language": hit.get("language"),
                                 "passage": snippet(str(hit["text"]), 400)})
                if not best:
                    rows.append({"document": name, "page": None, "score": None,
                                 "language": None, "passage": "(nothing relevant)"})
            table = await vdreamer.aemit_artifact(
                "comparison", ArtifactKind.TABLE, rows, metadata={"question": question}
            )
            return json.dumps(
                {"comparison_artifact_id": table.artifact_id, "sources": numbered(found)},
                ensure_ascii=False,
            )

        kwargs = ctx.agent_kwargs()
        middleware = [*kwargs.pop("middleware"), CitationMiddleware()]
        return create_agent(
            ctx.models["reasoner"],
            [search_documents, compare_documents],
            system_prompt=SYSTEM_PROMPT,
            middleware=middleware,
            **kwargs,
        )


ingest_agent = PdfIngestAgent()
research_agent = PdfResearchAgent()
