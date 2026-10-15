"""UC3 acceptance (PRD §7.3): multi-document research & comparison, EN + AR, end to end.

A service ingests the documents (``vdreamer run pdf-ingest --service``): text
documents are chunked, embedded (dense + sparse) and upserted into Qdrant; a
PDF is parsed by ``pypdf`` in the sandbox (POSIX) or reported skipped. Then a
user asks ``pdf-research`` in chat: the model calls ``search_documents``
(hybrid search, RRF fusion in Qdrant, rerank), answers with ``[n]`` markers
that become ``citation`` events bound to the answer, and its reasoning streams
as thinking. ``compare_documents`` puts the EN and AR documents side by side
in a ``table`` artifact.

Qdrant is real (``AsyncQdrantClient(":memory:")``, shared by both agents'
runners as both workers would share the server); fastembed and the reasoner
are scripted at the boundary.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from _uc_kit import (
    LexicalDense,
    LexicalRerank,
    LexicalSparse,
    ScriptedModel,
    scripted,
    serve,
    vdreamer_run,
)
from langchain_core.messages import AIMessage
from pdf_research import PdfIngestAgent, PdfResearchAgent
from pdf_research.retrieval import COLLECTION

EN = (
    "Data Retention Policy.\n"
    "Customer records are retained for seven years after the contract ends.\n"
    "\f"
    "Backups are encrypted and kept in the primary data centre for ninety days."
)
AR = (
    "سياسة الاحتفاظ بالبيانات.\n"
    "يتم الاحتفاظ بسجلات العملاء لمدة خمس سنوات بعد انتهاء العقد.\n"
    "\f"
    "تُحفظ النسخ الاحتياطية مشفرة لمدة ثلاثين يوما."
)
PDF_TEXT = "Audit logs are retained for twelve months"


def minimal_pdf(text: str) -> bytes:
    """A one-page PDF whose content stream draws ``text`` (enough for pypdf to extract)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream
        + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _stack(reasoner: ScriptedModel, client: Any) -> Any:
    from _api import make_stack

    embedders = {"dense": LexicalDense(), "sparse": LexicalSparse()}

    @asynccontextmanager
    async def ingest() -> AsyncIterator[dict[str, Any]]:
        yield {"models": embedders, "resources": {"vectors": client}}

    @asynccontextmanager
    async def research() -> AsyncIterator[dict[str, Any]]:
        yield {"models": {**embedders, "reasoner": reasoner, "rerank": LexicalRerank()},
               "resources": {"vectors": client}}

    return make_stack(
        workers=True,
        extra_agents=[PdfIngestAgent(), PdfResearchAgent()],
        worker_agents=("pdf-ingest", "pdf-research"),
        poll_interval_s=0.05,
        runner_options={"pdf-ingest": ingest, "pdf-research": research},
    )


@pytest.fixture
def uc3() -> Iterator[tuple[str, ScriptedModel, Any, Any]]:
    from qdrant_client import AsyncQdrantClient

    reasoner = scripted(
        _call("search_documents", {"query": "how long are customer records retained"}, "c1"),
        AIMessage(content=[
            {"type": "reasoning", "reasoning": "Source 1 is the English policy, page 1."},
            {"type": "text", "text": "Customer records are retained for seven years [1]."},
        ]),
        _call("compare_documents", {"question": "customer records retained years",
                                    "documents": ["policy_en.txt", "policy_ar.txt"]}, "c2"),
        AIMessage("The English policy keeps records seven years [1]; "
                  "the Arabic one five years [3]."),
    )
    client = AsyncQdrantClient(":memory:")
    stack = _stack(reasoner, client)
    try:
        with serve(stack) as url:
            yield url, reasoner, client, stack
    finally:
        stack.close()
        asyncio.run(client.close())


def _ingest(url: str, monkeypatch: pytest.MonkeyPatch, stack: Any) -> str:
    put = stack.artifacts.objects.put_bytes
    put("inputs/docs/policy_en.txt", EN.encode("utf-8"), "text/plain")
    put("inputs/docs/policy_ar.txt", AR.encode("utf-8"), "text/plain")
    put("inputs/docs/audit.pdf", minimal_pdf(PDF_TEXT), "application/pdf")
    documents = [{"object_key": f"inputs/docs/{name}"}
                 for name in ("policy_en.txt", "policy_ar.txt", "audit.pdf")]
    result = vdreamer_run(
        url, monkeypatch, agent="pdf-ingest", service=True, idempotency_key="ingest-1",
        input_json=json.dumps({"documents": documents}),
    )
    assert result.code == 0, result.out
    assert "[artifact] table 'ingested documents'" in result.out
    assert result.out.rstrip().endswith("[run completed]")
    return result.out


def test_documents_are_ingested_and_answers_are_cited(
    uc3: tuple[str, ScriptedModel, Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, reasoner, client, stack = uc3
    out = _ingest(url, monkeypatch, stack)
    points = asyncio.run(client.count(COLLECTION)).count
    if os.name == "posix":
        assert points == 5  # 2 + 2 text pages, 1 PDF page parsed in the sandbox
    else:
        assert points == 4
        assert "audit.pdf: PDFs are parsed in the sandbox" in out

    # Chat: search → a cited answer, with the reasoning shown as thinking.
    asked = vdreamer_run(url, monkeypatch, agent="pdf-research", once=True,
                         message="How long are customer records retained?")
    assert asked.code == 0, asked.out
    out = asked.out
    assert '-> search_documents({"query": "how long are customer records retained"})' in out
    assert "[thinking] Source 1 is the English policy, page 1." in out
    assert "Customer records are retained for seven years [1]." in out
    assert "[1] policy_en.txt — page 1, start 0" in out and "(doc://policy_en.txt)" in out
    # The model saw the numbered sources, the best one first.
    tool_result = json.loads(str(reasoner.seen[1][-1].content))
    best = tool_result["sources"][0]
    assert best["n"] == 1 and best["document"] == "policy_en.txt" and best["page"] == 1

    # Second turn on the same thread: EN vs AR side by side, sources numbered per run.
    compared = vdreamer_run(url, monkeypatch, agent="pdf-research", once=True,
                            thread=asked.thread_id(), message="Compare EN and AR policies")
    assert compared.code == 0, compared.out
    out = compared.out
    assert "-> compare_documents(" in out and "[artifact] table 'comparison'" in out
    sources = json.loads(str(reasoner.seen[3][-1].content))["sources"]
    assert {s["document"] for s in sources} == {"policy_en.txt", "policy_ar.txt"}
    assert [s["n"] for s in sources] == list(range(1, len(sources) + 1))
    assert {s["language"] for s in sources} == {"en", "ar"}
    assert out.count("(doc://") == 2  # [1] and [3] cited; unused sources are not


def test_an_answer_without_sources_has_no_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qdrant_client import AsyncQdrantClient

    # Nothing ingested: the search returns no sources, and a stray [1] cites nothing.
    reasoner = scripted(
        _call("search_documents", {"query": "revenue"}, "c1"),
        AIMessage("The documents do not cover revenue [1]."),
    )
    client = AsyncQdrantClient(":memory:")
    stack = _stack(reasoner, client)
    try:
        with serve(stack) as url:
            result = vdreamer_run(url, monkeypatch, agent="pdf-research", once=True,
                                  message="What was the revenue?")
    finally:
        stack.close()
        asyncio.run(client.close())
    assert result.code == 0, result.out
    assert '"sources": []' in str(reasoner.seen[1][-1].content)
    assert "(doc://" not in result.out
