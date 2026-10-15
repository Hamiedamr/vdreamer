"""Milestone 12 acceptance kit: the reference agents behind a real gateway, via ``vdreamer run``.

What is real: the gateway (uvicorn over TCP), the workers, the stores, the
outbox, ``vdreamer run`` (``run_terminal``), the SQLite warehouse (SQLAlchemy
async engine), Qdrant (``AsyncQdrantClient(":memory:")``), the UC2
site-inventory MCP server (stdio subprocess), ``safe_extract``.

What is scripted, at the boundary only: the chat/VLM models
(:class:`ScriptedModel`), the Neo4j driver (:class:`FakeNeo4j`, the semantic
model from the example's ``demo/semantic.json``) and fastembed
(:class:`LexicalDense`/:class:`LexicalSparse`/:class:`LexicalRerank` — the
fastembed method shapes over a deterministic bag of words). The live tier
(``test_uc_live_ollama.py``) swaps the models for Ollama's ``gemma4:31b-cloud``.
"""

from __future__ import annotations

import io
import json
import math
import re
import socket
import sys
import threading
import time
import zlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
API_DIR = REPO / "tests" / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from _api import SERVICE_TOKEN, TOKEN, Stack  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402

from vdreamer.client import RunOptions, run_terminal  # noqa: E402

EXAMPLES = REPO / "examples"

# -- models --------------------------------------------------------------------------------


class ScriptedModel(FakeMessagesListChatModel):
    """Answers in order; records every prompt it got. ``bind_tools`` keeps the script."""

    seen: list[Any] = []  # a pydantic field: each instance gets its own list

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def scripted(*answers: BaseMessage | str) -> ScriptedModel:
    return ScriptedModel(
        responses=[a if isinstance(a, BaseMessage) else AIMessage(a) for a in answers]
    )


# -- the semantic layer (Neo4j) --------------------------------------------------------------


@dataclass
class _Eager:
    records: list[dict[str, Any]]


class FakeNeo4j:
    """``execute_query`` answering the two UC1 semantic queries from the demo model."""

    def __init__(self) -> None:
        self.model = json.loads((EXAMPLES / "uc1_sql_analyst" / "demo" / "semantic.json")
                                .read_text(encoding="utf-8"))
        self.queries: list[str] = []

    async def execute_query(self, query: str, *args: Any, **kwargs: Any) -> _Eager:
        self.queries.append(query)
        assert kwargs.get("routing_") == "r", "the semantic layer is read through readers"
        if "JOINS" in query:
            return _Eager([dict(join) for join in self.model["joins"]])
        return _Eager([dict(table) for table in self.model["tables"]])


# -- fastembed stand-ins -----------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bucket(token: str, size: int) -> int:
    return zlib.crc32(token.encode("utf-8")) % size


class LexicalDense:
    """``TextEmbedding``'s shape: ``embed``/``query_embed`` yield unit vectors (64 dims)."""

    dim = 64

    def embed(self, texts: Sequence[str]) -> Iterator[list[float]]:
        for text in texts:
            vector = [0.0] * self.dim
            for token in _tokens(text):
                vector[_bucket(token, self.dim)] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            yield [v / norm for v in vector]

    query_embed = embed


class LexicalSparse:
    """``SparseTextEmbedding``'s shape: items with ``indices`` and ``values``."""

    def embed(self, texts: Sequence[str]) -> Iterator[SimpleNamespace]:
        for text in texts:
            counts: dict[int, float] = {}
            for token in _tokens(text):
                index = _bucket(token, 100_000)
                counts[index] = counts.get(index, 0.0) + 1.0
            indices = sorted(counts)
            yield SimpleNamespace(indices=indices, values=[counts[i] for i in indices])

    query_embed = embed


class LexicalRerank:
    """``TextCrossEncoder``'s shape: ``rerank(query, documents)`` yields one score each."""

    def rerank(self, query: str, documents: Sequence[str]) -> Iterator[float]:
        wanted = set(_tokens(query))
        for document in documents:
            words = _tokens(document)
            yield sum(1 for word in words if word in wanted) / (1 + len(words) ** 0.5)


# -- a real server, a real client ---------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def serve(stack: Stack) -> Iterator[str]:
    """``stack.app`` on uvicorn over TCP; yields the base URL."""
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(stack.app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("the gateway did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(15)


class Script:
    """Scripted terminal answers; records the prompts."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


@dataclass
class CliResult:
    code: int
    out: str
    script: Script

    def thread_id(self) -> str:
        match = re.search(r"(thread_[0-9a-f]{32})", self.out)
        assert match, self.out
        return match.group(1)


def vdreamer_run(url: str, monkeypatch: Any, *answers: str, **options: Any) -> CliResult:
    """``vdreamer run`` exactly as a developer types it, with scripted answers."""
    monkeypatch.setenv("VDREAMER_INTERNAL_TOKEN", TOKEN)
    monkeypatch.setenv("UC_SERVICE_TOKEN", SERVICE_TOKEN)
    if options.get("service"):
        options.setdefault("token_env", "UC_SERVICE_TOKEN")
    else:
        options.setdefault("user", "alice")
    script = Script(*answers)
    out = io.StringIO()
    code = run_terminal(RunOptions(url=url, **options), ask=script, out=out)
    return CliResult(code, out.getvalue(), script)


def wait_status(url: str, run_id: str, status: str, *, service: bool = False,
                timeout_s: float = 20.0) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {SERVICE_TOKEN if service else TOKEN}"}
    if not service:
        headers["X-User-Id"] = "alice"
    deadline = time.monotonic() + timeout_s
    with httpx.Client(base_url=url + "/api/v1", headers=headers) as client:
        while True:
            body = client.get(f"/runs/{run_id}").json()
            if body.get("status") == status:
                return dict(body)
            if time.monotonic() > deadline:
                raise AssertionError(f"run {run_id} never reached {status!r}: {body}")
            time.sleep(0.05)


def tiny_png(color: bytes = b"\xff\x00\x00") -> bytes:
    import struct

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + color * 4 for _ in range(3))
    ihdr = struct.pack(">IIBBBBB", 4, 3, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def zip_bytes(files: dict[str, bytes]) -> bytes:
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()
