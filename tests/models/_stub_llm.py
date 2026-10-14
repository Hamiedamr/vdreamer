"""Stub model servers for Milestone 7: an OpenAI-compatible (vLLM) and an Ollama API.

Both are small FastAPI apps served by uvicorn on a loopback port in a
background thread, so the real LangChain clients (``ChatOpenAI`` over the
``openai`` SDK, ``ChatOllama`` over the ``ollama`` SDK) talk to them over
real HTTP. Each server answers from a script of :class:`Reply` turns (the
last one repeats) and records every request body it saw.

- vLLM: ``GET /v1/models``, ``POST /v1/chat/completions`` (SSE streaming or
  one JSON body). Reasoning rides ``delta.reasoning_content`` (vLLM's
  ``--reasoning-parser`` field) when :attr:`Reply.reasoning` is set.
- Ollama: ``GET /api/tags``, ``GET /api/version``, ``POST /api/chat``
  (NDJSON). Reasoning rides ``message.thinking`` when the request asked for
  ``think``; otherwise it is inlined as ``<think>…</think>`` text, the way a
  think-tag model emits it.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/" + "think" + ">"


@dataclass
class Reply:
    """One scripted assistant turn."""

    content: str = "ok"
    reasoning: str | None = None
    tool_calls: Sequence[dict[str, Any]] = ()
    #: Streamed pieces: the reply is split into chunks of this many characters.
    chunk: int = 4
    #: For the think-tag style: put the reasoning inline in the content.
    inline_think: bool = False


@dataclass
class StubServer:
    """A running stub; ``base_url`` is what models.yaml points at."""

    kind: str
    base_url: str
    models: list[str]
    script: list[Reply]
    requests: list[dict[str, Any]] = field(default_factory=list)
    _server: Any = None
    _thread: Any = None

    def next_reply(self) -> Reply:
        if len(self.script) > 1:
            return self.script.pop(0)
        return self.script[0]

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def _pieces(text: str, size: int) -> Iterator[str]:
    for start in range(0, len(text), max(1, size)):
        yield text[start : start + size]


def _vllm_app(stub: StubServer) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": m, "object": "model"} for m in stub.models]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        body = await request.json()
        stub.requests.append(body)
        if body.get("model") not in stub.models:
            return JSONResponse(
                {"error": {"message": f"model {body.get('model')!r} not served"}}, 404
            )
        reply = stub.next_reply()
        created = int(time.time())
        tool_calls = [
            {
                "index": i,
                "id": call.get("id", f"call_{i}"),
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
            }
            for i, call in enumerate(reply.tool_calls)
        ]
        usage = {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
        if not body.get("stream"):
            message: dict[str, Any] = {"role": "assistant", "content": reply.content}
            if reply.reasoning is not None:
                message["reasoning_content"] = reply.reasoning
            if tool_calls:
                message["tool_calls"] = [
                    {k: v for k, v in call.items() if k != "index"} for call in tool_calls
                ]
            return {
                "id": "cmpl-stub",
                "object": "chat.completion",
                "created": created,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ],
                "usage": usage,
            }

        def frame(delta: dict[str, Any], finish: str | None = None) -> str:
            data = {
                "id": "cmpl-stub",
                "object": "chat.completion.chunk",
                "created": created,
                "model": body["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(data)}\n\n"

        def stream() -> Iterator[str]:
            yield frame({"role": "assistant", "content": ""})
            for piece in _pieces(reply.reasoning or "", reply.chunk):
                yield frame({"reasoning_content": piece})
            for piece in _pieces(reply.content, reply.chunk):
                yield frame({"content": piece})
            for call in tool_calls:
                yield frame({"tool_calls": [call]})
            yield frame({}, "tool_calls" if tool_calls else "stop")
            if (body.get("stream_options") or {}).get("include_usage"):
                data = {
                    "id": "cmpl-stub",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body["model"],
                    "choices": [],
                    "usage": usage,
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def _ollama_app(stub: StubServer) -> FastAPI:
    app = FastAPI()

    @app.get("/api/version")
    async def version() -> dict[str, Any]:
        return {"version": "0.12.0"}

    @app.get("/api/tags")
    async def tags() -> dict[str, Any]:
        return {"models": [{"name": m, "model": m, "size": 1} for m in stub.models]}

    @app.post("/api/chat")
    async def chat(request: Request) -> Any:
        body = await request.json()
        stub.requests.append(body)
        if body.get("model") not in stub.models:
            return JSONResponse({"error": f"model '{body.get('model')}' not found"}, 404)
        reply = stub.next_reply()
        think = bool(body.get("think"))
        content = reply.content
        if reply.reasoning is not None and (reply.inline_think or not think):
            content = f"{THINK_OPEN}{reply.reasoning}{THINK_CLOSE}{content}"

        def line(message: dict[str, Any], done: bool = False) -> str:
            data: dict[str, Any] = {
                "model": body["model"],
                "created_at": "2026-10-14T00:00:00Z",
                "message": {"role": "assistant", **message},
                "done": done,
            }
            if done:
                data |= {"done_reason": "stop", "prompt_eval_count": 7, "eval_count": 5}
            return json.dumps(data) + "\n"

        def stream() -> Iterator[str]:
            if think and reply.reasoning is not None and not reply.inline_think:
                for piece in _pieces(reply.reasoning, reply.chunk):
                    yield line({"content": "", "thinking": piece})
            for piece in _pieces(content, reply.chunk):
                yield line({"content": piece})
            if reply.tool_calls:
                calls = [
                    {"function": {"name": c["name"], "arguments": c["args"]}}
                    for c in reply.tool_calls
                ]
                yield line({"content": "", "tool_calls": calls})
            yield line({"content": ""}, done=True)

        if body.get("stream", True):
            return StreamingResponse(stream(), media_type="application/x-ndjson")
        parts = [json.loads(chunk) for chunk in stream()]
        final = parts[-1]
        final["message"]["content"] = "".join(p["message"].get("content", "") for p in parts)
        return final

    return app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_stub(kind: str, models: Sequence[str], script: Sequence[Reply] = ()) -> StubServer:
    """Start a ``"vllm"`` or ``"ollama"`` stub on a free loopback port."""
    port = _free_port()
    base = f"http://127.0.0.1:{port}" + ("/v1" if kind == "vllm" else "")
    stub = StubServer(kind, base, list(models), list(script) or [Reply()])
    app = _vllm_app(stub) if kind == "vllm" else _ollama_app(stub)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name=f"stub-{kind}", daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:  # pragma: no cover - a broken host
            raise RuntimeError(f"stub {kind} did not start")
        time.sleep(0.02)
    stub._server, stub._thread = server, thread
    return stub
