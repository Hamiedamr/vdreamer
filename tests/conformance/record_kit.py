"""Record the conformance kit from the real gateway (PRD §6.1).

``uv run python tests/conformance/record_kit.py`` re-records every scenario
into ``src/vdreamer/conformance/kit/`` — deliberately, when the protocol or a
driver changes (cross-cutting rule 4: a protocol change updates the kit in the
same PR). ``test_conformance_kit.py`` re-records in memory on every run and
checks the committed kit still folds to what the gateway produces today.

Each scenario drives the in-process REST/SSE stack (``tests/api/_api.py``'s
live stack: gateway + workers on the in-memory stores) exactly as a client
would — create a thread, start a run, read ``/runs/{id}/events``, drop the
connection, resume with ``Last-Event-ID``, resolve interrupts through
``/interrupts/{id}/resolve`` — and records the SSE bodies verbatim. Then
:func:`normalize` replaces every generated id and timestamp with a stable
placeholder (``run_1``, ``int_1``, ``2026-01-01T00:00:01Z`` …) so recordings
diff cleanly.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypedDict

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for extra in (HERE, REPO / "tests" / "api"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from _api import headers, make_stack  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from reference_fold import fold_recording  # noqa: E402

from vdreamer import (  # noqa: E402
    ArtifactKind,
    BaseAgent,
    BuildContext,
    Mode,
    emit_artifact,
    emit_citation,
    emit_log,
    emit_progress,
)
from vdreamer.events import PROTOCOL_VERSION  # noqa: E402

KIT_DIR = REPO / "src" / "vdreamer" / "conformance" / "kit"

# Warm the table stack before the workers start: the kit stack runs ~17 worker
# loops on one event-loop thread, and a first `import pyarrow` (numpy, pandas)
# inside a run can crawl for minutes under that GIL contention on Windows.
import pyarrow.parquet  # noqa: E402, F401

# -- the kit's own agents -------------------------------------------------------------------


class ConfChatAgent(BaseAgent):
    """Chat: progress, a log, thinking, then a streamed answer."""

    name = "conf-chat"
    mode = Mode.CHAT
    expose_thinking = True

    def build(self, ctx: BuildContext) -> Any:
        def chat(state: MessagesState) -> dict[str, Any]:
            emit_progress("reading the question", 25.0)
            emit_log("answering from general knowledge", level="info")
            model = GenericFakeChatModel(
                messages=iter(
                    [
                        AIMessage(
                            "Hello! How can I help you today?",
                            id="msg_answer",
                            additional_kwargs={"reasoning_content": "A greeting: reply briefly."},
                        )
                    ]
                )
            )
            answer = model.invoke(state["messages"])
            emit_progress("done", 100.0)
            return {"messages": [answer]}

        graph = StateGraph(MessagesState)
        graph.add_node("chat", chat)
        graph.add_edge(START, "chat")
        graph.add_edge("chat", END)
        return graph


class _ResearchState(TypedDict, total=False):
    messages: Any


class ConfCitationsAgent(BaseAgent):
    """Chat: an answer grounded in two sources — a PDF artifact page and a URI."""

    name = "conf-cite"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> Any:
        def answer(state: MessagesState) -> dict[str, Any]:
            report = emit_artifact(
                "q3-report.pdf", ArtifactKind.FILE, b"%PDF-1.4 fake", mime_type="application/pdf",
                filename="q3-report.pdf",
            )
            model = GenericFakeChatModel(
                messages=iter([AIMessage("Revenue grew 12% [1], led by EMEA [2].", id="msg_cited")])
            )
            message = model.invoke(state["messages"])
            emit_citation(
                str(message.id),
                {"artifact_id": report.artifact_id, "title": "Q3 report"},
                locator={"page": 4},
                snippet="Revenue grew 12% year over year.",
                score=0.91,
            )
            emit_citation(
                str(message.id),
                {"uri": "kb://regions/emea", "title": "EMEA overview"},
                locator={"section": "2.1"},
                snippet="EMEA led the growth.",
            )
            return {"messages": [message]}

        graph = StateGraph(MessagesState)
        graph.add_node("answer", answer)
        graph.add_edge(START, "answer")
        graph.add_edge("answer", END)
        return graph


KIT_AGENTS: tuple[BaseAgent, ...] = (ConfChatAgent(), ConfCitationsAgent())

#: The agents the scenarios run (only these get a worker: idle pollers starve the loop).
WORKER_AGENTS: tuple[str, ...] = (
    "conf-chat", "conf-cite", "tooling", "flow", "review", "elicit", "ask", "reports",
)

# -- recording helpers ----------------------------------------------------------------------


def _user(text: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


def _start(client: TestClient, agent: str, run_input: Any) -> tuple[str, str]:
    thread = client.post(f"/api/v1/agents/{agent}/threads", headers=headers())
    assert thread.status_code == 201, thread.text
    thread_id = str(thread.json()["thread_id"])
    run = client.post(
        f"/api/v1/threads/{thread_id}/runs", json={"input": run_input}, headers=headers()
    )
    assert run.status_code == 202, run.text
    return thread_id, str(run.json()["run_id"])


def _blocks(lines: Iterator[str]) -> Iterator[list[str]]:
    """SSE event blocks (comments/heartbeats dropped: their timing is not the contract)."""
    block: list[str] = []
    for line in lines:
        if line == "":
            if block:
                yield block
            block = []
        elif not line.startswith(":"):
            block.append(line)
    if block:
        yield block


def _status_of(block: list[str]) -> str | None:
    for line in block:
        if line.startswith("data:"):
            event = json.loads(line[5:].strip())
            if event["type"] == "run.status":
                return str(event["payload"]["status"])
    return None


def _sse(
    client: TestClient,
    path: str,
    *,
    last_event_id: str | None = None,
    stop: Callable[[list[str], int], bool] | None = None,
    reset: bool = False,
) -> tuple[dict[str, Any], str | None]:
    """One SSE response, recorded; ``stop(block, n)`` drops the connection early."""
    extra = {"Last-Event-ID": last_event_id} if last_event_id else {}
    body: list[str] = []
    last_id: str | None = None
    dropped = False
    with client.stream("GET", path, headers=headers(**extra)) as response:
        assert response.status_code == 200, response.read()
        for count, block in enumerate(_blocks(response.iter_lines()), start=1):
            body.append("\n".join(block) + "\n\n")
            for line in block:
                if line.startswith("id:"):
                    last_id = line[3:].strip()
            if stop is not None and stop(block, count):
                dropped = True
                break
    step: dict[str, Any] = {
        "kind": "sse",
        "path": path,
        "last_event_id": last_event_id,
        "dropped": dropped,
        "body": "".join(body),
    }
    if reset:
        step["reset"] = True
    return step, last_id


def _resolve(client: TestClient, run_id: str, body: dict[str, Any], **who: Any) -> dict[str, Any]:
    inbox = client.get("/api/v1/interrupts", headers=headers(**who)).json()["interrupts"]
    (record,) = [r for r in inbox if r["run_id"] == run_id]
    path = f"/api/v1/interrupts/{record['interrupt_id']}/resolve"
    sent = {"expected_version": record["version"], **body}
    response = client.post(path, json=sent, headers=headers(**who))
    assert response.status_code == 200, response.text
    return {"kind": "resolve", "path": path, "body": sent, "status": response.status_code}


def _recording(
    name: str, description: str, agent: str, steps: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "protocol_version": PROTOCOL_VERSION,
        "agent": agent,
        "steps": steps,
    }


# -- scenarios ------------------------------------------------------------------------------


def chat(client: TestClient) -> dict[str, Any]:
    _, run_id = _start(client, "conf-chat", _user("hi"))
    step, _ = _sse(client, f"/api/v1/runs/{run_id}/events")
    return _recording(
        "chat",
        "A chat turn: run.status, progress, a log, thinking and token deltas, the completed "
        "message, then the terminal run.status (the server closes the stream after it).",
        "conf-chat",
        [step],
    )


def reconnect(client: TestClient) -> dict[str, Any]:
    thread_id, run_id = _start(client, "conf-chat", _user("hi"))
    path = f"/api/v1/runs/{run_id}/events"
    _wait(client, run_id, "completed")  # TestClient cannot drop a live stream (see _interrupt)
    first, _ = _sse(client, path, stop=lambda block, n: n == 4)
    ids = re.findall(r"^id: (.+)$", first["body"], flags=re.MULTILINE)
    # The client persisted a lagging Last-Event-ID (two events behind what it rendered):
    # the resumed stream repeats those two, and the fold must dedup them by seq.
    second, _ = _sse(client, path, last_event_id=ids[-3])
    refresh, _ = _sse(client, f"/api/v1/threads/{thread_id}/events", reset=True)
    return _recording(
        "reconnect",
        "Mid-run reconnect: the first connection drops after 4 events; the client resumes with "
        "a Last-Event-ID two events behind (duplicates to dedup by seq); then a page refresh "
        "rebuilds the state from /threads/{id}/events (reset: true).",
        "conf-chat",
        [first, second, refresh],
    )


def tool_overflow(client: TestClient) -> dict[str, Any]:
    _, run_id = _start(client, "tooling", _user("build the report"))
    step, _ = _sse(client, f"/api/v1/runs/{run_id}/events")
    return _recording(
        "tool_overflow",
        "A governed tool whose result is over the inline limit (§2.7): tool.call with redacted "
        "args, the overflow stored as an artifact, tool.result naming it in artifact_ids.",
        "tooling",
        [step],
    )


def _interrupt(
    client: TestClient,
    name: str,
    description: str,
    agent: str,
    run_input: Any,
    body: dict[str, Any],
    **who: Any,
) -> dict[str, Any]:
    _, run_id = _start(client, agent, run_input)
    path = f"/api/v1/runs/{run_id}/events"
    # A live client reads until awaiting_approval, drops, resolves, resumes. The
    # in-process TestClient cannot drop a stream that is still open, so the pause
    # is awaited by polling, and the first segment is the replay cut at the very
    # frame a live client would have stopped on — the same bytes (§3.3: replay ==
    # live). The resumed segment is a real Last-Event-ID request.
    _wait(client, run_id, "awaiting_approval")
    resolve = _resolve(client, run_id, body, **who)
    _wait(client, run_id, "completed")
    full, _ = _sse(client, path)
    blocks = [block for block in full["body"].split("\n\n") if block]
    cut = next(i for i, block in enumerate(blocks) if _status_of(block.split("\n")) == (
        "awaiting_approval"
    ))
    first = {**full, "dropped": True, "body": "".join(b + "\n\n" for b in blocks[: cut + 1])}
    last_id = next(
        line[3:].strip() for line in blocks[cut].split("\n") if line.startswith("id:")
    )
    rest, _ = _sse(client, path, last_event_id=last_id)
    return _recording(name, description, agent, [first, resolve, rest])


def _wait(client: TestClient, run_id: str, status: str, timeout_s: float = 10.0) -> None:
    import time

    deadline = time.monotonic() + timeout_s
    while True:
        body = client.get(f"/api/v1/runs/{run_id}", headers=headers()).json()
        if body.get("status") == status:
            return
        if time.monotonic() > deadline:
            raise AssertionError(f"run {run_id} never reached {status!r}: {body}")
        time.sleep(0.02)


def interrupt_require_approval(client: TestClient) -> dict[str, Any]:
    return _interrupt(
        client,
        "interrupt_require_approval",
        "require_approval inside an ApprovalGate (kind sql_review, with edit_schema): the stream "
        "pauses at awaiting_approval, the client drops it, approves via POST /resolve, and "
        "resumes with Last-Event-ID to interrupt.resolved and completion.",
        "flow",
        {"question": "orders"},
        {"action": "approve"},
    )


def interrupt_tool_review(client: TestClient) -> dict[str, Any]:
    return _interrupt(
        client,
        "interrupt_tool_review",
        "HumanInTheLoopMiddleware → kind tool_review, gated to approver_role mail_reviewer: the "
        "reviewer submits decisions[] (one per action request, in order).",
        "review",
        _user("mail ops"),
        {"decisions": [{"action": "approve"}]},
        user="rita",
        roles=("mail_reviewer",),
    )


def interrupt_mcp_elicitation(client: TestClient) -> dict[str, Any]:
    return _interrupt(
        client,
        "interrupt_mcp_elicitation",
        "An MCP elicitation → kind mcp_elicitation: the client answers the form with action "
        "edit and a payload keyed by request.",
        "elicit",
        {"text": "fx"},
        {"action": "edit", "payload": {"k1": {"date": "2026-11-01"}}},
    )


def interrupt_generic(client: TestClient) -> dict[str, Any]:
    return _interrupt(
        client,
        "interrupt_generic",
        "A raw interrupt() → kind generic: the client responds with text.",
        "ask",
        {"question": "backup"},
        {"action": "respond", "text": "eu-west"},
    )


def artifacts(client: TestClient) -> dict[str, Any]:
    _, run_id = _start(client, "reports", {"text": "q3"})
    step, _ = _sse(client, f"/api/v1/runs/{run_id}/events")
    return _recording(
        "artifacts",
        "Every artifact kind (§3.4): table, image, html (linking the others by artifact://), "
        "json with renderer verdict.v1, and a plain file — each with its preview block.",
        "reports",
        [step],
    )


def citations(client: TestClient) -> dict[str, Any]:
    _, run_id = _start(client, "conf-cite", _user("how did Q3 go?"))
    step, _ = _sse(client, f"/api/v1/runs/{run_id}/events")
    return _recording(
        "citations",
        "An answer with two citations bound to its message_id: one to a PDF artifact "
        "(locator.page), one to a URI (locator.section).",
        "conf-cite",
        [step],
    )


def _frame(seq: int, type_: str, payload: dict[str, Any], v: str = PROTOCOL_VERSION) -> str:
    event = {
        "v": v, "type": type_, "run_id": "run_1", "thread_id": "thread_1", "agent": "future",
        "seq": seq, "ts": f"2026-01-01T00:00:{seq:02d}Z", "ns": [], "payload": payload,
    }
    return f"id: run_1:{seq}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"


def forward_compat(client: TestClient | None = None) -> dict[str, Any]:
    """Hand-written: what a 1.0 client must survive that 1.0 itself never emits."""
    art = {
        "question_id": "run_1", "session_id": "thread_1", "mime_type": "application/octet-stream",
        "size_bytes": 10, "preview": {}, "metadata": {},
    }
    body = "".join(
        [
            ": heartbeat\n\n",
            _frame(1, "run.status", {"status": "running", "attempt": 1}),
            # A newer minor's event type, and a new field on a known payload: ignore / tolerate.
            _frame(2, "telemetry.sample", {"cpu": 0.4}, v="1.3"),
            _frame(3, "progress", {"label": "rendering", "pct": 50.0, "eta_s": 4}),
            # Unknown kind and unknown renderer: both render as downloads (§6.1).
            _frame(4, "artifact.created", {
                **art, "artifact_id": "art_1", "kind": "model3d", "name": "tower.glb",
                "view_url": "/api/v1/artifacts/art_1/raw",
                "download_url": "/api/v1/artifacts/art_1/download",
            }),
            _frame(5, "artifact.created", {
                **art, "artifact_id": "art_2", "kind": "json", "renderer": "chart.v9",
                "name": "series", "mime_type": "application/json",
                "view_url": "/api/v1/artifacts/art_2/raw",
                "download_url": "/api/v1/artifacts/art_2/download",
            }),
            ": heartbeat\n\n",
            # A duplicate delivery of seq 3: dedup by seq.
            _frame(3, "progress", {"label": "rendering", "pct": 50.0, "eta_s": 4}),
            _frame(6, "run.status", {"status": "completed", "attempt": 1}),
        ]
    )
    recording = _recording(
        "forward_compat",
        "Hand-written (never emitted by 1.0): a heartbeat comment, an unknown event type from a "
        "newer minor (v 1.3), an unknown payload field, an artifact of unknown kind and one "
        "with an unknown renderer (both render as downloads), and a duplicate seq.",
        "future",
        [{"kind": "sse", "path": "/api/v1/runs/run_1/events", "last_event_id": None,
          "dropped": False, "body": body}],
    )
    recording["expected_state"] = fold_recording(recording)
    return recording


SCENARIOS: dict[str, Callable[[TestClient], dict[str, Any]]] = {
    "chat": chat,
    "reconnect": reconnect,
    "tool_overflow": tool_overflow,
    "interrupt_require_approval": interrupt_require_approval,
    "interrupt_tool_review": interrupt_tool_review,
    "interrupt_mcp_elicitation": interrupt_mcp_elicitation,
    "interrupt_generic": interrupt_generic,
    "artifacts": artifacts,
    "citations": citations,
}

# -- normalization --------------------------------------------------------------------------

_ID = re.compile(
    r"\b(?:(?P<prefix>run|thread|art|cit|msg|call|int|lc_run-)_?-?)?"
    r"(?P<hex>[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b"
)
_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})")


class Normalizer:
    """Generated ids → ``<prefix>_<n>`` (first-seen order); timestamps → a fixed clock."""

    def __init__(self) -> None:
        self.ids: dict[str, str] = {}
        self.counts: dict[str, int] = {}
        self.stamps: dict[str, str] = {}

    def _id(self, match: re.Match[str]) -> str:
        raw = match.group(0)
        if raw not in self.ids:
            prefix = (match.group("prefix") or "id").rstrip("-").replace("lc_run", "lc_run")
            self.counts[prefix] = self.counts.get(prefix, 0) + 1
            self.ids[raw] = f"{prefix}_{self.counts[prefix]}"
        return self.ids[raw]

    def _ts(self, match: re.Match[str]) -> str:
        raw = match.group(0)
        if raw not in self.stamps:
            minute, second = divmod(len(self.stamps), 60)
            self.stamps[raw] = f"2026-01-01T00:{minute:02d}:{second:02d}Z"
        return self.stamps[raw]

    def text(self, value: str) -> str:
        return _TS.sub(self._ts, _ID.sub(self._id, value))

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if key == "interrupt_id" and isinstance(item, str) and item not in self.ids:
                    self.counts["int"] = self.counts.get("int", 0) + 1
                    self.ids[item] = f"int_{self.counts['int']}"
                if key == "duration_ms" and isinstance(item, (int, float)):
                    out[key] = 1  # wall-clock, not contract
                else:
                    out[key] = self.value(item)
            return out
        return value

    def sse(self, body: str) -> str:
        lines = []
        for line in body.split("\n"):
            if line.startswith("data:"):
                event = json.loads(line[5:].strip())
                lines.append("data: " + json.dumps(self.value(event), separators=(",", ":")))
            else:
                lines.append(self.text(line))
        return "\n".join(lines)


def normalize(recording: dict[str, Any]) -> dict[str, Any]:
    norm = Normalizer()
    steps = []
    for step in recording["steps"]:
        out = {key: norm.value(value) for key, value in step.items() if key != "body"}
        if step["kind"] == "sse":
            out["body"] = norm.sse(step["body"])
        else:
            out["body"] = norm.value(step["body"])
        steps.append(out)
    normalized = {**recording, "steps": steps}
    normalized["expected_state"] = fold_recording(normalized)
    return normalized


def record_all() -> dict[str, dict[str, Any]]:
    """Every scenario, recorded live and normalized (expected_state from the reference fold)."""
    stack = make_stack(
        workers=True,
        extra_agents=KIT_AGENTS,
        worker_agents=WORKER_AGENTS,
        poll_interval_s=0.05,
    )
    try:
        with TestClient(stack.app) as client:
            recorded = {name: normalize(scenario(client)) for name, scenario in SCENARIOS.items()}
    finally:
        stack.close()
    recorded["forward_compat"] = forward_compat()
    return recorded


def dump(recording: dict[str, Any]) -> bytes:
    return (json.dumps(recording, indent=2, ensure_ascii=False) + "\n").encode()


def main() -> int:
    KIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, recording in record_all().items():
        (KIT_DIR / f"{name}.json").write_bytes(dump(recording))
        print(f"recorded {name}: {len(recording['steps'])} step(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
