"""Milestone 11: ``vdreamer run`` against a real gateway process (REST/SSE over TCP).

A live uvicorn server runs the API kit's stack (gateway + workers); the client
talks to it exactly as an app would. Prompts are scripted. Reconnect and the
protocol check use ``httpx.MockTransport`` to fail on purpose.
"""

from __future__ import annotations

import io
import json
import socket
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

HERE = Path(__file__).resolve().parent
for extra in (HERE.parent / "api", HERE.parent / "conformance"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from _api import SERVICE_TOKEN, TOKEN, headers, make_stack  # noqa: E402
from record_kit import KIT_AGENTS  # noqa: E402

from vdreamer.client import RunOptions, Session, Style, run_terminal, sse_events  # noqa: E402


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def gateway_url() -> Iterator[str]:
    import uvicorn

    stack = make_stack(
        workers=True,
        extra_agents=KIT_AGENTS,
        worker_agents=("conf-chat", "flow", "review", "ask", "sitecheck", "echo"),
        poll_interval_s=0.05,
    )
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(stack.app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(10)
        stack.close()


class Script:
    """Scripted answers to the client's prompts; records what it was asked."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def _run(url: str, monkeypatch: pytest.MonkeyPatch, *answers: str, **options: Any) -> tuple[
    int, str, Script
]:
    monkeypatch.setenv("VDREAMER_INTERNAL_TOKEN", TOKEN)
    monkeypatch.setenv("SVC_TOKEN", SERVICE_TOKEN)
    script = Script(*answers)
    out = io.StringIO()
    options.setdefault("user", "alice")
    code = run_terminal(RunOptions(url=url, **options), ask=script, out=out)
    return code, out.getvalue(), script


def test_a_chat_turn_streams_thinking_and_the_answer_once(
    gateway_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, out, _ = _run(gateway_url, monkeypatch, agent="conf-chat", message="hi", once=True)
    assert code == 0, out
    assert out.count("Hello! How can I help you today?") == 1  # completed before deltas: once
    assert "[thinking]" in out and "A greeting: reply briefly." in out
    assert "[progress] reading the question 25%" in out
    assert "[log info] answering from general knowledge" in out
    assert "[run completed]" in out and "\x1b[" not in out  # no color into a pipe


def test_the_chat_loop_continues_the_thread(
    gateway_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, out, script = _run(
        gateway_url, monkeypatch, "second", "", agent="echo", message="first", field_name="text"
    )
    assert code == 0, out
    assert out.count("[run completed]") == 2
    assert script.prompts.count("you> ") == 2  # asked again, then an empty line ended it


def test_a_generic_interrupt_is_answered_inline(
    gateway_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, out, script = _run(
        gateway_url, monkeypatch, "respond", "eu-west",
        agent="ask", input_json=json.dumps({"question": "backup"}), once=True,
    )
    assert code == 0, out
    assert "[interrupt generic]" in out and "[interrupt resolved: respond by alice]" in out
    assert any("action [" in prompt for prompt in script.prompts)
    assert out.rstrip().endswith("[run completed]")


def test_an_invalid_edit_is_shown_and_asked_again(
    gateway_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, out, script = _run(
        gateway_url, monkeypatch,
        "edit", json.dumps({"sql": "", "limit": 5000}),  # refused: 422 invalid_edit
        "edit", json.dumps({"sql": "SELECT 1", "limit": 10}),  # accepted, the run completes
        agent="flow", input_json=json.dumps({"question": "orders"}), once=True,
    )
    assert "422 invalid_edit" in out, out
    assert "edit_schema:" in out and "[interrupt resolved: edit by alice]" in out
    assert code == 0, out
    assert script.answers == []


def test_a_tool_review_sends_decisions(
    gateway_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, out, script = _run(
        gateway_url, monkeypatch, "approve",
        agent="review", message="mail ops", user="rita", roles=("mail_reviewer",), once=True,
    )
    assert code == 0, out
    assert "-> send_email(" in out and "<- ok sent to ops@example.internal" in out
    assert any("decision for send_email" in prompt for prompt in script.prompts)


def test_a_service_trigger_detaches_at_the_approval(
    gateway_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, out, _ = _run(
        gateway_url, monkeypatch,
        agent="sitecheck", input_json=json.dumps({"site_id": "S-7"}), service=True,
        token_env="SVC_TOKEN", user=None, idempotency_key="mail-7",
    )
    assert code == 3, out
    assert "[triggered " in out and "Idempotency-Key mail-7" in out
    assert "reattach with: vdreamer run sitecheck --thread" in out


def test_reattach_to_a_waiting_run(gateway_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with httpx.Client(base_url=gateway_url + "/api/v1", headers=headers()) as client:
        thread_id = client.post("/agents/ask/threads").json()["thread_id"]
        run_id = client.post(
            f"/threads/{thread_id}/runs", json={"input": {"question": "dr site"}}
        ).json()["run_id"]
        deadline = time.monotonic() + 10
        while client.get(f"/runs/{run_id}").json()["status"] != "awaiting_approval":
            assert time.monotonic() < deadline
            time.sleep(0.05)
    code, out, _ = _run(
        gateway_url, monkeypatch, "respond", "us-east", agent="ask", thread=thread_id, once=True
    )
    assert code == 0, out
    assert f"[reattaching to {run_id} on {thread_id}]" in out
    assert "[run completed]" in out


def test_a_missing_token_or_agent(gateway_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE", raising=False)
    assert run_terminal(RunOptions(agent="echo", url=gateway_url, token_env="NOPE")) == 1
    code, _, _ = _run(gateway_url, monkeypatch, agent="no-such-agent", message="x")
    assert code == 1


# -- reconnect, dedup and the protocol check, with a scripted transport -------------------


def _frame(seq: int, type_: str, payload: dict[str, Any], v: str = "1.0") -> str:
    event = {"v": v, "type": type_, "run_id": "r1", "thread_id": "t1", "agent": "a",
             "seq": seq, "ts": "x", "ns": [], "payload": payload}
    return f"id: r1:{seq}\ndata: {json.dumps(event)}\n\n"


def test_follow_reconnects_with_last_event_id_and_dedups() -> None:
    requests: list[httpx.Request] = []
    bodies = [
        _frame(1, "run.status", {"status": "running"})
        + _frame(2, "token.delta", {"message_id": "m", "text": "Hel"}),  # then the drop
        _frame(2, "token.delta", {"message_id": "m", "text": "Hel"})  # replayed again
        + ": heartbeat\n\n"
        + _frame(3, "token.delta", {"message_id": "m", "text": "lo"})
        + _frame(4, "brand.new", {})
        + _frame(5, "run.status", {"status": "completed"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=bodies[len(requests) - 1],
                              headers={"Content-Type": "text/event-stream"})

    from vdreamer.client import Gateway

    gateway = Gateway("http://gw", "token-0123456789abcdef", "alice",
                      transport=httpx.MockTransport(handler))
    out = io.StringIO()
    delays: list[float] = []
    session = Session(gateway, "a", Script(), out, Style(False), sleep=delays.append)
    assert session.follow("r1") == "completed"
    assert "Last-Event-ID" not in requests[0].headers
    assert requests[1].headers["Last-Event-ID"] == "r1:2"
    assert requests[0].headers["X-Protocol-Version"] == "1"
    text = out.getvalue()
    assert text.count("Hel") == 1 and "\nlo\n" in text  # the replayed seq 2 printed once
    assert "(ignoring unknown event 'brand.new')" in text
    assert len(delays) == 1 and 0.3 <= delays[0] <= 0.7  # backoff with jitter


def test_another_protocol_major_is_refused() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"agents": []}, headers={"X-Protocol-Version": "2.0"})

    from vdreamer.client import ClientError, Gateway

    gateway = Gateway("http://gw", "t" * 20, "alice", transport=httpx.MockTransport(handler))
    with pytest.raises(ClientError, match="protocol 2.0"):
        gateway.get("/agents")


def test_sse_parsing() -> None:
    lines = [": hb", "id: r:1", "data: {\"a\":", "data: 1}", "", "data: {\"b\":2}"]
    assert list(sse_events(iter(lines))) == [("r:1", {"a": 1}), (None, {"b": 2})]
