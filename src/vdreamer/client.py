"""``vdreamer run <agent>`` — a terminal client for AI developers.

It talks to the gateway over REST/SSE only, exactly as a client app must
(``docs/protocol.md``). So it is the contract exercised from a terminal, not
a privileged backdoor, and not a client SDK: nothing here is importable API
for apps.

- **chat** (default): create a thread (or ``--thread``), send ``--message``
  / ``--input`` or prompt for one, stream the run, then prompt for the next
  turn. ``--once`` exits after one run.
- **trigger** (``--trigger``, automatic for a ``HEADLESS`` agent):
  ``POST /agents/{name}/trigger`` with an ``Idempotency-Key``, then stream.
- **reattach** (``--thread`` with an active run): follow that run from the
  thread's replay, then continue.
- **inbox** (``--inbox``): resolve the pending interrupts of this agent the
  user may see — how an approver (``approver_role``, four-eyes) acts on runs
  they do not own.

Streaming follows the protocol: ``X-Protocol-Version: 1``; reconnect with backoff and
jitter and ``Last-Event-ID``; dedup by ``(run_id, seq)``; unknown event types
are shown dimmed and skipped. An ``interrupt.requested`` is resolved inline:
the client fetches the record, offers its ``actions[]``, builds the body the
interrupt's source needs, and sends ``expected_version``; a ``409``
refreshes and asks again. The run then continues on the same stream.

Identity comes from flags (``--user``, ``--roles``) and the token from an
environment variable (``--token-env``, default ``VDREAMER_INTERNAL_TOKEN``):
the CLI plays the Web Backend.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, TextIO

import httpx

from vdreamer.events import HEARTBEAT_INTERVAL_S, PROTOCOL_VERSION

PROTOCOL_MAJOR: Final = PROTOCOL_VERSION.split(".", 1)[0]
TERMINAL: Final[frozenset[str]] = frozenset(
    {"completed", "failed", "rejected", "expired", "cancelled"}
)
ACTIVE: Final[frozenset[str]] = frozenset({"queued", "running", "awaiting_approval"})
KNOWN_TYPES: Final[frozenset[str]] = frozenset(
    {
        "run.status", "token.delta", "thinking.delta", "message.completed", "tool.call",
        "tool.result", "artifact.created", "citation", "interrupt.requested",
        "interrupt.resolved", "progress", "log",
    }
)

Ask = Callable[[str], str]


class ClientError(RuntimeError):
    """The gateway refused a request; the message is the gateway's own."""


class Detached(Exception):
    """The user stopped following; the run goes on server-side."""


@dataclass
class Style:
    """ANSI styling, off for pipes and ``--no-color``."""

    enabled: bool

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)


@dataclass
class Gateway:
    """The REST side: one ``httpx.Client`` with the Web Backend's headers."""

    base_url: str
    token: str
    user: str | None
    roles: tuple[str, ...] = ()
    transport: httpx.BaseTransport | None = None
    _client: httpx.Client = field(init=False)

    def __post_init__(self) -> None:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Protocol-Version": PROTOCOL_MAJOR,
        }
        if self.user:
            headers["X-User-Id"] = self.user
        if self.roles:
            headers["X-User-Roles"] = ",".join(self.roles)
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/") + "/api/v1",
            headers=headers,
            transport=self.transport,
            trust_env=False,
            # Reads block for heartbeats: 2x the interval is a dead connection.
            timeout=httpx.Timeout(30.0, read=2 * HEARTBEAT_INTERVAL_S + 5),
        )

    def close(self) -> None:
        self._client.close()

    def _check(self, response: httpx.Response) -> Any:
        server = response.headers.get("X-Protocol-Version", PROTOCOL_VERSION)
        if server.split(".", 1)[0] != PROTOCOL_MAJOR:
            raise ClientError(
                f"the gateway speaks protocol {server}, this client {PROTOCOL_MAJOR}.x"
            )
        if response.is_success:
            return response.json() if response.content else None
        try:
            error = response.json()["error"]
        except (ValueError, KeyError, TypeError):
            raise ClientError(f"HTTP {response.status_code}: {response.text[:300]}") from None
        raise ClientError(f"{response.status_code} {error.get('code')}: {error.get('message')}")

    def get(self, path: str, **params: Any) -> Any:
        return self._check(self._client.get(path, params=params or None))

    def post(self, path: str, body: Any = None, *, headers: Mapping[str, str] | None = None,
             params: Mapping[str, Any] | None = None) -> Any:
        return self._check(self._client.post(path, json=body, headers=headers, params=params))

    def post_raw(self, path: str, body: Any) -> httpx.Response:
        return self._client.post(path, json=body)

    def stream(self, path: str, last_event_id: str | None) -> Any:
        headers = {"Last-Event-ID": last_event_id} if last_event_id else None
        return self._client.stream("GET", path, headers=headers)


def sse_events(lines: Iterator[str]) -> Iterator[tuple[str | None, dict[str, Any]]]:
    """``(id, envelope)`` per SSE event; comments (heartbeats) skipped."""
    event_id: str | None = None
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                yield event_id, json.loads("\n".join(data))
            event_id, data = None, []
        elif line.startswith(":"):
            continue
        else:
            name, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if name == "id":
                event_id = value
            elif name == "data":
                data.append(value)
    if data:
        yield event_id, json.loads("\n".join(data))


@dataclass
class Printer:
    """Renders events to the terminal; keeps just enough state to print deltas once."""

    out: TextIO
    style: Style
    printed: dict[str, int] = field(default_factory=dict)
    completed: set[str] = field(default_factory=set)
    open_line: bool = False
    thinking_open: str | None = None
    citations: int = 0

    def _line(self, text: str) -> None:
        if self.open_line:
            self.out.write("\n")
            self.open_line = False
        self.out.write(text + "\n")
        self.out.flush()

    def _inline(self, text: str) -> None:
        self.out.write(text)
        self.out.flush()
        self.open_line = True

    def event(self, envelope: Mapping[str, Any]) -> None:
        kind = str(envelope.get("type"))
        payload: Mapping[str, Any] = envelope.get("payload") or {}
        handler = getattr(self, "_" + kind.replace(".", "_"), None)
        if kind not in KNOWN_TYPES or handler is None:
            self._line(self.style.dim(f"  (ignoring unknown event {kind!r})"))
            return
        handler(payload)

    def _run_status(self, p: Mapping[str, Any]) -> None:
        status = str(p.get("status"))
        color = {"completed": self.style.green, "failed": self.style.red,
                 "awaiting_approval": self.style.yellow}.get(status, self.style.dim)
        detail = ""
        error = p.get("error")
        if isinstance(error, Mapping):
            detail = f" — {error.get('code')}: {error.get('message')}"
        self._line(color(f"[run {status}]{detail}"))

    def _thinking_delta(self, p: Mapping[str, Any]) -> None:
        if str(p["message_id"]) in self.completed:
            return
        if self.thinking_open != p["message_id"]:
            if self.open_line:
                self.out.write("\n")
            # Same shape as a completed message's thinking: "[thinking] text".
            self._inline(self.style.dim("[thinking] "))
            self.thinking_open = str(p["message_id"])
        self._inline(self.style.dim(str(p["text"])))

    def _token_delta(self, p: Mapping[str, Any]) -> None:
        message_id = str(p["message_id"])
        if message_id in self.completed:
            return  # message.completed already printed the full text
        if message_id not in self.printed:
            self._line(self.style.bold("assistant>"))
            self.printed[message_id] = 0
        self._inline(str(p["text"]))
        self.printed[message_id] += len(str(p["text"]))

    def _message_completed(self, p: Mapping[str, Any]) -> None:
        message_id = str(p["message_id"])
        if message_id in self.completed:
            return  # a resumed node re-emits it with the same id
        text = str(p.get("text") or "")
        thinking = p.get("thinking")
        if thinking and self.thinking_open != message_id:
            self._line(self.style.dim(f"[thinking] {thinking}"))
        done = self.printed.get(message_id)
        if done is None and text:
            self._line(self.style.bold("assistant>"))
            self._inline(text)
        elif done is not None and len(text) > done:
            self._inline(text[done:])
        self.completed.add(message_id)
        if self.open_line:
            self._line("")

    def _tool_call(self, p: Mapping[str, Any]) -> None:
        args = json.dumps(p.get("args"), ensure_ascii=False)
        self._line(self.style.cyan(f"-> {p.get('name')}({args[:200]})") + self.style.dim(
            f" [{p.get('source')}]"
        ))

    def _tool_result(self, p: Mapping[str, Any]) -> None:
        ok = p.get("status") == "ok"
        mark = self.style.green("<- ok") if ok else self.style.red("<- error")
        summary = str(p.get("summary") or "").replace("\n", " ")[:200]
        extra = f" artifacts: {', '.join(p['artifact_ids'])}" if p.get("artifact_ids") else ""
        self._line(f"{mark} {summary}" + self.style.dim(f" ({p.get('duration_ms')} ms){extra}"))

    def _artifact_created(self, p: Mapping[str, Any]) -> None:
        renderer = f", renderer {p['renderer']}" if p.get("renderer") else ""
        self._line(self.style.cyan(f"[artifact] {p.get('kind')} {p.get('name')!r}{renderer}")
                   + self.style.dim(f" {p.get('download_url')}"))

    def _citation(self, p: Mapping[str, Any]) -> None:
        self.citations += 1
        source = p.get("source") or {}
        locator = p.get("locator") or {}
        where = ", ".join(f"{k} {v}" for k, v in locator.items())
        target = source.get("uri") or source.get("artifact_id") or ""
        self._line(self.style.dim(
            f"[{self.citations}] {source.get('title')}{' — ' + where if where else ''} ({target})"
        ))

    def _interrupt_requested(self, p: Mapping[str, Any]) -> None:
        role = f", approver role {p['approver_role']}" if p.get("approver_role") else ""
        self._line(self.style.yellow(
            f"[interrupt {p.get('kind')}] actions: {', '.join(p.get('actions') or [])}{role}"
        ))

    def _interrupt_resolved(self, p: Mapping[str, Any]) -> None:
        who = p.get("resolved_by")
        self._line(self.style.dim(f"[interrupt resolved: {p.get('action')} by {who}]"))

    def _progress(self, p: Mapping[str, Any]) -> None:
        # pct is 0–100; ``:g`` keeps a fraction visible (0.2 → "0.2%", not "0%").
        pct = f" {p['pct']:g}%" if isinstance(p.get("pct"), (int, float)) else ""
        self._line(self.style.dim(f"[progress] {p.get('label')}{pct}"))

    def _log(self, p: Mapping[str, Any]) -> None:
        self._line(self.style.dim(f"[log {p.get('level')}] {p.get('message')}"))


# -- interrupts ---------------------------------------------------------------------------


def _ask_json(ask: Ask, prompt: str) -> Any:
    while True:
        raw = ask(prompt).strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            ask(f"  not JSON ({exc.msg}) — press Enter to retry ")


def _choose(ask: Ask, prompt: str, options: list[str]) -> str:
    while True:
        answer = ask(f"{prompt} [{'/'.join(options)}]: ").strip()
        if answer in options:
            return answer
        matches = [option for option in options if option.startswith(answer)] if answer else []
        if len(matches) == 1:
            return matches[0]


def build_resolution(record: Mapping[str, Any], ask: Ask, out: TextIO) -> dict[str, Any]:
    """The resolve body for ``record``, asked of the user; without ``expected_version``."""
    kind = str(record.get("kind"))
    actions = [str(action) for action in record.get("actions") or []]
    payload = record.get("payload") or {}
    if kind == "tool_review":
        configs = {
            str(c.get("action_name")): c for c in payload.get("review_configs") or []
            if isinstance(c, Mapping)
        }
        decisions = []
        for request in payload.get("action_requests") or []:
            name = str(request.get("name"))
            config = configs.get(name) or {}
            allowed = [str(d) for d in config.get("allowed_decisions") or actions]
            out.write(f"  tool {name}({json.dumps(request.get('args'), ensure_ascii=False)})\n")
            action = _choose(ask, f"  decision for {name}", allowed)
            decision: dict[str, Any] = {"action": action}
            if action == "edit":
                decision["payload"] = _ask_json(ask, "  edited args (JSON): ") or {}
            elif action in ("reject", "respond"):
                text = ask("  message: ").strip()
                if text:
                    decision["text"] = text
            decisions.append(decision)
        return {"decisions": decisions}

    action = _choose(ask, "  action", actions)
    body: dict[str, Any] = {"action": action}
    if kind == "mcp_elicitation" and action == "edit":
        answers: dict[str, Any] = {}
        for request in payload.get("requests") or []:
            out.write(f"  {request.get('message')}  schema: "
                      f"{json.dumps(request.get('requested_schema'), ensure_ascii=False)}\n")
            answers[str(request.get("key"))] = _ask_json(ask, "  answer (JSON): ") or {}
        body["payload"] = answers
    elif action == "edit":
        if record.get("edit_schema"):
            out.write(f"  edit_schema: {json.dumps(record['edit_schema'], ensure_ascii=False)}\n")
        body["payload"] = _ask_json(ask, "  edited payload (JSON): ") or {}
    elif action in ("feedback", "respond"):
        body["text"] = ask("  text: ")
    elif action == "reject":
        text = ask("  reason (optional): ").strip()
        if text:
            body["text"] = text
    elif kind == "generic" and action not in ("approve",):
        value = _ask_json(ask, "  payload (JSON, Enter for none): ")
        if value is not None:
            body["payload"] = value
    return body


# -- the session --------------------------------------------------------------------------


@dataclass
class Session:
    gateway: Gateway
    agent: str
    ask: Ask
    out: TextIO
    style: Style
    sleep: Callable[[float], None] = time.sleep
    max_reconnects: int = 20

    def __post_init__(self) -> None:
        self.printer = Printer(self.out, self.style)
        self.seen: set[tuple[str, int]] = set()

    def say(self, text: str) -> None:
        self.printer._line(text)

    def follow(self, run_id: str, *, path: str | None = None) -> str:
        """Stream ``run_id`` to a terminal status, resolving interrupts; the final status."""
        path = path or f"/runs/{run_id}/events"
        last_id: str | None = None
        status: str | None = None
        failures = 0
        while True:
            try:
                with self.gateway.stream(path, last_id) as response:
                    if response.status_code != 200:
                        response.read()
                        self.gateway._check(response)
                    failures = 0
                    for event_id, envelope in sse_events(response.iter_lines()):
                        major = str(envelope.get("v", "")).split(".", 1)[0]
                        if major != PROTOCOL_MAJOR:
                            raise ClientError(f"event of protocol {envelope.get('v')}: refusing")
                        if event_id:
                            last_id = event_id
                        key = (str(envelope["run_id"]), int(envelope["seq"]))
                        if key in self.seen:
                            continue  # dedup by seq (a replay after reconnect)
                        self.seen.add(key)
                        self.printer.event(envelope)
                        if envelope["type"] == "run.status" and envelope["run_id"] == run_id:
                            status = str(envelope["payload"].get("status"))
                            if status == "awaiting_approval":
                                self.resolve_pending(run_id)
                            elif status in TERMINAL:
                                return status
                if status in TERMINAL:
                    return str(status)
                raise httpx.RemoteProtocolError("the stream ended before a terminal status")
            except (httpx.TransportError, httpx.RemoteProtocolError) as exc:
                failures += 1
                if failures > self.max_reconnects:
                    raise ClientError(f"lost the stream: {exc}") from exc
                delay = min(30.0, 0.5 * 2 ** (failures - 1)) * random.uniform(0.75, 1.25)
                self.say(self.style.dim(f"[reconnecting in {delay:.1f}s: {type(exc).__name__}]"))
                self.sleep(delay)

    def resolve_pending(self, run_id: str) -> None:
        """Resolve every pending interrupt of ``run_id`` this user may see."""
        try:
            inbox = self.gateway.get("/interrupts")["interrupts"]
        except ClientError as exc:  # a service token reads no inbox
            raise Detached(f"the run waits for an approver ({exc})") from exc
        records = [
            r for r in inbox if r.get("run_id") == run_id and r.get("status") == "pending"
        ]
        if not records:
            current = self.gateway.get(f"/runs/{run_id}").get("status")
            if current != "awaiting_approval":
                return  # a replayed pause, resolved since: keep reading
            raise Detached(
                "the run waits for an approver this user cannot act for "
                "(another user/role must resolve it)"
            )
        for record in records:
            self.resolve(record)

    def resolve(self, record: Mapping[str, Any]) -> None:
        interrupt_id = str(record["interrupt_id"])
        while True:
            self.say(self.style.yellow(f"interrupt {record.get('kind')} ({interrupt_id})"))
            shown = json.dumps(record.get("payload"), indent=2, ensure_ascii=False)
            self.say(shown if len(shown) <= 2000 else shown[:2000] + "\n  …")
            body = build_resolution(record, self.ask, self.out)
            body["expected_version"] = record["version"]
            response = self.gateway.post_raw(f"/interrupts/{interrupt_id}/resolve", body)
            if response.status_code == 409:
                self.say(self.style.yellow("[409: it changed meanwhile — refreshing]"))
                record = self.gateway.get(f"/interrupts/{interrupt_id}")
                if record.get("status") != "pending":
                    return
                continue
            if response.status_code == 422:
                error = response.json().get("error", {})
                self.say(self.style.red(f"[422 {error.get('code')}: {error.get('message')}]"))
                for detail in (error.get("details") or {}).get("errors") or []:
                    self.say(self.style.red(f"  {detail.get('loc')}: {detail.get('msg')}"))
                continue
            if response.status_code == 403:
                error = response.json().get("error", {})
                raise Detached(f"403 {error.get('code')}: {error.get('message')}")
            self.gateway._check(response)
            return

    # -- starting runs ----------------------------------------------------------------------

    def start_chat(self, thread_id: str, run_input: Any) -> str:
        created = self.gateway.post(f"/threads/{thread_id}/runs", {"input": run_input})
        return str(created["run_id"])

    def trigger(self, run_input: Any, key: str, callback: str | None) -> tuple[str, str]:
        body: dict[str, Any] = {"input": run_input}
        if callback:
            body["callback"] = callback
        created = self.gateway.post(
            f"/agents/{self.agent}/trigger", body, headers={"Idempotency-Key": key}
        )
        return str(created["run_id"]), str(created["thread_id"])

    def latest_run(self, thread_id: str) -> tuple[str, str] | None:
        """``(run_id, status)`` of the thread's active run, else its most recent one."""
        runs = list(self.gateway.get(f"/threads/{thread_id}").get("runs") or [])
        if not runs:
            return None
        active = [run for run in runs if run.get("status") in ACTIVE]
        chosen = active[-1] if active else max(runs, key=lambda run: str(run["created_at"]))
        return str(chosen["run_id"]), str(chosen["status"])


@dataclass(frozen=True)
class RunOptions:
    agent: str
    url: str = "http://127.0.0.1:8000"
    token_env: str = "VDREAMER_INTERNAL_TOKEN"
    user: str | None = None
    roles: tuple[str, ...] = ()
    thread: str | None = None
    input_json: str | None = None
    message: str | None = None
    field_name: str | None = None
    trigger: bool = False
    idempotency_key: str | None = None
    callback: str | None = None
    service: bool = False
    inbox: bool = False
    once: bool = False
    color: bool = True


def _input_from_text(text: str, field_name: str | None) -> Any:
    """A typed line → the run input: ``{field: text}``, or the LangGraph messages shape."""
    if field_name:
        return {field_name: text}
    return {"messages": [{"role": "user", "content": text}]}


def _text_field(agent: Mapping[str, Any], field_name: str | None) -> str | None:
    """Where typed text goes for this agent: ``--field``, else its schema's one text field.

    An agent without an ``input_schema``, or whose schema has ``messages``, takes
    the messages shape (``None``). A schema with exactly one string property
    takes the text there; any other schema needs ``--field`` or ``--input`` —
    the gateway would silently drop a ``messages`` key the schema does not have.
    """
    if field_name:
        return field_name
    schema = agent.get("input_schema")
    properties = schema.get("properties") if isinstance(schema, Mapping) else None
    if not isinstance(properties, Mapping) or "messages" in properties:
        return None
    text_fields = [
        name for name, spec in properties.items()
        if isinstance(spec, Mapping) and spec.get("type") == "string"
    ]
    if len(text_fields) == 1:
        return str(text_fields[0])
    raise ClientError(
        f"agent {agent.get('name')!r} takes {sorted(properties)} (its input_schema), not "
        "messages: pass --input JSON, or --field NAME to send the text as {NAME: text}"
    )


def run_terminal(
    options: RunOptions,
    *,
    ask: Ask | None = None,
    out: TextIO | None = None,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """CLI entry point for ``vdreamer run``."""
    stream = out or sys.stdout
    prompt: Ask = ask or input
    token = os.environ.get(options.token_env, "")
    if not token:
        print(f"run: {options.token_env} is not set (the gateway's bearer token)", file=sys.stderr)
        return 1
    service = options.service
    user = None if service else (
        options.user or os.environ.get("USER") or os.environ.get("USERNAME") or "developer"
    )
    gateway = Gateway(options.url, token, user, options.roles, transport=transport)
    style = Style(options.color and stream.isatty())
    session = Session(gateway, options.agent, prompt, stream, style, sleep=sleep)
    thread_id = options.thread
    try:
        if service:  # a service token opens /trigger and its runs only
            agent: Mapping[str, Any] | None = {"name": options.agent, "mode": "headless"}
        else:
            agents = gateway.get("/agents")["agents"]
            agent = next((a for a in agents if a["name"] == options.agent), None)
        if agent is None:
            print(f"run: no agent {options.agent!r} at {options.url}", file=sys.stderr)
            return 1
        headless = service or options.trigger or agent.get("mode") == "headless"
        resolved_agent: Mapping[str, Any] = agent

        def text_input(text: str) -> Any:
            return _input_from_text(text, _text_field(resolved_agent, options.field_name))

        def first_input() -> Any:
            if options.input_json is not None:
                return json.loads(options.input_json)
            if options.message is not None:
                return text_input(options.message)
            return None

        if options.inbox:
            # An approver acts on runs they do not own (another user's, a service's):
            # the inbox is their view, not the thread (owner-scoped threads).
            pending = [
                r for r in gateway.get("/interrupts")["interrupts"]
                if r.get("agent") == options.agent and r.get("status") == "pending"
            ]
            if not pending:
                session.say(style.dim(f"[no pending interrupts for {options.agent}]"))
                return 0
            for record in pending:
                session.say(style.dim(f"[run {record['run_id']} on {record['thread_id']}]"))
                session.resolve(record)
            return 0
        if thread_id is not None:
            latest = session.latest_run(thread_id)
            # A new message continues the thread: a finished run is replayed only without one.
            fresh = not headless and first_input() is not None
            if latest is not None and (latest[1] in ACTIVE or not fresh):
                run_id, current = latest
                verb = "reattaching to" if current in ACTIVE else "replaying"
                session.say(style.dim(f"[{verb} {run_id} on {thread_id}]"))
                # The run's stream replays from its first event, then goes live.
                status = session.follow(run_id)
                if headless or (options.once and not fresh):
                    return 0 if status == "completed" else 2
        if headless:
            run_input = first_input()
            if run_input is None:
                run_input = _ask_json(prompt, "input (JSON): ") or {}
            key = options.idempotency_key or f"cli-{uuid.uuid4().hex}"
            run_id, thread_id = session.trigger(run_input, key, options.callback)
            session.say(style.dim(f"[triggered {run_id} on {thread_id}, Idempotency-Key {key}]"))
            status = session.follow(run_id)
            return 0 if status == "completed" else 2

        if thread_id is None:
            thread_id = str(gateway.post(f"/agents/{options.agent}/threads")["thread_id"])
            session.say(style.dim(f"[thread {thread_id}]"))
        run_input = first_input()
        status = "completed"
        while True:
            if run_input is None:
                try:
                    text = prompt(style.bold("you> "))
                except (EOFError, KeyboardInterrupt):
                    # Quitting at an idle prompt (Ctrl+C, Ctrl+D / Ctrl+Z): nothing
                    # is running, so this is a plain exit, not a detach.
                    text = ""
                    session.say("")
                if not text.strip():
                    session.say(style.dim(
                        f"[thread kept — continue it with: vdreamer run {options.agent} "
                        f"--thread {thread_id}]"
                    ))
                    break
                run_input = text_input(text)
            run_id = session.start_chat(thread_id, run_input)
            status = session.follow(run_id)
            run_input = None
            if options.once:
                break
        return 0 if status == "completed" else 2
    except Detached as exc:
        session.say(style.yellow(f"[detached: {exc}]"))
        if thread_id:
            session.say(f"reattach with: vdreamer run {options.agent} --thread {thread_id}")
        return 3
    except KeyboardInterrupt:
        session.say(style.yellow("[detached — the run continues on the server]"))
        if thread_id:
            session.say(f"reattach with: vdreamer run {options.agent} --thread {thread_id}")
        return 130
    except (ClientError, httpx.HTTPError, json.JSONDecodeError) as exc:
        print(f"run: {exc}", file=sys.stderr)
        return 1
    finally:
        gateway.close()


__all__ = ["RunOptions", "build_resolution", "run_terminal", "sse_events"]
