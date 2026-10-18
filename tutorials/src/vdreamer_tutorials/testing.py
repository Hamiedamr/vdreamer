"""Test support for the tutorials: a scripted chat model and an in-process agent session.

Lesson 6 (``t06_testing_and_traces``) walks through this module; every lesson's
tests use it. It is built on public APIs only:

- :class:`ScriptedChatModel` is a LangChain chat model that answers from a fixed
  script, records every prompt it receives, and fails loudly when the script runs
  out. Tests use it in place of the ``chat`` alias so they need no model server.
- :class:`AgentSession` drives one agent in-process with ``vdreamer.AgentRunner``
  (the runner a vdreamer worker uses), a LangGraph checkpointer, and a
  ``vdreamer.InMemoryEventSink`` that collects the protocol events a client would
  receive over SSE. Each :class:`RunResult` holds those events plus the final state.

What the session does *not* do is what the gateway and the worker add on top:
authentication, the run queue, ``awaiting_approval`` bookkeeping, permission
checks on who may resolve an approval, and artifact storage. Tests that need those
run the agent behind a real gateway instead (see ``tests/tutorials/_tutorial_kit.py``).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pydantic
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from vdreamer import AgentRunner, BaseAgent, EventEnvelope, InMemoryEventSink, VDreamerContext

__all__ = [
    "AgentSession",
    "RoutedChatModel",
    "RunResult",
    "ScriptExhausted",
    "ScriptedChatModel",
    "call",
    "calls",
    "reply",
    "scripted",
    "structured",
]


# -- the scripted model ---------------------------------------------------------------------


class ScriptExhausted(AssertionError):
    """The agent called the model more often than the test scripted."""


def _tool_name(tool: Any) -> str:
    if isinstance(tool, Mapping):
        function = tool.get("function")
        if isinstance(function, Mapping) and "name" in function:
            return str(function["name"])
        return str(tool.get("name", "?"))
    if isinstance(tool, type):
        return tool.__name__
    return str(getattr(tool, "name", type(tool).__name__))


class ScriptedChatModel(BaseChatModel):
    """Answers with ``replies`` in order, one per model call.

    ``prompts[i]`` is the exact message list the i-th call received, so a test
    can assert what the model *saw* (the system prompt, the tool results, the
    conversation history). ``bound_tools[i]`` lists the tool names of the i-th
    ``bind_tools`` call. An exception in the script is raised by that call — a
    model server failing (lesson 10). Running past the end of the script raises
    :class:`ScriptExhausted` instead of silently repeating an answer.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    replies: list[BaseMessage | Exception]
    delay_s: float = 0.0
    """Seconds each call takes: a slow model server (lesson 24's timeouts)."""
    prompts: list[list[BaseMessage]] = pydantic.Field(default_factory=list)
    bound_tools: list[list[str]] = pydantic.Field(default_factory=list)
    _lock: threading.Lock = pydantic.PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def calls_made(self) -> int:
        return len(self.prompts)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        """Record which tools the agent offered; the script decides what gets called."""
        self.bound_tools.append([_tool_name(tool) for tool in tools])
        return self

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[Any, Any]:
        """Parse the scripted reply's tool call into ``schema`` (see :func:`structured`).

        This is what ``method="function_calling"`` does on a real model: the
        model answers with one tool call named after the schema.
        """
        from langchain_core.output_parsers.openai_tools import PydanticToolsParser

        if not (isinstance(schema, type) and issubclass(schema, pydantic.BaseModel)):
            raise TypeError("ScriptedChatModel supports pydantic schemas only")
        return self | PydanticToolsParser(tools=[schema], first_tool_only=True)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        with self._lock:
            position = len(self.prompts)
            self.prompts.append(list(messages))
        if self.delay_s:
            time.sleep(self.delay_s)
        if position >= len(self.replies):
            raise ScriptExhausted(
                f"the model was called {position + 1} times but only "
                f"{len(self.replies)} replies were scripted"
            )
        scripted_reply = self.replies[position]
        if isinstance(scripted_reply, Exception):
            raise scripted_reply
        message = scripted_reply.model_copy(deep=True)
        return ChatResult(generations=[ChatGeneration(message=message)])


class RoutedChatModel(BaseChatModel):
    """One model alias shared by several agents, each with its own script.

    A supervisor and its specialists (lessons 22–24) often use the same alias, and
    specialists asked in parallel would consume one shared script in a random
    order. Each call here goes to the script whose key appears in the call's
    system prompt, so every agent's script stays in order.
    """

    routes: dict[str, ScriptedChatModel]

    @property
    def _llm_type(self) -> str:
        return "scripted-routes"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> RoutedChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = messages[0].text if messages and messages[0].type == "system" else ""
        for key, model in self.routes.items():
            if key in system:
                return model._generate(messages, stop, run_manager, **kwargs)
        raise ScriptExhausted(f"no script routes the system prompt {system[:80]!r}")


def reply(text: str) -> AIMessage:
    """A final answer: plain text, no tool calls."""
    return AIMessage(content=text)


def call(name: str, call_id: str | None = None, /, **args: Any) -> AIMessage:
    """A model turn that calls one tool: ``call("get_order_status", order_id="A1")``."""
    return calls((name, args, call_id))


def calls(*specs: tuple[str, Mapping[str, Any]] | tuple[str, Mapping[str, Any], str | None]
          ) -> AIMessage:
    """A model turn with several tool calls at once (they run in parallel)."""
    tool_calls = []
    for spec in specs:
        name, args = spec[0], spec[1]
        call_id = spec[2] if len(spec) > 2 else None
        tool_calls.append(
            {"name": name, "args": dict(args), "id": call_id or f"call_{uuid.uuid4().hex[:8]}"}
        )
    return AIMessage(content="", tool_calls=tool_calls)


def structured(value: pydantic.BaseModel, call_id: str | None = None) -> AIMessage:
    """A structured answer: one tool call named after ``value``'s class."""
    return call(type(value).__name__, call_id, **value.model_dump(mode="json"))


def scripted(*replies: str | BaseMessage | Exception) -> ScriptedChatModel:
    """``scripted("hello")``, ``scripted(call("tool", x=1), "done")``: strings are answers,
    exceptions are raised by that call."""
    return ScriptedChatModel(
        replies=[reply(r) if isinstance(r, str) else r for r in replies]
    )


# -- running an agent in-process --------------------------------------------------------------


@dataclass
class RunResult:
    """One run: the protocol events a client would see, and the graph's final state."""

    events: list[EventEnvelope]
    values: dict[str, Any]
    error: BaseException | None = None

    def payloads(self, event_type: str) -> list[dict[str, Any]]:
        """Every payload of one event type, in order, e.g. ``payloads("tool.call")``."""
        return [dict(e.payload) for e in self.events if e.type.value == event_type]

    @property
    def types(self) -> list[str]:
        """The event types in order, with consecutive ``token.delta`` events folded."""
        folded: list[str] = []
        for event in self.events:
            name = event.type.value
            if name == "token.delta" and folded and folded[-1] == name:
                continue
            folded.append(name)
        return folded

    @property
    def status(self) -> str:
        """The run's final status: ``completed``, ``failed`` or ``awaiting_approval``."""
        statuses = self.payloads("run.status")
        return str(statuses[-1]["status"]) if statuses else "unknown"

    @property
    def answer(self) -> str:
        """The text of the last non-empty message the client saw."""
        texts = [p["text"] for p in self.payloads("message.completed") if p.get("text")]
        return str(texts[-1]) if texts else ""

    @property
    def tool_calls(self) -> list[tuple[str, dict[str, Any]]]:
        """``(name, args)`` per ``tool.call`` event; args are already redacted."""
        return [(p["name"], dict(p["args"])) for p in self.payloads("tool.call")]

    @property
    def tool_results(self) -> list[dict[str, Any]]:
        return self.payloads("tool.result")

    @property
    def interrupts(self) -> list[dict[str, Any]]:
        return self.payloads("interrupt.requested")

    def trace(self) -> str:
        """A compact, human-readable rendering of the event stream (lesson 6)."""
        lines: list[str] = []
        for event in self.events:
            p = event.payload
            kind = event.type.value
            where = f" [{'/'.join(event.ns)}]" if event.ns else ""
            if kind == "token.delta":
                continue
            if kind == "run.status":
                error = p.get("error")
                suffix = f" ({error['code']}: {error['message']})" if error else ""
                lines.append(f"run.status        {p['status']}{suffix}")
            elif kind == "message.completed":
                lines.append(f"message.completed {p['role']}: {p['text']!r}{where}")
            elif kind == "tool.call":
                lines.append(f"tool.call         {p['name']}({p['args']}){where}")
            elif kind == "tool.result":
                detail = p.get("summary") if p["status"] == "ok" else p.get("error")
                lines.append(f"tool.result       {p['status']}: {detail}{where}")
            elif kind == "interrupt.requested":
                lines.append(f"interrupt         {p['kind']} actions={p.get('actions')}")
            elif kind == "progress":
                lines.append(f"progress          {p['label']} {p.get('pct')}%")
            elif kind == "log":
                lines.append(f"log               [{p['level']}] {p['message']}")
            else:
                lines.append(f"{kind:<17} {p}")
        return "\n".join(lines)


@dataclass
class AgentSession:
    """One agent, one checkpointer, many runs, the way one vdreamer worker holds it.

    ``models`` / ``resources`` become ``ctx.models`` / ``ctx.resources`` in
    ``build()``; ``store`` becomes ``ctx.store``. ``build()`` runs once, on the
    first run, exactly as in a worker, so a scripted model is shared by every
    run of the session.
    """

    agent: BaseAgent
    models: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    checkpointer: Any = None
    store: Any = None
    user_id: str = "alice"
    roles: tuple[str, ...] = ()
    runner: AgentRunner = field(init=False)

    def __post_init__(self) -> None:
        if self.checkpointer is None:
            self.checkpointer = InMemorySaver()
        self.runner = AgentRunner(
            self.agent,
            checkpointer=self.checkpointer,
            store=self.store,
            models=self.models,
            resources=self.resources,
        )

    def _context(
        self, thread_id: str, user_id: str | None, roles: Sequence[str] | None, headless: bool
    ) -> VDreamerContext:
        return VDreamerContext(
            run_id=f"run_{uuid.uuid4().hex}",
            thread_id=thread_id,
            agent_name=self.agent.name,
            user_id=self.user_id if user_id is None else user_id,
            roles=tuple(self.roles if roles is None else roles),
            headless=headless,
        )

    async def run(
        self,
        input: Any,
        *,
        thread_id: str = "thread-1",
        user_id: str | None = None,
        roles: Sequence[str] | None = None,
        headless: bool = False,
        recursion_limit: int | None = None,
    ) -> RunResult:
        """Start (or continue) a run on ``thread_id``; never raises for a failed run.

        A plain string becomes ``{"messages": [{"role": "user", "content": ...}]}``,
        the input a chat client sends. A failed run comes back with
        ``status == "failed"`` and the exception in ``error``. ``recursion_limit``
        is LangGraph's superstep cap; a client sets it per run with
        ``{"input": ..., "config": {"recursion_limit": N}}``.
        """
        if isinstance(input, str):
            input = {"messages": [{"role": "user", "content": input}]}
        if self.agent.input_schema is not None and not isinstance(input, Command):
            input = self.agent.input_schema.model_validate(input).model_dump(mode="json")
        sink = InMemoryEventSink()
        context = self._context(thread_id, user_id, roles, headless)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        if recursion_limit is not None:
            config["recursion_limit"] = recursion_limit
        values: Any = None
        error: BaseException | None = None
        try:
            values = await self.runner.arun(input, config=config, context=context, sink=sink)
        except Exception as exc:  # the run failed; the events say how
            error = exc
        state = values if isinstance(values, dict) else {}
        return RunResult(events=list(sink.events), values=dict(state), error=error)

    async def resume(
        self,
        value: Any,
        *,
        thread_id: str = "thread-1",
        user_id: str | None = None,
        roles: Sequence[str] | None = None,
        headless: bool = False,
    ) -> RunResult:
        """Resume the thread's pending interrupt with ``value``.

        The gateway builds this value for you in production; here the test
        passes it explicitly, e.g. ``{"action": "approve", "resolved_by": "bob"}``
        for ``require_approval``.
        """
        return await self.run(
            Command(resume=value),
            thread_id=thread_id,
            user_id=user_id,
            roles=roles,
            headless=headless,
        )

    async def state(self, thread_id: str = "thread-1") -> Any:
        """The thread's latest LangGraph ``StateSnapshot``: ``values``, ``next``, ``interrupts``."""
        return await self.runner.aget_state({"configurable": {"thread_id": thread_id}})
