"""The governed tool pipeline.

Every governed call — through :class:`GovernedTool` (``ctx.tools.govern``,
``ctx.tools.mcp``) or :class:`~vdreamer.tools.VDreamerToolMiddleware`
(``create_agent`` via ``ctx.agent_kwargs()``) — runs the same seven steps:

1. **Authorize** — ``requires_roles`` (any one of them) and ``scope`` (granted
   by a role equal to it or an ``fnmatch`` pattern over it, e.g. ``db:*``)
   against the initiator's roles. A denial is contained (``denied``).
2. **Emit ``tool.call``** — the model-visible args, redacted by policy keys
   plus the defaults. Injected args (``runtime``, state, store) never
   reach the event.
3. **Approve** — an optional per-tool gate (``Policy.approval``) through
   :func:`~vdreamer.context.require_approval` with ``approve``/``reject``;
   a rejection is contained (``denied``). The resume re-runs the tool node,
   so a resumed attempt re-emits its ``tool.call`` (same ``tool_call_id``).
4. **Execute** under ``timeout_s``. A ``side_effects=True`` tool sees
   ``ctx.idempotency_key`` = ``{run key}:{tool_call_id}`` — the model's call id
   is checkpointed, so a recovered attempt re-executing the call gets the same
   key.
5. **Normalize** — a returned ``vdreamer.Artifact`` (or a list of them, or one
   as the ``ToolMessage.artifact``) is stored like ``emit_artifact`` and the
   model gets a reference naming its ``artifact_id``. Text over
   ``max_result_chars`` becomes a ``json`` (when it parses as an object or
   list, ≤1 MB) or ``file`` artifact, and the model gets the truncated body
   plus the ``artifact_id``. MCP structured content (the
   ``ToolMessage.artifact``) and base64 media blocks map the same way.
6. **Contain** — failures become ``ToolMessage(status="error")`` with a code:
   ``tool_error | timeout | denied | transport | invalid_args``. The run goes
   on unless ``fail_run=True``, which raises :class:`GovernedToolError`.
7. **Emit ``tool.result``** — status, error, summary, artifact ids,
   ``source`` (on the matching ``tool.call``) and ``duration_ms``.

The ``tool.call``/``tool.result`` records ride the run's custom stream like
every other helper record (``context.emit_record``); the stream mapper
(``vdreamer.streaming``) turns them into envelopes and drops its own
ungoverned mapping for those call ids — governed tools are the single emitter.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import contextlib
import contextvars
import dataclasses
import fnmatch
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, TypeVar, cast

import pydantic
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp

from vdreamer import metrics
from vdreamer.artifact_data import json_preview
from vdreamer.artifacts import Artifact, run_blocking, store_returned, write_artifact
from vdreamer.context import (
    VDreamerContext,
    bind_context,
    current_context_or_none,
    emit_record,
    require_approval,
)
from vdreamer.tools import (
    VDREAMER_METADATA_KEY,
    Policy,
    redact_args,
    redact_text,
    resolve_policy,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from vdreamer.mcp import MCPRegistry

_T = TypeVar("_T")

#: ``tool.metadata`` key naming a tool's ``source`` explicitly (the MCP layer
#: stamps ``mcp:{server}``); otherwise :func:`tool_source` infers it.
SOURCE_METADATA_KEY: Final = "vdreamer_source"

ErrorCode = Literal["tool_error", "timeout", "denied", "transport", "invalid_args"]

#: ``json`` artifacts hold at most 1 MB; larger JSON results are files.
JSON_ARTIFACT_MAX_BYTES: Final = 1024 * 1024
#: A ``json`` artifact's SSE preview carries the full body up to 64 KB.
JSON_PREVIEW_MAX_BYTES: Final = 64 * 1024
#: ``tool.result`` summaries and error messages are capped here.
SUMMARY_MAX_CHARS: Final = 8000
ERROR_MESSAGE_MAX_CHARS: Final = 1000

_LANGCHAIN_ROOTS: Final = frozenset({"langchain", "langchain_core", "langgraph"})
#: Module roots whose exceptions, raised by an MCP call, are transport failures.
_TRANSPORT_ROOTS: Final = frozenset({"httpx", "httpx2", "httpcore", "anyio", "mcp", "fastmcp"})


class GovernedToolError(Exception):
    """A governed tool failed with ``fail_run=True``: the run fails with it."""

    def __init__(self, code: ErrorCode, message: str, *, tool: str) -> None:
        super().__init__(f"tool {tool!r} failed ({code}): {message}")
        self.code = code
        self.tool = tool


class _Contained(Exception):
    """Internal: one contained failure, carried from a step to step 6."""

    def __init__(
        self, code: ErrorCode, message: str, *, reply: ToolMessage | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.reply = reply


# ---------------------------------------------------------------------------
# Tool facts: source, policy, model-visible args
# ---------------------------------------------------------------------------


def _is_langchain_module(module: str) -> bool:
    root = module.split(".", 1)[0]
    return root in _LANGCHAIN_ROOTS or root.startswith("langchain_")


def tool_source(tool: BaseTool) -> str:
    """``custom | langchain | mcp:{server}`` (``tool.call.source``).

    An explicit ``metadata["vdreamer_source"]`` wins (MCP tools carry one).
    A ``@vdreamer.tool`` is ``custom``; so is a plain ``@tool`` over the
    developer's own function. A tool class — or a ``@tool`` function — that
    lives in a LangChain package is ``langchain``.
    """
    meta = tool.metadata or {}
    explicit = meta.get(SOURCE_METADATA_KEY)
    if isinstance(explicit, str) and explicit:
        return explicit
    if VDREAMER_METADATA_KEY in meta:
        return "custom"
    module = type(tool).__module__
    if module.startswith("langchain_core.tools"):
        fn = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
        fn_module = getattr(fn, "__module__", "") or ""
        return "langchain" if _is_langchain_module(fn_module) else "custom"
    return "langchain" if _is_langchain_module(module) else "custom"


def effective_policy(tool: BaseTool, policies: Mapping[str, Policy]) -> Policy:
    """The decorator's policy (if any) overlaid by the agent's ``tool_policies``."""
    meta = (tool.metadata or {}).get(VDREAMER_METADATA_KEY)
    decorated = Policy.from_metadata(meta) if isinstance(meta, Mapping) else None
    return resolve_policy(tool.name, decorated, policies)


def _schema_keys(tool: BaseTool) -> set[str] | None:
    """The model-visible argument names, or ``None`` when unknowable."""
    try:
        schema = tool.tool_call_schema
    except Exception:
        return None
    if isinstance(schema, type) and issubclass(schema, pydantic.BaseModel):
        return set(schema.model_fields)
    if isinstance(schema, Mapping):
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            return set(properties)
    return None


def model_args(tool: BaseTool, args: Mapping[str, Any]) -> dict[str, Any]:
    """``args`` without the injected ones (``runtime``, state, store)."""
    keys = _schema_keys(tool)
    if keys is None:
        return dict(args)
    return {key: value for key, value in args.items() if key in keys}


def _invalid_args(tool: BaseTool, args: Mapping[str, Any]) -> str | None:
    """Validate against a pydantic call schema; JSON-schema tools validate server-side."""
    try:
        schema = tool.tool_call_schema
    except Exception:
        return None
    if not (isinstance(schema, type) and issubclass(schema, pydantic.BaseModel)):
        return None
    try:
        schema.model_validate(dict(args))
    except pydantic.ValidationError as exc:
        # include_input=False: the offending values may be the secrets masks.
        problems = [
            f"{'.'.join(str(part) for part in error['loc']) or '(args)'}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        ]
        return "invalid arguments — " + "; ".join(problems)
    return None


def authorize(policy: Policy, roles: Sequence[str]) -> str | None:
    """Step 1: the denial reason, or ``None`` when the initiator may call the tool."""
    held = set(roles)
    if policy.requires_roles and not held.intersection(policy.requires_roles):
        return f"requires one of the roles {sorted(policy.requires_roles)}"
    if policy.scope and not any(fnmatch.fnmatchcase(policy.scope, grant) for grant in held):
        return f"the scope {policy.scope!r} is not granted to the initiator"
    return None


def _error_message(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text[:ERROR_MESSAGE_MAX_CHARS]


def _classify(exc: BaseException, source: str) -> ErrorCode:
    """Step 6: an execution exception's error code."""
    from langchain_core.tools import ToolException

    if isinstance(exc, pydantic.ValidationError):
        return "invalid_args"
    if isinstance(exc, ToolException):
        return "tool_error"
    if isinstance(exc, ConnectionError):
        return "transport"
    if source.startswith("mcp:"):
        # MCP tool errors arrive as isError results (an error ToolMessage);
        # anything raised instead is the connection or the protocol failing.
        return "transport"
    if type(exc).__module__.split(".", 1)[0] in _TRANSPORT_ROOTS:
        return "transport"
    return "tool_error"


# ---------------------------------------------------------------------------
# Step 5: normalization
# ---------------------------------------------------------------------------


def _truncated(text: str, limit: int, artifact_id: str) -> str:
    return (
        f"{text[:limit]}\n\n[vdreamer: result truncated to {limit} of {len(text)} characters; "
        f"the full result is artifact {artifact_id}]"
    )


async def _text_artifact(text: str, call: GovernedCall) -> str:
    """Store an oversized text result: ``json`` when it is a JSON object/list."""
    data = text.encode("utf-8")
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    stem = f"{call.name}-{call.tool_call_id}"
    if isinstance(parsed, (dict, list)) and len(data) <= JSON_ARTIFACT_MAX_BYTES:
        return await write_artifact(
            kind="json",
            name=f"{call.name} result",
            data=data,
            mime_type="application/json",
            preview=json_preview(parsed, len(data)),
            metadata=_artifact_metadata(call),
            filename=f"{stem}.json",
        )
    mime = "application/json" if isinstance(parsed, (dict, list)) else "text/plain"
    filename = f"{stem}.json" if mime == "application/json" else f"{stem}.txt"
    return await write_artifact(
        kind="file",
        name=f"{call.name} result",
        data=data,
        mime_type=mime,
        preview={"filename": filename, "mime": mime},
        metadata=_artifact_metadata(call),
        filename=filename,
    )


def _artifact_metadata(call: GovernedCall) -> dict[str, Any]:
    return {"tool": call.name, "tool_call_id": call.tool_call_id, "source": call.source}


async def _media_artifact(block: Mapping[str, Any], call: GovernedCall, index: int) -> str | None:
    """Store one base64 media block as a ``file`` artifact; ``None`` if undecodable."""
    encoded = block.get("base64") or block.get("data")
    if not isinstance(encoded, str):
        return None
    try:
        data = base64.b64decode(encoded, validate=False)
    except ValueError:
        return None
    mime = str(block.get("mime_type") or "application/octet-stream")
    filename = f"{call.name}-{call.tool_call_id}-{index}"
    return await write_artifact(
        kind="file",
        name=f"{call.name} {block.get('type', 'media')}",
        data=data,
        mime_type=mime,
        preview={"filename": filename, "mime": mime},
        metadata=_artifact_metadata(call),
        filename=filename,
    )


def _block_text(block: Any) -> str | None:
    if isinstance(block, str):
        return block
    if isinstance(block, Mapping) and block.get("type") == "text":
        return str(block.get("text") or "")
    return None


def _is_artifacts(value: Any) -> bool:
    if isinstance(value, Artifact):
        return True
    return (
        isinstance(value, list) and bool(value) and all(isinstance(v, Artifact) for v in value)
    )


async def returned_artifacts(result: Any, call: GovernedCall) -> tuple[Any, list[str]]:
    """Store a returned ``Artifact`` (or list), or one riding ``ToolMessage.artifact``."""
    carrier = result.artifact if isinstance(result, ToolMessage) else result
    if not _is_artifacts(carrier):
        return result, []
    items = [carrier] if isinstance(carrier, Artifact) else list(carrier)
    refs = [await store_returned(item) for item in items]
    ids = [ref.artifact_id for ref in refs]
    pointer = {"artifact_ids": ids}
    if isinstance(result, ToolMessage) and not result.additional_kwargs.get(_RETURNED_MARK):
        return result.model_copy(update={"artifact": pointer}), ids
    text = "\n".join(
        f"[vdreamer: stored {ref.kind.value} artifact {ref.artifact_id} ({ref.name!r})]"
        for ref in refs
    )
    message = ToolMessage(
        content=text, artifact=pointer, name=call.name, tool_call_id=call.tool_call_id
    )
    return message, ids


async def _normalized(result: Any, call: GovernedCall) -> tuple[Any, list[str]]:
    """Step 5 in full: returned artifacts first, then overflow."""
    result, artifact_ids = await returned_artifacts(result, call)
    if isinstance(result, ToolMessage):
        result, overflow = await normalize(result, call)
        artifact_ids = [*artifact_ids, *overflow]
    return result, artifact_ids


async def normalize(result: ToolMessage, call: GovernedCall) -> tuple[ToolMessage, list[str]]:
    """Step 5: move what exceeds ``max_result_chars`` into artifacts."""
    limit = call.policy.max_result_chars
    artifact_ids: list[str] = []
    content: Any = result.content
    if isinstance(content, str):
        if len(content) > limit:
            artifact_id = await _text_artifact(content, call)
            artifact_ids.append(artifact_id)
            content = _truncated(content, limit, artifact_id)
    elif isinstance(content, list) and len(json.dumps(content, default=str)) > limit:
        texts = [text for text in (_block_text(block) for block in content) if text is not None]
        joined = "\n".join(texts)
        collapse = len(joined) > limit
        blocks: list[Any] = []
        text_placed = False
        for index, block in enumerate(content):
            if _block_text(block) is not None:
                if not collapse:
                    blocks.append(block)
                elif not text_placed:
                    artifact_id = await _text_artifact(joined, call)
                    artifact_ids.append(artifact_id)
                    blocks.append({"type": "text", "text": _truncated(joined, limit, artifact_id)})
                    text_placed = True
                continue
            if isinstance(block, Mapping) and (block.get("base64") or block.get("data")):
                media_id = await _media_artifact(block, call, index)
                if media_id is not None:
                    artifact_ids.append(media_id)
                    blocks.append(
                        {
                            "type": "text",
                            "text": f"[vdreamer: {block.get('mime_type') or 'binary'} content "
                            f"stored as artifact {media_id}]",
                        }
                    )
                    continue
            blocks.append(block)
        content = blocks
    artifact: Any = result.artifact
    if artifact is not None:
        serialized = json.dumps(artifact, default=str)
        if len(serialized) > limit:
            artifact_id = await _text_artifact(serialized, call)
            artifact_ids.append(artifact_id)
            artifact = {"artifact_id": artifact_id}
    if not artifact_ids:
        return result, []
    return result.model_copy(update={"content": content, "artifact": artifact}), artifact_ids


def _summary(result: Any, artifact_ids: Sequence[str] = ()) -> str:
    """The ``tool.result`` summary: the model-facing text, capped, always naming
    the overflow artifacts (their note sits at the end of a truncated body)."""
    if not isinstance(result, ToolMessage):
        return ""
    text = result.text
    if len(text) <= SUMMARY_MAX_CHARS:
        return text
    if not artifact_ids:
        return text[:SUMMARY_MAX_CHARS]
    note = f"\n\n[vdreamer: full result in artifact {', '.join(artifact_ids)}]"
    return text[: SUMMARY_MAX_CHARS - len(note)] + note


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GovernedCall:
    """One tool call under governance: what every step reads."""

    tool: BaseTool
    name: str
    tool_call_id: str
    args: Mapping[str, Any]
    policy: Policy
    source: str
    #: True when the caller passed bare args (``tool.invoke(tool_call["args"])``),
    #: so the id is ours, not the model's: the stream mapper may then bind this
    #: call to the model's matching tool call (one ``tool.call``, the model's id).
    id_assigned: bool = False


def governed_call(
    tool: BaseTool,
    tool_call: Mapping[str, Any],
    *,
    policy: Policy,
    source: str | None = None,
) -> GovernedCall:
    """Build the :class:`GovernedCall` for a ``ToolCall`` dict on ``tool``."""
    args = tool_call.get("args")
    return GovernedCall(
        tool=tool,
        name=str(tool_call.get("name") or tool.name),
        tool_call_id=str(tool_call.get("id") or f"call_{uuid.uuid4().hex}"),
        args=model_args(tool, args if isinstance(args, Mapping) else {}),
        policy=policy,
        source=source if source is not None else tool_source(tool),
        id_assigned=not tool_call.get("id"),
    )


def _emit(record: dict[str, Any]) -> None:
    """Publish a record when a run is bound (governance also works outside one)."""
    if current_context_or_none() is not None:
        emit_record(record)


def _call_context(ctx: VDreamerContext | None, call: GovernedCall) -> VDreamerContext | None:
    """The context the tool executes under: a per-call key for side effects."""
    if ctx is None or not call.policy.side_effects:
        return ctx
    base = ctx.idempotency_key or ctx.run_id
    # model_copy is shallow: ``pending`` stays the run's own buffer.
    return ctx.model_copy(update={"idempotency_key": f"{base}:{call.tool_call_id}"})


@contextlib.contextmanager
def _bound(ctx: VDreamerContext | None) -> Iterator[None]:
    if ctx is None:
        yield
        return
    with bind_context(ctx):
        yield


def _approve(call: GovernedCall, redacted: Mapping[str, Any]) -> None:
    """Step 3: the optional per-tool gate. Raises ``GraphInterrupt`` until resolved."""
    spec = call.policy.approval
    if spec is None:
        return
    decision = require_approval(
        kind=spec.kind,
        payload={"tool": call.name, "source": call.source, "args": dict(redacted)},
        actions=("approve", "reject"),
        approver_role=spec.approver_role,
        scope=spec.scope,
        ttl_hours=spec.ttl_hours,
    )
    action = decision.get("action") if isinstance(decision, Mapping) else decision
    if action != "approve":
        text = decision.get("text") if isinstance(decision, Mapping) else None
        reason = f"rejected by the approver: {text}" if text else "rejected by the approver"
        raise _Contained("denied", reason)


class _Pipeline:
    """Steps 1–3 before execution, 5–7 after; shared by the async and sync paths."""

    def __init__(self, call: GovernedCall) -> None:
        self.call = call
        self.ctx = current_context_or_none()
        self.started = time.monotonic()

    def before(self) -> VDreamerContext | None:
        call = self.call
        redacted = redact_args(call.args, call.policy.redact)
        record: dict[str, Any] = {
            "type": "tool.call",
            "tool_call_id": call.tool_call_id,
            "name": call.name,
            "source": call.source,
            "args": redacted,
        }
        if call.id_assigned:
            record["id_assigned"] = True
        _emit(record)
        denial = authorize(call.policy, self.ctx.roles if self.ctx is not None else ())
        if denial is not None:
            raise _Contained("denied", denial)
        _approve(call, redacted)
        invalid = _invalid_args(call.tool, call.args)
        if invalid is not None:
            raise _Contained("invalid_args", invalid)
        return _call_context(self.ctx, call)

    def check(self, result: Any) -> Any:
        """A tool that reported failure itself (an error ``ToolMessage``) is ``tool_error``."""
        if isinstance(result, ToolMessage) and result.status == "error":
            raise _Contained("tool_error", result.text[:ERROR_MESSAGE_MAX_CHARS], reply=result)
        return result

    def contain(self, exc: BaseException) -> _Contained:
        if isinstance(exc, _Contained):
            return exc
        if isinstance(exc, TimeoutError):
            return _Contained(
                "timeout", f"the tool did not finish within {self.call.policy.timeout_s:g}s"
            )
        return _Contained(_classify(exc, self.call.source), _error_message(exc))

    def _duration_ms(self) -> int:
        return int((time.monotonic() - self.started) * 1000)

    def succeeded(self, result: Any, artifact_ids: Sequence[str]) -> Any:
        call = self.call
        metrics.TOOL_LATENCY.labels(source=call.source).observe(time.monotonic() - self.started)
        _emit(
            {
                "type": "tool.result",
                "tool_call_id": call.tool_call_id,
                "status": "ok",
                "error": None,
                "summary": _summary(result, artifact_ids),
                "artifact_ids": list(artifact_ids),
                "duration_ms": self._duration_ms(),
            }
        )
        return result

    def failed(self, failure: _Contained) -> ToolMessage:
        call = self.call
        metrics.TOOL_LATENCY.labels(source=call.source).observe(time.monotonic() - self.started)
        metrics.TOOL_ERRORS.labels(source=call.source, code=failure.code).inc()
        # Exceptions echo what they were given: mask the call's secrets in the
        # message before the model, the event log or the client see it.
        message = redact_text(failure.message, call.args, call.policy.redact)
        reply = failure.reply or ToolMessage(
            content=f"Error ({failure.code}): {message}",
            name=call.name,
            tool_call_id=call.tool_call_id,
            status="error",
        )
        _emit(
            {
                "type": "tool.result",
                "tool_call_id": call.tool_call_id,
                "status": "error",
                "error": {"code": failure.code, "message": message},
                "summary": redact_text(reply.text, call.args, call.policy.redact)[
                    :SUMMARY_MAX_CHARS
                ],
                "artifact_ids": [],
                "duration_ms": self._duration_ms(),
            }
        )
        if call.policy.fail_run:
            raise GovernedToolError(failure.code, message, tool=call.name) from failure
        return reply


async def run_governed(
    call: GovernedCall,
    execute: Callable[[VDreamerContext | None], Awaitable[_T]],
) -> _T | ToolMessage:
    """Run ``execute`` through the seven steps (async — vdreamer's own path).

    ``execute`` receives the call-scoped context (``None`` outside a run) and
    must forward it to the tool's ``runtime.context`` when it injects one.
    ``GraphInterrupt`` and every other ``GraphBubbleUp`` pass through untouched:
    an approval or an MCP elicitation pauses the run, it does not fail the call.
    """
    pipeline = _Pipeline(call)
    try:
        call_ctx = pipeline.before()
        try:
            with _bound(call_ctx):
                result = await asyncio.wait_for(execute(call_ctx), timeout=call.policy.timeout_s)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            raise pipeline.contain(exc) from exc
        result = pipeline.check(result)
        result, artifact_ids = await _normalized(result, call)
    except _Contained as failure:
        return pipeline.failed(failure)
    return pipeline.succeeded(result, artifact_ids)  # type: ignore[no-any-return]


def run_governed_sync(
    call: GovernedCall,
    execute: Callable[[VDreamerContext | None], _T],
) -> _T | ToolMessage:
    """The synchronous twin of :func:`run_governed` (plain ``invoke`` callers).

    The timeout runs the tool on a helper thread (carrying the context); a
    timed-out thread cannot be killed, only abandoned — the async path is the
    one vdreamer's runtime drives.
    """
    pipeline = _Pipeline(call)
    try:
        call_ctx = pipeline.before()
        context = contextvars.copy_context()

        def _target() -> _T:
            with _bound(call_ctx):
                return execute(call_ctx)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(context.run, _target)
            result = future.result(timeout=call.policy.timeout_s)
        except GraphBubbleUp:
            raise
        except concurrent.futures.TimeoutError as exc:
            raise pipeline.contain(TimeoutError()) from exc
        except Exception as exc:
            raise pipeline.contain(exc) from exc
        finally:
            pool.shutdown(wait=False)
        result = pipeline.check(result)
        result, artifact_ids = run_blocking(_normalized(result, call))
    except _Contained as failure:
        return pipeline.failed(failure)
    return pipeline.succeeded(result, artifact_ids)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# The proxy (ctx.tools.govern) and the toolkit (BuildContext.tools)
# ---------------------------------------------------------------------------


def _with_call_context(payload: Any, call_ctx: VDreamerContext | None) -> Any:
    """Point an injected ``runtime``'s context at the call-scoped one."""
    if call_ctx is None or not isinstance(payload, Mapping):
        return payload
    args = payload.get("args")
    if not isinstance(args, Mapping):
        return payload
    runtime = args.get("runtime")
    if runtime is None or not dataclasses.is_dataclass(runtime):
        return payload
    if not isinstance(getattr(runtime, "context", None), VDreamerContext):
        return payload
    runtime = dataclasses.replace(runtime, context=call_ctx)  # type: ignore[type-var]
    return {**payload, "args": {**args, "runtime": runtime}}


def _as_tool_call(input: Any) -> tuple[dict[str, Any], bool]:
    """``(tool_call, from_model)`` — a direct ``invoke(args)`` gets a synthetic call."""
    if isinstance(input, Mapping) and input.get("type") == "tool_call":
        return dict(input), True
    args = dict(input) if isinstance(input, Mapping) else {"input": input}
    # No id: ``governed_call`` mints one and flags it ``id_assigned``.
    synthetic = {"type": "tool_call", "name": None, "args": args, "id": None}
    return synthetic, False


def _unwrap(result: Any, from_model: bool) -> Any:
    """A direct invocation gets the plain output back, as ``BaseTool.invoke`` would."""
    if from_model or not isinstance(result, ToolMessage):
        return result
    return result.content


class GovernedTool(BaseTool):
    """A governance proxy around one ``BaseTool``: same name, description,
    and schema, every call through the pipeline.

    LangGraph's ``ToolNode`` reads injection needs (``runtime``, state, store)
    from ``get_input_schema()`` and the function's annotations, so both are
    delegated to the wrapped tool — the proxy receives exactly the arguments
    the tool itself would have.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    inner: BaseTool
    policy: Policy
    source: str

    @classmethod
    def wrap(cls, tool: BaseTool, *, policy: Policy, source: str | None = None) -> GovernedTool:
        return cls(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            return_direct=tool.return_direct,
            response_format=tool.response_format,
            tags=tool.tags,
            metadata=tool.metadata,
            extras=getattr(tool, "extras", None),
            inner=tool,
            policy=policy,
            source=source if source is not None else tool_source(tool),
        )

    # Injection and schema facts come from the wrapped tool.

    def get_input_schema(self, config: RunnableConfig | None = None) -> type[pydantic.BaseModel]:
        return cast("type[pydantic.BaseModel]", self.inner.get_input_schema(config))

    @property
    def tool_call_schema(self) -> Any:
        return self.inner.tool_call_schema

    @property
    def args(self) -> dict[str, Any]:
        return self.inner.args

    @property
    def func(self) -> Callable[..., Any] | None:
        return getattr(self.inner, "func", None)

    @property
    def coroutine(self) -> Callable[..., Awaitable[Any]] | None:
        return getattr(self.inner, "coroutine", None)

    def _call(self, tool_call: Mapping[str, Any]) -> GovernedCall:
        return governed_call(
            self.inner,
            {**tool_call, "name": tool_call.get("name") or self.name},
            policy=self.policy,
            source=self.source,
        )

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        tool_call, from_model = _as_tool_call(input)
        call = self._call(tool_call)
        tool_call["name"] = call.name
        tool_call["id"] = call.tool_call_id

        async def execute(call_ctx: VDreamerContext | None) -> Any:
            payload = _with_call_context(tool_call, call_ctx)
            return await self.inner.ainvoke(payload, config, **kwargs)

        return _unwrap(await run_governed(call, execute), from_model)

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        tool_call, from_model = _as_tool_call(input)
        call = self._call(tool_call)
        tool_call["name"] = call.name
        tool_call["id"] = call.tool_call_id

        def execute(call_ctx: VDreamerContext | None) -> Any:
            payload = _with_call_context(tool_call, call_ctx)
            return self.inner.invoke(payload, config, **kwargs)

        return _unwrap(run_governed_sync(call, execute), from_model)

    def _run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - invoke is overridden
        return self.inner._run(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return await self.inner._arun(*args, **kwargs)


#: ``ToolMessage.additional_kwargs`` flag: the message only carries returned artifacts.
_RETURNED_MARK: Final = "vdreamer_returned_artifacts"


def _carry_artifacts(result: Any, tool_call: Mapping[str, Any], name: str) -> Any:
    """An ``Artifact`` return as a ``ToolMessage`` ``ToolNode`` accepts (step 5 stores it)."""
    if not _is_artifacts(result):
        return result
    return ToolMessage(
        content="",
        artifact=result,
        name=name,
        tool_call_id=str(tool_call.get("id") or ""),
        additional_kwargs={_RETURNED_MARK: True},
    )


class ArtifactReturnBridge(GovernedTool):
    """What the middleware hands ``ToolNode``: the tool itself, with ``Artifact``
    returns carried in a ``ToolMessage`` — ``ToolNode`` validates the return
    type before the middleware sees it. No governance of its own: the
    middleware's pipeline runs around it.
    """

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        result = await self.inner.ainvoke(input, config, **kwargs)
        return _carry_artifacts(result, input if isinstance(input, Mapping) else {}, self.name)

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        result = self.inner.invoke(input, config, **kwargs)
        return _carry_artifacts(result, input if isinstance(input, Mapping) else {}, self.name)


def bridge_artifact_returns(tool: BaseTool) -> BaseTool:
    """``tool`` wrapped in :class:`ArtifactReturnBridge` (the middleware path)."""
    return ArtifactReturnBridge.wrap(tool, policy=Policy())


def is_governed(tool: Any) -> bool:
    """Whether ``tool`` is already a governance proxy."""
    return isinstance(tool, GovernedTool) and not isinstance(tool, ArtifactReturnBridge)


class ToolKit:
    """``BuildContext.tools``: govern tools, select MCP tools.

    ``govern`` wraps each tool in a :class:`GovernedTool` with its effective
    policy — the decorator's, overlaid by the agent's ``tool_policies`` — and
    passes already-governed tools through. ``mcp`` returns a configured
    server's discovered tools (``{server}_{tool}``), governed, optionally
    filtered by ``include`` globs on the qualified name.
    """

    def __init__(
        self,
        policies: Mapping[str, Policy] | None = None,
        mcp: MCPRegistry | None = None,
    ) -> None:
        self._policies: Mapping[str, Policy] = policies if policies is not None else {}
        self._mcp = mcp

    @property
    def policies(self) -> Mapping[str, Policy]:
        return self._policies

    def govern(self, tools: Iterable[BaseTool | Callable[..., Any]]) -> list[BaseTool]:
        from langchain_core.tools import tool as as_tool

        governed: list[BaseTool] = []
        for candidate in tools:
            if isinstance(candidate, GovernedTool):
                governed.append(candidate)
                continue
            built = candidate if isinstance(candidate, BaseTool) else as_tool(candidate)
            if not isinstance(built, BaseTool):
                raise TypeError(f"cannot govern {candidate!r}: not a tool or a function")
            governed.append(
                GovernedTool.wrap(built, policy=effective_policy(built, self._policies))
            )
        return governed

    def mcp(self, server: str, include: Sequence[str] | None = None) -> list[BaseTool]:
        if self._mcp is None:
            raise LookupError(
                f"no MCP server {server!r}: no MCP servers are configured (resources.yaml)"
            )
        return self.govern(self._mcp.tools(server, include=include))


__all__ = [
    "ErrorCode",
    "GovernedCall",
    "GovernedTool",
    "GovernedToolError",
    "SOURCE_METADATA_KEY",
    "ToolKit",
    "authorize",
    "effective_policy",
    "governed_call",
    "is_governed",
    "model_args",
    "normalize",
    "run_governed",
    "run_governed_sync",
    "tool_source",
]
