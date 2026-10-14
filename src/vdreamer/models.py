"""``models.yaml`` resolution (PRD §3.6, D17, A21, A23, A36).

Agents name model *aliases* (``BaseAgent.models = ["reasoner", "dense"]``);
``models.yaml`` maps each alias to a backend:

- ``provider: vllm`` / ``provider: ollama``: a LangChain chat model built
  with ``output_version="v1"`` (§2.4): :class:`VllmChatModel` (a
  ``ChatOpenAI`` against vLLM's OpenAI-compatible API, Chat Completions only)
  or :class:`OllamaChatModel` (a ``ChatOllama``). Each alias picks a
  ``thinking`` mode, and the model turns it into standard ``reasoning``
  content blocks, so both stream drivers, the final message, and the
  conversation history see one shape:

  - ``reasoning_content``: the server separates reasoning (vLLM
    ``--reasoning-parser``, A21: ``delta.reasoning_content``, or ``reasoning``
    on newer vLLM; Ollama's ``think``).
  - ``think_tags``: the model writes ``<think>…</think>`` at the start of its
    text; the tags are parsed out of the stream, even split across chunks.
  - ``none`` (the default): no extraction beyond what the client does.

- ``provider: fastembed``: ``kind: embedding | sparse | rerank`` →
  ``TextEmbedding`` / ``SparseTextEmbedding`` / ``TextCrossEncoder``, loaded
  once per worker from the ``fastembed.cache_dir`` with
  ``local_files_only=True`` and the configured ONNX providers (A36). These
  are the native objects, not wrapped.

Secrets never appear in the file (A23): ``api_key_env`` names the
environment variable. Endpoints must match ``NETWORK_ALLOWLIST`` when it is
set (A40), at load and again when they are probed. Per-alias latency and
token counts feed the ``vdreamer_model_*`` metrics (§5.4).
"""

from __future__ import annotations

import os
import time
import urllib.parse
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast
from uuid import UUID

import yaml
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGenerationChunk, ChatResult, LLMResult
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from vdreamer.network import EndpointError, NetworkAllowlist, check_url, default_port
from vdreamer.resources import SECRET_KEYS

#: Where the CLI looks when ``--models`` is not given.
MODELS_ENV: Final = "VDREAMER_MODELS"

ThinkingMode = Literal["none", "reasoning_content", "think_tags"]
THINKING_MODES: Final[tuple[str, ...]] = ("none", "reasoning_content", "think_tags")
CHAT_PROVIDERS: Final[tuple[str, ...]] = ("vllm", "ollama")
FASTEMBED_KINDS: Final[tuple[str, ...]] = ("embedding", "sparse", "rerank")
MODALITIES: Final[tuple[str, ...]] = ("text", "image")

#: vLLM accepts any key when it runs without ``--api-key``; the SDK needs one.
_NO_API_KEY: Final = "EMPTY"

_THINK_OPEN: Final = "<" + "think" + ">"
_THINK_CLOSE: Final = "<" + "/" + "think" + ">"


class ModelsError(ValueError):
    """``models.yaml`` is missing, unparsable, or has the wrong shape."""


class ModelUnavailable(RuntimeError):
    """An alias cannot be served: endpoint down, model not served, files missing."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

#: Optional generation settings per chat provider, passed to the client.
_VLLM_PARAMS: Final = frozenset(
    {"max_tokens", "temperature", "top_p", "stop", "seed", "timeout_s", "max_retries",
     "extra_body", "context_window"}
)
_OLLAMA_PARAMS: Final = frozenset(
    {"max_tokens", "temperature", "top_p", "top_k", "stop", "seed", "timeout_s", "num_ctx",
     "keep_alive", "context_window"}
)
_CHAT_KEYS: Final = frozenset({"provider", "base_url", "model", "api_key_env", "thinking",
                               "modalities"})
_FASTEMBED_KEYS: Final = frozenset({"provider", "kind", "model"})
_FASTEMBED_SETTINGS_KEYS: Final = frozenset(
    {"cache_dir", "local_files_only", "providers", "threads"}
)


@dataclass(frozen=True)
class ChatModelConfig:
    """One ``provider: vllm | ollama`` alias."""

    alias: str
    provider: Literal["vllm", "ollama"]
    base_url: str
    model: str
    api_key_env: str | None = None
    thinking: ThinkingMode = "none"
    modalities: tuple[str, ...] = ("text",)
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FastembedModelConfig:
    """One ``provider: fastembed`` alias."""

    alias: str
    kind: Literal["embedding", "sparse", "rerank"]
    model: str
    provider: Literal["fastembed"] = "fastembed"


@dataclass(frozen=True)
class FastembedSettings:
    """The top-level ``fastembed:`` block (A36)."""

    cache_dir: str
    providers: tuple[str, ...] = ("CPUExecutionProvider",)
    threads: int | None = None
    local_files_only: bool = True


ModelConfig = ChatModelConfig | FastembedModelConfig


@dataclass(frozen=True)
class ModelsConfig:
    """A validated ``models.yaml``."""

    aliases: Mapping[str, ModelConfig] = field(default_factory=dict)
    fastembed: FastembedSettings | None = None
    origin: str = "models.yaml"

    def get(self, alias: str) -> ModelConfig:
        try:
            return self.aliases[alias]
        except KeyError:
            raise ModelsError(
                f"unknown model alias {alias!r} (models.yaml has: {sorted(self.aliases) or 'none'})"
            ) from None


def _fail(alias: str, message: str) -> ModelsError:
    return ModelsError(f"model {alias!r}: {message}")


def _check_secrets(alias: str, entry: Mapping[str, Any]) -> None:
    inline = sorted(str(key) for key in entry if str(key).lower() in SECRET_KEYS)
    if inline:
        raise _fail(
            alias,
            f"{', '.join(inline)} would put a secret in YAML (A23); reference an "
            "environment variable with api_key_env instead",
        )


def check_endpoint_url(
    label: str, url: str, allowlist: NetworkAllowlist | None
) -> urllib.parse.SplitResult:
    """:func:`vdreamer.network.check_url`, as a :class:`ModelsError`."""
    try:
        return check_url(label, url, allowlist)
    except EndpointError as exc:
        raise ModelsError(str(exc)) from exc


def _str(alias: str, entry: Mapping[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise _fail(alias, f"needs a non-empty string {key!r}")
    return value


def _parse_chat(
    alias: str, entry: Mapping[str, Any], allowlist: NetworkAllowlist | None
) -> ChatModelConfig:
    provider = cast(Literal["vllm", "ollama"], entry["provider"])
    allowed = _VLLM_PARAMS if provider == "vllm" else _OLLAMA_PARAMS
    unknown = sorted(set(map(str, entry)) - _CHAT_KEYS - allowed)
    if unknown:
        raise _fail(
            alias, f"unknown key(s) {unknown} for provider {provider} (known: "
            f"{sorted(_CHAT_KEYS | allowed)})"
        )
    base_url = _str(alias, entry, "base_url")
    check_endpoint_url(f"model {alias!r} base_url", base_url, allowlist)
    api_key_env = entry.get("api_key_env")
    if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env):
        raise _fail(alias, "api_key_env must name an environment variable")
    thinking = entry.get("thinking", "none")
    if thinking is None:
        thinking = "none"
    if thinking not in THINKING_MODES:
        raise _fail(alias, f"thinking must be one of {list(THINKING_MODES)}, got {thinking!r}")
    modalities = entry.get("modalities", ["text"])
    if (
        isinstance(modalities, (str, bytes))
        or not isinstance(modalities, Sequence)
        or not modalities
        or any(m not in MODALITIES for m in modalities)
    ):
        raise _fail(alias, f"modalities must be a list drawn from {list(MODALITIES)}")
    params = {str(k): v for k, v in entry.items() if k in allowed}
    for key in ("max_tokens", "num_ctx", "context_window", "max_retries", "seed", "top_k"):
        if key in params and (not isinstance(params[key], int) or isinstance(params[key], bool)
                              or params[key] < 0):
            raise _fail(alias, f"{key} must be a non-negative integer")
    for key in ("temperature", "top_p", "timeout_s"):
        if key in params and (
            not isinstance(params[key], (int, float)) or isinstance(params[key], bool)
        ):
            raise _fail(alias, f"{key} must be a number")
    return ChatModelConfig(
        alias=alias,
        provider=provider,
        base_url=base_url,
        model=_str(alias, entry, "model"),
        api_key_env=api_key_env,
        thinking=cast(ThinkingMode, thinking),
        modalities=tuple(modalities),
        params=params,
    )


def _parse_fastembed(alias: str, entry: Mapping[str, Any]) -> FastembedModelConfig:
    unknown = sorted(set(map(str, entry)) - _FASTEMBED_KEYS)
    if unknown:
        raise _fail(alias, f"unknown key(s) {unknown} for provider fastembed")
    kind = entry.get("kind")
    if kind not in FASTEMBED_KINDS:
        raise _fail(alias, f"kind must be one of {list(FASTEMBED_KINDS)}, got {kind!r}")
    return FastembedModelConfig(
        alias=alias,
        kind=cast(Literal["embedding", "sparse", "rerank"], kind),
        model=_str(alias, entry, "model"),
    )


def _parse_fastembed_settings(value: Any, origin: str) -> FastembedSettings:
    if not isinstance(value, Mapping):
        raise ModelsError(f"{origin}: fastembed must be a mapping")
    unknown = sorted(set(map(str, value)) - _FASTEMBED_SETTINGS_KEYS)
    if unknown:
        raise ModelsError(f"{origin}: fastembed: unknown key(s) {unknown}")
    cache_dir = value.get("cache_dir")
    if not isinstance(cache_dir, str) or not cache_dir:
        raise ModelsError(f"{origin}: fastembed needs a cache_dir (the vendored model cache, A36)")
    if value.get("local_files_only", True) is not True:
        raise ModelsError(
            f"{origin}: fastembed.local_files_only must be true: models load from the "
            "vendored cache only, never the network (A36, §5.2)"
        )
    providers = value.get("providers", ["CPUExecutionProvider"])
    if (
        isinstance(providers, (str, bytes))
        or not isinstance(providers, Sequence)
        or not providers
        or not all(isinstance(p, str) and p for p in providers)
    ):
        raise ModelsError(f"{origin}: fastembed.providers must be a list of ONNX provider names")
    threads = value.get("threads")
    if threads is not None and (not isinstance(threads, int) or threads < 1):
        raise ModelsError(f"{origin}: fastembed.threads must be a positive integer")
    return FastembedSettings(cache_dir=cache_dir, providers=tuple(providers), threads=threads)


def parse_models(
    data: Any,
    *,
    origin: str = "models.yaml",
    allowlist: NetworkAllowlist | None = None,
) -> ModelsConfig:
    """Validate a loaded document: ``{aliases: {...}, fastembed: {...}}``."""
    if data is None:
        return ModelsConfig(origin=origin)
    if not isinstance(data, Mapping):
        raise ModelsError(f"{origin}: expected a mapping with 'aliases' (and 'fastembed')")
    unknown = sorted(set(map(str, data)) - {"aliases", "fastembed"})
    if unknown:
        raise ModelsError(f"{origin}: unknown top-level key(s) {unknown}")
    raw = data.get("aliases") or {}
    if not isinstance(raw, Mapping):
        raise ModelsError(f"{origin}: aliases must be a mapping of alias -> entry")
    aliases: dict[str, ModelConfig] = {}
    for alias, entry in raw.items():
        if not isinstance(alias, str) or not alias:
            raise ModelsError(f"{origin}: aliases must be non-empty strings, got {alias!r}")
        if not isinstance(entry, Mapping):
            raise _fail(alias, f"expected a mapping, got {type(entry).__name__}")
        _check_secrets(alias, entry)
        provider = entry.get("provider")
        if provider in CHAT_PROVIDERS:
            aliases[alias] = _parse_chat(alias, entry, allowlist)
        elif provider == "fastembed":
            aliases[alias] = _parse_fastembed(alias, entry)
        else:
            raise _fail(
                alias, f"provider must be one of {[*CHAT_PROVIDERS, 'fastembed']}, got {provider!r}"
            )
    settings = (
        _parse_fastembed_settings(data["fastembed"], origin)
        if data.get("fastembed") is not None
        else None
    )
    if settings is None and any(isinstance(c, FastembedModelConfig) for c in aliases.values()):
        raise ModelsError(f"{origin}: fastembed aliases need a top-level fastembed.cache_dir")
    return ModelsConfig(aliases=aliases, fastembed=settings, origin=origin)


def load_models(path: str | Path, *, allowlist: NetworkAllowlist | None = None) -> ModelsConfig:
    """Load and validate ``models.yaml`` (``yaml.safe_load``)."""
    file = Path(path)
    try:
        raw = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelsError(f"cannot read {file}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ModelsError(f"{file}: invalid YAML: {exc}") from exc
    return parse_models(data, origin=str(file), allowlist=allowlist)


def resolve_models_path(path: str | Path | None) -> Path | None:
    """The explicit path, else ``$VDREAMER_MODELS``, else none."""
    if path is not None:
        return Path(path)
    from_env = os.environ.get(MODELS_ENV)
    return Path(from_env) if from_env else None


# ---------------------------------------------------------------------------
# Thinking extraction at the model layer
# ---------------------------------------------------------------------------


class ThinkTagParser:
    """Splits a ``<think>…</think>`` preamble out of streamed text.

    The model must open with the tag (leading whitespace allowed); text that
    starts any other way is all answer. Tags split across chunks are held
    back until they resolve. A stream that ends inside the think block
    routes its tail as thinking (the model was cut off mid-thought).
    """

    def __init__(self) -> None:
        self._state: Literal["start", "think", "text"] = "start"
        self._buffer = ""

    def feed(self, text: str) -> tuple[str, str]:
        """Feed one chunk; returns ``(thinking_delta, text_delta)``."""
        if self._state == "text":
            return "", text
        data = self._buffer + text
        self._buffer = ""
        if self._state == "start":
            head = data.lstrip()
            if not head or (len(head) < len(_THINK_OPEN) and _THINK_OPEN.startswith(head)):
                self._buffer = data
                return "", ""
            if not head.startswith(_THINK_OPEN):
                self._state = "text"
                return "", data
            self._state = "think"
            data = head[len(_THINK_OPEN) :]
        index = data.find(_THINK_CLOSE)
        if index >= 0:
            self._state = "text"
            return data[:index], data[index + len(_THINK_CLOSE) :].lstrip("\n")
        for size in range(min(len(data), len(_THINK_CLOSE) - 1), 0, -1):
            if _THINK_CLOSE.startswith(data[-size:]):
                self._buffer = data[-size:]
                return data[:-size], ""
        return data, ""

    def close(self) -> tuple[str, str]:
        """Finish the stream; returns any held-back remainder."""
        tail, self._buffer = self._buffer, ""
        if self._state == "think":
            return tail, ""
        return "", tail


def split_think_tags(text: str) -> tuple[str, str]:
    """A whole message's ``(thinking, text)``."""
    parser = ThinkTagParser()
    thinking, answer = parser.feed(text)
    more_thinking, more_answer = parser.close()
    return thinking + more_thinking, answer + more_answer


def _with_reasoning(message: Any, reasoning: str) -> None:
    """Put ``reasoning`` first in ``message``'s v1 content blocks."""
    blocks = list(message.content_blocks)
    if not any(block.get("type") == "reasoning" for block in blocks):
        blocks.insert(0, {"type": "reasoning", "reasoning": reasoning})
    message.content = blocks
    message.response_metadata["output_version"] = "v1"


class _ThinkingMixin:
    """The per-alias thinking mode, applied to every chunk and result."""

    vdreamer_thinking: ThinkingMode
    output_version: str | None

    def _server_reasoning(self, message: Any) -> str:
        """Reasoning the server sent separately and the client left aside."""
        return ""

    def _rewrite(self, message: Any, parser: ThinkTagParser | None, *, final: bool) -> None:
        reasoning = self._server_reasoning(message)
        if parser is not None and isinstance(message.content, str):
            thinking, text = parser.feed(message.content)
            if final:
                more_thinking, more_text = parser.close()
                thinking, text = thinking + more_thinking, text + more_text
            message.content = text
            reasoning += thinking
        if not reasoning:
            return
        if self.output_version == "v1":
            _with_reasoning(message, reasoning)
        else:
            message.additional_kwargs["reasoning_content"] = reasoning

    def _parser(self) -> ThinkTagParser | None:
        return ThinkTagParser() if self.vdreamer_thinking == "think_tags" else None

    def _tail(self, parser: ThinkTagParser | None) -> ChatGenerationChunk | None:
        if parser is None:
            return None
        thinking, text = parser.close()
        if not thinking and not text:
            return None
        from langchain_core.messages import AIMessageChunk

        message = AIMessageChunk(content=text)
        if thinking:
            if self.output_version == "v1":
                _with_reasoning(message, thinking)
            else:
                message.additional_kwargs["reasoning_content"] = thinking
        return ChatGenerationChunk(message=message)

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        parser = self._parser()
        for chunk in super()._stream(*args, **kwargs):  # type: ignore[misc]
            self._rewrite(chunk.message, parser, final=False)
            yield chunk
        tail = self._tail(parser)
        if tail is not None:
            yield tail

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        parser = self._parser()
        async for chunk in super()._astream(*args, **kwargs):  # type: ignore[misc]
            self._rewrite(chunk.message, parser, final=False)
            yield chunk
        tail = self._tail(parser)
        if tail is not None:
            yield tail

    def _rewrite_result(self, result: ChatResult) -> ChatResult:
        for generation in result.generations:
            self._rewrite(generation.message, self._parser(), final=True)
        return result

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        return self._rewrite_result(super()._generate(*args, **kwargs))  # type: ignore[misc]

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        return self._rewrite_result(await super()._agenerate(*args, **kwargs))  # type: ignore[misc]


class VllmChatModel(_ThinkingMixin, ChatOpenAI):
    """``ChatOpenAI`` against vLLM, keeping its separate reasoning field (A21).

    ``ChatOpenAI`` targets the official API and drops vLLM's
    ``reasoning_content`` (``reasoning`` on newer vLLM); with
    ``thinking: reasoning_content`` this subclass keeps it and the mixin
    turns it into a standard ``reasoning`` block.
    """

    vdreamer_alias: str = ""
    vdreamer_thinking: ThinkingMode = "none"

    @staticmethod
    def _reasoning_of(payload: Any) -> str:
        if not isinstance(payload, Mapping):
            return ""
        value = payload.get("reasoning_content") or payload.get("reasoning")
        return value if isinstance(value, str) else ""

    def _server_reasoning(self, message: Any) -> str:
        if self.vdreamer_thinking != "reasoning_content":
            return ""
        return str(message.additional_kwargs.pop("vdreamer_reasoning", "") or "")

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation is None or self.vdreamer_thinking != "reasoning_content":
            return generation
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        if choices:
            reasoning = self._reasoning_of(choices[0].get("delta"))
            if reasoning:
                generation.message.additional_kwargs["vdreamer_reasoning"] = reasoning
        return generation

    def _create_chat_result(
        self, response: Any, generation_info: dict[str, Any] | None = None
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        if self.vdreamer_thinking != "reasoning_content":
            return result
        payload = response if isinstance(response, dict) else response.model_dump()
        choices = payload.get("choices") or []
        for generation, choice in zip(result.generations, choices, strict=False):
            reasoning = self._reasoning_of(choice.get("message"))
            if reasoning:
                generation.message.additional_kwargs["vdreamer_reasoning"] = reasoning
        return result


class OllamaChatModel(_ThinkingMixin, ChatOllama):
    """``ChatOllama``: ``reasoning_content`` uses Ollama's own ``think``
    separation; ``think_tags`` parses tags out of the text."""

    vdreamer_alias: str = ""
    vdreamer_thinking: ThinkingMode = "none"


# ---------------------------------------------------------------------------
# Metrics (§5.4: model latency/tokens per alias)
# ---------------------------------------------------------------------------


class ModelMetricsHandler(BaseCallbackHandler):
    """Records ``vdreamer_model_latency_seconds`` / ``vdreamer_model_tokens``."""

    run_inline = True

    def __init__(self, alias: str) -> None:
        self.alias = alias
        self._started: dict[UUID, float] = {}

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._started[run_id] = time.monotonic()

    def _observe(self, run_id: UUID) -> None:
        from vdreamer import metrics

        started = self._started.pop(run_id, None)
        if started is not None:
            metrics.MODEL_LATENCY.labels(alias=self.alias).observe(time.monotonic() - started)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        from vdreamer import metrics

        self._observe(run_id)
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if not usage:
                    continue
                for direction, key in (("input", "input_tokens"), ("output", "output_tokens")):
                    count = usage.get(key) or 0
                    if count:
                        metrics.MODEL_TOKENS.labels(alias=self.alias, direction=direction).inc(
                            count
                        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._observe(run_id)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def _api_key(config: ChatModelConfig, environ: Mapping[str, str]) -> str | None:
    if config.api_key_env is None:
        return None
    value = environ.get(config.api_key_env, "")
    if not value:
        raise ModelUnavailable(
            f"model {config.alias!r}: environment variable {config.api_key_env} "
            "(api_key_env) is not set"
        )
    return value


def _profile(config: ChatModelConfig) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "text_inputs": True,
        "image_inputs": "image" in config.modalities,
        "text_outputs": True,
        "reasoning_output": config.thinking != "none",
        "tool_calling": True,
    }
    if "max_tokens" in config.params:
        profile["max_output_tokens"] = config.params["max_tokens"]
    if "context_window" in config.params:
        profile["max_input_tokens"] = config.params["context_window"]
    return profile


def build_chat_model(
    config: ChatModelConfig, *, environ: Mapping[str, str] | None = None
) -> VllmChatModel | OllamaChatModel:
    """The alias's LangChain chat model (``output_version="v1"``). Makes no request."""
    env = environ if environ is not None else os.environ
    params = dict(config.params)
    common: dict[str, Any] = {
        "model": config.model,
        "output_version": "v1",
        "vdreamer_alias": config.alias,
        "vdreamer_thinking": config.thinking,
        "profile": _profile(config),
        "callbacks": [ModelMetricsHandler(config.alias)],
        "metadata": {"vdreamer_model_alias": config.alias},
    }
    for name in ("temperature", "top_p", "stop", "seed"):
        if name in params:
            common[name] = params[name]
    key = _api_key(config, env)
    if config.provider == "vllm":
        kwargs: dict[str, Any] = {
            **common,
            "base_url": config.base_url,
            "api_key": key or _NO_API_KEY,
            "use_responses_api": False,
            "stream_usage": True,
        }
        if "max_tokens" in params:
            kwargs["max_tokens"] = params["max_tokens"]
        if "timeout_s" in params:
            kwargs["timeout"] = params["timeout_s"]
        if "max_retries" in params:
            kwargs["max_retries"] = params["max_retries"]
        if "extra_body" in params:
            kwargs["extra_body"] = params["extra_body"]
        return VllmChatModel(**kwargs)
    client: dict[str, Any] = {}
    if key:
        client["headers"] = {"Authorization": f"Bearer {key}"}
    if "timeout_s" in params:
        client["timeout"] = params["timeout_s"]
    kwargs = {
        **common,
        "base_url": config.base_url,
        "reasoning": True if config.thinking == "reasoning_content" else None,
        "client_kwargs": client,
    }
    if "max_tokens" in params:
        kwargs["num_predict"] = params["max_tokens"]
    for name in ("top_k", "num_ctx", "keep_alive"):
        if name in params:
            kwargs[name] = params[name]
    return OllamaChatModel(**kwargs)


def _force_offline() -> None:
    """§5.2: the HuggingFace stack never reaches the hub from a worker."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _fastembed_class(kind: str) -> Any:
    _force_offline()
    try:
        if kind == "embedding":
            from fastembed import TextEmbedding

            return TextEmbedding
        if kind == "sparse":
            from fastembed import SparseTextEmbedding

            return SparseTextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder
    except ImportError as exc:
        raise ModelUnavailable(
            "fastembed aliases need the [fastembed] extra (pip install 'vdreamer[fastembed]')"
        ) from exc


def load_fastembed(
    config: FastembedModelConfig, settings: FastembedSettings, *, lazy: bool = False
) -> Any:
    """The native fastembed object, from the vendored cache only (A36).

    ``lazy=True`` locates the files without loading the ONNX session — the
    "missing model files" check of ``validate``/``doctor``.
    """
    cls = _fastembed_class(config.kind)
    kwargs: dict[str, Any] = {
        "model_name": config.model,
        "cache_dir": settings.cache_dir,
        "providers": list(settings.providers),
        "local_files_only": True,
        "lazy_load": lazy,
    }
    if settings.threads is not None:
        kwargs["threads"] = settings.threads
    try:
        return cls(**kwargs)
    except Exception as exc:
        raise ModelUnavailable(
            f"model {config.alias!r}: cannot load {config.kind} model {config.model!r} from "
            f"{settings.cache_dir}: {type(exc).__name__}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Probing (validate / doctor / worker boot)
# ---------------------------------------------------------------------------


async def probe_chat_model(
    config: ChatModelConfig,
    *,
    allowlist: NetworkAllowlist | None = None,
    environ: Mapping[str, str] | None = None,
    timeout_s: float = 5.0,
) -> str:
    """Reach the endpoint and confirm it serves ``config.model``; a short status line."""
    import httpx

    env = environ if environ is not None else os.environ
    parsed = urllib.parse.urlsplit(config.base_url)
    host = parsed.hostname or ""
    if allowlist is not None:
        try:
            await allowlist.check_connect(host, default_port(parsed))
        except Exception as exc:
            raise ModelUnavailable(f"model {config.alias!r}: {exc}") from exc
    key = _api_key(config, env)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    base = config.base_url.rstrip("/")
    url = f"{base}/models" if config.provider == "vllm" else f"{base}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=timeout_s, trust_env=False) as client:
            response = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        raise ModelUnavailable(
            f"model {config.alias!r}: {config.provider} endpoint {base} unreachable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if response.status_code != 200:
        raise ModelUnavailable(
            f"model {config.alias!r}: {url} answered {response.status_code}"
        )
    try:
        body = response.json()
        if config.provider == "vllm":
            served = [str(item.get("id")) for item in body.get("data") or []]
        else:
            served = [
                str(item.get("name") or item.get("model")) for item in body.get("models") or []
            ]
    except (ValueError, AttributeError) as exc:
        raise ModelUnavailable(f"model {config.alias!r}: {url} sent an unexpected body") from exc
    if config.model not in served and not (
        config.provider == "ollama" and f"{config.model}:latest" in served
    ):
        raise ModelUnavailable(
            f"model {config.alias!r}: {base} does not serve {config.model!r} (serves: "
            f"{sorted(served) or 'nothing'})"
        )
    return f"{config.provider} {base} serves {config.model}"


def check_fastembed_files(config: FastembedModelConfig, settings: FastembedSettings) -> str:
    """Confirm the model files are in the cache without loading them."""
    load_fastembed(config, settings, lazy=True)
    return f"fastembed {config.kind} {config.model} in {settings.cache_dir}"


async def probe_model(
    models: ModelsConfig,
    alias: str,
    *,
    allowlist: NetworkAllowlist | None = None,
    network: bool = True,
) -> str:
    """One alias's readiness: endpoint + served model, or files in the cache."""
    import asyncio

    config = models.get(alias)
    if isinstance(config, ChatModelConfig):
        if not network:
            _api_key(config, os.environ)
            return f"{config.provider} {config.base_url} (not contacted)"
        return await probe_chat_model(config, allowlist=allowlist)
    assert models.fastembed is not None  # parse_models guarantees it
    return await asyncio.to_thread(check_fastembed_files, config, models.fastembed)


# ---------------------------------------------------------------------------
# The worker's model set
# ---------------------------------------------------------------------------


class AliasMap(Mapping[str, Any]):
    """``ctx.models`` / ``ctx.resources``: the declared aliases only.

    Agents reference aliases only (§3.6); looking up one the agent did not
    declare is a ``KeyError`` that says so.
    """

    def __init__(self, label: str, owner: str, values: Mapping[str, Any]) -> None:
        self._label = label
        self._owner = owner
        self._values = dict(values)

    def __getitem__(self, alias: str) -> Any:
        try:
            return self._values[alias]
        except KeyError:
            raise KeyError(
                f"{self._label} alias {alias!r} is not declared by {self._owner} "
                f"(declared: {sorted(self._values) or 'none'})"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"AliasMap({self._label}: {sorted(self._values)})"


def missing_aliases(models: ModelsConfig | None, declared: Sequence[str]) -> list[str]:
    known = models.aliases if models is not None else {}
    return [alias for alias in declared if alias not in known]


def build_models(
    models: ModelsConfig | None,
    declared: Sequence[str],
    *,
    lazy_embeddings: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Every declared alias's object: chat models, and fastembed loaded once."""
    missing = missing_aliases(models, declared)
    if missing:
        where = (
            models.origin
            if models is not None
            else f"no models.yaml (--models / ${MODELS_ENV})"
        )
        raise ModelsError(f"unknown model alias(es) {missing}: {where}")
    assert models is not None or not declared
    built: dict[str, Any] = {}
    for alias in declared:
        assert models is not None
        config = models.aliases[alias]
        if isinstance(config, ChatModelConfig):
            built[alias] = build_chat_model(config, environ=environ)
        else:
            assert models.fastembed is not None
            built[alias] = load_fastembed(config, models.fastembed, lazy=lazy_embeddings)
    return built


__all__ = [
    "CHAT_PROVIDERS",
    "MODELS_ENV",
    "AliasMap",
    "ChatModelConfig",
    "FastembedModelConfig",
    "FastembedSettings",
    "ModelConfig",
    "ModelMetricsHandler",
    "ModelUnavailable",
    "ModelsConfig",
    "ModelsError",
    "OllamaChatModel",
    "ThinkTagParser",
    "VllmChatModel",
    "build_chat_model",
    "build_models",
    "check_endpoint_url",
    "check_fastembed_files",
    "load_fastembed",
    "load_models",
    "missing_aliases",
    "parse_models",
    "probe_chat_model",
    "probe_model",
    "resolve_models_path",
    "split_think_tags",
]
