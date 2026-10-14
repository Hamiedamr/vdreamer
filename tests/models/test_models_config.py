"""``models.yaml`` parsing (PRD §3.6, A23, A36, A40)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vdreamer.models import (
    MODELS_ENV,
    ChatModelConfig,
    FastembedModelConfig,
    ModelsError,
    load_models,
    parse_models,
    resolve_models_path,
)
from vdreamer.network import NetworkAllowlist

#: The PRD §3.6 example, with concrete model names.
PRD_EXAMPLE = """
aliases:
  reasoner: {provider: vllm, base_url: http://10.20.0.5:8000/v1, model: qwen3-32b,
             api_key_env: VLLM_KEY, max_tokens: 8192, temperature: 0.2,
             thinking: reasoning_content}
  vision:   {provider: vllm, base_url: http://10.20.0.6:8000/v1, model: qwen2.5-vl,
             modalities: [text, image]}
  local:    {provider: ollama, base_url: http://10.20.0.7:11434, model: llama3,
             thinking: think_tags}
  dense:    {provider: fastembed, kind: embedding, model: intfloat/multilingual-e5-large}
  sparse:   {provider: fastembed, kind: sparse, model: Qdrant/bm25}
  rerank:   {provider: fastembed, kind: rerank, model: jinaai/jina-reranker-v2-base-multilingual}
fastembed: {cache_dir: /opt/models/fastembed, local_files_only: true,
            providers: [CPUExecutionProvider]}
"""


def test_the_prd_example_parses(tmp_path: Path) -> None:
    file = tmp_path / "models.yaml"
    file.write_text(PRD_EXAMPLE, encoding="utf-8")
    config = load_models(file, allowlist=NetworkAllowlist.parse("10.20.0.0/16"))
    reasoner = config.aliases["reasoner"]
    assert isinstance(reasoner, ChatModelConfig)
    assert (reasoner.provider, reasoner.thinking, reasoner.api_key_env) == (
        "vllm",
        "reasoning_content",
        "VLLM_KEY",
    )
    assert reasoner.params == {"max_tokens": 8192, "temperature": 0.2}
    vision = config.aliases["vision"]
    assert isinstance(vision, ChatModelConfig)
    assert vision.modalities == ("text", "image") and vision.thinking == "none"
    assert isinstance(config.aliases["local"], ChatModelConfig)
    kinds = {
        alias: c.kind for alias, c in config.aliases.items() if isinstance(c, FastembedModelConfig)
    }
    assert kinds == {"dense": "embedding", "sparse": "sparse", "rerank": "rerank"}
    assert config.fastembed is not None
    assert config.fastembed.cache_dir == "/opt/models/fastembed"
    assert config.fastembed.providers == ("CPUExecutionProvider",)
    assert config.get("dense").model == "intfloat/multilingual-e5-large"
    with pytest.raises(ModelsError, match="unknown model alias 'nope'"):
        config.get("nope")


def _one(entry: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"aliases": {"m": entry}, **extra}


VLLM = {"provider": "vllm", "base_url": "http://10.0.0.1:8000/v1", "model": "x"}


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({**VLLM, "api_key": "sk-123"}, "secret in YAML"),
        ({**VLLM, "base_url": "http://user:pw@10.0.0.1/v1"}, "credentials in the URL"),
        ({**VLLM, "base_url": "ftp://10.0.0.1"}, "http/https URL"),
        ({**VLLM, "thinking": "sometimes"}, "thinking must be one of"),
        ({**VLLM, "modalities": ["text", "audio"]}, "modalities"),
        ({**VLLM, "temprature": 0.1}, "unknown key"),
        ({**VLLM, "num_ctx": 4096}, "unknown key"),  # an Ollama setting on a vLLM alias
        ({**VLLM, "max_tokens": "lots"}, "max_tokens must be a non-negative integer"),
        ({**VLLM, "model": ""}, "non-empty string 'model'"),
        ({"provider": "openai", "model": "gpt"}, "provider must be one of"),
        ({"provider": "fastembed", "kind": "colbert", "model": "x"}, "kind must be one of"),
    ],
)
def test_bad_entries_are_refused(entry: dict[str, Any], message: str) -> None:
    with pytest.raises(ModelsError, match=message):
        parse_models(_one(entry))


def test_fastembed_needs_the_vendored_cache_only() -> None:
    dense = {"provider": "fastembed", "kind": "embedding", "model": "BAAI/bge-small-en-v1.5"}
    with pytest.raises(ModelsError, match="need a top-level fastembed.cache_dir"):
        parse_models(_one(dense))
    with pytest.raises(ModelsError, match="local_files_only must be true"):
        parse_models(_one(dense, fastembed={"cache_dir": "/m", "local_files_only": False}))
    with pytest.raises(ModelsError, match="providers"):
        parse_models(_one(dense, fastembed={"cache_dir": "/m", "providers": "CPU"}))
    parsed = parse_models(_one(dense, fastembed={"cache_dir": "/m", "threads": 2}))
    assert parsed.fastembed is not None and parsed.fastembed.threads == 2


def test_endpoints_must_be_on_the_allowlist() -> None:
    with pytest.raises(ModelsError, match="NETWORK_ALLOWLIST"):
        parse_models(_one(VLLM), allowlist=NetworkAllowlist.parse("192.168.0.0/16"))
    parse_models(_one(VLLM), allowlist=NetworkAllowlist.parse("10.0.0.0/8"))


def test_top_level_shape() -> None:
    assert parse_models(None).aliases == {}
    with pytest.raises(ModelsError, match="unknown top-level"):
        parse_models({"models": {}})
    with pytest.raises(ModelsError, match="expected a mapping"):
        parse_models(["reasoner"])


def test_the_path_comes_from_the_flag_then_the_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(MODELS_ENV, raising=False)
    assert resolve_models_path(None) is None
    monkeypatch.setenv(MODELS_ENV, str(tmp_path / "env.yaml"))
    assert resolve_models_path(None) == tmp_path / "env.yaml"
    assert resolve_models_path("flag.yaml") == Path("flag.yaml")
    with pytest.raises(ModelsError, match="cannot read"):
        load_models(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("aliases: [", encoding="utf-8")
    with pytest.raises(ModelsError, match="invalid YAML"):
        load_models(bad)
