"""vLLM/Ollama aliases as LangChain chat models, against stub servers (§3.6, A21)."""

from __future__ import annotations

from typing import Any

import pytest
from _stub_llm import Reply, StubServer
from langchain_core.messages import HumanMessage

from vdreamer import metrics
from vdreamer.models import (
    ChatModelConfig,
    ModelUnavailable,
    OllamaChatModel,
    VllmChatModel,
    build_chat_model,
    parse_models,
    probe_chat_model,
    split_think_tags,
)
from vdreamer.network import NetworkAllowlist


def _config(stub: StubServer, **entry: Any) -> ChatModelConfig:
    model = stub.models[0]
    parsed = parse_models(
        {"aliases": {"m": {"provider": stub.kind, "base_url": stub.base_url, "model": model,
                           **entry}}}
    )
    config = parsed.aliases["m"]
    assert isinstance(config, ChatModelConfig)
    return config


def _blocks(message: Any) -> list[tuple[str, str]]:
    out = []
    for block in message.content:
        if block["type"] == "reasoning":
            out.append(("reasoning", block["reasoning"]))
        elif block["type"] == "text":
            out.append(("text", block["text"]))
    return out


def _aggregate(chunks: list[Any]) -> Any:
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


THINKING = Reply(content="The answer is 4.", reasoning="2 plus 2 is 4")


# -- vLLM --------------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
async def test_vllm_reasoning_content_becomes_a_reasoning_block(
    vllm: StubServer, stream: bool
) -> None:
    vllm.script[:] = [THINKING]
    model = build_chat_model(_config(vllm, thinking="reasoning_content"))
    assert isinstance(model, VllmChatModel) and model.output_version == "v1"
    if stream:
        message = _aggregate([chunk async for chunk in model.astream("2+2?")])
    else:
        message = await model.ainvoke("2+2?")
    assert _blocks(message) == [("reasoning", "2 plus 2 is 4"), ("text", "The answer is 4.")]
    assert bool(vllm.requests[-1].get("stream")) is stream


def test_vllm_sync_stream_too(vllm: StubServer) -> None:
    vllm.script[:] = [THINKING]
    model = build_chat_model(_config(vllm, thinking="reasoning_content"))
    chunks = list(model.stream("2+2?"))
    reasoning = [b for c in chunks for b in c.content if b["type"] == "reasoning"]
    assert "".join(b["reasoning"] for b in reasoning) == "2 plus 2 is 4"
    assert _blocks(model.invoke("2+2?"))[0] == ("reasoning", "2 plus 2 is 4")


def test_without_a_thinking_mode_the_field_is_left_out(vllm: StubServer) -> None:
    vllm.script[:] = [THINKING]
    model = build_chat_model(_config(vllm))
    assert _blocks(model.invoke("2+2?")) == [("text", "The answer is 4.")]


def test_vllm_tool_calls_and_history(vllm: StubServer) -> None:
    vllm.script[:] = [
        Reply(content="", reasoning="need a tool", tool_calls=[{"name": "add", "args": {"a": 1}}]),
        Reply(content="done"),
    ]
    model = build_chat_model(_config(vllm, thinking="reasoning_content"))
    first = model.invoke("add")
    assert first.tool_calls == [
        {"name": "add", "args": {"a": 1}, "id": "call_0", "type": "tool_call"}
    ]
    model.invoke([HumanMessage("add"), first, HumanMessage("again")])
    sent = vllm.requests[-1]["messages"][1]
    # Reasoning stays out of the history sent back; the tool call goes as OpenAI wants it.
    assert sent["role"] == "assistant"
    assert "reasoning" not in str(sent["content"])
    assert sent["tool_calls"][0]["function"]["name"] == "add"


def test_generation_settings_and_key_reach_the_server(
    vllm: StubServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_VLLM_KEY", "s3cret")
    config = _config(vllm, api_key_env="STUB_VLLM_KEY", max_tokens=64, temperature=0.2, seed=7)
    model = build_chat_model(config)
    assert model.openai_api_key is not None
    assert model.openai_api_key.get_secret_value() == "s3cret"
    model.invoke("hi")
    body = vllm.requests[-1]
    assert body["temperature"] == 0.2 and body["seed"] == 7
    assert 64 in (body.get("max_tokens"), body.get("max_completion_tokens"))
    assert model.profile is not None
    assert model.profile["max_output_tokens"] == 64 and model.profile["image_inputs"] is False
    monkeypatch.delenv("STUB_VLLM_KEY")
    with pytest.raises(ModelUnavailable, match="STUB_VLLM_KEY"):
        build_chat_model(config)


def test_modalities_land_in_the_profile(vllm: StubServer) -> None:
    model = build_chat_model(_config(vllm, modalities=["text", "image"], thinking="think_tags"))
    assert model.profile is not None
    assert model.profile["image_inputs"] is True and model.profile["reasoning_output"] is True


def test_latency_and_tokens_are_recorded_per_alias(vllm: StubServer) -> None:
    def sample(name: str, **labels: str) -> float:
        value = metrics.REGISTRY.get_sample_value(name, {"alias": "m", **labels})
        return value or 0.0

    before_in = sample("vdreamer_model_tokens_total", direction="input")
    before_out = sample("vdreamer_model_tokens_total", direction="output")
    before_calls = sample("vdreamer_model_latency_seconds_count")
    model = build_chat_model(_config(vllm))
    model.invoke("hi")
    list(model.stream("hi"))
    assert sample("vdreamer_model_latency_seconds_count") == before_calls + 2
    assert sample("vdreamer_model_tokens_total", direction="input") == before_in + 14
    assert sample("vdreamer_model_tokens_total", direction="output") == before_out + 10


# -- Ollama ------------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
async def test_ollama_think_tags_are_parsed_out(ollama: StubServer, stream: bool) -> None:
    ollama.script[:] = [Reply(content="Four.", reasoning="adding up", inline_think=True, chunk=3)]
    model = build_chat_model(_config(ollama, thinking="think_tags"))
    assert isinstance(model, OllamaChatModel)
    if stream:
        chunks = [chunk async for chunk in model.astream("2+2?")]
        # The tags never reach a chunk's text, even split across chunk boundaries.
        assert all("think" not in b.get("text", "") for c in chunks for b in c.content)
        message = _aggregate(chunks)
    else:
        message = await model.ainvoke("2+2?")
    assert _blocks(message) == [("reasoning", "adding up"), ("text", "Four.")]


def test_ollama_reasoning_content_uses_ollama_think(ollama: StubServer) -> None:
    ollama.script[:] = [Reply(content="Four.", reasoning="adding up")]
    model = build_chat_model(_config(ollama, thinking="reasoning_content", max_tokens=32))
    message = model.invoke("2+2?")
    assert ollama.requests[-1]["think"] is True
    assert ollama.requests[-1]["options"]["num_predict"] == 32
    assert _blocks(message) == [("reasoning", "adding up"), ("text", "Four.")]


def test_a_think_tags_model_that_does_not_think_is_all_text(ollama: StubServer) -> None:
    ollama.script[:] = [Reply(content="Just text, mentioning <think> later.")]
    model = build_chat_model(_config(ollama, thinking="think_tags"))
    assert _blocks(model.invoke("hi")) == [("text", "Just text, mentioning <think> later.")]


def test_split_think_tags() -> None:
    assert split_think_tags("  <think>a</think>\nb") == ("a", "b")
    assert split_think_tags("no tags") == ("", "no tags")
    assert split_think_tags("<think>cut off") == ("cut off", "")
    assert split_think_tags("") == ("", "")


# -- Probing -----------------------------------------------------------------------------


async def test_probes(vllm: StubServer, ollama: StubServer) -> None:
    assert "serves served-chat" in await probe_chat_model(_config(vllm))
    assert "serves local-chat" in await probe_chat_model(_config(ollama))
    other = parse_models(
        {"aliases": {"m": {"provider": "vllm", "base_url": vllm.base_url, "model": "absent"}}}
    ).aliases["m"]
    assert isinstance(other, ChatModelConfig)
    with pytest.raises(ModelUnavailable, match="does not serve 'absent'"):
        await probe_chat_model(other)
    with pytest.raises(ModelUnavailable, match="outside NETWORK_ALLOWLIST"):
        await probe_chat_model(
            _config(vllm), allowlist=NetworkAllowlist.parse("localhost-only.internal, 10.0.0.0/8")
        )


async def test_an_unreachable_endpoint() -> None:
    from _stub_llm import _free_port

    config = parse_models(
        {"aliases": {"m": {"provider": "vllm", "model": "x",
                           "base_url": f"http://127.0.0.1:{_free_port()}/v1"}}}
    ).aliases["m"]
    assert isinstance(config, ChatModelConfig)
    with pytest.raises(ModelUnavailable, match="unreachable"):
        await probe_chat_model(config, timeout_s=2)


def test_building_makes_no_request(vllm: StubServer) -> None:
    build_chat_model(_config(vllm))
    assert vllm.requests == []
