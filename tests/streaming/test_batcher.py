"""Delta batching (§3.3: 50 ms / 32 tokens) and think-tag splitting (D18/A20).

The batcher and splitter are unit-tested standalone — the drivers feed both,
so their thresholds/semantics are pinned here and the end-to-end tests only
assert wiring.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessageChunk

from vdreamer.streaming import DeltaBatcher, ThinkTagSplitter, chunk_text_and_thinking

_TAG_OPEN = "<" + "think" + ">"
_TAG_CLOSE = "<" + "/" + "think" + ">"


class FakeClock:
    """A mutable monotonic fake."""

    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_batcher_flushes_on_token_count_threshold() -> None:
    clock = FakeClock()
    batcher = DeltaBatcher(interval_s=60.0, max_deltas=3, clock=clock)

    assert batcher.add(("token", "m1"), "a ") == []
    assert batcher.add(("token", "m1"), "b ") == []
    # The third delta reaches the max-tokens threshold and flushes the batch.
    flushed = batcher.add(("token", "m1"), "c")
    assert flushed == [(("token", "m1"), "a b c")]
    # The batch is drained: nothing left to flush.
    assert batcher.flush_all() == []


def test_batcher_keys_batch_independently() -> None:
    clock = FakeClock()
    batcher = DeltaBatcher(interval_s=60.0, max_deltas=1, clock=clock)
    # max_deltas=1: every add flushes immediately, keyed per (kind, message).
    assert batcher.add(("token", "m1"), "x") == [(("token", "m1"), "x")]
    assert batcher.add(("thinking", "m1"), "y") == [(("thinking", "m1"), "y")]
    assert batcher.add(("token", "m2"), "z") == [(("token", "m2"), "z")]


def test_batcher_flushes_on_interval_threshold() -> None:
    clock = FakeClock()
    batcher = DeltaBatcher(interval_s=0.05, max_deltas=1000, clock=clock)
    batcher.add(("token", "m1"), "hel")
    # Not yet: within the interval, under the token threshold.
    assert batcher.flush_ready() == []
    clock.advance(0.06)
    assert batcher.flush_ready() == [(("token", "m1"), "hel")]
    assert batcher.flush_ready() == []


def test_batcher_flush_all_drains_every_open_batch() -> None:
    clock = FakeClock()
    batcher = DeltaBatcher(interval_s=60.0, max_deltas=1000, clock=clock)
    batcher.add(("token", "m1"), "one ")
    batcher.add(("token", "m1"), "two")
    batcher.add(("thinking", "m1"), "hmm")
    batcher.add(("token", "m2"), "other")
    assert batcher.flush_all() == [
        (("token", "m1"), "one two"),
        (("thinking", "m1"), "hmm"),
        (("token", "m2"), "other"),
    ]
    assert batcher.flush_all() == []


def test_batcher_default_thresholds_are_the_prd_constants() -> None:
    from vdreamer.events import DELTA_BATCH_INTERVAL_S, DELTA_BATCH_MAX_TOKENS

    batcher = DeltaBatcher()
    assert batcher._interval_s == DELTA_BATCH_INTERVAL_S
    assert batcher._max_deltas == DELTA_BATCH_MAX_TOKENS


def test_splitter_routes_around_the_closing_tag() -> None:
    splitter = ThinkTagSplitter()
    thinking, text = splitter.feed("let me " + _TAG_CLOSE + " answer")
    assert thinking == "let me "
    assert text == " answer"
    # After the closing tag everything is text.
    thinking2, text2 = splitter.feed(" more")
    assert (thinking2, text2) == ("", " more")


def test_splitter_handles_tag_split_across_three_chunks() -> None:
    splitter = ThinkTagSplitter()
    # The closing tag split into three fragments across chunk boundaries.
    thinking, text = splitter.feed("deeper ")
    assert (thinking, text) == ("deeper ", "")
    thinking, text = splitter.feed("thoughts <" + "/")
    # Held back (partial tag at the end); "thoughts " emitted as thinking.
    assert thinking == "thoughts "
    assert text == ""
    thinking, text = splitter.feed("thi")
    assert thinking == ""
    assert text == ""
    thinking, text = splitter.feed("nk> done")
    assert thinking == ""
    assert text == " done"
    assert splitter.close() == ("", "")


def test_splitter_close_routes_unseen_tail_as_text() -> None:
    """No closing tag ever seen → the model did not use the think format."""
    splitter = ThinkTagSplitter()
    thinking, _ = splitter.feed("plain ")
    assert thinking == "plain "
    thinking, text = splitter.close()
    assert thinking == ""
    assert text == ""


def test_splitter_multiple_messages_do_not_share_state() -> None:
    first = ThinkTagSplitter()
    second = ThinkTagSplitter()
    first.feed("a" + _TAG_CLOSE)
    thinking, _ = second.feed("still thinking")
    assert thinking == "still thinking"
    assert not second._saw_close


def test_chunk_extraction_prefers_reasoning_blocks() -> None:
    chunk: Any = AIMessageChunk(
        content=[
            {"type": "reasoning", "reasoning": "pondering"},
            {"type": "text", "text": "answer"},
        ],
        id="m1",
    )
    text, thinking = chunk_text_and_thinking(chunk)
    assert text == "answer"
    assert thinking == "pondering"


def test_chunk_extraction_thinking_block_key() -> None:
    chunk: Any = AIMessageChunk(
        content=[{"type": "thinking", "thinking": "deep"}],
        id="m1",
    )
    assert chunk_text_and_thinking(chunk) == ("", "deep")


def test_chunk_extraction_reasoning_content_alias() -> None:
    chunk: Any = AIMessageChunk(
        content="answer",
        additional_kwargs={"reasoning_content": "vllm reasoning"},
        id="m1",
    )
    assert chunk_text_and_thinking(chunk) == ("answer", "vllm reasoning")


def test_chunk_extraction_plain_text_has_no_thinking() -> None:
    chunk: Any = AIMessageChunk(content="just text", id="m1")
    assert chunk_text_and_thinking(chunk) == ("just text", "")


def test_splitter_end_to_end_routing() -> None:
    """The splitter primitive, driven as M7 will drive it per alias."""
    splitter = ThinkTagSplitter()
    text_out: list[str] = []
    think_out: list[str] = []
    for fragment in ("working ", "hard", _TAG_CLOSE, " done"):
        thinking, text = splitter.feed(fragment)
        think_out.append(thinking)
        text_out.append(text)
    assert "".join(think_out) == "working hard"
    assert "".join(text_out) == " done"
