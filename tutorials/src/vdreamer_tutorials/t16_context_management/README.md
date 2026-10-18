# Lesson 16 — Managing context growth and conversation boundaries

> **Outcome:** an assistant that can talk for hours on one thread: once a conversation passes a number of turns, the oldest turns are summarised by a background model call, the summary lives in thread state and the system prompt, and the old messages are deleted — without breaking tool-call pairs, leaking the summary to the client, or re-sending earlier answers.

## Learning objectives

- Explain why context grows, what it costs, and the three remedies: **trim**, **summarise**, **start a new thread**.
- Summarise old turns in a `@before_model` hook, keep the summary in its own state key, and delete old messages by id with `RemoveMessage`.
- Cut history only at turn boundaries, so a tool call is never separated from its result.
- Keep background model calls out of the client's stream (the `nostream` tag).
- Know what LangChain's prebuilt `SummarizationMiddleware` does and its two client-visible side effects under vdreamer.

## Prerequisites

- Lessons [13](../t13_state_and_history/README.md) (state, `RemoveMessage`, checkpoints) and [14](../t14_short_term_memory/README.md) (`state_schema`, `@dynamic_prompt`).

## Concepts

The transcript is resent on every model call (lesson 14). A long conversation therefore gets **slower and more expensive with every turn**, eventually exceeds the model's context window, and — well before that — quality drops as the important facts drown in old small talk.

| Remedy | What it does | Loses |
| --- | --- | --- |
| Trim | send only the last N messages/tokens | everything older, abruptly |
| Summarise | replace old turns with a model-written summary | detail (the summary is lossy) |
| New thread | start a fresh conversation for a new topic | the whole thread (keep durable facts in long-term memory, lesson 15) |

This lesson summarises. Three details make the difference between "works in a demo" and "works for users":

1. **Cut at turn boundaries.** An `AIMessage` with tool calls must be followed by their `ToolMessage`s; deleting one without the other makes the next request invalid. Cutting just before a user message never splits a pair.
2. **Keep the summary out of the transcript.** Storing it in its own state key (rendered into the system prompt by `@dynamic_prompt`) means old messages can simply be *deleted by id*. Nothing already shown to the user is re-added to state — and vdreamer streams every AI message that appears in a state update, so re-added messages would be shown again.
3. **Keep the summariser silent.** Every chat-model call inside a run streams to the client as `token.delta` events. LangGraph skips calls on a model tagged `nostream`; `background_model()` makes such a copy.

LangChain's `SummarizationMiddleware` implements summarisation too. Under vdreamer it has two side effects, both shown in this lesson's tests: it rewrites the message list (so the kept AI messages are re-sent to clients as `message.completed`, with their original ids), and, unless given a `nostream` model, its summary streams to the client as deltas. A client that folds messages by `message_id` hides the first; the second needs the tag.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `@before_model`, `@dynamic_prompt`, `state_schema`, `SummarizationMiddleware` | LangChain |
| `RemoveMessage`, `add_messages` delete-by-id, the `nostream` tag | LangGraph |
| Streaming of model calls and of AI messages in state updates; `emit_log` | vdreamer |
| Turn-boundary cut, state-held summary, `background_model` | this lesson's code |

## How it works

`LongConversationAgent(max_turns=12, keep_turns=4)`: before each model call, if the thread holds more than 12 user turns:

```
messages: [q1 a1 q2 a2 … q13]                summary: "…"(earlier, if any)
cut = start of the last 4 turns  ──► old = everything before it
summarizer (nostream copy of "chat"):  [instructions, "Summary of what came before: …", *old] → new summary
update: summary = new summary;  messages = [RemoveMessage(id) for each old message]
system prompt (dynamic): BASE_PROMPT + "\nEarlier in this conversation: <summary>"
```

The client sees one `log` event ("summarised 18 earlier messages") and the new answer — nothing else.

## Step by step

**1. A state key for the summary.**

```python
# file: agent.py
class SummarizedState(AgentState[Any]):
    summary: NotRequired[str]
```

**2. Cut at a user message.**

```python
# file: agent.py
def turn_cut(messages: Sequence[AnyMessage], keep_turns: int) -> int:
    """Index where the last ``keep_turns`` user turns start (0: nothing to cut).

    Cutting at a user message keeps every tool call with its result.
    """
    starts = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    return starts[-keep_turns] if len(starts) > keep_turns else 0
```

**3. Summarise silently, fold in the previous summary, delete by id.**

```python
# file: agent.py
            old = messages[: turn_cut(messages, self.keep_turns)]
            earlier = state.get("summary")
            transcript = [SystemMessage(SUMMARY_INSTRUCTIONS)]
            if earlier:
                transcript.append(HumanMessage(f"Summary of what came before: {earlier}"))
            summary = await summarizer.ainvoke([*transcript, *old])
            vdreamer.emit_log(f"summarised {len(old)} earlier messages")
            return {
                "summary": summary.text,
                "messages": [RemoveMessage(id=m.id) for m in old if m.id is not None],
            }
```

`summarizer` is `background_model(ctx.models["chat"])`:

```python
# file: agent.py
def background_model(model: BaseChatModel) -> BaseChatModel:
    """A copy of ``model`` whose calls are not streamed to clients.

    Every chat-model call inside a run streams to the client as ``token.delta``
    events, a summariser's included. LangGraph skips calls tagged ``nostream``.
    """
    return model.model_copy(update={"tags": [*(model.tags or []), "nostream"]})
```

**4. Render the summary into every prompt.**

```python
# file: agent.py
        @dynamic_prompt
        def prompt_with_summary(request: ModelRequest) -> str:
            summary = request.state.get("summary")
            return f"{BASE_PROMPT}\nEarlier in this conversation: {summary}" if summary \
                else BASE_PROMPT
```

**5. Draw conversation boundaries in the client.** A thread is one conversation. When the user starts a new topic, start a new thread (`POST /api/v1/agents/{name}/threads`; `vdreamer run` without `--thread`). Delete finished threads with `DELETE /api/v1/threads/{id}`. Facts that must survive go to long-term memory.

## The complete example

[`agent.py`](agent.py) — `SummarizedState`, `background_model`, `turn_cut`, and `LongConversationAgent` (`t16-long-conversation`; `max_turns`/`keep_turns` are constructor arguments so tests use small numbers).

## Run it

From `tutorials/`. To see a summary quickly, register a variant with small limits in your own registry, or use the tests below; with the defaults the 13th user turn triggers the first summary:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t16-long-conversation --models config/models.ollama.yaml
uv run vdreamer run t16-long-conversation
```

Then read the thread: `GET /api/v1/threads/<id>` shows the `summary` key and the remaining messages.

## Expected behavior

On the turn that triggers summarisation, `vdreamer run` shows:

```
[log info] summarised 18 earlier messages
assistant>
…the answer to the current question…
```

and no earlier answer is repeated. The model keeps knowing what was discussed (order ids, decisions) through the summary.

## Test it

[`test_t16_context_management.py`](../../../../tests/tutorials/test_t16_context_management.py) checks the cut, the state after a summary, what the answering model saw, that the client saw only the new answer, that summaries fold into the next one, that tool pairs survive — and pins the prebuilt middleware's side effects:

```python
# file: tests/tutorials/test_t16_context_management.py
async def test_clients_see_only_the_new_answer() -> None:
    _, results = await _four_turns(scripted("one", "two", "three", SUMMARY, "four"))
    last = results[-1]
    assert [p["text"] for p in last.payloads("message.completed")] == ["four"]
    assert SUMMARY not in json.dumps([e.payload for e in last.events])
```

```bash
uv run pytest tests/tutorials/test_t16_context_management.py -v
```

## Common mistakes and debugging

- **Cutting by message count** — sooner or later the cut lands between a tool call and its result, and the next request fails (lesson 3). Cut at user turns.
- **The summary appears in the chat UI** — the summariser's model call streamed. Use a `nostream` copy of the model.
- **Old answers reappear after summarising** — something re-added AI messages to state (a rewrite instead of deletes). Delete by id, keep the summary elsewhere, or make your client fold by `message_id`.
- **Summaries of summaries drift** — each fold loses detail. Put durable facts in structured state (lesson 13) or long-term memory (lesson 15), not only in the summary.
- **Summarising on every turn** — a model call per turn. Use a threshold with headroom (`max_turns` well above `keep_turns`).
- **Counting tokens with the wrong tokenizer** — if you switch to a token threshold, use the model's tokenizer or a conservative approximation; turn counts are simpler and predictable.

## Design trade-offs

- **Turns vs. tokens.** A turn threshold is predictable and model-independent; a token threshold tracks the real constraint. Long tool results make turns a poor proxy — combine with result limits (lesson 5) or `ContextEditingMiddleware`, which clears old tool outputs.
- **Own hook vs. `SummarizationMiddleware`.** The prebuilt one supports token/fraction triggers and context-overflow retries; the hook here is ~30 lines, keeps the client stream clean, and is easy to test. Prefer the prebuilt one when your client folds by `message_id` and you pass a `nostream` model.
- **Summarise in the run vs. in the background.** Summarising before the model call adds latency to that one turn. A background job could do it between turns, at the cost of complexity.

## Exercises

1. Trigger on approximate tokens instead of turns (`langchain_core.messages.utils.count_tokens_approximately`) and test both thresholds.
2. Keep order ids out of the summary's lossiness: extract them into a state key (lesson 13) before deleting old messages.
3. Add a `/new` command in your client that starts a new thread and copies the summary into the first message.

Next: Lesson 17 — Testing memory persistence, retrieval and isolation.
