# Lesson 10 — Handling failures, retries and unexpected outputs

> **Outcome:** a parcel-tracking agent that retries a flaky carrier API without the user noticing, retries a failing model and then falls back to a second model, and never shows a customer an empty answer — each failure path scripted in a test.

## Learning objectives

- Tell **transient** failures (retry) from **permanent** ones (report) and from **unexpected outputs** (repair or replace).
- Retry tool calls with `ToolRetryMiddleware` and model calls with `ModelRetryMiddleware`, with exponential backoff.
- Fall back to another model alias with `ModelFallbackMiddleware`, and order these middlewares correctly.
- Guard the model's output with an `@after_model` hook — and make sure the client actually sees the repair.
- Know which failures vdreamer itself recovers from (crashed workers) and which it reports (failed runs).

## Prerequisites

- Lessons [5](../t05_tool_errors/README.md) (containment, error codes), [8](../t08_tool_contracts/README.md) (dependency injection, idempotency) and [9](../t09_tool_selection/README.md) (middleware and its order).

## Concepts

Failures come in three kinds, and each needs a different response:

| Kind | Example | Response |
| --- | --- | --- |
| Transient | carrier API returns 503; model server connection reset | **retry** with backoff — the same call will probably work soon |
| Permanent | unknown tracking id; model server down for an hour | **report** (to the model, which tells the user) or **fall back** to an alternative |
| Unexpected output | the model returns an empty reply, or text where JSON was required | **validate and repair**, or replace with a safe answer |

LangChain ships middleware for the first two, and `create_agent` lets you add your own hooks for the third:

- `ToolRetryMiddleware(tools=[...], max_retries, retry_on, initial_delay, backoff_factor)` re-runs a failing tool call. With `retry_on=(ConnectionError, TimeoutError)` only transient errors are retried; a `ToolException` ("unknown tracking id") is not. After the last attempt (`on_failure="continue"`) the model receives an error message.
- `ModelRetryMiddleware` does the same for model calls. With `on_failure="error"` it re-raises after its last attempt — which is what lets an outer `ModelFallbackMiddleware` take over.
- `ModelFallbackMiddleware(backup_model)` calls the next model when the current one raises. Here the backup is another vdreamer alias, `ctx.models["backup"]`.
- `@after_model` runs after every model call with the agent state, and may return a state update.

**Retries must be safe.** Retrying a read is harmless; retrying a write can duplicate it. Only retry side-effecting tools that are idempotent (lesson 8).

**What vdreamer adds.** Governance wraps the whole retry loop: one `tool.call`, one `tool.result`, and `timeout_s` covering every attempt. And vdreamer recovers from *crashes*: when a worker dies mid-run, its lease expires and the run is requeued (up to 3 attempts) and resumes from the last checkpoint. A run whose code *raises* is not retried; it ends `failed`, because running the same failing code again rarely helps.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `ToolRetryMiddleware`, `ModelRetryMiddleware`, `ModelFallbackMiddleware`, `@after_model` | LangChain (`langchain.agents.middleware`) |
| `RemoveMessage` | LangChain (`langchain_core`) / LangGraph's `add_messages` |
| `models = ("chat", "backup")`, both aliases in `models.yaml` | vdreamer |
| One `tool.call`/`tool.result` per call however many retries; `timeout_s` over all attempts | vdreamer governance |
| Crash recovery: requeue on an expired lease, resume from checkpoint (max 3 attempts) | vdreamer worker |

## How it works

```
middleware (outermost first):
  vdreamer governance ─ ModelFallback(backup) ─ ModelRetry(2, re-raise) ─ ToolRetry(2, transient) ─ replace_empty_answer

model call:  chat ✗ → retry ✗ → retry ✗ → (re-raise) → fallback: backup ✓
tool call:   governance: tool.call ─► locate ✗ 503 → wait → locate ✗ 503 → wait → locate ✓ ─► tool.result ok
reply "":    after_model → RemoveMessage(empty) + AIMessage(FALLBACK_ANSWER) → client sees the fallback
```

## Step by step

**1. Inject the unreliable dependency** (lesson 8), so tests can decide how flaky it is:

```python
# file: agent.py
class FlakyCarrier:
    """A stand-in for a carrier API that fails a few times before answering."""
```

**2. Retry transient tool errors only.** The delay is a constructor argument so tests run without waiting:

```python
# file: agent.py
            ToolRetryMiddleware(tools=["track_parcel"], max_retries=2, retry_on=TRANSIENT,
                                initial_delay=self.retry_delay_s),
```

**3. Retry the model, then fall back.** The fallback must be *outside* the retry, and the retry must re-raise when it gives up — otherwise (with the default `on_failure="continue"`) it would return an error message as the model's answer and the fallback would never run:

```python
# file: agent.py
            # Outer: if the primary model still fails after its retries, use the backup.
            ModelFallbackMiddleware(ctx.models["backup"]),
            # Inner: retry each model, re-raising when out of retries so the fallback runs.
            ModelRetryMiddleware(max_retries=2, initial_delay=self.retry_delay_s,
                                 on_failure="error"),
```

**4. Guard the output.** An empty reply is replaced. The replacement must be a **new** message: the empty one has already streamed to the client, and replacing it in state under the same id would change the thread's history but send the client nothing.

```python
# file: agent.py
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and not last.tool_calls and not last.text.strip():
        vdreamer.emit_log("the model returned an empty answer; replaced it", level="warning")
        assert last.id is not None
        return {"messages": [RemoveMessage(id=last.id), AIMessage(FALLBACK_ANSWER)]}
    return None
```

**5. Declare and configure the second model.** `models = ("chat", "backup")`; both config files define `backup` (a smaller local model in [`models.ollama.yaml`](../../../config/models.ollama.yaml)).

## The complete example

[`agent.py`](agent.py) — `FlakyCarrier`, `make_track_parcel`, `replace_empty_answer`, and `ResilientTrackerAgent` (`t10-resilient-tracker`; the registered instance's carrier fails once, then answers).

## Run it

From `tutorials/`. vdreamer probes every declared alias at startup, so both `chat` and `backup` must be served (`ollama pull qwen3:8b` for the default backup):

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t10-resilient-tracker --models config/models.ollama.yaml
uv run vdreamer run t10-resilient-tracker --message "Where is parcel TRK-1001?"
```

## Expected behavior

The carrier fails on the first attempt, and the user sees one clean call after a one-second pause:

```
-> track_parcel({"tracking_id": "TRK-1001"}) [custom]
<- ok TRK-1001: in transit, at the Frankfurt hub (~1000 ms)
```

The duration includes the retry's backoff (1 s ± 25% jitter). If the carrier kept failing, the result would read `error Error (tool_error): Tool 'track_parcel' failed after 3 attempts with ConnectionError: carrier API: 503 Service Unavailable. Please try again.`

## Test it

[`test_t10_failures_and_retries.py`](../../../../tests/tutorials/test_t10_failures_and_retries.py) scripts each path. Scripted models can include exceptions, which that model call raises — a model server failing on cue:

```python
# file: tests/tutorials/test_t10_failures_and_retries.py
    down = ConnectionError("model server down")
    chat, backup = scripted(down, down, down), scripted("Answer from the backup model.")
    result = await _run(FlakyCarrier(), chat, backup)

    assert chat.calls_made == 3 and backup.calls_made == 1
```

The empty-answer test asserts on what the *client* received, which is how the id problem in step 4 was found:

```python
# file: tests/tutorials/test_t10_failures_and_retries.py
    result = await _run(FlakyCarrier(), scripted(""))

    assert result.answer == FALLBACK_ANSWER
```

```bash
uv run pytest tests/tutorials/test_t10_failures_and_retries.py -v
```

## Common mistakes and debugging

- **Retrying everything** (`retry_on` left at its default for a tool that raises `ToolException` for "not found") — three identical failures and three times the latency. Retry transient errors only.
- **Fallback that never happens** — `ModelRetryMiddleware` with the default `on_failure="continue"` swallows the last error into an AI message; or the fallback is listed *inside* the retry. Re-raise, and put the fallback outside.
- **Retry delays longer than the tool's `timeout_s`** — governance's timeout covers all attempts, so the last retries are cut off with `timeout`. Budget `timeout_s ≥ attempts × call time + total backoff`.
- **A repair the client never sees** — editing an already-streamed message in state (same id). Clients receive events, not state: emit a new message.
- **Retrying non-idempotent writes** — a refund issued twice. Make the write idempotent first (lesson 8), or do not retry it.
- **`dev: model 'backup': … unreachable`** — every declared alias is probed at startup, even one used only on failure. Serve it, or point it at the same server as `chat` while developing.

## Design trade-offs

- **Where to retry.** Retrying inside the HTTP client (most SDKs have `max_retries`) is closest to the failure; middleware retries are visible to the agent's design and testable; LangGraph's node-level `retry_policy=RetryPolicy(...)` (for hand-written graphs) retries a whole node. Avoid stacking all three: retries multiply.
- **Fallback model quality.** A smaller backup keeps the service up but may answer worse, and may not support the same tools. Test the agent against the backup too (the live tier, lesson 6).
- **Fail vs. degrade.** Replacing an empty answer with an apology keeps the conversation going but hides a problem; the `log` event and your monitoring (lesson 12) make sure someone still sees it.

## Exercises

1. Add `ToolException` for unknown tracking ids (`"TRK-"` prefix check) and test that it is **not** retried (`carrier.calls == 1`).
2. Make `ModelRetryMiddleware` retry only `ConnectionError` and `TimeoutError`, and test that a `ValueError` from the model goes straight to the fallback.
3. Extend `replace_empty_answer` to also catch a reply that just repeats the user's question.
4. Use `ToolCallLimitMiddleware` (lesson 9) together with the retry: which one counts retries?

Next: [Lesson 11 — Structuring an agent application as a Python package](../t11_package_structure/README.md).
