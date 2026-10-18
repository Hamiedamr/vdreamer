# Lesson 24 — Handling failures and controlling multi-agent execution

> **Outcome:** the lesson-22 supervisor made safe to run unattended — every specialist capped in model calls, every delegation bounded by a timeout and contained on failure, each specialist and the whole run limited in delegations, and LangGraph's recursion limit as the last backstop — each limit triggered on purpose in a test.

## Learning objectives

- Identify how multi-agent systems fail: loops inside a specialist, slow or failing specialists, over-delegation, ping-pong.
- Put a limit at each level: specialist (`ModelCallLimitMiddleware`), delegation (`timeout_s`, containment), supervisor (`ToolCallLimitMiddleware` per tool and in total), run (`recursion_limit`).
- Choose for each limit whether it **ends gracefully**, **reports an error to the caller agent**, or **fails the run**.
- Test every limit deterministically, including timeouts (a slow scripted model).

## Prerequisites

- Lessons [22](../t22_supervisor/README.md) (the supervisor), [10](../t10_failures_and_retries/README.md) (retries and fallbacks), [9](../t09_tool_selection/README.md) (`ToolCallLimitMiddleware`), [5](../t05_tool_errors/README.md) (containment) and [3](../t03_react_loop/README.md) (budgets vs. recursion limits).

## Concepts

One agent can loop; several agents can loop *inside each other*. A supervisor that re-asks a confused specialist, which itself keeps calling tools, multiplies cost and latency. Unattended multi-agent systems need limits at **every** level, each failing in the mildest useful way:

| Level | Risk | Limit | When hit |
| --- | --- | --- | --- |
| Specialist | keeps calling tools/the model | `ModelCallLimitMiddleware(run_limit=N, exit_behavior="end")` | the specialist stops with "Model call limits exceeded…" — returned to the supervisor as its answer |
| Delegation | slow model server, hanging tool | governed tool `timeout_s` | `Error (timeout): …` to the supervisor |
| Delegation | specialist's model down, tool crash | governance containment | `Error (transport): …` / `Error (tool_error): …` to the supervisor |
| Supervisor | asks one specialist over and over | `ToolCallLimitMiddleware(tool_name="ask_x", run_limit=N)` | the extra call is refused with an error message |
| Supervisor | too many delegations overall | `ToolCallLimitMiddleware(run_limit=N)` | same |
| Run | anything else, e.g. handoff ping-pong (lesson 23) | `recursion_limit` (client-settable per run) | the run **fails** (`node_error: Recursion limit …`) |

Two principles:

- **Contain at the boundary.** A delegation is a governed tool call, so whatever goes wrong inside a specialist arrives at the supervisor as an error message it can explain ("order lookups are unavailable right now"), never as a crashed run. The run fails only at the outermost backstop.
- **Tell the coordinator what to do.** Limits produce messages; the supervisor's prompt says how to react to them (`GUARDRAILS`: do not retry endlessly; report what you could and could not find out).

## What comes from where

| Piece | Provided by |
| --- | --- |
| `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware` | LangChain |
| `timeout_s` for a delegation, containment with error codes, `tool.result` events | vdreamer governance |
| `recursion_limit`, settable per run by clients (`{"config": {"recursion_limit": N}}`) | LangGraph / vdreamer run config |
| `Limits`, the guarded delegate, the guardrail prompt | this lesson's code |
| `ScriptedChatModel(delay_s=...)`, `RoutedChatModel` | the series' test support |

## How it works

```
customer ─► supervisor  [per-specialist ask limits, total delegation limit]
              └─ ask_orders (timeout_s)          ─► orders specialist  [model-call cap, end gracefully]
                   ✗ model down     → Error (transport): orders model down      ─┐
                   ✗ too slow       → Error (timeout): … within 60s              ├─► supervisor explains
                   ✗ loops          → "Model call limits exceeded: run limit (4/4)" ─┘
              └─ ask_orders again (3rd time)     → "Tool call limit exceeded. Do not call 'ask_orders' again."
run: recursion_limit ─► if all else fails: run.status failed
```

## Step by step

**1. Name the limits in one place.**

```python
# file: agent.py
@dataclass(frozen=True)
class Limits:
    delegation_timeout_s: float = 60
    max_specialist_model_calls: int = 4
    max_asks_per_specialist: int = 2
    max_delegations: int = 5
```

**2. Cap each specialist, ending gracefully.**

```python
# file: agent.py
    specialist = build_specialist(spec, model, middleware=[
        VDreamerToolMiddleware(),
        # "end": stop the specialist with a message instead of failing the run.
        ModelCallLimitMiddleware(run_limit=limits.max_specialist_model_calls,
                                 exit_behavior="end"),
    ])
```

**3. Bound each delegation** with the governed tool's timeout (it covers the whole sub-conversation):

```python
# file: agent.py
    @vdreamer.tool(
        f"ask_{spec.name}",
        description=f"{spec.responsibility} Ask one self-contained question.",
        timeout_s=limits.delegation_timeout_s,
    )
```

**4. Limit the supervisor**, per specialist and in total — after vdreamer's governance middleware:

```python
# file: agent.py
        per_specialist = [
            ToolCallLimitMiddleware(tool_name=f"ask_{spec.name}",
                                    run_limit=limits.max_asks_per_specialist)
            for spec in SPECIALISTS
        ]
```

**5. Tell the supervisor how to react.**

```python
# file: agent.py
GUARDRAILS = (
    "\nIf a specialist reports an error, times out, or you hit a limit, do not retry "
    "endlessly: tell the customer what you could and could not find out."
)
```

## The complete example

[`agent.py`](agent.py) — `Limits`, `make_guarded_delegate`, `GUARDRAILS`, `GuardedSupervisorAgent` (`t24-guarded-supervisor`). Specialists come from [lesson 21](../t21_specialists/specialists.py), the supervisor prompt from [lesson 22](../t22_supervisor/agent.py).

## Run it

From `tutorials/`:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t24-guarded-supervisor --models config/models.ollama.yaml
uv run vdreamer run t24-guarded-supervisor --message "Where is A1001 and do you have family tents?"
```

To see a backstop, start a run over HTTP with `"config": {"recursion_limit": 3}`. To see containment, stop Ollama in the middle of a run.

## Expected behavior

Normally, the same flow as lesson 22. When a specialist is too slow (the gateway test sets a 0.3 s timeout and a 1 s specialist):

```
-> ask_orders({"question": "A1001?"}) [custom]
<- error Error (timeout): the tool did not finish within 0.3s (300 ms)
assistant>
Order lookups are slow right now; please try again soon.
```

The run still completes.

## Test it

[`test_t24_multi_agent_control.py`](../../../../tests/tutorials/test_t24_multi_agent_control.py) triggers every limit on purpose. A slow specialist is a scripted model with `delay_s`; a failing one has an exception in its script:

```python
# file: tests/tutorials/test_t24_multi_agent_control.py
    supervisor = scripted(call("ask_orders", question="Status of A1001?"),
                          "Order lookups are unavailable right now.")
    result = await _run(Limits(), supervisor=supervisor,
                        orders=scripted(ConnectionError("orders model down")))

    assert result.tool_results[0]["error"]["code"] == "transport"
    seen = supervisor.prompts[1][-1]
    assert seen.text == "Error (transport): orders model down"  # the supervisor can explain
```

```bash
uv run pytest tests/tutorials/test_t24_multi_agent_control.py -v
```

## Common mistakes and debugging

- **Limits only at the top** — a recursion limit stops a runaway system by *failing* it, after it has spent its budget. Put graceful limits at the inner levels.
- **`exit_behavior="error"` on a specialist** — its limit then fails the delegation as an error; fine, but `"end"` gives the supervisor a usable partial answer.
- **A delegation timeout shorter than a healthy specialist needs** — every question times out. Measure real durations (the `tool.result` `duration_ms`) and add headroom.
- **Retrying a failing specialist** — the supervisor's prompt must say not to; the per-specialist limit enforces it.
- **Limits that hide incidents** — a contained failure looks like a polite answer. vdreamer's metrics (`vdreamer_tool_errors_total{code="timeout"}`, lesson 12) make them visible.

## Design trade-offs

- **Graceful vs. loud.** Ending gracefully keeps customers served; failing loudly gets problems fixed. Use graceful limits plus metrics and alerts.
- **Static limits vs. budgets.** Fixed counts are simple; a cost or token budget shared across agents tracks what you actually pay for, at the cost of bookkeeping.
- **Fallbacks per specialist** (lesson 10's `ModelFallbackMiddleware` inside each specialist) can turn many `transport` errors into answers — consider it for specialists whose answers matter most.

## Exercises

1. Add `ModelFallbackMiddleware` to the orders specialist with the `backup` alias (lesson 10) and test that a `ConnectionError` now produces an answer.
2. Apply a per-turn handoff budget to lesson 23's desk (count transfers in state; refuse the third).
3. Emit a `log` warning event whenever a delegation fails, so the client can show "some information is unavailable".

Next: Lesson 25 — Introducing human-in-the-loop.
