# Lesson 20 — Planning and task decomposition

> **Outcome:** an assistant that answers a message with several questions by planning — a model splits it into typed tasks, LangGraph runs one worker per task in parallel, and a model writes one answer from all the results — with the plan published as an artifact and state reset correctly every turn.

## Learning objectives

- Recognise when to plan: a request that is several independent sub-requests.
- Build the **orchestrator–worker** pattern: a structured plan, `Send` for dynamic fan-out, a reducer to collect parallel results, a join node.
- Bound and validate a plan, and handle the empty and unusable cases.
- Reset per-turn data held under an appending reducer.
- Know what vdreamer provides here (serving, events) and what it does not (a planner).

## Prerequisites

- Lessons [18](../t18_structured_output/README.md) (structured output), [19](../t19_workflows/README.md) (workflows, conditional edges) and [13](../t13_state_and_history/README.md) (reducers).

## Concepts

Lesson 19's workflow had one path per request. Some requests contain several: "Where is A1001, can I still return A1003, and what is your warranty on boots?" A single ReAct loop can handle that one tool call at a time; **planning** makes the structure explicit:

1. **Plan** — a model turns the request into a list of typed tasks (structured output, so the plan can be validated, bounded, logged and tested).
2. **Fan out** — one worker per task. The number of tasks is only known at run time, so the edges are created at run time: a routing function returns `[Send("work", {"task": …}), …]`, and LangGraph runs those worker invocations **in parallel**, each with its own input.
3. **Join** — the workers' results are combined by a reducer, and a node with an edge from `work` runs once all of them have finished.
4. **Synthesise** — a model writes one answer, from the results only.

vdreamer has no planner primitive — by design, its core stays generic (the PRD's design rule). The pattern is pure LangGraph, and vdreamer serves it like any graph: the plan artifact, worker logs and the answer reach the client as events, and every step is checkpointed. LangChain's `TodoListMiddleware` is a different, agent-driven approach: it gives a `create_agent` agent a to-do list tool to plan for itself.

**Per-turn data in a thread.** `results` uses an appending reducer so parallel workers can all write to it. But state persists across turns (lesson 13): the second question would see the first question's results. Returning `[]` does not help — an appending reducer appends nothing. The reducer here treats `None` as "reset", and `plan` sends `None` at the start of every turn. (The tests caught the bug in the first version.)

## What comes from where

| Piece | Provided by |
| --- | --- |
| `Send`, parallel supersteps, reducers, joins | LangGraph |
| `with_structured_output` | LangChain |
| Plan artifact, logs, progress, checkpoints per step | vdreamer |
| Plan schema, workers, the reset reducer | this lesson's code |

## How it works

```
"Where is A1001, can I still return A1003, and what is your warranty on boots?"
plan ─► Plan(tasks=[order_status A1001, return_check A1003, policy warranty])   (+ artifact "plan")
fan_out ─► Send("work", {task: …}) ×3 ─┬─ work: "Order A1001: shipped, arriving 2026-10-04"
                                        ├─ work: "Order A1003: delivered 47 days ago, can no longer be returned"
                                        └─ work: "Policy (warranty): … 2-year warranty …"
results (gather reducer) ──► answer ─► one reply, from those facts only
```

## Step by step

**1. A bounded, typed plan.**

```python
# file: agent.py
class Plan(BaseModel):
    """The independent tasks needed to answer the customer's message."""

    tasks: list[Task] = Field(max_length=MAX_TASKS)
```

**2. A reducer that collects in parallel and can be reset.**

```python
# file: agent.py
def gather(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Parallel workers append their results; ``None`` resets the list for a new turn.

    (Returning ``[]`` would not clear it: an appending reducer appends nothing.)
    """
    return [] if new is None else [*(existing or []), *new]
```

**3. Dynamic fan-out** — one `Send` per task, or straight to the answer when there is nothing to do:

```python
# file: agent.py
def fan_out(state: PlannerState) -> list[Send] | str:
    tasks = state.get("tasks", [])
    if not tasks:
        return "answer"  # nothing to look up: answer directly
    return [Send("work", {"task": task}) for task in tasks]
```

**4. A worker receives its own input** (the `Send` payload), not the whole state:

```python
# file: agent.py
async def work(payload: dict[str, Any]) -> dict[str, Any]:
    """Runs once per planned task, in parallel (``Send``)."""
    task = Task.model_validate(payload["task"])
    vdreamer.emit_log(f"working on {task.kind} {task.order_id or task.topic}")
    return {"results": [do_task(task)]}
```

**5. Wire it.** The edge `work → answer` makes `answer` wait for every worker:

```python
# file: agent.py
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", fan_out, ["work", "answer"])
        graph.add_edge("work", "answer")  # answer waits for every worker
        graph.add_edge("answer", END)
```

## The complete example

[`agent.py`](agent.py) — `Task`, `Plan`, `gather`, `PlannerState`, `do_task`, `work`, `fan_out`, and `PlannerAgent` (`t20-planner`: `plan → work ×N → answer`).

## Run it

From `tutorials/`:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t20-planner --models config/models.ollama.yaml
uv run vdreamer run t20-planner --message \
    "Where is A1001, can I still return A1003, and what is your warranty on boots?"
```

## Expected behavior

```
[artifact] json 'plan' /api/v1/artifacts/art_…/download
[progress] planned 3 tasks 20%
[log info] working on order_status A1001
[log info] working on return_check A1003
[log info] working on policy warranty
assistant>
…one reply covering all three questions…
```

The three log lines may come in any order: the workers run in parallel. "Thanks!" plans no tasks and is answered directly.

## Test it

[`test_t20_planning.py`](../../../../tests/tutorials/test_t20_planning.py) tests the workers and the reducer as plain functions, the bound on the plan, the full fan-out (the answering model saw all three facts), the empty and unusable plans, the per-turn reset, and the gateway:

```python
# file: tests/tutorials/test_t20_planning.py
def test_the_resettable_reducer() -> None:
    assert gather(["a"], ["b"]) == ["a", "b"]
    assert gather(["a"], None) == []
    assert gather(["a"], []) == ["a"]  # why [] cannot reset an appending reducer
```

```bash
uv run pytest tests/tutorials/test_t20_planning.py -v
```

## Common mistakes and debugging

- **Writing parallel results to a key without a reducer** — LangGraph raises `InvalidUpdateError` (several writes to one value in one step). Give it a reducer.
- **Results from the last turn in this turn's answer** — appending reducers never forget. Reset explicitly at the start of a turn.
- **Unbounded plans** — a model can produce twenty tasks for a simple question. Bound the schema (`max_length`) and the cost.
- **Workers that call the model** for lookups code can do — slow and less exact. Plan with a model; execute deterministically where you can.
- **`json data must be a Pydantic model or a dict, got list`** — a JSON artifact holds an object. Wrap lists: `{"tasks": [...]}`.
- **Assuming worker order** — parallel workers finish in any order; sort or key results if order matters.

## Design trade-offs

- **Plan-and-execute vs. ReAct.** Planning makes the decomposition explicit, testable and parallel, but it plans once: if a result changes what should happen next, it cannot react (unless you add a re-plan step). ReAct reacts after every step but serialises work and hides its plan.
- **Deterministic vs. agentic workers.** Code workers are exact and cheap; agent workers (a sub-agent per task, lessons 21–22) handle open-ended subtasks at higher cost and risk.
- **One synthesis call** keeps the answer coherent; streaming per-task answers would be faster to first token but disjointed.

## Exercises

1. Add a `delivery_estimate` task kind (lesson 4's shipping rates) and a test where it runs alongside the others.
2. Add a re-plan step: if any result says "not found", ask the model once whether another task is needed.
3. Emit progress per finished worker (`completed k of n`) — what must you store to know `n`?
4. Bound the total cost: refuse plans whose tasks would call an external API more than twice.

Next: [Lesson 21 — Specialised agents with clear responsibilities](../t21_specialists/README.md).
