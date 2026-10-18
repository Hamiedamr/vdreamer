# Lesson 6 — Testing agents and reading execution traces

> **Outcome:** a returns-eligibility agent with an injectable clock, tested at four levels — the business rule, the tool, the agent with a scripted model, the full gateway — plus model-server and live-model tiers; and the skill to read any run's event trace.

## Learning objectives

- Split an agent's behavior into layers that can each be tested deterministically, and know what each layer's tests can and cannot prove.
- Use `ScriptedChatModel` and `AgentSession` (this series' test support, built on vdreamer's public `AgentRunner`) and understand how they work.
- Remove hidden inputs — here, today's date — so tests do not depend on when they run.
- Read an execution trace: the twelve protocol event types and what each step of a run produces.
- Run agents behind a real gateway in tests, against stub model servers, and (opt-in) against a live model.

## Prerequisites

- Lessons [2](../t02_minimal_react_agent/README.md)–[5](../t05_tool_errors/README.md). Basic [pytest](https://docs.pytest.org/) (`assert`, fixtures, `parametrize`); async tests run automatically in this repository (`asyncio_mode = "auto"`).

## Concepts

An agent mixes deterministic code (tools, routing, state) with a non-deterministic component (the model). You cannot unit-test the model's judgement, but you do not have to: most bugs live in the deterministic parts and in the wiring between them. So test in layers:

| Level | What runs | Proves | Speed |
| --- | --- | --- | --- |
| 1. Business rule | a plain function | the logic, every edge case | µs |
| 2. Tool | `tool.invoke(...)` | the contract and the wiring to dependencies | µs |
| 3. Agent | the graph with a **scripted model** | tool routing, state, events, what the model is *shown* | ms |
| 4. Gateway | the real server, worker, SSE and `vdreamer run` | the deployment path, auth, artifacts | ~0.5 s |
| 5. Model server | the CLI against stub Ollama/vLLM servers | config, HTTP clients, wire formats | ~2 s |
| 6. Live model | a real model (opt-in) | the model's judgement on your prompts and tools | seconds, costs, flaky |

A **scripted model** replaces the LLM with a fixed list of replies: "first ask for `check_return(A1003)`, then answer with this text". It cannot tell you whether a real model *would* do that, but it lets you prove everything that happens around such a decision — that the tool runs with those arguments, that the result is shown to the model, that the right events reach the client — and it never flakes.

The other enemy of deterministic tests is **hidden input**: the clock, random numbers, environment variables, network calls. Make them explicit parameters. Here the agent receives its clock in the constructor.

An **execution trace** is the ordered list of protocol events of a run — exactly what a client receives over SSE. Learning to read it is the fastest way to debug an agent:

| Event | Emitted when |
| --- | --- |
| `run.status` | the run starts (`running`) and ends: `completed`, `failed` (+ error), `awaiting_approval`, `rejected`, `expired`, `cancelled` |
| `token.delta` / `thinking.delta` | text (or model reasoning) streams, batched |
| `message.completed` | an `AIMessage` is finished — tool-call messages included (empty text) |
| `tool.call` / `tool.result` | a governed tool starts / finishes (`ok` or `error` + code) |
| `artifact.created` | a table, chart, JSON, HTML or file is stored |
| `citation` | an answer is bound to a source (`emit_citation`) |
| `interrupt.requested` / `interrupt.resolved` | the run pauses for / resumes after a human decision |
| `progress` / `log` | your code calls `emit_progress` / `emit_log` |

## What comes from where

| Piece | Provided by |
| --- | --- |
| `AgentRunner`, `InMemoryEventSink`, `VDreamerContext`, `EventEnvelope` | vdreamer (public API) |
| `ScriptedChatModel`, `AgentSession`, `RunResult`, `call`/`calls`/`scripted` | this series: [`vdreamer_tutorials/testing.py`](../testing.py) |
| `BaseChatModel`, `InMemorySaver`, `Command(resume=...)` | LangChain / LangGraph |
| Gateway kit (`served`, `vdreamer_run`) | this repository's test kits: [`tests/tutorials/_tutorial_kit.py`](../../../../tests/tutorials/_tutorial_kit.py) over `tests/api/_api.py` and `tests/acceptance/_uc_kit.py` |
| Stub model servers | this repository's test kit: [`tests/models/_stub_llm.py`](../../../../tests/models/_stub_llm.py) |

## How it works

`AgentSession` does what a vdreamer worker does for one run, minus the queue and the database:

```
AgentSession(agent, models={"chat": scripted(...)})
   └─ AgentRunner(agent, checkpointer=InMemorySaver(), models=...)   ← vdreamer's own runner
        ├─ build(ctx) once (ctx.models["chat"] is the scripted model)
        └─ arun(input, config={thread_id}, context=VDreamerContext(user, roles…),
                sink=InMemoryEventSink())                               ← the events a client sees
RunResult(events, values) → .status .answer .tool_calls .tool_results .interrupts .trace()
```

The scripted model is a LangChain `BaseChatModel`, so the agent cannot tell it from a real one. Each call pops the next reply and records the full prompt:

```python
# file: tutorials/src/vdreamer_tutorials/testing.py
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
```

A script may also contain exceptions (raised by that call — a failing model server, lesson 10) and a `delay_s` (a slow one, lesson 24). Running past the end of the script raises instead of repeating a reply, so an agent that loops more than you expected fails the test loudly. `bind_tools` returns the model itself (and records the tool names): the *script*, not the tool list, decides what gets called.

## Step by step

**1. Pull the business rule out into a plain function** — it needs no LangChain at all, so it can be tested exhaustively, edge cases included:

```python
# file: agent.py
def return_decision(order_id: str, today: date) -> str:
    """The business rule, as a plain function: easy to test exhaustively."""
```

**2. Inject the hidden input.** The tool is built around a clock; the agent takes the clock in its constructor; the registered instance uses the real one.

```python
# file: agent.py
def make_check_return(today: Callable[[], date]) -> BaseTool:
    """Build the tool around a clock, so tests can choose what "today" is."""

    @vdreamer.tool
    def check_return(order_id: str) -> str:
        """Check whether an order can still be returned (30 days from delivery)."""
        return return_decision(order_id, today())

    return check_return
...
class ReturnsAgent(BaseAgent):
    name = "t06-returns"
    mode = Mode.CHAT
    models = ("chat",)

    def __init__(self, today: Callable[[], date] = date.today) -> None:
        self.today = today
```

**3. Test the boundaries of the rule.** The interesting days are the last day inside the window, the first day outside, and the delivery day itself:

```python
# file: tests/tutorials/test_t06_testing_and_traces.py
@pytest.mark.parametrize(
    ("today", "verdict"),
    [
        (date(2026, 10, 1), "can be returned: delivered 30 days ago"),     # the last day
        (date(2026, 10, 2), "cannot be returned: delivered 31 days ago"),  # one day late
        (date(2026, 9, 1), "can be returned: delivered 0 days ago"),       # delivery day
    ],
)
```

**4. Test the agent: behavior, not wording.** Assert on the tool calls, the run status and *what the model was shown* — never on the exact phrasing of the scripted answer, which you wrote yourself:

```python
# file: tests/tutorials/test_t06_testing_and_traces.py
    # Behavior, not wording: the right tool, the right argument, a completed run.
    assert result.status == "completed"
    assert result.tool_calls == [("check_return", {"order_id": "A1003"})]
    # What the model was given: the system prompt, then the tool's verdict.
```

**5. Pin a trace when the sequence matters.** `RunResult.trace()` renders the events without ids or timings, so it is stable enough to compare whole:

```python
# file: tests/tutorials/test_t06_testing_and_traces.py
    assert result.trace() == "\n".join([
        "run.status        running",
        "message.completed ai: ''",
        "tool.call         check_return({'order_id': 'A1002'})",
```

Use whole-trace assertions sparingly: they break on any harmless change. Prefer targeted assertions (`result.tool_calls`, `result.status`) and keep one golden trace per important flow.

**6. Go through the real stack.** `served([...])` starts the gateway and a worker on a free port; `vdreamer_run` is the terminal client, with scripted answers for any prompts:

```python
# file: tests/tutorials/test_t06_testing_and_traces.py
    with served([agent], {agent.name: backends}) as url:
        result = vdreamer_run(url, monkeypatch, agent=agent.name,
                              message="Return A1003?", once=True, color=False)
    assert result.code == 0, result.out
```

**7. Check the model servers and the real model.** [`test_tutorials_model_servers.py`](../../../../tests/tutorials/test_tutorials_model_servers.py) runs the documented `vdreamer dev … --models …` command against stub Ollama and vLLM servers. [`test_tutorials_live_ollama.py`](../../../../tests/tutorials/test_tutorials_live_ollama.py) runs agents against your real model; it is skipped unless you opt in, because it is slow, may cost money, and a model's choices vary.

## The complete example

- [`agent.py`](agent.py) — `return_decision`, `make_check_return`, `ReturnsAgent` (`t06-returns`).
- [`vdreamer_tutorials/testing.py`](../testing.py) — the test support every lesson uses.
- [`test_t06_testing_and_traces.py`](../../../../tests/tutorials/test_t06_testing_and_traces.py) — one test file, all four in-process levels.

## Run it

From the repository root:

```bash
uv run pytest tests/tutorials/test_t06_testing_and_traces.py -v      # levels 1–4
uv run pytest tests/tutorials/test_t03_react_loop.py -s -k cycles     # print a trace
uv run pytest tests/tutorials/test_tutorials_model_servers.py -v      # level 5
uv run pytest tests/tutorials -v                                      # the whole series
```

The live tier (level 6) — needs Ollama with the model from `tutorials/config/models.ollama.yaml`; `gemma4:31b-cloud` causes egress:

```bash
VDREAMER_TEST_OLLAMA=1 uv run pytest tests/tutorials/test_tutorials_live_ollama.py -v -s
# another server: VDREAMER_TUTORIALS_MODELS=tutorials/config/models.vllm.yaml VDREAMER_TEST_OLLAMA=1 …
```

The agent itself runs like the others: `uv run vdreamer dev --registry vdreamer.yaml --agent t06-returns --models config/models.ollama.yaml` from `tutorials/`.

## Expected behavior

All in-process and gateway tests pass in about a second. The trace of "Return A1002?":

```
run.status        running
message.completed ai: ''
tool.call         check_return({'order_id': 'A1002'})
tool.result       ok: Order A1002 has not been delivered yet, so it cannot be returned yet.
message.completed ai: 'Not delivered yet.'
run.status        completed
```

Without `VDREAMER_TEST_OLLAMA=1` the live tests report `SKIPPED … live model tier`.

## Test it

This lesson *is* the test pattern. One more worth reading: the gateway test once caught a real bug. With the test clock fixed at 2026-10-01, order A1001 (delivered 2026-10-04) printed `can be returned: delivered -3 days ago`. The fix — a delivery date in the future counts as "not delivered yet" — now has its own level-1 test:

```python
# file: tests/tutorials/test_t06_testing_and_traces.py
    # A1001 is delivered on 2026-10-04: after TODAY. Found by the gateway test below,
    # which once printed "delivered -3 days ago".
    assert "has not been delivered yet" in return_decision("A1001", TODAY)
```

Fixing the date exposed an edge case that `date.today()` would only have hit on certain days of the year — or in a timezone you do not test in.

## Common mistakes and debugging

- **`ScriptExhausted: the model was called 3 times but only 2 replies were scripted`** — the agent made one more model call than you planned: usually the final answer after a tool call. Script the whole conversation, one reply per model call.
- **Asserting the scripted answer's wording** proves nothing about your agent: you wrote that text. Assert on tool calls, arguments, statuses, events and prompts.
- **A test that passes alone and fails in the suite** — shared state: a module-level dictionary a tool mutates, or a flag set without `monkeypatch`. Use `monkeypatch.setattr` so it is restored.
- **One scripted model shared across runs** — `AgentSession` builds the agent once, like a worker, so one script serves every run of the session: script all turns in order. The gateway kit builds a fresh model per `served(...)` block.
- **`RuntimeError: no active vdreamer run context`** when calling a node function directly — `emit_*` helpers need a run. Test nodes through `AgentSession`, or keep logic in plain functions.
- **Live tests that assert exact text** — they will flake. Assert on the workflow, with tolerance (`a["order_id"].upper()`).
- **Reading a trace:** no `tool.call` after a tool-call message means the tool node never ran (routing); `tool.result error` tells you the code; a `run.status failed` right after `tool.call` means an *ungoverned* tool raised (or a governed one with `fail_run=True`).

## Design trade-offs

- **Scripted vs. live.** Scripted tests are fast and deterministic but only as realistic as your script. Live tests are realistic and flaky. Keep the bulk scripted, a handful live, and run the live ones before releases or model changes.
- **In-process vs. gateway.** `AgentSession` skips authentication, the queue and artifact storage; the gateway kit includes them at ~0.5 s per test. Use the gateway for what only it covers.
- **Golden traces.** They catch unintended changes in event order, but break on intended ones. One per important flow is plenty.
- **Constructor injection.** `ReturnsAgent(today=...)` adds an `__init__` to the agent. Alternatives — monkeypatching `date.today`, or a freezing library — hide the dependency. Explicit is easier to read and needs no extra dependency.

## Exercises

1. Add level-1 tests for whitespace and lower-case order ids (`" a1003 "`).
2. Write a scripted test where the model first calls `check_return` with an unknown id, gets "not found", then asks the user for the id. Assert that the second prompt contains that observation.
3. Add a golden trace for the "order not found" flow.
4. Add a live test for `t03-react-loop` that asks about two orders and asserts both ids were looked up, in any order.

Next: [Lesson 7 — An agent with multiple tools](../t07_multiple_tools/README.md).
