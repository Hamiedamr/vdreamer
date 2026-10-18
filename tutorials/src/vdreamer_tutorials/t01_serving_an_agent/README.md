# Lesson 1 — How vdreamer serves an agent

> **Outcome:** a one-node agent served by vdreamer: you start it with `vdreamer dev`, talk to it with `vdreamer run` and `curl`, and read the events a client receives. No model server needed.

## Learning objectives

- Say what an AI agent is and which part of it vdreamer is responsible for.
- Write the smallest valid `BaseAgent`: a `name`, a `mode`, and a `build(ctx)` that returns a LangGraph graph.
- Run an agent three ways: a one-shot `vdreamer dev --input`, the dev server with `vdreamer run`, and plain HTTP.
- Recognise the protocol events (`run.status`, `progress`, `log`, `token.delta`, `message.completed`) and see that a *thread* keeps history between runs.
- Find your way around the tutorial series: where lessons live, how they are run and tested.

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/), the repository cloned, and `uv sync --all-extras` run once at the repository root.
- Comfortable with Python type hints and `TypedDict`. No prior LangGraph or agent experience is assumed.

## Concepts

**An AI agent** is a program that uses a language model to decide what to do next. The model reads the conversation, either answers or asks for a *tool* to be called (a function you wrote), sees the tool's result, and repeats until it can answer. That loop — reason, act, observe — is called **ReAct** and is the subject of lessons 2 and 3.

Writing the loop is the easy part. Running it for real users needs much more: an HTTP API, streaming output, conversations that survive restarts, concurrency, retries after a crash, tool calls that cannot leak secrets, pausing for a human's approval, and a contract a frontend team can build against. **vdreamer is that serving layer.** You write the agent as a LangGraph graph inside a small `BaseAgent` class; vdreamer turns it into a REST/SSE API without generated server code.

This lesson's agent has no model at all. That is deliberate: it isolates the serving contract, so everything you see here — threads, runs, events — is exactly what every later, model-driven lesson also gets.

Three vocabulary words used throughout the series:

- **Thread** — one conversation. It owns the agent's state (here: the message list), stored by a LangGraph *checkpointer*.
- **Run** — one execution of the graph on a thread, from `queued` to `completed`, `failed` or `awaiting_approval`.
- **Event** — what clients receive while a run executes, over Server-Sent Events. Your graph's state is private; only events reach clients.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `BaseAgent`, `BuildContext`, `Mode`, `emit_progress`, `emit_log` | vdreamer |
| `vdreamer dev`, `vdreamer run`, `vdreamer validate`, the REST/SSE API, the event protocol | vdreamer |
| `StateGraph`, `START`/`END`, `add_messages`, checkpointers | LangGraph |
| `AIMessage`, `HumanMessage` | LangChain (`langchain_core`) |

## How it works

```
 vdreamer run / curl / your app
        │  POST /api/v1/threads/{id}/runs        GET …/events (SSE)
        ▼                                              ▲
 ┌──────────────┐  queue row   ┌───────────────┐       │ events
 │   gateway    │ ───────────► │    worker     │ ──────┘
 │ (REST + SSE) │              │ build() once, │
 └──────────────┘              │ run the graph │──► checkpointer (thread state)
                               └───────────────┘
```

1. A client creates a thread and starts a run with an input: `{"messages": [{"role": "user", "content": "hello"}]}`.
2. The gateway validates the request and queues the run. A worker claims it.
3. The worker calls your agent's `build(ctx)` **once** (then caches the graph), compiles it with its checkpointer, loads the thread's saved state and runs the graph with the new input.
4. While the graph runs, the worker turns what happens into protocol events: `emit_progress`/`emit_log` calls, and every `AIMessage` the graph produces (as `token.delta` chunks, then `message.completed`).
5. The run ends with a final `run.status`. The thread's state is checkpointed, so the next run on the same thread continues from it.

`vdreamer dev` runs the gateway and one worker in a single process with SQLite (`./.vdreamer/dev.db`). In production, `vdreamer serve` and `vdreamer worker` are separate processes that share Postgres — your agent code is identical.

## Step by step

**1. Describe the state.** A graph's state is a `TypedDict`. `add_messages` is a *reducer*: when a node returns `{"messages": [...]}`, the new messages are appended rather than replacing the list.

```python
# file: agent.py
class State(TypedDict):
    # add_messages appends new messages to the list instead of replacing it.
    messages: Annotated[list[AnyMessage], add_messages]
```

**2. Write a node.** A node is a function from the state to a *partial update*. This one reads the latest user message, reports progress, and answers with an `AIMessage` — which is what makes the answer visible to clients.

```python
# file: agent.py
def respond(state: State) -> dict[str, Any]:
    """Echo the latest user message and count the turns in this thread."""
    vdreamer.emit_progress("reading your message", 50)
    question = state["messages"][-1].text
    turn = sum(isinstance(message, HumanMessage) for message in state["messages"])
    vdreamer.emit_log(f"turn {turn}: received {len(question)} characters")
    return {"messages": [AIMessage(f"You said: {question} (message {turn} in this thread)")]}
```

`.text` gives a message's content as a plain string; prefer it over `.content` (lesson 2 explains why).

**3. Wrap the graph in a `BaseAgent`.** `name` is the agent's name in the API (lowercase letters, digits and `-`). `build()` returns the `StateGraph` *uncompiled*: vdreamer compiles it with its own durable checkpointer.

```python
# file: agent.py
class EchoAgent(BaseAgent):
    name = "t01-echo"
    mode = Mode.CHAT

    def build(self, ctx: BuildContext) -> StateGraph:
        graph = StateGraph(State)
        graph.add_node("respond", respond)
        graph.add_edge(START, "respond")
        graph.add_edge("respond", END)
        return graph  # uncompiled: vdreamer compiles it with its checkpointer


agent = EchoAgent()
```

**4. Register it.** `tutorials/vdreamer.yaml` maps short names to `module:attribute` import specs, so commands can say `--agent t01-echo`:

```yaml
agents:
  t01-echo: vdreamer_tutorials.t01_serving_an_agent.agent:agent
```

## The complete example

[`agent.py`](agent.py) is the whole agent — the three snippets above plus imports. The tutorial series is organised the same way for every lesson:

```
tutorials/
  README.md                 the index and recommended order
  vdreamer.yaml             registry: every lesson's agent by name
  config/                   models.ollama.yaml, models.vllm.yaml
  .env.example              copy to .env (the API token)
  src/vdreamer_tutorials/
    tNN_<topic>/            one lesson: README.md + agent code
    testing.py              shared test support (lesson 6)
tests/tutorials/
  test_tNN_<topic>.py       each lesson's tests
```

## Run it

From the repository root, once:

```bash
uv sync --all-extras
cd tutorials
cp .env.example .env        # sets VDREAMER_INTERNAL_TOKEN for every vdreamer command
```

Check the agent against the developer contract:

```bash
uv run vdreamer validate --registry vdreamer.yaml t01-echo
```

Run it once, without a server — vdreamer validates the input, drives one run, prints every event and exits:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t01-echo \
    --input '{"messages": [{"role": "user", "content": "hello"}]}'
```

Serve it, then talk to it from a second terminal (also in `tutorials/`):

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t01-echo
uv run vdreamer run t01-echo --message "hello"
```

`vdreamer run` keeps prompting (`you>`) on the same thread; Ctrl+C quits. Or use plain HTTP (token from your `.env`), or the Swagger UI at http://127.0.0.1:8000/api/v1/docs:

```bash
H=(-H "Authorization: Bearer tutorial-token-change-me-0123456789" -H "X-User-Id: alice" -H "Content-Type: application/json")
curl "${H[@]}" -X POST http://127.0.0.1:8000/api/v1/agents/t01-echo/threads
curl "${H[@]}" -X POST http://127.0.0.1:8000/api/v1/threads/<thread_id>/runs \
     -d '{"input": {"messages": [{"role": "user", "content": "hello"}]}}'
curl -N "${H[@]}" http://127.0.0.1:8000/api/v1/threads/<thread_id>/events
```

## Expected behavior

`vdreamer validate` ends with `validate EchoAgent: PASS`. The one-shot run prints:

```
[run running]
[progress] reading your message 50%
[log info] turn 1: received 5 characters
assistant>
You said: hello (message 1 in this thread)

[run completed]
vdreamer dev: run run_… -> completed (attempt 1, 6 events)
```

The SSE stream carries the same six events as JSON envelopes (`id:` is `run_id:seq`, so a client can reconnect with `Last-Event-ID` and miss nothing):

```
id: run_56bd…:2
data: {"v":"1.0","type":"progress","run_id":"run_56bd…","thread_id":"thread_288c…","agent":"t01-echo","seq":2,"ts":"…","ns":[],"payload":{"label":"reading your message","pct":50.0}}
```

A second message on the same thread answers `(message 2 in this thread)`: the checkpointer restored the earlier messages. A new thread starts again at 1. `GET /api/v1/threads/<thread_id>` returns the thread's runs and its current state `values`.

## Test it

[`tests/tutorials/test_t01_serving_an_agent.py`](../../../../tests/tutorials/test_t01_serving_an_agent.py) checks the agent three ways: in-process, through a real gateway with `vdreamer run`, and through the `vdreamer dev --input` command line. The in-process tests use `AgentSession` from `vdreamer_tutorials.testing` — about a hundred lines over vdreamer's public `AgentRunner`, explained in lesson 6. For now, read it as "run the agent like a worker would and collect the events":

```python
# file: tests/tutorials/test_t01_serving_an_agent.py
async def test_the_agent_answers_with_progress_and_a_log() -> None:
    result = await AgentSession(agent).run("hello")

    assert result.status == "completed"
    assert result.answer == "You said: hello (message 1 in this thread)"
    assert result.types == [
        "run.status", "progress", "log", "token.delta", "message.completed", "run.status",
    ]
```

```bash
uv run pytest tests/tutorials/test_t01_serving_an_agent.py -v      # from the repository root
```

## Common mistakes and debugging

- **`agent spec … could not be imported`** — run commands from `tutorials/` (the registry path is relative), or pass the full spec: `--agent vdreamer_tutorials.t01_serving_an_agent.agent:agent`.
- **`401` from the API or `vdreamer run`** — the token is missing or differs between the server and the client. Both read `.env` from their working directory; start both in `tutorials/`.
- **The answer is never shown to the client** — only messages, progress, logs, artifacts, citations, tool calls and approvals become events. A value you store in another state key (say `state["answer"]`) is private. Return an `AIMessage`.
- **`build()` returned `graph.compile()`** — `vdreamer validate` fails with `checkpointer: build() compiled the graph without a checkpointer`, and `dev`/`worker`/`serve` refuse to start the agent: without vdreamer's checkpointer, threads would forget everything and crashed runs could not resume. Return the uncompiled `StateGraph`, or compile with `checkpointer=ctx.checkpointer`.
- **Calling `emit_progress` outside a run** (in a script or a test without a session) raises `RuntimeError: no active vdreamer run context`. The helpers work inside runs only.
- **Port 8000 in use** — `vdreamer dev --port 8010`, then `vdreamer run --url http://127.0.0.1:8010 …`.

## Design trade-offs

- **A graph, not a callback.** vdreamer asks for a LangGraph graph rather than a plain function so it can checkpoint after every step, resume after a crash, pause for approvals and stream intermediate output. The price is learning LangGraph's state model, which the next lessons do gradually.
- **`build()` runs once per worker.** Expensive setup (clients, models, compiled graphs) is paid once, but `build()` must not hold per-run or per-user data: one compiled graph serves every thread.
- **Dev vs. production.** `vdreamer dev` keeps everything in one SQLite file and one process — ideal for learning, not for load. The same agent runs unchanged under `vdreamer serve` + `vdreamer worker` with Postgres.

## Exercises

1. Make the agent answer in upper case when the message ends with `!`. Add a test for it.
2. Emit a second progress event at 100% just before returning, and watch both in `vdreamer run`.
3. Add a node `count_words` before `respond` that stores the word count in a new state key, and use it in the answer. Confirm with `GET /api/v1/threads/<id>` that the key is saved in the thread's `values` but never appears as an event.

Next: [Lesson 2 — A minimal ReAct agent](../t02_minimal_react_agent/README.md).
