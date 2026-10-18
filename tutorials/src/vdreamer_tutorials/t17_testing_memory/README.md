# Lesson 17 — Testing memory persistence, retrieval and isolation

> **Outcome:** a memory test suite you can copy: retrieval in order, an isolation matrix (same thread, same user, other user), concurrent users, scoped deletion, persistence across a reopened database — and a real `vdreamer dev` server stopped and restarted between conversations.

## Learning objectives

- List the properties a memory system must guarantee, and write one test per property.
- Make an agent's memory behavior deterministic enough to test exactly (here: no model at all).
- Test persistence with the same durable backends vdreamer uses (`AsyncSqliteSaver`, `AsyncSqliteStore`) and across real process restarts.
- Test isolation at both levels: your namespacing (store) and vdreamer's thread ownership (gateway).

## Prerequisites

- Lessons [13](../t13_state_and_history/README.md)–[15](../t15_long_term_memory/README.md) (thread state, working memory, the store) and [6](../t06_testing_and_traces/README.md) (test levels).

## Concepts

Memory bugs are quiet and serious: a customer sees another customer's data, a preference silently disappears after a deploy, a "deleted" note keeps coming back. They rarely show up in a single-conversation demo. Write down the guarantees and test each:

| Property | Question | Test |
| --- | --- | --- |
| Retrieval | does what was stored come back, complete and in order? | store three, recall three |
| Thread isolation | does thread memory stay in its thread? | same user, other thread: no session notes |
| User isolation | does user memory stay with its user? | other user: nothing at all |
| Concurrency | do simultaneous users interfere? | eight users write at once, each reads only their own |
| Deletion | is "forget" complete and scoped? | delete one user's notes, another's remain |
| Durability | does memory survive restarts? | reopen the database; restart `vdreamer dev` |
| Access control | can a user reach another user's thread? | the gateway answers 404 |

Two choices make these tests trustworthy:

- **A deterministic subject.** This lesson's agent parses commands instead of calling a model, so every answer is exact. For a model-driven agent, test the memory layer the same way through scripted models (lessons 14–16) and keep live tests for judgement.
- **Real backends where it matters.** In-memory stores prove logic; durability can only be proven with the backends vdreamer actually uses — the same SQLite file layout as `vdreamer dev` and, ideally, a real process restart.

## What comes from where

| Piece | Provided by |
| --- | --- |
| Thread ownership (another user's thread is `404 thread_not_found`) | vdreamer gateway |
| `vdreamer dev`'s single SQLite file for runs, checkpoints and store | vdreamer |
| `AsyncSqliteSaver`, `AsyncSqliteStore`, `InMemoryStore`, `Runtime.store` in nodes | LangGraph |
| The memory desk agent and the test suite | this lesson |

## How it works

```
"remember for now: blue tent"  ─► state.session_notes += ["blue tent"]        (thread t-1, checkpointed)
"remember forever: size 44"    ─► store.aput(("memory-desk", "ada"), …)       (user ada, any thread)
"recall"                       ─► "This conversation: ['blue tent']. Always: ['size 44']."

restart test:  vdreamer dev --db dev.db  ─ run, run ─ SIGINT ─ vdreamer dev --db dev.db ─ recall ×3
```

## Step by step

**1. A deterministic agent with both memories.** The node reads the store through LangGraph's `Runtime` (vdreamer compiled the graph with `ctx.store`) and the user from the run context:

```python
# file: agent.py
async def handle(state: MemoryState, runtime: Runtime[Any]) -> dict[str, Any]:
    text = state["messages"][-1].text.strip()
    command = text.lower()
    user_id = vdreamer.current_context().user_id
    store = runtime.store
```

**2. An isolation matrix**, one parametrized test instead of three similar ones:

```python
# file: tests/tutorials/test_t17_testing_memory.py
@pytest.mark.parametrize(
    ("user", "thread", "expected"),
    [
        ("ada", "t-1", "This conversation: ['blue tent']. Always: ['size 44']."),  # same both
        ("ada", "t-2", "This conversation: nothing. Always: ['size 44']."),  # same user
        ("lin", "t-3", "This conversation: nothing. Always: nothing."),  # another user
    ],
)
```

**3. Concurrency** — runs on different threads execute concurrently in a worker; so should the test:

```python
# file: tests/tutorials/test_t17_testing_memory.py
    await asyncio.gather(*(
        session.run(f"remember forever: note of {u}", user_id=u, thread_id=f"t-{u}")
        for u in users
    ))
```

**4. Durability with the real backends**: write with one saver/store pair, close it, open a new pair on the same file, read.

**5. A real restart.** `dev_server(db)` starts `vdreamer dev` as a subprocess, waits for `/api/v1/healthz`, and stops it with SIGINT (Ctrl+C):

```python
# file: tests/tutorials/test_t17_testing_memory.py
    with dev_server(db) as url:
        first = vdreamer_run(url, monkeypatch, agent="t17-memory-desk", user="ada", once=True,
                             message="remember for now: blue tent", color=False)
        vdreamer_run(url, monkeypatch, agent="t17-memory-desk", user="ada", once=True,
                     thread=first.thread_id(), message="remember forever: size 44", color=False)
    # The process is gone; only ./dev.db remains.
```

**6. Access control at the gateway**: another user's thread is not just unreadable but invisible — `404`, never `403`, so its existence does not leak.

## The complete example

- [`agent.py`](agent.py) — `MemoryDeskAgent` (`t17-memory-desk`), no model.
- [`test_t17_testing_memory.py`](../../../../tests/tutorials/test_t17_testing_memory.py) — the suite.

## Run it

```bash
uv run pytest tests/tutorials/test_t17_testing_memory.py -v        # from the repository root
```

Or by hand, from `tutorials/` (no model needed):

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t17-memory-desk
uv run vdreamer run t17-memory-desk --user ada --message "remember forever: size 44" --once
# Ctrl+C the server, start it again, then:
uv run vdreamer run t17-memory-desk --user ada --message "recall" --once
uv run vdreamer run t17-memory-desk --user lin --message "recall" --once
```

## Expected behavior

All ten tests pass in a few seconds (the restart test starts two server processes). By hand, after the restart, Ada gets `This conversation: nothing. Always: ['size 44'].` and Lin gets `This conversation: nothing. Always: nothing.`

## Test it

This lesson is its own test. The property most often forgotten is deletion *scope*:

```python
# file: tests/tutorials/test_t17_testing_memory.py
    assert (await session.run("forget forever", user_id="ada", thread_id="1")).answer == (
        "Forgot 2 notes."
    )
    assert await store.asearch(("memory-desk", "ada")) == []
    assert len(await store.asearch(("memory-desk", "lin"))) == 1
```

## Common mistakes and debugging

- **Testing only with `InMemoryStore`/`InMemorySaver`** — proves logic, not durability. Add one test on the real file-backed backends.
- **Asserting on counts only** ("two notes") — a wrong user's two notes pass. Assert on content.
- **Shared fixtures across tests** — a module-level store leaks notes from one test into the next. Create stores per test.
- **A restart test that never actually restarts** — make sure the first process is gone (`wait()` after SIGINT) before starting the second.
- **Port clashes in CI** — pick a free port per server (`_free_port()`), never 8000.
- **`RuntimeError: vdreamer dev did not start`** — the helper prints the process output; usually a missing `VDREAMER_INTERNAL_TOKEN` or an import error.

## Design trade-offs

- **Deterministic subject vs. the real agent.** A command-parsing agent makes memory tests exact but is not your production agent. Keep memory logic in testable functions and nodes (lessons 11, 13) so the same code runs in both.
- **Process-level tests are slower** (~3 s here). Keep one or two for durability; do the rest in-process.
- **Postgres.** Production uses Postgres (`vdreamer serve`/`worker`). The repository's live-Postgres tests run when `VDREAMER_TEST_PG_DSN` is set; the same suite can be pointed at `AsyncPostgresSaver`/`AsyncPostgresStore`.

## Exercises

1. Add a property test: for random sequences of remember/forget commands, `recall` always matches a simple Python model of the expected memory (hint: `hypothesis`, or a seeded `random`).
2. Add a "retention" rule — notes older than N days are not recalled — with an injectable clock (lesson 6) and tests.
3. Point the durability test at Postgres (`langgraph-checkpoint-postgres` is already a vdreamer dependency) and skip it unless `VDREAMER_TEST_PG_DSN` is set.

Next: Lesson 18 — Structured outputs and typed data models.
