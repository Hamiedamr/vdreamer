# Lesson 5 — Tool inputs, outputs, errors and validation

> **Outcome:** three order-desk tools hardened for production: arguments validated by a pydantic schema, explainable errors returned to the model, unexpected failures contained with their secrets masked, a hanging call cut off by a timeout, and an oversized result truncated and kept as an artifact — each behavior pinned by a test.

## Learning objectives

- Validate tool arguments with a pydantic `args_schema` (patterns, ranges, descriptions) and know what happens when the model sends something invalid.
- Choose between returning a message, raising `ToolException`, and letting an exception escape — and know what the model and the client see in each case.
- Use vdreamer's per-tool policy: `timeout_s`, `redact`, `max_result_chars`, `fail_run`.
- Explain the five error codes of a governed tool — `invalid_args`, `tool_error`, `timeout`, `denied`, `transport` — and what `fail_run` changes.
- Prove with a test why tool calls must go through governance.

## Prerequisites

- [Lesson 4](../t04_custom_tool/README.md): tool schemas and `@vdreamer.tool`.
- [Lesson 3](../t03_react_loop/README.md): `ToolMessage`s and how the model observes results.
- Basic pydantic (`BaseModel`, `Field`).

## Concepts

Tools are where an agent touches the real world, so they are where most failures happen. A model sends malformed arguments; a record does not exist; a database is down; an external API hangs; a query returns ten megabytes. The question for each is: **who needs to know, and what should happen next?**

vdreamer runs every governed tool call through one pipeline (the *governance pipeline*):

1. **Authorize** — roles and scopes (lesson 28); a refusal is `denied`.
2. **Emit `tool.call`** — the arguments the model sent, with sensitive ones masked.
3. **Approve** — an optional human gate (lesson 27).
4. **Validate and execute** — arguments checked against the tool's schema (`invalid_args`), then the tool runs under `timeout_s` (`timeout`).
5. **Normalize** — results longer than `max_result_chars` are stored as an **artifact**; the model gets a truncated body plus the artifact id.
6. **Contain** — any failure becomes a `ToolMessage(status="error")` whose text starts with `Error (<code>):`. **The run continues**, so the model can explain or retry. With `fail_run=True` the run fails instead.
7. **Emit `tool.result`** — status, error code, summary, artifact ids, duration.

Error messages are written for the model *and* shown to clients, so the pipeline masks secrets in them: the values of redacted arguments, and anything that looks like `password=…`, `token=…`, `secret=…`, `api_key=…` or `authorization: …`. Arguments with those names are always masked in `tool.call`; `redact=[...]` adds your own (here, `email`).

Three ways for a tool to report a problem:

| The tool… | The model sees | Use it when |
| --- | --- | --- |
| returns a normal result ("No order found") | an ordinary observation | "nothing found" is a normal answer |
| raises `ToolException("…")` | `Error (tool_error): …` | the request cannot be served and the message helps the model explain or retry |
| raises anything else | `Error (tool_error): …` (masked) | a bug or outage — the message is not written for anyone, so keep it free of secrets anyway |

## What comes from where

| Piece | Provided by |
| --- | --- |
| `args_schema`, `ToolException` | LangChain |
| `Field(pattern=…, description=…)`, validation | pydantic |
| `timeout_s`, `redact`, `max_result_chars`, `fail_run`, `Policy` | vdreamer (`@vdreamer.tool`, agent `tool_policies`) |
| Containment into `Error (<code>): …`, masking, overflow artifacts, `tool.result` | vdreamer governance pipeline |
| Artifact storage (`vdreamer dev`: next to the `.db`; production: MinIO) | vdreamer |

## How it works

```
model: get_order(order_id="1001", email="ada")
   │
   ├─ tool.call  {"order_id": "1001", "email": "***"}          ← redacted
   ├─ validate against OrderLookup ✗  → ToolMessage("Error (invalid_args): invalid arguments — order_id: …")
   │                                     (the function never ran)
   └─ tool.result {status: "error", error: {code: "invalid_args", …}}
model: "The order id should look like A1001 and I need your full email address…"
run.status completed
```

An oversized result follows the success path with one extra step:

```
order_history(...) → 1,370 characters > max_result_chars=400
   ├─ artifact.created  file "order_history result" (the full text)
   └─ ToolMessage = first 400 characters + "[vdreamer: result truncated to 400 of 1370 characters; the full result is artifact art_…]"
```

## Step by step

**1. Validate inputs with a schema.** The model's arguments are checked before your function runs. Field descriptions also land in the tool's schema, so they guide the model too.

```python
# file: agent.py
class OrderLookup(BaseModel):
    """The arguments of ``get_order``, validated before the tool runs."""

    order_id: str = Field(
        pattern=r"^A\d{4}$", description='The order id: "A" and four digits, e.g. "A1001".'
    )
    email: str = Field(
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        description="The email address the order was placed with.",
    )
```

**2. Separate expected failures from unexpected ones.** `ToolException` carries a message meant for the model. Anything else is a bug or an outage; it is contained too, and its secrets are masked.

```python
# file: agent.py
@vdreamer.tool(args_schema=OrderLookup, redact=["email"], timeout_s=5)
def get_order(order_id: str, email: str) -> dict[str, Any]:
    """Look up one order. The customer must give the order id and their email address."""
    if not ORDERS_DB_ONLINE:
        # An unexpected failure: governance contains it and masks the secret.
        raise RuntimeError("orders-db login failed: password=hunter2")
    order = ORDERS.get(order_id)
    if order is None or order["email"] != email.lower():
        # An expected, explainable failure: the message is for the model.
        raise ToolException(f"No order {order_id} was found for that email address.")
    return {"order_id": order_id, "status": order["status"], "tracking_id": order["tracking_id"]}
```

The not-found message does not reveal whether the order exists under another email: error messages are output, and output can leak.

**3. Bound slow calls.** `timeout_s` cancels the call; the model receives `Error (timeout): …`. Use `async def` for I/O-bound tools so a cancelled call really stops.

```python
# file: agent.py
@vdreamer.tool(timeout_s=1.0)
async def track_parcel(tracking_id: str) -> str:
    """Ask the carrier where a parcel is, by tracking id (from get_order)."""
    if tracking_id in SLOW_PARCELS:
        await asyncio.sleep(30)  # the carrier never answers in time
    return PARCELS.get(tracking_id, f"The carrier does not know {tracking_id}.")
```

**4. Bound large outputs.** Everything a tool returns is sent to the model on every later call in the turn. `max_result_chars` caps it; the full result is stored as an artifact the client can download.

```python
# file: agent.py
@vdreamer.tool(max_result_chars=400, redact=["email"])
def order_history(email: str) -> str:
```

**5. Tell the model what to do with errors.** Containment only helps if the model knows how to react:

```python
# file: agent.py
SYSTEM_PROMPT = (
    "You are an order-desk assistant. To look up an order you need its id and the "
    "customer's email address: ask for whatever is missing. If a tool reports an error, "
    "explain it plainly and suggest what the customer can do; never make up data."
)
```

## The complete example

[`agent.py`](agent.py) — `get_order`, `track_parcel`, `order_history`, and `OrderDeskAgent` (`t05-order-desk`, a `create_agent` agent). Two module flags let tests simulate trouble: `ORDERS_DB_ONLINE = False` (an outage) and the tracking id `TRK-SLOW` (a hanging carrier).

## Run it

From `tutorials/`:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t05-order-desk --models config/models.ollama.yaml
uv run vdreamer run t05-order-desk     # then try the messages below at the you> prompt
```

- `Where is order A1001? My email is ada@example.com` — a normal lookup, then perhaps `track_parcel`.
- `Order 1001, email ada` — invalid arguments; the model should ask you to check both.
- `Order A1003, email ada@example.com` — the email does not match: a `tool_error` the model explains.
- `Show my order history, ada@example.com` — a truncated result plus an artifact link.

## Expected behavior

Invalid arguments (exact tool lines; model wording varies):

```
-> get_order({"order_id": "1001", "email": "***"}) [custom]
<- error Error (invalid_args): invalid arguments — order_id: String should match pattern '^A\d{4}$'; email: *** should match pattern '^[^@\s]+@[^@\s]+\.[^@\s]+$' (0 ms)
assistant>
The order id should look like A1001 and I need your full email address. Could you check both?
```

Notice `email: *** should match pattern`: pydantic wrote `email: String should match…`, and the masking rule hid the text after `email:` — it errs on the side of hiding. The email address itself never appears in any event.

An oversized result:

```
-> order_history({"email": "***"}) [custom]
[artifact] file 'order_history result' /api/v1/artifacts/art_fe7a…/download
<- ok 2026-09-30: order event #30 for this customer 2026-09-29: … (2 ms) artifacts: art_fe7a…
```

## Test it

[`test_t05_tool_errors.py`](../../../../tests/tutorials/test_t05_tool_errors.py) runs one scripted tool call per failure mode and asserts on the event *and* on the `ToolMessage` the model received. The most instructive test runs the same failing tool twice from a one-node graph — once raw, once governed:

```python
# file: tests/tutorials/test_t05_tool_errors.py
    monkeypatch.setattr(lesson, "ORDERS_DB_ONLINE", False)
    raw = await AgentSession(_direct_caller(governed=False)).run({})
    governed = await AgentSession(_direct_caller(governed=True)).run({})

    assert raw.status == "failed"
    assert "password=hunter2" in raw.payloads("run.status")[-1]["error"]["message"]
    assert governed.status == "completed"
    assert governed.values["result"] == "Error (tool_error): orders-db login failed: password=***"
```

Ungoverned, the outage fails the whole run and sends the database password to the client. Governed, the run continues and the secret is masked.

```bash
uv run pytest tests/tutorials/test_t05_tool_errors.py -v
```

## Common mistakes and debugging

- **Catching every exception inside the tool and returning `"error"`** — the model gets no useful explanation, the client sees `status: ok`, and your metrics count a failure as a success. Raise `ToolException` with a clear message instead.
- **Secrets in error messages** — masking catches `password=…`-style text and the values of redacted arguments, not every format (a DSN like `postgresql://user:pw@host` is *not* recognised). Do not put credentials in exception messages in the first place.
- **A sync tool that blocks past its timeout** — the call is abandoned and the model moves on, but a blocked thread cannot be killed and keeps running. Make I/O-bound tools `async`, and give their clients their own timeouts too.
- **`max_result_chars` too small** — the model works from a truncated view and may answer wrongly. Size it to what the model needs; return summaries rather than raw dumps (lesson 8).
- **`args_schema` and the function signature disagree** (a field missing from one of them) — validation passes, then the call itself fails: `tool_error: get_order() got an unexpected keyword argument …`. Keep the names identical.
- **Expecting the artifact bytes in a unit test** — `AgentSession` announces the artifact (`artifact.created`) but stores nothing; only a real gateway stores it (see the gateway test).

## Design trade-offs

- **Contain vs. fail the run.** Containment (the default) lets the model recover and the conversation continue; it is right for most tool failures. `fail_run=True` is right when continuing would be unsafe or meaningless (a failed payment capture in the middle of a checkout flow).
- **Strict schemas.** They reject garbage early and document the contract, but each rejection costs a model round trip. Keep descriptions precise so the model gets it right the first time.
- **Timeouts.** Too short and slow-but-healthy calls fail; too long and the user waits. Base them on the dependency's real latency (p99) and leave the model time to answer.
- **Truncation vs. pagination.** Truncation plus an artifact is a safety net. A tool designed for agents should return the few items that matter, or accept `limit`/`offset` arguments.

## Exercises

1. Add a `quantity: int` argument with `Field(gt=0, le=10)` to a new `reorder` tool and test both bounds.
2. Make `track_parcel` raise `ConnectionError` for a tracking id `TRK-DOWN`, and check which error code the client sees (hint: `transport`).
3. Lower `max_result_chars` to 100 and look at what the model receives. Then change `order_history` to accept `limit: int = 5` instead.
4. Give `get_order` `fail_run=True` with `@vdreamer.tool(..., fail_run=True)` and compare the run's final status with the agent-level override used in the test.

Next: [Lesson 6 — Testing agents and reading execution traces](../t06_testing_and_traces/README.md).
