# Lesson 18 — Structured outputs and typed data models

> **Outcome:** a ticket-triage agent with a typed input (validated by the gateway before anything runs) and a typed output (a pydantic `TicketTriage` the model must produce): invalid answers are retried once with the exact validation errors, then replaced by a safe fallback, and the result reaches clients as a JSON artifact plus a readable line.

## Learning objectives

- Define the agent's input with `input_schema` and know what the gateway does with it.
- Get structured data from a model with `with_structured_output(Model, method="function_calling")`, and know the prebuilt alternative (`create_agent(response_format=ToolStrategy(Model))`).
- Treat model output as untrusted input: validate it, retry with actionable feedback, fall back safely.
- Deliver structured results to clients as a JSON artifact (`aemit_artifact`).

## Prerequisites

- Lessons [4](../t04_custom_tool/README.md)–[5](../t05_tool_errors/README.md) (schemas, validation) and [13](../t13_state_and_history/README.md) (state).
- pydantic models, `Literal`, validators.

## Concepts

Free text is fine for people; programs need data. When an agent's result feeds another system — a ticket queue, a dashboard, a workflow branch (lesson 19) — define it as a **typed contract** and make the model fill it.

**Input.** `BaseAgent.input_schema` is a pydantic model. The gateway validates every run's input against it and answers `422` (with the field errors) before queuing anything, so the graph only ever sees valid input. It is required for `HEADLESS` and `BOTH` agents, and it is exported in the client contract (`vdreamer export-schema`).

**Output.** `with_structured_output(TicketTriage, method="function_calling")` sends the schema to the model as a tool and parses the model's tool call into a `TicketTriage` instance. `function_calling` works on every tool-capable model served by vLLM or Ollama; other methods depend on server features (the playground's README explains why). `create_agent(..., response_format=ToolStrategy(TicketTriage))` does the same inside a ReAct agent and puts the result in `state["structured_response"]`.

**Validation is the point.** The schema is a request, not a guarantee: models invent categories, misformat ids, or answer in prose. Pydantic turns each problem into a precise message ("category: Input should be 'order_status', …"). Feed that back once — models usually fix a pointed error — and if it still fails, return a conservative **fallback** (`UNCLASSIFIED`: needs a human) rather than failing the run or guessing.

**Delivery.** State keys are private (lesson 13). A JSON artifact gives clients the exact structure (with its full body in the `artifact.created` preview, up to 64 KB, and a download URL); a short `AIMessage` gives people something to read.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `input_schema`, the gateway's `422`, `export-schema` | vdreamer |
| `aemit_artifact(..., ArtifactKind.JSON, ...)`, `artifact.created` with preview, artifact download | vdreamer |
| `with_structured_output`, `create_agent(response_format=ToolStrategy(...))` | LangChain |
| `BaseModel`, `Field`, `Literal`, `field_validator`, `ValidationError` | pydantic |
| Retry with feedback, fallback | this lesson's code |

## How it works

```
POST …/runs {"input": {"subject": "", …}}  ──► 422 (input_schema), nothing runs
POST …/runs {"input": {"subject": "Charged twice", "body": "Order A1002 …"}}
  triage:  model ─► TicketTriage(category="shipping", …)  ✗ category: Input should be …
           model ◄─ "Your previous answer was invalid (category: …). Call TicketTriage again…"
           model ─► TicketTriage(category="billing", priority="urgent", order_ids=["A1002"], …) ✓
           (still invalid → UNCLASSIFIED + log warning)
  report:  artifact.created json "triage" {…}  +  message "billing / urgent — needs a human: …"
```

## Step by step

**1. The input contract.**

```python
# file: models.py
class TicketInput(BaseModel):
    """The run input: validated by the gateway (422 if invalid) before anything runs."""

    subject: str = Field(min_length=3, max_length=200)
    body: str = Field(min_length=1, max_length=5000)
    customer_email: str | None = None
```

**2. The output contract**, with enums, descriptions for the model, and a validator for what types cannot express:

```python
# file: models.py
class TicketTriage(BaseModel):
    """Triage of one customer-support ticket."""

    category: Literal["order_status", "returns", "billing", "product_question", "complaint",
                      "other"]
    priority: Literal["low", "normal", "high", "urgent"] = Field(
        description="urgent only for safety issues or money taken twice"
    )
```

**3. Declare the input schema on the agent** (`mode = Mode.BOTH` so both chat clients and systems can start it):

```python
# file: agent.py
class TicketTriageAgent(BaseAgent):
    name = "t18-ticket-triage"
    mode = Mode.BOTH  # chat clients and other systems (lesson 19) alike
    models = ("chat",)
    input_schema = TicketInput
```

**4. Ask for the structure, validate, retry once, fall back.**

```python
# file: agent.py
            for _attempt in range(2):  # one retry, with the error shown to the model
                try:
                    result = await classifier.ainvoke(prompt)
                except (ValidationError, OutputParserException) as exc:
                    result, problem = None, describe(exc)
                else:
                    problem = "no TicketTriage call"
                if isinstance(result, TicketTriage):
                    return {"triage": result.model_dump(), "validation_errors": errors}
```

`describe(exc)` lists one entry per invalid field — "1 validation error for TicketTriage" would not help the model.

**5. Deliver it.** The state must declare `messages` for the readable line to be kept (an undeclared key is silently dropped):

```python
# file: agent.py
        async def report(state: TriageState) -> dict[str, Any]:
            result = TicketTriage.model_validate(state["triage"])
            await vdreamer.aemit_artifact("triage", ArtifactKind.JSON, result.model_dump())
            flag = " — needs a human" if result.needs_human else ""
            line = f"{result.category} / {result.priority}{flag}: {result.summary}"
            return {"messages": [AIMessage(line)]}
```

## The complete example

- [`models.py`](models.py) — `TicketInput`, `TicketTriage`, `UNCLASSIFIED`.
- [`agent.py`](agent.py) — `describe`, `TriageState`, `TicketTriageAgent` (`t18-ticket-triage`: `triage → report`).

## Run it

From `tutorials/`:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t18-ticket-triage --models config/models.ollama.yaml
uv run vdreamer run t18-ticket-triage --once \
    --input '{"subject": "Charged twice", "body": "Order A1002 was charged twice!"}'
uv run vdreamer run t18-ticket-triage --once --input '{"subject": "", "body": "x"}'   # rejected
```

The artifact's bytes: `GET /api/v1/artifacts/<id>/download` (the path is printed by `vdreamer run`).

## Expected behavior

```
[artifact] json 'triage' /api/v1/artifacts/art_2666…/download
assistant>
billing / urgent — needs a human: Order A1002 was charged twice.
```

The empty subject never starts a run: the gateway answers `422` with error code `invalid_input` and `{"loc": ["subject"], "msg": "String should have at least 3 characters"}` in its details.

## Test it

[`test_t18_structured_output.py`](../../../../tests/tutorials/test_t18_structured_output.py) covers the contracts, the happy path (artifact + message), the retry — asserting the model was shown the precise field error — the fallback, the `create_agent` alternative, and the gateway's `422`:

```python
# file: tests/tutorials/test_t18_structured_output.py
    wrong = call("TicketTriage", category="shipping", priority="high", summary="x",
                 needs_human=False)
    model = scripted(wrong, structured(GOOD))
    result = await AgentSession(agent, models={"chat": model}).run(TICKET)

    assert result.values["triage"] == GOOD.model_dump()
    feedback = model.prompts[1][-1]
```

`structured(GOOD)` scripts what a model does under `function_calling`: one tool call named after the schema.

```bash
uv run pytest tests/tutorials/test_t18_structured_output.py -v
```

## Common mistakes and debugging

- **Trusting the structure blindly** — always handle `ValidationError`/`OutputParserException` and the "no tool call" case (`None`). Small models often answer in prose.
- **The default `with_structured_output` method on Ollama `-cloud` models** — they ignore the JSON-schema `format` field and answer in prose. Use `method="function_calling"`.
- **Retrying forever** — one or two retries with feedback, then a fallback. Unbounded retries turn a confused model into a cost spike.
- **Returning `messages` from a node when the state has no `messages` key** — silently dropped; the client sees nothing. Declare it.
- **Schemas too strict to be fillable** (a `summary` limit of 20 characters) — the model will fail validation forever. Strict where the data needs it, generous elsewhere.
- **Huge structured results in state** — use an artifact; keep state small.

## Design trade-offs

- **Structured output vs. tools.** Asking for a final structure is right when the result *is* the product (a triage, an extraction). When the model must act along the way, use tools and add `response_format` at the end.
- **Fallback vs. failure.** `UNCLASSIFIED` keeps the pipeline flowing and routes the ticket to a person. Failing the run is right when a wrong default would be harmful.
- **Literal enums vs. free strings.** Enums make downstream code simple and safe; adding a category becomes a code change. That is usually what you want for anything that drives routing.

## Exercises

1. Add `language: Literal["en", "de", "fr", "other"]` to `TicketTriage` and route German tickets differently in lesson 19's style.
2. Make the number of retries a constructor argument and test 0 and 2.
3. Add `sentiment: float = Field(ge=-1, le=1)` and test that `1.5` triggers a retry.
4. Generate the TypeScript type for `TicketInput` with `vdreamer export-schema --out schema` and find it in the output.

Next: Lesson 19 — Sequential and conditional workflows.
