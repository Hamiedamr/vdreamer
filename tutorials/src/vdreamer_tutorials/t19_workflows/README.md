# Lesson 19 — Sequential and conditional workflows

> **Outcome:** a headless ticket workflow that other systems trigger over HTTP with a service token and an idempotency key: a model classifies the ticket, code routes it down a fixed path (payment check, return-window check, or escalation to a person), a model drafts the reply from checked facts, and the resolution is published as a JSON artifact.

## Learning objectives

- Decide which steps need a model and which must be plain code, and build a **workflow** (a fixed graph) rather than an open-ended agent loop.
- Branch with `add_conditional_edges` on a structured classification.
- Ground model output in facts that code has checked, and let code overrule the model where it can check itself.
- Serve a `HEADLESS` agent: `/trigger`, service tokens, `Idempotency-Key`, and why chat routes are refused.
- Make every model call visible to clients exactly once.

## Prerequisites

- Lessons [18](../t18_structured_output/README.md) (structured output, `input_schema`, artifacts), [13](../t13_state_and_history/README.md) (state) and [3](../t03_react_loop/README.md) (conditional edges).

## Concepts

An **agent** decides its own steps (the ReAct loop). A **workflow** has steps you decided in advance, with models used *inside* some of them. Workflows are more predictable, cheaper and easier to test; agents are more flexible. Most production systems are workflows with small agentic parts.

Good workflow design asks, step by step, *does this need judgement?*

| Step | Judgement? | So |
| --- | --- | --- |
| What is this ticket about? | yes (free text) | model, structured output |
| Which order id is mentioned? | no (a pattern) | regex — and it overrules the model |
| Was the order charged twice? | no (a lookup) | code |
| Is it within 30 days? | no (arithmetic) | code |
| Should a model answer an angry complaint? | policy says no | escalate to a person |
| How do we phrase the reply? | yes | model, given only checked facts |

**Headless agents** are started by systems — a helpdesk, an ERP, a scheduler — not by people in a chat. `mode = Mode.HEADLESS` makes vdreamer refuse chat routes (`400 mode_not_supported`) and accept `POST /api/v1/agents/{name}/trigger`, authenticated with a **service token** (`VDREAMER_SERVICE_TOKENS=name:token`). The caller sends an **`Idempotency-Key`** (say, the ticket id): retries with the same key return the same run instead of starting a second one — essential, since the caller cannot tell a lost response from a failed request. A `callback` URL (hosts must be in `CALLBACK_ALLOWLIST`) is notified when the run completes, fails or waits for approval. Headless runs also change approvals (lesson 26): there is no user to ask, so every approval needs an `approver_role`.

**Every model call streams.** Whatever a model call produces is streamed to the client as `token.delta` events. If you then also return a separate `AIMessage` with the same text, clients see it twice. Put the model's own message into `messages` (it is then streamed and completed once), or call the model through a `nostream` copy (lesson 16) when its output is internal.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `Mode.HEADLESS`, `input_schema`, `/trigger`, service tokens, `Idempotency-Key`, callbacks | vdreamer |
| `emit_progress`, `aemit_artifact`, `bind_context` (tests) | vdreamer |
| `StateGraph`, `add_conditional_edges` | LangGraph |
| `with_structured_output(..., method="function_calling")` | LangChain |
| The routing, checks and escalation policy | this lesson's code |

## How it works

```
POST /api/v1/agents/t19-ticket-workflow/trigger   (service token, Idempotency-Key: ticket-T-123)
  classify ─► Classification(category="billing")   + regex: order_id = A1002
  route    ─► check_payment ─► "charges: [EUR 89.00 …, EUR 89.00 …]; DUPLICATE CHARGE"
  draft_reply ─► model, told only those facts ─► streamed reply
  finish   ─► artifact "resolution" {ticket_id, category, order_id, findings, escalated, reply}
```

## Step by step

**1. Typed input, headless mode.**

```python
# file: agent.py
class TicketWorkflowAgent(BaseAgent):
    name = "t19-ticket-workflow"
    mode = Mode.HEADLESS
    models = ("chat",)
    input_schema = TicketIn
```

**2. Classify with a model; let code fix what code can check.**

```python
# file: agent.py
            if not isinstance(result, Classification):
                result = Classification(category="other")
            # Code, not the model, has the last word on facts it can check itself.
            order_id = find_order_id(state["body"]) or result.order_id
            return {"category": result.category, "order_id": order_id}
```

**3. Deterministic steps are plain functions** (testable without any model):

```python
# file: agent.py
def check_payment(state: WorkflowState) -> dict[str, Any]:
    """Deterministic step: what the payment system says about the order."""
    vdreamer.emit_progress("checking payments", 40)
    charges = PAYMENTS.get(state.get("order_id") or "", [])
    duplicate = len(charges) != len(set(charges))
    note = f"charges: {charges or 'none found'}" + ("; DUPLICATE CHARGE" if duplicate else "")
    return {"findings": note}
```

**4. Route on the classification.**

```python
# file: agent.py
def route(state: WorkflowState) -> str:
    return {"billing": "check_payment", "returns": "check_return",
            "complaint": "escalate"}.get(state["category"], "draft_reply")
```

```python
# file: agent.py
        graph.add_conditional_edges(
            "classify", route, ["check_payment", "check_return", "escalate", "draft_reply"]
        )
```

**5. Draft from checked facts only, and stream the model's message once.**

```python
# file: agent.py
            facts = state.get("findings", "no system checks were needed")
            answer = await model.ainvoke([
                SystemMessage("Write a short, friendly reply to the customer. Use only these "
                              f"facts from our systems: {facts}."),
                HumanMessage(state["body"]),
            ])
            # The model's own message goes into `messages`: clients see it stream once.
            return {"reply": answer.text, "messages": [answer]}
```

## The complete example

[`agent.py`](agent.py) — `TicketIn`, `Classification`, the four step functions, `route`, `finish`, and `TicketWorkflowAgent` (`t19-ticket-workflow`).

## Run it

From `tutorials/`. Add a service token to `.env` (format `name:token`):

```bash
echo "VDREAMER_SERVICE_TOKENS=helpdesk:helpdesk-token-0123456789" >> .env
uv run vdreamer dev --registry vdreamer.yaml --agent t19-ticket-workflow --models config/models.ollama.yaml
```

Trigger it as the helpdesk system would — from the terminal client or with `curl`:

```bash
HELPDESK_TOKEN=helpdesk-token-0123456789 uv run vdreamer run t19-ticket-workflow \
    --trigger --service --token-env HELPDESK_TOKEN --idempotency-key ticket-T-123 \
    --input '{"ticket_id": "T-123", "body": "Was order A1002 charged twice?"}'

curl -X POST http://127.0.0.1:8000/api/v1/agents/t19-ticket-workflow/trigger \
     -H "Authorization: Bearer helpdesk-token-0123456789" -H "Idempotency-Key: ticket-T-123" \
     -H "Content-Type: application/json" \
     -d '{"input": {"ticket_id": "T-123", "body": "Was order A1002 charged twice?"}}'
```

## Expected behavior

```
[triggered run_f2d8… on thread_6374…, Idempotency-Key ticket-T-123]
[run running]
[progress] classifying 10%
[progress] checking payments 40%
[progress] drafting a reply 70%
assistant>
Sorry about that: order A1002 was charged twice, and we have started a refund of the duplicate EUR 89.00.

[artifact] json 'resolution' /api/v1/artifacts/art_a1db…/download
[progress] done 100%
[run completed]
```

The `curl` with the same `Idempotency-Key` returns the same `run_id` — no second run. Creating a chat thread for this agent returns `400 mode_not_supported`.

## Test it

[`test_t19_workflows.py`](../../../../tests/tutorials/test_t19_workflows.py) tests the steps as functions (binding a run context so `emit_progress` works outside a worker), every branch's path through its progress labels, grounding, the regex overruling the model, the fallback, and the trigger with idempotency through the real gateway:

```python
# file: tests/tutorials/test_t19_workflows.py
@pytest.mark.parametrize(
    ("category", "steps"),
    [
        ("billing", ["classifying", "checking payments", "drafting a reply", "done"]),
        ("returns", ["classifying", "checking the return window", "drafting a reply", "done"]),
        ("other", ["classifying", "drafting a reply", "done"]),
        ("complaint", ["classifying", "escalated to the support team", "done"]),
    ],
)
```

```bash
uv run pytest tests/tutorials/test_t19_workflows.py -v
```

## Common mistakes and debugging

- **Letting the model do arithmetic or lookups** — it will sometimes be wrong, confidently. Compute in code and give the model the result.
- **The reply appears twice** — a model call streamed and a separate `AIMessage` repeated it. Return the model's message itself (this lesson), or use a `nostream` model for internal calls.
- **`400 mode_not_supported`** — you started a headless agent through the chat routes. Use `/trigger` (`vdreamer run --trigger`).
- **`401` on `/trigger`** — service tokens are separate from the internal token: `VDREAMER_SERVICE_TOKENS=name:token`, and the client sends that token (`--service --token-env …`).
- **Duplicate runs on retries** — no `Idempotency-Key`. Use a stable business key (the ticket id), not a random one.
- **`RuntimeError: no active vdreamer run context`** in a unit test of a step that emits progress — bind one with `vdreamer.bind_context(VDreamerContext(...))`.

## Design trade-offs

- **Workflow vs. agent.** The fixed graph cannot handle a ticket type nobody planned for — it falls into `other`. An agent could improvise; it could also improvise badly. Use workflows where the process is known and audited; use agents for open-ended help, and combine them (lesson 22).
- **Escalation policy in code.** Hard-coding "complaints go to a person" is a business decision made visible and testable — better than a prompt instruction a model might ignore.
- **Classification granularity.** Few categories classify reliably; many categories route precisely. Start coarse, add branches when data shows the need.

## Exercises

1. Add a `delivery` category with a `track_parcel` step (lesson 10's carrier) and its test.
2. Escalate billing tickets with a duplicate charge above EUR 500 instead of drafting a reply.
3. Trigger with a `callback` URL on a host listed in `CALLBACK_ALLOWLIST`, and receive the HMAC-signed notification with a small HTTP server (see the root README's "Headless agents" section).
4. Make the agent `Mode.BOTH` and start it from a chat with `vdreamer run t19-ticket-workflow --input …` — what changes for approvals (lesson 26)?

Next: Lesson 20 — Planning and task decomposition.
