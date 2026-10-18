# Lesson 27 — Approving, rejecting and editing proposed actions

> **Outcome:** two agents whose actions a person can approve, reject, **edit** or send back with **feedback**: a goodwill-credit proposal reviewed through vdreamer's `ApprovalGate` (with a validated edit schema and a re-plan loop), and a ReAct agent whose tool calls are reviewed before they run — by LangChain's `HumanInTheLoopMiddleware` and by a vdreamer tool policy.

## Learning objectives

- Use `vdreamer.ApprovalGate` to build the full review loop: `prepare → gate → {execute, on_reject, replan}`.
- Let approvers change a proposal safely with an `edit_schema`, validated by the gateway (`422 invalid_edit`) and again in the run.
- Turn reviewer feedback into a new proposal (re-plan) instead of a rejection.
- Review a ReAct agent's tool calls: `HumanInTheLoopMiddleware` (approve / edit arguments / reject with a message) and `@vdreamer.tool(approval=ApprovalSpec(...))` (approve / reject, with a role).
- Keep drafts meant for the approver out of the customer's stream.

## Prerequisites

- Lessons [25](../t25_human_in_the_loop/README.md)–[26](../t26_asking_for_input/README.md) (approvals, roles), [9](../t09_tool_selection/README.md) (middleware) and [16](../t16_context_management/README.md) (`nostream`).

## Concepts

Approve-or-reject wastes good proposals: a refund that is right but slightly too high gets rejected and redone. Two more actions make review practical:

- **edit** — the approver changes the proposal (a lower amount, a better sentence). The change must stay inside what the approver is allowed to change: an **edit schema** (a pydantic model) bounds it. The gateway validates the edit before resuming (`422 invalid_edit` otherwise) and `require_approval` validates it again inside the run.
- **feedback** — the approver explains what is wrong ("warmer tone, please"); the agent **re-plans** and asks again.

`ApprovalGate` builds this loop as a subgraph and routes each action for you; you supply the nodes:

```
prepare ─► gate ─┬─ approve ──────────────────► execute
                 ├─ edit (payload → state) ───► execute
                 ├─ reject ───────────────────► on_reject      (headless: the run ends "rejected")
                 └─ feedback ─► replan ─► prepare ─► gate …
```

For **ReAct agents**, the proposal *is* the model's tool call. Two ways to review it:

| | `HumanInTheLoopMiddleware` (LangChain) | `approval=ApprovalSpec(...)` (vdreamer tool policy) |
| --- | --- | --- |
| Actions | approve, **edit args**, reject (with a message for the model), respond | approve, reject |
| Configured | per tool name in the middleware | on the tool (or via `tool_policies`) |
| Approver | the initiator, or via `interrupt_policies` | `approver_role` on the spec |
| Interrupt kind | `tool_review` | the spec's `kind` |

Both go through the same `/interrupts` endpoint and the same `vdreamer run` prompts.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `ApprovalGate`, `edit_schema` validation (gateway `422` + in-run), `decision_key` | vdreamer |
| `ApprovalSpec` tool gates, `interrupt_policies`, the `tool_review` interrupt kind | vdreamer |
| `HumanInTheLoopMiddleware` and its decisions | LangChain |
| `nostream` tag for drafts | LangGraph |

## How it works

```
"Order A1001 arrived a week late"
prepare  ─► order_id=A1001, amount_eur=14.90 (10%, capped), message=<draft, not streamed>
gate     ⏸ credit_review {order_id, amount_eur, message}   actions: approve, reject, edit, feedback
   feedback "warmer, please" ─► replan (redraft) ─► prepare (keeps it) ─► gate ⏸ again
   edit {"amount_eur": 25, "message": "…"} ─► validated by CreditEdit ─► written to state
execute  ─► credit issued ─► the final message to the customer, "(… approved by bob)"
```

## Step by step

**1. Bound what an approver may change.** The edit schema's fields are state keys:

```python
# file: credit.py
class CreditEdit(BaseModel):
    """What an approver may change. Its fields are state keys."""

    amount_eur: float = Field(gt=0, le=100)
    message: str = Field(min_length=10, max_length=500)
```

**2. Prepare without overwriting.** `prepare` runs again after every re-plan, so it only fills in what is missing:

```python
# file: credit.py
            if "amount_eur" not in state:
                update["amount_eur"] = default_credit(update.get("order_id")
                                                      or state["order_id"])
            if "message" not in state:
                update["message"] = await draft({**state, **update})
```

**3. Draft for the approver, not the customer.**

```python
# file: credit.py
        # Drafts go to the approver, not the customer: keep them out of the stream.
        drafter = ctx.models["chat"].model_copy(update={"tags": ["nostream"]})
```

**4. Assemble the gate.**

```python
# file: credit.py
        gate = ApprovalGate(
            CreditState,
            kind="credit_review",
            payload=lambda s: {"order_id": s["order_id"], "amount_eur": s["amount_eur"],
                               "message": s["message"]},
            prepare=prepare,
            execute=execute,
            on_reject=on_reject,
            replan=replan,
            edit_schema=CreditEdit,
            decision_key="review",
        )
        return gate.build()
```

`decision_key="review"` stores the whole decision in state; `replan` reads the feedback from `state["review"]["text"]`, `execute` the approver from `resolved_by`.

**5. Review tool calls.** Middleware for one tool, a policy for another:

```python
# file: tool_review.py
@vdreamer.tool(approval=ApprovalSpec(kind="cancel_review", approver_role="supervisor"))
def cancel_order(order_id: str) -> str:
```

```python
# file: tool_review.py
        review = HumanInTheLoopMiddleware(
            interrupt_on={"issue_credit": {"allowed_decisions": ["approve", "edit", "reject"]}},
            description_prefix="Credit pending review",
        )
```

## The complete example

- [`credit.py`](credit.py) — `CreditEdit`, `CreditState`, `CreditApprovalAgent` (`t27-credit-approval`).
- [`tool_review.py`](tool_review.py) — `issue_credit`, `cancel_order`, `ToolReviewAgent` (`t27-tool-review`).

## Run it

From `tutorials/`:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t27-credit-approval --models config/models.ollama.yaml
uv run vdreamer run t27-credit-approval --message "Order A1001 arrived a week late"
```

At the `action` prompt try `feedback` (then type it), `edit` (then type JSON such as `{"amount_eur": 20, "message": "We are very sorry; please accept EUR 20."}`), or `approve`. For tool review:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t27-tool-review --models config/models.ollama.yaml
uv run vdreamer run t27-tool-review --roles supervisor --message "A1002 was late, give EUR 30 credit"
```

## Expected behavior

The credit review (with `approve`):

```
[interrupt credit_review] actions: approve, reject, edit, feedback
[run awaiting_approval]
interrupt credit_review (81ff…)
{
  "order_id": "A1001",
  "amount_eur": 14.9,
  "message": "We're sorry your order A1001 arrived late. Please accept a EUR 14.90 credit."
}
[interrupt resolved: approve by alice]
assistant>
We're sorry your order A1001 arrived late. Please accept a EUR 14.90 credit.

assistant>
(EUR 14.90 credited; approved by alice)
```

A tool review edited to EUR 15: the `tool.call` line still shows the model's proposal (`"amount_eur": 30`); the result begins "Note: a human reviewer replaced this tool call before it ran…" and EUR 15 is credited. An edit with `"amount_eur": 500` on the credit review is refused with `422 invalid_edit`.

## Test it

[`test_t27_approve_edit_reject.py`](../../../../tests/tutorials/test_t27_approve_edit_reject.py) drives every action of the gate, the gateway's edit validation, and both tool-review mechanisms — including an edit typed into `vdreamer run`:

```python
# file: tests/tutorials/test_t27_approve_edit_reject.py
    edited = {"type": "edit", "edited_action": {"name": "issue_credit",
                                                "args": {"order_id": "A1002", "amount_eur": 15}}}
    done = await session.resume({"decisions": [edited]})
    assert tool_review.CREDITS == {"A1002": 15}
    # tool.call shows what the model proposed; the result says what actually ran.
    assert done.tool_calls[0][1]["amount_eur"] == 30
```

In-process, tests pass the resume value the gateway would build: `{"action": …}` for `require_approval`/`ApprovalGate`, `{"decisions": [...]}` for `HumanInTheLoopMiddleware`.

```bash
uv run pytest tests/tutorials/test_t27_approve_edit_reject.py -v
```

## Common mistakes and debugging

- **`prepare` overwrites the edited or replanned proposal** — it runs again after every re-plan. Fill in only what is missing.
- **Edit schema fields that are not state keys** — the default `apply_edit` writes the payload's keys into state; mismatched names change nothing. Name them the same, or pass `apply_edit=`.
- **A `feedback` action with no `replan` node** — `validate` reports the unwired action (`approval.routes`). Wire it or do not offer it.
- **Customers see drafts** — model calls stream. Draft with a `nostream` model.
- **Trusting the `tool.call` event after an edit** — it shows the proposal; the `tool.result` (and your own records) show what ran.
- **A `reject` in a headless run** ends the run `rejected`; `on_reject` runs only in chat runs.

## Design trade-offs

- **Gate a proposal vs. gate a tool call.** A proposal gate (`ApprovalGate`) reviews a whole, well-shaped decision with context. Tool-call review is quicker to add to an existing ReAct agent but shows raw arguments.
- **Edit freedom.** A tight edit schema keeps approvers from creating new risks (a EUR 5,000 "goodwill" credit); a loose one saves rejections. Bound amounts, free-text messages.
- **Feedback loops** can run long; cap re-plans the way lesson 26 capped questions.

## Exercises

1. Cap re-plans at two: after that, offer only approve/reject.
2. Give the credit gate `approver_role="supervisor"` and run it headless via `/trigger` (make the agent `BOTH` with an input schema).
3. Use `interrupt_policies = {"tool_review": InterruptPolicy(approver_role="supervisor")}` so only supervisors can review credits.

Next: [Lesson 28 — Middleware for logging, validation, authorization and execution control](../t28_middleware/README.md).
