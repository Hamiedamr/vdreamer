# Lesson 25 — Introducing human-in-the-loop

> **Outcome:** a refund agent that prepares a refund, **pauses the run** until a person approves or rejects it, and then executes or declines — resolved inline in `vdreamer run` or over the REST API, surviving restarts while it waits. No model needed.

## Learning objectives

- Pause a run for a human decision with `vdreamer.require_approval(kind, payload, actions)`.
- Follow the approval lifecycle: `interrupt.requested` → run `awaiting_approval` → resolve → `interrupt.resolved` → the run resumes.
- Place work correctly around the pause: a resume re-runs the approval node, not the nodes before it.
- Resolve approvals from the terminal client and from the API (`GET /interrupts`, `POST /interrupts/{id}/resolve` with `expected_version`).

## Prerequisites

- Lessons [1](../t01_serving_an_agent/README.md) (threads, runs, events), [13](../t13_state_and_history/README.md) (state, checkpoints) and [19](../t19_workflows/README.md) (routing).

## Concepts

Some actions should not happen just because a model — or a workflow — decided they should: money moves, data is deleted, emails go to thousands of customers. **Human-in-the-loop (HITL)** puts a person's decision into the run.

`require_approval(kind, payload, actions)` is vdreamer's one approval primitive. Called inside a node, it:

1. **Pauses the run.** It raises LangGraph's interrupt; the run's state is checkpointed and the run moves to **`awaiting_approval`**. No worker holds it while it waits — it costs nothing, and it survives restarts and deploys.
2. **Publishes a request** — an `interrupt.requested` event with an `interrupt_id`, the `kind` (a hint for how clients render it), the `payload` (what the approver sees), the offered `actions`, and an expiry (`ttl_hours`, default 24).
3. **Resumes with the decision.** When someone resolves it, the gateway checks they may, records `interrupt.resolved`, and the run continues: `require_approval` *returns* `{"action": "approve", "resolved_by": "alice", …}`.

**Who may decide?** By default, the user who started the run. Lesson 26 adds approver roles (`approver_role`), four-eyes rules and headless runs, where there is no user to ask.

**Resume re-runs the node.** LangGraph resumes by running the interrupted node again from the top; this time `require_approval` returns the decision instead of pausing. So anything before the call *in the same node* runs twice, and anything in *earlier nodes* does not. Do the work (look-ups, computations, side effects) in a node **before** the approval node.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `require_approval`, the `interrupt.requested`/`interrupt.resolved` events, `awaiting_approval`, expiry | vdreamer |
| Interrupt records, `GET /interrupts`, `POST /interrupts/{id}/resolve`, `resolved_by` | vdreamer gateway |
| Inline approval prompts and `--inbox` in `vdreamer run` | vdreamer CLI |
| `interrupt()` / resume, checkpoints | LangGraph |

## How it works

```
"Please refund order A1001"
prepare  ─► refund = {order_id: A1001, amount_eur: 149.0}
approve  ─► require_approval("refund_review", refund, actions=("approve", "reject"))
            ⏸ interrupt.requested ─► run.status awaiting_approval      (checkpointed; no worker busy)
                  … minutes or hours later …
            POST /interrupts/{id}/resolve {"action": "approve", "expected_version": 1}
            ▶ interrupt.resolved ─► run.status running
approve (re-run) ─► returns {"action": "approve", "resolved_by": "alice"}
execute  ─► refund issued ─► "Refunded EUR 149.00 for A1001 (approved by alice)."
```

## Step by step

**1. Do the work before the pause.**

```python
# file: agent.py
def prepare(state: RefundState) -> dict[str, Any]:
    """Work before the pause. A resumed run restarts ``approve``, never this node."""
```

**2. Ask.** The approval node only asks, and stores the decision:

```python
# file: agent.py
def approve(state: RefundState) -> dict[str, Any]:
    """Pause here until someone decides. The decision is this function's return value."""
    decision = vdreamer.require_approval(
        "refund_review",
        state["refund"],  # what the approver sees
        actions=("approve", "reject"),
    )
    return {"decision": decision}
```

**3. Route on the decision.**

```python
# file: agent.py
def after_decision(state: RefundState) -> Literal["execute", "decline"]:
    return "execute" if state["decision"].get("action") == "approve" else "decline"
```

**4. Act, and record who approved** — `resolved_by` is stamped by the gateway with the deciding user's id, not supplied by the client:

```python
# file: agent.py
    who = state["decision"].get("resolved_by")
```

## The complete example

[`agent.py`](agent.py) — `prepare`, `approve`, `execute`, `decline`, the routers, and `RefundApprovalAgent` (`t25-refund-approval`). `REFUNDED` stands in for the payment system.

## Run it

From `tutorials/` — no model needed:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t25-refund-approval
uv run vdreamer run t25-refund-approval --message "Please refund order A1001"
```

`vdreamer run` shows the request and prompts for the decision. To decide from somewhere else, leave `vdreamer run` (the run keeps waiting, `awaiting_approval`, on the server) and use the API:

```bash
curl -s "${H[@]}" http://127.0.0.1:8000/api/v1/interrupts            # the pending requests
curl -s "${H[@]}" -X POST http://127.0.0.1:8000/api/v1/interrupts/<interrupt_id>/resolve \
     -d '{"action": "approve", "expected_version": 1}'
uv run vdreamer run t25-refund-approval --thread <thread_id>          # reattach and see the end
```

(`H` is the header array from lesson 1.)

## Expected behavior

```
[run running]
[interrupt refund_review] actions: approve, reject
[run awaiting_approval]
interrupt refund_review (37f78e97…)
{
  "order_id": "A1001",
  "amount_eur": 149.0
}
  action [approve/reject]: approve
[interrupt resolved: approve by alice]
[run running]
assistant>
Refunded EUR 149.00 for A1001 (approved by alice).

[run completed]
```

Resolving the same interrupt twice returns `409`: a decision is final.

## Test it

[`test_t25_human_in_the_loop.py`](../../../../tests/tutorials/test_t25_human_in_the_loop.py) pauses and resumes in-process (`AgentSession.resume`), checks that the resumed run did not repeat `prepare`, and resolves through the terminal client and the REST API:

```python
# file: tests/tutorials/test_t25_human_in_the_loop.py
        (pending,) = api.get("/interrupts").json()["interrupts"]
        resolved = api.post(f"/interrupts/{pending['interrupt_id']}/resolve",
                            json={"action": "approve", "expected_version": pending["version"]})
        wait_status(url, run, "completed")
        stale = api.post(f"/interrupts/{pending['interrupt_id']}/resolve",
                         json={"action": "reject", "expected_version": pending["version"]})
```

In-process, the test passes the resume value the gateway would build, including `resolved_by`.

```bash
uv run pytest tests/tutorials/test_t25_human_in_the_loop.py -v
```

## Common mistakes and debugging

- **Side effects before `require_approval` in the same node** — they run again on resume (twice in total). Move them to an earlier node, or after the approval.
- **Calling `require_approval` outside a node** (in a script, a test without a session) — it needs a running graph (`RuntimeError`).
- **Trusting a client-supplied approver** — the gateway stamps `resolved_by` from the authenticated caller; use that value.
- **`409` on resolve** — someone already decided, or your `expected_version` is stale. Re-read the interrupt.
- **The run seems stuck** — it is `awaiting_approval`. Check `GET /interrupts` (and `vdreamer run <agent> --inbox`); after `ttl_hours` an undecided request expires and the run ends `expired`.
- **A huge payload** — the approver must understand it at a glance. Send what they need to decide, not the whole state.

## Design trade-offs

- **Approve everything vs. approve what matters.** Each approval costs a person's time and adds latency. Gate irreversible or expensive actions, and set thresholds (lesson 27 shows per-tool gates; lesson 31 approves only refunds above a limit).
- **Where to pause.** A dedicated approval node keeps the resume semantics obvious. Approvals inside tools (lesson 27) gate exactly one action but re-run the tool node on resume.
- **Approve/reject only** keeps decisions simple. Edit and feedback (lesson 27) let approvers fix a proposal instead of rejecting it.

## Exercises

1. Approve automatically below EUR 50 (no interrupt), and ask above. Test both paths.
2. Add `ttl_hours=1` and read the `expires_at` of the `interrupt.requested` event.
3. Show the customer's original message in the payload so the approver has context.

Next: Lesson 26 — Pausing for approval or additional input.
