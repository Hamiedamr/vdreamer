# Lesson 26 — Pausing for approval or additional input

> **Outcome:** a warranty-claims agent that pauses for two different reasons — to ask the customer for a missing serial number (and re-ask if the answer is wrong), and to get a decision from a warranty manager that the customer cannot give — and that never pauses for a customer when a system started the run.

## Learning objectives

- Pause for **information** (`actions=("respond",)`, the answer in `decision["text"]`) as well as for **decisions**.
- Route a decision to someone other than the initiator with `approver_role`, and know how the gateway enforces it (`403 forbidden_role`).
- Resolve as an approver from the terminal: `vdreamer run <agent> --inbox --roles <role>`.
- Handle **headless** runs, where there is no user to ask, and know vdreamer's rules for them.
- Use LangGraph's raw `interrupt()` through the same endpoint.

## Prerequisites

- Lesson [25](../t25_human_in_the_loop/README.md) (`require_approval`, the lifecycle) and lesson [19](../t19_workflows/README.md) (headless agents, `Mode.BOTH`).

## Concepts

`require_approval` is about more than yes/no. Its `actions` choose what the person does:

| Action | Resolver sends | `require_approval` returns |
| --- | --- | --- |
| `approve` / `reject` | the action | `{"action": "approve", "resolved_by": …}` |
| `respond` | the action and a **text** | `{"action": "respond", "text": "…", "resolved_by": …}` |
| `feedback` | the action and a text | `{"action": "feedback", "text": "…"}` (lesson 27) |
| `edit` | the action and a **payload** (validated) | `{"action": "edit", "payload": {…}}` (lesson 27) |

So "ask the customer for their serial number" is a pause with `actions=("respond",)`; the gateway refuses a `respond` without text (`422`).

**Who may resolve.** Without `approver_role`, only the run's initiator. With `approver_role="warranty_manager"`, only callers holding that role (from `X-User-Roles`) — the initiator included only if they hold it. Anyone else gets `403 forbidden_role`. Approvers find what they may decide in the **inbox**: `GET /api/v1/interrupts`, or `vdreamer run <agent> --inbox`. With `HITL_STRICT_SCOPES` (lesson 30), some scopes additionally forbid the initiator from approving their own request ("four eyes").

**Headless runs** (started by `/trigger`, or any run of a `HEADLESS` agent — lesson 19) have no user in the loop. vdreamer's rules: every `require_approval` there must name an `approver_role` (otherwise it raises `ValueError`), and a `reject` ends the run `rejected` instead of resuming it. An agent that wants to *ask the customer* must therefore check `vdreamer.current_context().headless` and do something else — here, reply that details are needed.

**Raw interrupts.** LangGraph's own `interrupt(value)` also works: it appears as an interrupt of kind `generic`, and a `respond` resolves it — with the text (or the given payload) passed back verbatim.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `require_approval(..., actions=("respond",))`, `approver_role`, `ttl_hours` | vdreamer |
| Authority checks (`403 forbidden_role`), `422` for a respond without text, the inbox, `--inbox` | vdreamer gateway and CLI |
| `current_context().headless`, headless approval rules | vdreamer |
| `interrupt()` (shown as kind `generic`) | LangGraph |

## How it works

```
ada: "My tent pole broke, order A1001"           (no serial in the message)
  intake ─► ask_serial ⏸ claim_details  actions=[respond]   — ada may resolve (initiator)
  ada: respond "it says SN-123456"
  ask_serial (re-run) ─► serial = SN-123456 ─► assess: in warranty
  manager_review ⏸ warranty_replacement  actions=[approve, reject]  approver_role=warranty_manager
  ada: approve  ─► 403 forbidden_role                        — not hers to decide
  sam (--roles warranty_manager --inbox): approve
  conclude ─► "Your claim for A1001 was approved by sam: a replacement is on its way."

POST /trigger (headless): intake ─► need_details  ─► "We need the product's serial number…"  (no pause)
```

## Step by step

**1. Ask for information.**

```python
# file: agent.py
    answer = vdreamer.require_approval("claim_details", {"question": question},
                                       actions=("respond",))
    found = SERIAL.search(str(answer.get("text", "")).upper())
    return {"serial": found.group(0) if found else None, "questions_asked": asked + 1}
```

**2. Validate the answer and ask again — a bounded number of times.**

```python
# file: agent.py
def after_question(state: ClaimState) -> Literal["ask_serial", "assess", "need_details"]:
    if state.get("serial"):
        return "assess"
    return "ask_serial" if state.get("questions_asked", 0) < MAX_QUESTIONS else "need_details"
```

**3. Ask someone with authority.**

```python
# file: agent.py
    decision = vdreamer.require_approval(
        "warranty_replacement",
        {"order_id": state.get("order_id"), "serial": state.get("serial"),
         "claim": state["message"]},
        actions=("approve", "reject"),
        approver_role=MANAGER_ROLE,
        ttl_hours=72,
    )
```

**4. Never ask a customer who is not there.**

```python
# file: agent.py
    # Headless runs have nobody to ask: stop instead of pausing.
    return "need_details" if vdreamer.current_context().headless else "ask_serial"
```

## The complete example

[`agent.py`](agent.py) — `ClaimInput`, the nodes (`intake`, `ask_serial`, `assess`, `manager_review`, `conclude` and three short replies), the routers, and `WarrantyClaimsAgent` (`t26-warranty-claims`, `Mode.BOTH`, no model).

## Run it

From `tutorials/`, three terminals:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t26-warranty-claims
uv run vdreamer run t26-warranty-claims --user ada --message "My tent pole broke, order A1001"
uv run vdreamer run t26-warranty-claims --user sam --roles warranty_manager --inbox
```

Ada answers the serial question (`respond`, then the text). If she tries to approve the replacement herself, she gets `403`. Sam approves from the inbox; Ada reattaches with `--thread` to see the outcome.

## Expected behavior

Ada's terminal:

```
[interrupt claim_details] actions: respond
…
[interrupt resolved: respond by ada]
[interrupt warranty_replacement] actions: approve, reject, approver role warranty_manager
…
[detached: 403 forbidden_role: resolving needs the role 'warranty_manager']
reattach with: vdreamer run t26-warranty-claims --thread thread_…
```

After Sam approves, reattaching shows `[interrupt resolved: approve by sam]` and `Your claim for A1001 was approved by sam: a replacement is on its way.`

## Test it

[`test_t26_asking_for_input.py`](../../../../tests/tutorials/test_t26_asking_for_input.py) covers asking, re-asking and giving up, the manager's role on the request, the out-of-warranty and headless paths, the full two-person flow through the gateway, and a raw `interrupt()`:

```python
# file: tests/tutorials/test_t26_asking_for_input.py
    assert "[interrupt resolved: respond by ada]" in ada.out
    assert "403 forbidden_role: resolving needs the role 'warranty_manager'" in ada.out
    assert sam.code == 0, sam.out
    assert "[interrupt resolved: approve by sam]" in after.out
```

```bash
uv run pytest tests/tutorials/test_t26_asking_for_input.py -v
```

## Common mistakes and debugging

- **`ValueError: require_approval(...) in a headless run needs an approver_role`** — a system-started run reached a question meant for a person. Branch on `current_context().headless`, or give the approval a role.
- **Unbounded re-asking** — a confused customer can loop forever. Count questions in state and give up politely.
- **`422 … a 'respond' resolution needs text`** — a `respond` must carry text.
- **The initiator can approve their own request** — without `approver_role` that is the design. Name a role for decisions that need authority, and use `HITL_STRICT_SCOPES` where four eyes are mandatory.
- **Questions that are really decisions** (or the reverse) — use `respond` to gather facts, approve/reject for authority; mixing them confuses approvers and inboxes.
- **Reading roles from the conversation** — roles come only from the authenticated caller.

## Design trade-offs

- **Asking vs. guessing.** Each question adds latency for the customer but avoids wrong decisions; ask only for what you cannot look up.
- **Role-based vs. named approvers.** Roles let any manager decide and survive staff changes; named approvers are more accountable. vdreamer records `resolved_by` either way.
- **Expiry.** `ttl_hours=72` for the manager gives them time; an expired request ends the run `expired`, which your client should explain.

## Exercises

1. Ask for a photo reference too (a second `respond` question) only when the claim mentions damage.
2. Give rejected claims a reason: offer `feedback` to the manager and include the text in the reply.
3. Trigger the agent headlessly with a serial in the message and check that the manager review still happens (headless approvals with a role are allowed).

Next: [Lesson 27 — Approving, rejecting and editing proposed actions](../t27_approve_edit_reject/README.md).
