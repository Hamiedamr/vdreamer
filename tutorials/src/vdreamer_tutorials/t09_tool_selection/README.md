# Lesson 9 — Tool selection and execution control

> **Outcome:** a support agent whose staff-only tools are invisible to customers, refused at execution if called anyway, and limited to one refund per run — selection controlled by middleware, execution by vdreamer governance.

## Learning objectives

- Change, per request, which tools the model is offered, with a LangChain `@wrap_model_call` middleware reading the caller's roles from vdreamer's run context.
- Enforce the same rule at execution with `requires_roles` — and explain why both layers are needed.
- Cap how often a tool may run with `ToolCallLimitMiddleware`.
- Order middleware deliberately around vdreamer's governance middleware.

## Prerequisites

- Lessons [7](../t07_multiple_tools/README.md) (tool sets) and [8](../t08_tool_contracts/README.md) (side effects, `tool_policies`).
- Lesson [2](../t02_minimal_react_agent/README.md): what `ctx.agent_kwargs()` contains.

## Concepts

Lesson 7 let the model choose freely among all tools. Production agents need to control that choice:

- **Selection** — what the model is *offered*. The model cannot pick a tool it never saw, and fewer tools mean shorter prompts and fewer wrong choices. LangChain's `create_agent` runs every model call through *middleware*; a `wrap_model_call` middleware receives the `ModelRequest` (messages, tools, state, runtime) and can change it before the model sees it.
- **Execution** — what may actually *run*. Hiding a tool is not security: a model can name a tool from an earlier turn, from a prompt injection, or by guessing. vdreamer's governance checks every call's `requires_roles` / `scope` against the caller's roles and refuses with `denied`. The rule lives in the tool's policy, so it holds whatever the model does.
- **Frequency** — how often. `ToolCallLimitMiddleware(tool_name=..., run_limit=...)` counts calls per run (or per thread with `thread_limit`) and blocks the extra ones with an error the model sees.

Where do roles come from? The client sends `X-User-Roles` (a trusted backend asserts who the user is); vdreamer puts them in the run's `VDreamerContext.roles`. Middleware reads it as `request.runtime.context`, because `ctx.agent_kwargs()` declares `context_schema=VDreamerContext`.

**Middleware order.** In `create_agent(middleware=[a, b, c])`, the first entry is the outermost wrapper. `ctx.agent_kwargs()["middleware"]` holds vdreamer's governance middleware; keep it first so every tool call — including ones another middleware blocks or retries — is governed and evented exactly once.

## What comes from where

| Piece | Provided by |
| --- | --- |
| `@wrap_model_call`, `ModelRequest.override(tools=...)`, `ToolCallLimitMiddleware` | LangChain (`langchain.agents.middleware`) |
| `requires_roles` (and `scope`) → `denied` at execution | vdreamer governance |
| Caller roles in `VDreamerContext.roles` (from `X-User-Roles`), `request.runtime.context` | vdreamer (`context_schema` from `agent_kwargs()`) |
| `vdreamer run --roles staff` | vdreamer CLI |

## How it works

```
request (roles=())                       request (roles=("staff",))
  │                                        │
  ▼ staff_tools_for_staff_only             ▼
model sees: get_order_status             model sees: get_order_status, get_order_details, issue_refund
  │ (calls issue_refund anyway?)           │ issue_refund ×2 in one step
  ▼                                        ▼ ToolCallLimitMiddleware(run_limit=1)
governance: requires_roles → denied      1st: ok — 2nd: "Tool call limit exceeded…"
```

## Step by step

**1. Mark staff tools in their policy** — this is the execution-time rule:

```python
# file: agent.py
@vdreamer.tool(requires_roles=[STAFF_ROLE], side_effects=True)
def issue_refund(order_id: str, reason: str) -> str:
    """Staff only: refund an order in full. Changes money: use only when asked to refund."""
```

**2. Filter what the model is offered** — the selection rule, as middleware:

```python
# file: agent.py
@wrap_model_call
async def staff_tools_for_staff_only(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse:
    """Offer staff tools to the model only when the caller has the staff role."""
    context = request.runtime.context
    roles = context.roles if isinstance(context, VDreamerContext) else ()
    if STAFF_ROLE not in roles:
        visible = [t for t in request.tools if getattr(t, "name", None) not in STAFF_TOOLS]
        request = request.override(tools=visible)
    return await handler(request)
```

`override` returns a modified copy; the agent's tool list is unchanged, so the tool node can still execute (and governance can still refuse) a call to any of them.

**3. Assemble the middleware, governance first:**

```python
# file: agent.py
        kwargs = ctx.agent_kwargs()
        middleware = [
            *kwargs.pop("middleware"),  # vdreamer's governance stays first (outermost)
            staff_tools_for_staff_only,
            ToolCallLimitMiddleware(tool_name="issue_refund", run_limit=1),
        ]
```

Then `create_agent(..., middleware=middleware, **kwargs)` — the remaining kwargs are the checkpointer, store and context schema.

## The complete example

[`agent.py`](agent.py) — three tools (`get_order_status` for everyone; `get_order_details` and `issue_refund` for staff), the selection middleware, and `SupportDeskAgent` (`t09-support-desk`).

## Run it

From `tutorials/`:

```bash
uv run vdreamer dev --registry vdreamer.yaml --agent t09-support-desk --models config/models.ollama.yaml
uv run vdreamer run t09-support-desk --message "Please refund order A1001"              # a customer
uv run vdreamer run t09-support-desk --roles staff --message "Refund order A1001, damaged"
```

Over HTTP, roles are the `X-User-Roles: staff` header.

## Expected behavior

As a customer, the model has no refund tool, so it should say a staff member must handle it (no tool line). As staff:

```
-> issue_refund({"order_id": "A1001", "reason": "damaged"}) [custom]
<- ok Refunded EUR 149.00 for A1001 (damaged). (1 ms)
```

A second refund in the same run gets `<- error … Tool call limit exceeded. Do not call 'issue_refund' again.` A customer run in which the model calls `issue_refund` anyway gets `<- error Error (denied): requires one of the roles ['staff']`.

## Test it

[`test_t09_tool_selection.py`](../../../../tests/tutorials/test_t09_tool_selection.py) checks each layer separately — what is offered per role (`model.bound_tools`), refusal at execution, the per-run limit — and that roles flow from `vdreamer run --roles` through the gateway:

```python
# file: tests/tutorials/test_t09_tool_selection.py
    model = scripted(call("issue_refund", order_id="A1001", reason="angry"), "I cannot.")
    result = await AgentSession(agent, models={"chat": model}).run("Refund A1001!")

    assert result.tool_results[0]["error"] == {
        "code": "denied", "message": "requires one of the roles ['staff']"
    }
```

The scripted model calls a tool it was not offered — exactly what a real model can do — and governance still refuses it.

```bash
uv run pytest tests/tutorials/test_t09_tool_selection.py -v
```

## Common mistakes and debugging

- **Relying on hiding alone.** Filtering is a usability and cost measure, not access control. Always pair it with `requires_roles`/`scope` (or an approval, lesson 27) on the tool itself.
- **Reading roles from the message** ("I am staff") — never. Roles come from the authenticated caller (`X-User-Roles`, set by your backend), via `VDreamerContext`.
- **`request.runtime.context` is `None`** — the agent was built without `**ctx.agent_kwargs()` (no `context_schema`), or you are running the graph outside vdreamer. The `isinstance` check keeps the middleware safe in both cases (it falls back to "no roles").
- **Governance middleware not first** — another middleware that short-circuits a call (a limit, a cache) would then skip governance, so no `tool.call`/`tool.result` events and no authorization for that call. Keep `*kwargs.pop("middleware")` at the front.
- **A limit that seems to persist across messages** — `thread_limit` counts across the whole thread; `run_limit` resets each run.
- **Mutating `request.tools` in place** — use `request.override(...)`; requests are shared objects.

## Design trade-offs

- **Static vs. dynamic tool sets.** A separate agent per audience (customer agent, staff agent) is simpler to reason about; one agent with dynamic selection avoids duplicating prompts and logic. Here the difference is one tool list, so dynamic selection is cheap.
- **Rule-based vs. model-based selection.** LangChain's `LLMToolSelectorMiddleware` asks a (smaller) model to choose relevant tools for large toolboxes. It saves prompt space but adds a model call and its own failure mode; rules like role checks should never be delegated to a model.
- **Limits: `continue` vs. `end` vs. `error`.** `continue` (used here) lets the model explain; `end` stops the run with a message; `error` fails the run. Choose by how harmful an extra call would be.

## Exercises

1. Hide `get_order_details` from staff too until the conversation mentions an order id (read `request.messages`).
2. Add `thread_limit=3` for `issue_refund` and test that the fourth refund in a thread is blocked.
3. Change the refund's execution rule from `requires_roles` to `scope="refunds:write"` and give staff the role `refunds:*` (an `fnmatch` pattern). Which test needs updating?
4. Try `ToolCallLimitMiddleware(..., exit_behavior="end")` and look at the trace of a double refund.

Next: [Lesson 10 — Handling failures, retries and unexpected outputs](../t10_failures_and_retries/README.md).
