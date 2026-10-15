# Building a client for vdreamer protocol 1.0

For client teams (web apps, back ends, integrations). vdreamer ships no client code (PRD §6, A45). What it ships is:

- `vdreamer export-schema` output:
  - `openapi.json` (OpenAPI 3.1)
  - `events.schema.json` (the SSE envelope)
  - per-agent input schemas
  - `vdreamer.d.ts` (TypeScript types)
- the conformance kit in `conformance/`: recorded streams + expected state, with its own `README.md` defining the fold.

Every release attaches both. This page is the normative behavior around them. "MUST" here is the contract the kit tests.

## 1. Talking to the gateway

Browsers never talk to the gateway directly. A Web Backend authenticates the user (LDAP, SSO …; vdreamer has no end-user auth, D13) and proxies `/api/v1/**` (§3). Every request it forwards carries:

| Header | Value |
|---|---|
| `Authorization` | `Bearer <VDREAMER_INTERNAL_TOKEN>`, never exposed to the browser |
| `X-User-Id` | the authenticated user |
| `X-User-Roles` | comma-separated roles (may be empty) |
| `X-Protocol-Version` | `1`. The client MUST send it and MUST refuse a response whose `X-Protocol-Version` major is not `1` (§6.1). |

Errors are always `{"error": {"code", "message", "details?"}}`. Branch on `code`, never on `message`.

Upstream systems (a mail worker, a scheduler) call `POST /agents/{name}/trigger` with a service token instead (A16). See `openapi.json`.

## 2. A run's lifecycle, from the client's side

1. `POST /api/v1/agents/{name}/threads` → `{thread_id}` (chat agents).
2. `POST /api/v1/threads/{thread_id}/runs` with `{input, config?}` → `202 {run_id, thread_id, status}`. The input must match the agent's input schema (`agents/<name>.input.schema.json`), else `422 invalid_input`.
   - One active run per thread: while a run is `running`/`awaiting_approval`, a new one is `409 run_conflict` (`details.active_run_id`).
   - Block the prompt box meanwhile, or pass `?wait_for_idle=true` to queue behind it.
3. `GET /api/v1/runs/{run_id}/events` (SSE) until a terminal `run.status`: `completed`, `failed`, `rejected`, `expired`, `cancelled`. The server closes the stream after it.
4. `awaiting_approval` is **not** terminal. The stream stays open (heartbeats) while the run waits for a human (§5).
5. `POST /api/v1/runs/{run_id}/cancel` stops it.

## 3. Consuming SSE (MUST)

- **Use `fetch` streaming** (D14), not `EventSource`: custom headers are needed, and the Web Backend adds them server-side.
- **Heartbeats** are SSE comments (`: …`) every 15 s. Ignore them, but treat a 2× heartbeat silence as a dead connection.
- **Reconnect** on any drop with exponential backoff plus jitter (for example 0.5 s doubling to 30 s, ±25 %). Send `Last-Event-ID: <the last id you applied>`. The server replays everything after it, so nothing is lost.
- **Page refresh:** rebuild from `GET /api/v1/threads/{thread_id}/events`, which replays every run of the thread in order. Then continue on the active run's stream.
- **Dedup by `seq`:** the key is `(run_id, seq)`. A resume or replay may repeat events you already applied.
- **Ignore unknown event types** and tolerate unknown fields. A newer minor (`v: "1.3"`) may add them; a different major (`2.x`) must be refused.
- **Unknown artifact kinds and renderers** render as downloads (`download_url`).

## 4. The fold (MUST)

Fold the events into one state per thread view:

```
{status, error, messages[{id, text, thinking, citations[]}], toolCalls[], artifacts[], interrupts[], progress, logs}
```

`conformance/README.md` defines the per-event rules. They are short, and two are easy to get wrong:

- `message.completed` is authoritative and may arrive *before* the last `token.delta`s of that message. After it, ignore that message's deltas.
- Things are identified by id, not by arrival. When a run resumes after an interrupt, the paused node runs again from the top and re-emits its `message.completed` / `tool.call` with the **same ids**. Upsert.

Prove it: replay every `conformance/*.json` recording (only the `sse` steps, honoring `reset`) and compare with its `expected_state`. Exact equality is expected for the fields listed. A client may keep more.

## 5. Interrupts (MUST, §6.4)

An `interrupt.requested` event (and `GET /api/v1/interrupts` for the inbox) carries:

- `interrupt_id`, `kind`, `actions[]`, `payload`
- `edit_schema?`, `approver_role?`, `scope?`, `expires_at?`
- `version` (on the REST record).

- **Offer only the `actions[]` listed.** The server rejects anything else (`422 invalid_action`).
- **Build edit forms from `edit_schema`** (JSON Schema 2020-12). The server validates the edit too (`422 invalid_edit` with `details.errors[{loc, msg}]`).
- **Resolve** with `POST /api/v1/interrupts/{id}/resolve`. Always send `expected_version`. The body depends on the source:

| `kind` | Body |
|---|---|
| any `require_approval` kind (`sql_review`, `site_verdict`, …) | `{action, payload?, text?, expected_version}`: `edit` carries `payload`; `feedback` carries `text` |
| `tool_review` | `{decisions: [{action, payload?, text?}, ...], expected_version}`, one per action request **in order** |
| `mcp_elicitation` | `{action: "edit", payload: {<request key>: {...}}}`, or `{action: "reject"}` to decline |
| `generic` | `{action: "respond", text}` or `{action, payload}`; the payload is handed to the graph verbatim |

- **`409`** means someone resolved or edited first. Refresh the record (`GET /interrupts/{id}`) and show the new state.
- **`410`** means it expired: the run ended `expired`.
- **`403 forbidden_role`** means this user may not decide. `details.reason: "four_eyes"` means the initiator may not approve their own `scope` (A32).
- Headless runs (`/trigger`) end `rejected` on a `reject`, and nothing resumes.

After a successful resolve, the run is re-queued and the **same `run_id`** continues: keep reading (or reconnect to) `/runs/{run_id}/events`.

## 6. Artifacts and citations (§6.3, non-normative hints)

| Kind | Rendering | Source |
|---|---|---|
| `table` | server-paged grid; CSV/XLSX export | `GET /artifacts/{id}/preview?offset&limit&sort`, `/download?format=csv|xlsx` |
| `image` | `<img>` | `/raw` |
| `html` | `<iframe sandbox="allow-scripts">`, never `allow-same-origin` | `/raw` |
| `json` | a tree, or the view registered for `renderer` | preview, `/raw` |
| `file` | a download; PDFs at `#page=N` | `/raw`, `/download` |

- `html` artifacts link to other artifacts with `artifact://<id>`. Rewrite those to the artifact's URL.
- A citation binds to `message_id` and opens `source.artifact_id` (or `source.uri`) at `locator.page`.

## 7. Web Backend proxy requirements (§6.2)

- Proxy `/api/v1/**` and inject the token and identity headers server-side.
- **No SSE buffering** anywhere on the path (nginx: `proxy_buffering off`, plus `X-Accel-Buffering: no`).
- Idle/read timeout **> 2× heartbeat**, so above 30 s.
- **Forward `Last-Event-ID`** both ways.
- Serve artifact URLs same-origin, so `<img>`/`<a>` links work with the session cookie and no JS-attached headers (A25).

## 8. Trying it from a terminal

`vdreamer run <agent> --url http://127.0.0.1:8000` is a terminal client built on exactly this contract, REST and SSE only:

- It streams a run and prompts inline for interrupts.
- `--thread` reattaches after a disconnect.

Use it next to `vdreamer dev` to see the events a client will receive.
