# vdreamer protocol 1.0 — conformance kit

Recorded SSE streams from a real vdreamer gateway, each paired with the client state a conforming client must reach. Port your client's SSE handling and event fold, replay every recording through them, and compare with `expected_state`. vdreamer ships no client code. These files and `docs/protocol.md` are the contract.

## Recordings

| File | What it exercises |
|---|---|
| `chat.json` | a chat turn: progress, log, thinking + token deltas, `message.completed`, terminal status |
| `reconnect.json` | a dropped connection, `Last-Event-ID` resume with duplicates, then a page refresh from `/threads/{id}/events` |
| `tool_overflow.json` | a governed tool with redacted args whose oversized result becomes an artifact |
| `interrupt_require_approval.json` | `require_approval` in an `ApprovalGate` (`edit_schema`), approved |
| `interrupt_tool_review.json` | `HumanInTheLoopMiddleware` → `tool_review`, role-gated, resolved with `decisions[]` |
| `interrupt_mcp_elicitation.json` | MCP elicitation → `mcp_elicitation`, answered with `edit` + `payload` |
| `interrupt_generic.json` | a raw `interrupt()` → `generic`, answered with `respond` + `text` |
| `artifacts.json` | every artifact kind: `table`, `image`, `html`, `json` (`renderer: verdict.v1`), `file` |
| `citations.json` | two citations bound to one `message_id`: an artifact page and a URI |
| `forward_compat.json` | hand-written: heartbeat comments, an unknown event type from a newer minor, an unknown payload field, unknown artifact kind and renderer, a duplicate `seq` |

Generated ids and timestamps are normalized (`run_1`, `thread_1`, `int_1`, `art_1`, `2026-01-01T00:00:01Z`) so recordings diff cleanly between releases. Treat them as opaque strings, exactly like real ids.

## File format

```json
{
  "name": "interrupt_generic",
  "description": "...",
  "protocol_version": "1.0",
  "agent": "ask",
  "steps": [
    {"kind": "sse", "path": "/api/v1/runs/run_1/events", "last_event_id": null,
     "dropped": true, "body": "id: run_1:1\ndata: {...}\n\n..."},
    {"kind": "resolve", "path": "/api/v1/interrupts/int_1/resolve", "status": 200,
     "body": {"expected_version": 1, "action": "respond", "text": "eu-west"}},
    {"kind": "sse", "path": "/api/v1/runs/run_1/events", "last_event_id": "run_1:5",
     "dropped": false, "body": "..."}
  ],
  "expected_state": { "...": "the fold after every step" }
}
```

- **`sse` steps:** one HTTP SSE response, byte for byte as the gateway sent it, minus heartbeat timing.
  - `last_event_id` is the `Last-Event-ID` header the client sent.
  - `dropped: true` means the client closed the connection after the last event shown (the server never sent an end).
  - `reset: true` (in `reconnect.json`) means the client reloaded the page: it discards its state and rebuilds it from that response.
- **`resolve` steps:** the REST call the client made between streams, with the body it sent. Replay tools only need the SSE steps. The resolve steps show the matching request shape per interrupt source.

## The fold (normative)

Feed every `sse` step's events in order, keeping state across steps (unless `reset`).

1. **Parse SSE.** An `id:` and one or more `data:` lines make an event; lines starting with `:` are comments (heartbeats) and are ignored.
2. **Check the version.** The envelope's `v` must have major `1`; any other major means refuse the stream. Newer minors (`1.3`) are fine.
3. **Dedup.** Drop an event whose `(run_id, seq)` was already applied. Remember the last SSE `id` for `Last-Event-ID`.
4. **Ignore unknown event types.** Tolerate unknown payload fields.
5. **Apply known events to the state.** State shape:

```
{
  "status": null,            // run.status → payload.status (the latest)
  "error": null,             // run.status → payload.error
  "messages": [],            // {id, text, thinking, citations[]}
  "toolCalls": [],           // {tool_call_id, name, source, args, status, summary, artifact_ids, error}
  "artifacts": [],           // {artifact_id, kind, renderer, name, mime_type, view_url, download_url, render}
  "interrupts": [],          // {interrupt_id, kind, actions, approver_role, scope, status, action, resolved_by}
  "progress": null,          // {label, pct}: the latest
  "logs": []                 // {level, message}
}
```

| Event | Effect |
|---|---|
| `run.status` | `status` and `error` |
| `token.delta` | append `text` to the message `message_id` (create it if new), unless that message is completed |
| `thinking.delta` | the same, into `thinking` |
| `message.completed` | set the message's `text` (and `thinking`, when present); the message is completed, and later deltas for it are ignored. `message.completed` can arrive *before* the last deltas are flushed: it is authoritative. |
| `citation` | append to the message `message_id`'s `citations` (create the message if new), deduplicated by `citation_id`; stored without `message_id` |
| `tool.call` | upsert by `tool_call_id`: status `running` |
| `tool.result` | update that call: `status` (`ok`/`error`), `summary`, `artifact_ids`, `error` |
| `artifact.created` | append (dedupe by `artifact_id`). `render` is the kind, but `"download"` when the kind is not one of `table image html json file`, or the `renderer` is one this client does not implement (the reference client implements `verdict.v1`). |
| `interrupt.requested` | append (dedupe by `interrupt_id`), status `pending` |
| `interrupt.resolved` | set that interrupt's `status: resolved`, `action`, `resolved_by` |
| `progress` | replace `progress` |
| `log` | append |

**Ids, not order, identify things.** When an interrupt is resolved, the run resumes by re-executing the interrupted node from the top. The events that node emitted before the pause are emitted again with the same ids (`message.completed`, `tool.call`). Upsert by id; never append a second entry. `interrupt_tool_review.json` shows this.
