# AGENTS.md

## Repository state

- Spec-only: `docs/PRD.md` is the entire content. No code, build system, tests, or CI exist yet — never assume an implementation exists.
- The PRD is the contract for everything built here. Read it before designing or writing code; when it conflicts with intuition, the PRD wins.
- PRD map: §2 architecture/LangGraph compat/tool governance, §3 events & artifacts, §4 HITL, §5 air-gapped packaging & zero-egress, §7 scope + non-goals. Cross-refs like `A41`/`D11` are assumption/decision IDs; §7.4 defines A24–A44.

## What this is

A Python SDK + CLI (`vdreamer`) that serves developer-written `BaseAgent` subclasses (LangGraph graphs) as a production REST/SSE API with zero generated server code, for air-gapped enterprise deployment. Python ≥3.11.

## Hard constraints — violating these means rework

- **Version pins (A41):** `langchain>=1.4,<2` (needed for `langchain.mcp`), `langgraph>=1.2,<2`, `langgraph-checkpoint-postgres`. A 2.0 of either dependency requires a vdreamer major. CI must run the reference agents against both the pinned floor and the latest 1.x minor.
- **Design rule:** core ships generic primitives only; domain logic (retrieval, SQL semantics, validation rules) lives in agent code. A capability enters core only if ≥2 reference agents need it and it can't be built on existing primitives.
- **Streaming isolation (A31):** all LangGraph streaming specifics live in one adapter module — two drivers (`v2` default, `v3` opt-in via `VDREAMER_STREAM_DRIVER`) emitting identical protocol events, verified by golden-event tests. Don't spread stream handling elsewhere.
- **Persistence:** `durability="sync"` by default (A42) — crash recovery depends on every step being checkpointed. `async` is a per-agent opt-in.
- **Topology:** gateway ↔ worker communication is Postgres only (queue rows + `LISTEN/NOTIFY`, A7); only workers reach model servers, MinIO writes, and agent resources. The gateway is stateless (N replicas).
- **Tools are plain LangChain `BaseTool`s (A37):** no vdreamer base class. Governance is a proxy/middleware that preserves each tool's name and schema.
- **Protocol version `1.0`:** event envelope field `"v"` and the `X-Protocol-Version` header.
- **Secrets never appear in YAML:** config references env vars via `*_env` keys only (A23).
- **Zero egress is mandatory (§5.2):** no CDNs, telemetry, or update checks; `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`; fastembed `local_files_only`; vendored chart JS. `doctor --offline` runs in CI with egress blocked and must assert zero egress.

## Planned layout (from the PRD; none of this exists yet)

- One OCI image, two entrypoints: `vdreamer serve` (gateway) and `vdreamer worker`. `vdreamer dev` runs everything in one process on SQLite + a local filesystem store (D11); `serve`/`worker` use Postgres + MinIO.
- CLI surface: `dev`, `serve`, `worker`, `validate`, `migrate`, `export-schema`, `doctor --offline`, `bundle`.
- REST under `/api/v1`; SSE with `Last-Event-ID` replay; artifact bytes in MinIO, state/metadata in Postgres.
- `examples/` holds reference agents UC1–UC3. They are the acceptance test: core is P0-complete only when they build **without touching vdreamer internals** (§7.3).
- Python extras: `[postgres]`, `[minio]`, `[fastembed]`, `[qdrant]`, `[neo4j]`, `[mcp]`, `[sandbox]`.

## Non-goals — do not build these (PRD §7.2)

LLM serving/fine-tuning; end-user auth/LDAP; mail fetching; cron; retrieval pipelines, chunking, ingestion, or vector-schema management in core; Neo4j semantic-layer sync; MCP prompts/resources/sampling or exposing vdreamer agents as MCP servers; generated server code; LangGraph Platform API compatibility; non-LangChain frameworks; cloud model providers; a complete Angular app.

## Gotchas

- The reference topology is Linux VMs/containers (§5.1). Sandbox primitives (dedicated uid, rlimits, namespaces — D19/A19) are Unix-only and canbe exercised on a Windows dev host.
- `docs/.PRD.md.un~` is a vim undo artifact, not content — ignore it.
- Once real build/test tooling lands, record the actual commands here.
