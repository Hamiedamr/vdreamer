# UC1 — SQL & Data Analysis (chat)

The reference chat agent, `sql-analyst`. It is built on vdreamer's public API only (`import vdreamer`).

```
understand → schema_link → generate_sql ─┬─ read review:  dry_run → sql_review gate → execute → table → chart → report
                                         └─ write review: dry_run → sql_review gate (dba, db:write:warehouse) → execute
```

- **`schema_link`** queries the semantic layer, a Neo4j graph of `(:Table)-[:HAS_COLUMN]->(:Column)` and `[:JOINS]`. It goes through `ctx.resources["semantic"]` (read-only) and keeps the tables the question talks about.
- **`generate_sql`** asks `ctx.models["analyst"]` for one statement.
- **`dry_run`** `EXPLAIN`s it on `ctx.resources["warehouse"]`.
- **The review** is a `vdreamer.ApprovalGate` of kind `sql_review`. The payload is the SQL, the plan and the lineage. The reviewer can `approve`, `edit` the SQL (validated against `SqlEdit`), send `feedback` (the SQL is regenerated) or `reject`.
- **Writes** (any `INSERT/UPDATE/DELETE/DDL…`) go to a second gate with `approver_role="dba"` and scope `db:write:warehouse`. With `HITL_STRICT_SCOPES=db:write:*`, the person who asked cannot approve their own write (four-eyes).
- **After `execute`:**
  - the rows become a `table` artifact (paged, with CSV/XLSX export);
  - a bar chart is drawn by matplotlib in the sandbox (POSIX workers; elsewhere it is skipped with a log line);
  - an `html` report links both through `artifact://` URLs.

## Run it

```
python demo/seed.py --warehouse demo/warehouse.db          # SQLite demo warehouse
# load the semantic layer into Neo4j: demo/seed.py --neo4j bolt://... (see its docstring)
export WAREHOUSE_URL=sqlite+aiosqlite:///demo/warehouse.db
export SEMANTIC_NEO4J_USER=neo4j SEMANTIC_NEO4J_PASSWORD=...
vdreamer validate --models config/models.yaml --resources config/resources.yaml sql_analyst.agent:agent
vdreamer dev --agent sql_analyst.agent:agent --models config/models.ollama.yaml --resources config/resources.yaml
vdreamer run sql-analyst --message "What is the revenue per region?"
```

`config/models.yaml` is the production alias (vLLM). `config/models.ollama.yaml` is a workstation alias for Ollama's `gemma4:31b-cloud`. The `-cloud` tag sends prompts to ollama.com; pull `gemma4:31b` to keep them local.

## Install with pip

```
pip install ./examples/uc1_sql_analyst       # vdreamer[sql,neo4j,sandbox] + aiosqlite
```
