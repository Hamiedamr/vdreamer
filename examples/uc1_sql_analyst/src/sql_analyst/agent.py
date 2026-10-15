"""UC1 — SQL & Data Analysis (Chat). The PRD §7.3 reference agent.

``understand → schema_link → generate_sql → route``, then one of two
approval gates (``vdreamer.ApprovalGate``):

- **read review** — ``dry_run`` (``EXPLAIN``) → ``sql_review`` gate, the
  initiator decides (``approve`` / ``edit`` / ``feedback`` / ``reject``);
- **write review** — the same, plus ``approver_role="dba"`` and the scope
  ``db:write:<table>``, so ``HITL_STRICT_SCOPES=db:write:*`` makes it four-eyes.

``execute`` runs the approved (or edited) statement on the warehouse, then the
result becomes a ``table`` artifact, a chart drawn in the sandbox (POSIX
workers; skipped with a log line elsewhere) and an ``html`` report linking
both. The semantic layer is a Neo4j graph of tables, columns and joins
(``ctx.resources["semantic"]``); the warehouse is any SQLAlchemy async engine
(``ctx.resources["warehouse"]``).

Everything here is domain logic built on vdreamer's public API (``import
vdreamer``) — the acceptance rule of PRD §7.3.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, TypedDict

import pydantic
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import vdreamer
from vdreamer import ApprovalGate, ArtifactKind, BaseAgent, BuildContext, Mode

#: Rows kept from one query (the table artifact pages them for the client).
MAX_ROWS = 10_000

#: Statements that change data or schema: their review needs the dba (and four eyes).
WRITE_KEYWORDS = frozenset(
    {
        "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "REPLACE", "DROP", "ALTER", "CREATE",
        "TRUNCATE", "GRANT", "REVOKE", "COPY", "VACUUM", "ATTACH", "DETACH", "PRAGMA",
    }
)

SEMANTIC_TABLES = """
MATCH (t:Table)
OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
WITH t, c ORDER BY c.name
RETURN t.name AS table, t.description AS description,
       collect({name: c.name, type: c.type, description: c.description}) AS columns
ORDER BY table
"""

SEMANTIC_JOINS = """
MATCH (a:Table)-[j:JOINS]->(b:Table)
RETURN a.name AS left, b.name AS right, j.on AS on
"""

SYSTEM_PROMPT = """You write one SQL statement for the {dialect} warehouse described below.
Answer with the statement only, in a ```sql block. Use only these tables and columns.

{schema}"""


class AnalystState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    schema: list[dict[str, Any]]
    joins: list[dict[str, Any]]
    sql: str
    write: bool
    target: str | None
    plan: list[str]
    review: dict[str, Any]
    error: str | None
    row_count: int
    table_id: str | None
    chart_id: str | None
    report_id: str | None


class SqlEdit(pydantic.BaseModel):
    """What a reviewer may change: the statement itself."""

    model_config = pydantic.ConfigDict(extra="forbid")

    sql: str = pydantic.Field(min_length=1, max_length=20_000, description="The SQL to run.")


# -- SQL helpers (domain logic: core stays generic) -----------------------------------------


def extract_sql(text: str) -> str:
    """The statement from a model answer (a ```sql block, else the whole text)."""
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    sql = (match.group(1) if match else text).strip()
    return sql.rstrip(";").strip()


def _strip_literals(sql: str) -> str:
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)


def is_single_statement(sql: str) -> bool:
    return ";" not in _strip_literals(sql)


def is_write(sql: str) -> bool:
    """Any write keyword anywhere (a CTE can hide a DELETE): conservative by design."""
    words = set(re.findall(r"[A-Za-z_]+", _strip_literals(sql).upper()))
    return bool(words & WRITE_KEYWORDS)


def write_target(sql: str) -> str | None:
    match = re.search(
        r"\b(?:INTO|UPDATE|FROM|TABLE)\s+[\"`\[]?([A-Za-z_][A-Za-z0-9_.]*)",
        _strip_literals(sql),
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else None


def referenced_tables(sql: str, known: Sequence[str]) -> list[str]:
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", _strip_literals(sql).lower()))
    return [table for table in known if table.lower() in words]


def link_schema(
    question: str, tables: Sequence[Mapping[str, Any]], limit: int = 4
) -> list[dict[str, Any]]:
    """Tables whose names, descriptions or columns share words with the question."""
    words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 2}

    def stems(text: str) -> set[str]:
        return {w.rstrip("s") for w in re.findall(r"\w+", text.lower()) if len(w) > 2}

    wanted = {w.rstrip("s") for w in words}
    scored = []
    for table in tables:
        text = " ".join(
            [str(table["table"]), str(table.get("description") or "")]
            + [f"{c.get('name')} {c.get('description') or ''}" for c in table["columns"]]
        )
        score = len(wanted & stems(text)) + (3 if str(table["table"]).rstrip("s") in wanted else 0)
        if score:
            scored.append((score, table))
    scored.sort(key=lambda pair: (-pair[0], str(pair[1]["table"])))
    picked = [dict(table) for _, table in scored[:limit]]
    return picked or [dict(table) for table in tables[:limit]]


def describe(schema: Sequence[Mapping[str, Any]], joins: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for table in schema:
        columns = ", ".join(
            f"{c['name']} {c.get('type') or ''}".strip() for c in table["columns"] if c.get("name")
        )
        lines.append(f"- {table['table']}({columns}): {table.get('description') or ''}")
    names = {t["table"] for t in schema}
    for join in joins:
        if join["left"] in names and join["right"] in names:
            lines.append(f"- join {join['left']} -> {join['right']} on {join['on']}")
    return "\n".join(lines)


def _last_user_text(messages: Sequence[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            content = message.content
            return content if isinstance(content, str) else json.dumps(content)
    return ""


CHART_CODE = r"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.load(open("data.json", encoding="utf-8"))
labels = [str(row[data["label"]]) for row in data["rows"]][:30]
values = [float(row[data["value"]] or 0) for row in data["rows"]][:30]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(labels, values, color="#4c72b0")
ax.set_title(data["title"])
ax.set_ylabel(data["value"])
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.savefig("out/chart.png", dpi=120)
"""


def chart_columns(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str] | None:
    """A (label, numeric value) column pair worth a bar chart, if the result has one."""
    if len(rows) < 2:
        return None
    first = rows[0]
    numeric = [k for k, v in first.items() if isinstance(v, (int, float)) and not isinstance(
        v, bool
    )]
    labels = [k for k in first if k not in numeric]
    if not numeric:
        return None
    return (labels[0] if labels else numeric[0]), numeric[-1]


def report_html(state: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(k))}</th>" for k in (rows[0] if rows else {}))
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row.values()) + "</tr>"
        for row in rows[:20]
    )
    chart = (
        f'<img alt="chart" src="artifact://{state["chart_id"]}">' if state.get("chart_id") else ""
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(state.get('question') or 'Query report')}</title></head><body>"
        f"<h1>{html.escape(state.get('question') or '')}</h1>"
        f"<pre><code>{html.escape(state.get('sql') or '')}</code></pre>"
        f"<p>{len(rows)} row(s) — <a href=\"artifact://{state['table_id']}\">full result</a></p>"
        f"{chart}<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        "</body></html>"
    )


class SqlAnalystAgent(BaseAgent):
    """Chat: question → linked schema → SQL → reviewed → executed → table, chart, report."""

    name = "sql-analyst"
    mode = Mode.CHAT
    models = ["analyst"]
    resources = ["semantic", "warehouse"]
    expose_thinking = True

    def build(self, ctx: BuildContext) -> Any:
        model = ctx.models["analyst"]
        semantic = ctx.resources["semantic"]
        warehouse = ctx.resources["warehouse"]

        def dialect() -> str:
            """The warehouse's SQL dialect — read at run time: resources are not
            connected while ``build()`` runs (``vdreamer validate``)."""
            return str(getattr(getattr(warehouse, "dialect", None), "name", "sql"))

        def understand(state: AnalystState) -> dict[str, Any]:
            return {"question": _last_user_text(state["messages"]), "error": None}

        async def schema_link(state: AnalystState) -> dict[str, Any]:
            vdreamer.emit_progress("reading the semantic layer", 10.0)
            tables = (await semantic.execute_query(SEMANTIC_TABLES, routing_="r")).records
            joins = (await semantic.execute_query(SEMANTIC_JOINS, routing_="r")).records
            all_tables = [dict(record) for record in tables]
            linked = link_schema(state["question"], all_tables)
            vdreamer.emit_log(
                f"linked tables: {', '.join(t['table'] for t in linked)}", level="info"
            )
            return {"schema": linked, "joins": [dict(record) for record in joins]}

        async def generate_sql(state: AnalystState) -> dict[str, Any]:
            vdreamer.emit_progress("writing SQL", 30.0)
            prompt = [
                SystemMessage(
                    SYSTEM_PROMPT.format(
                        dialect=dialect(), schema=describe(state["schema"], state["joins"])
                    )
                ),
                HumanMessage(state["question"]),
            ]
            review = state.get("review") or {}
            if review.get("action") == "feedback" and review.get("text"):
                prompt += [
                    AIMessage(f"```sql\n{state.get('sql', '')}\n```"),
                    HumanMessage(f"Reviewer feedback: {review['text']}. Write it again."),
                ]
            answer = await model.ainvoke(prompt)
            sql = extract_sql(str(answer.content))
            return {"sql": sql, "write": is_write(sql), "target": write_target(sql)}

        async def dry_run(state: AnalystState) -> dict[str, Any]:
            sql = state["sql"]
            if not is_single_statement(sql):
                return {"plan": ["refused: more than one statement"],
                        "error": "multiple statements"}
            explain = "EXPLAIN QUERY PLAN " if dialect() == "sqlite" else "EXPLAIN "
            from sqlalchemy import text

            try:
                async with warehouse.connect() as conn:
                    result = await conn.execute(text(explain + sql))
                    plan = [" ".join(str(v) for v in row) for row in result.fetchall()]
                    await conn.rollback()
            except Exception as exc:  # the plan failing is what the reviewer must see
                return {"plan": [f"EXPLAIN failed: {exc}"], "error": str(exc)}
            vdreamer.emit_progress("waiting for review", 50.0)
            return {"plan": plan, "error": None}

        def review_payload(state: AnalystState) -> dict[str, Any]:
            known = [t["table"] for t in state.get("schema") or []]
            used = referenced_tables(state["sql"], known)
            lineage = [
                {"table": t["table"], "columns": [c["name"] for c in t["columns"]
                                                  if c.get("name")]}
                for t in state.get("schema") or [] if t["table"] in used
            ]
            return {
                "question": state.get("question"),
                "sql": state["sql"],
                "dialect": dialect(),
                "write": bool(state.get("write")),
                "target": state.get("target"),
                "plan": state.get("plan") or [],
                "lineage": lineage,
                "error": state.get("error"),
            }

        def execute_node(allow_write: bool) -> Any:
            async def execute(state: AnalystState) -> dict[str, Any]:
                from sqlalchemy import text

                sql = state["sql"]
                if not is_single_statement(sql):
                    return {"error": "multiple statements", "messages": [AIMessage(
                        "Not run: the statement must be a single SQL statement.")]}
                if is_write(sql) and not allow_write:
                    return {"error": "write after read review", "messages": [AIMessage(
                        "Not run: the reviewed statement now changes data, which needs the "
                        "write review (dba). Ask again and it will be routed there.")]}
                vdreamer.emit_progress("running the query", 70.0)
                try:
                    if is_write(sql):
                        async with warehouse.begin() as conn:
                            result = await conn.execute(text(sql))
                            count = max(int(result.rowcount or 0), 0)
                        where = state.get("target") or "the warehouse"
                        return {"row_count": count, "error": None, "messages": [AIMessage(
                            f"Done: {count} row(s) changed in {where}.")]}
                    async with warehouse.connect() as conn:
                        result = await conn.execute(text(sql))
                        rows = [dict(row) for row in result.mappings().fetchmany(MAX_ROWS)]
                except Exception as exc:
                    return {"error": str(exc), "messages": [AIMessage(f"The query failed: {exc}")]}
                return await present(state, rows)

            return execute

        async def present(state: AnalystState, rows: list[dict[str, Any]]) -> dict[str, Any]:
            vdreamer.emit_progress("building the report", 85.0)
            table = await vdreamer.aemit_artifact(
                "result", ArtifactKind.TABLE, rows or [{"result": "(no rows)"}],
                metadata={"sql": state["sql"]},
            )
            chart_id: str | None = None
            pair = chart_columns(rows)
            if pair is not None:
                payload = {"rows": rows[:30], "label": pair[0], "value": pair[1],
                           "title": state.get("question") or "result"}
                try:
                    run = await vdreamer.current_context().sandbox.arun(
                        CHART_CODE, inputs={"data.json": json.dumps(payload, default=str)},
                        timeout_s=60,
                    )
                    if run.ok and run.artifacts:
                        chart_id = run.artifacts[0].artifact_id
                    else:
                        vdreamer.emit_log(f"chart failed: {run.stderr[-300:]}", level="warning")
                except vdreamer.SandboxUnavailable as exc:
                    vdreamer.emit_log(f"chart skipped: {exc}", level="warning")
            updated = {**state, "table_id": table.artifact_id, "chart_id": chart_id}
            report = await vdreamer.aemit_artifact(
                "report", ArtifactKind.HTML, report_html(updated, rows)
            )
            vdreamer.emit_progress("done", 100.0)
            summary = f"The query returned {len(rows)} row(s)"
            if rows:
                summary += f"; first row: {json.dumps(rows[0], default=str)}"
            return {
                "row_count": len(rows),
                "table_id": table.artifact_id,
                "chart_id": chart_id,
                "report_id": report.artifact_id,
                "messages": [AIMessage(summary + ".")],
            }

        def rejected(state: AnalystState) -> dict[str, Any]:
            reason = (state.get("review") or {}).get("text")
            note = f" ({reason})" if reason else ""
            return {"messages": [AIMessage(f"Understood — the query was not run{note}.")]}

        def route(state: AnalystState) -> str:
            return "write_review" if state.get("write") else "read_review"

        def gate(*, write: bool) -> ApprovalGate:
            return ApprovalGate(
                state_schema=AnalystState,
                kind="sql_review",
                payload=review_payload,
                prepare=dry_run,
                execute=execute_node(allow_write=write),
                on_reject=rejected,
                replan=generate_sql,
                edit_schema=SqlEdit,
                decision_key="review",
                approver_role="dba" if write else None,
                scope="db:write:warehouse" if write else None,
                ttl_hours=24,
            )

        graph = StateGraph(AnalystState)
        graph.add_node("understand", understand)
        graph.add_node("schema_link", schema_link)
        graph.add_node("generate_sql", generate_sql)
        graph.add_node("read_review", gate(write=False).compile())
        graph.add_node("write_review", gate(write=True).compile())
        graph.add_edge(START, "understand")
        graph.add_edge("understand", "schema_link")
        graph.add_edge("schema_link", "generate_sql")
        graph.add_conditional_edges("generate_sql", route, ["read_review", "write_review"])
        graph.add_edge("read_review", END)
        graph.add_edge("write_review", END)
        return graph


agent = SqlAnalystAgent()
