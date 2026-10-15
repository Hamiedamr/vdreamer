"""UC2 — Telecom Site & Antenna Validation (Headless). The PRD §7.3 reference agent.

A mail worker uploads a survey ZIP to ``inputs/`` and calls
``POST /agents/site-validation/trigger`` with ``{site_id, package: FileRef}``
(and a ``callback``). The run:

``unpack`` (``ctx.files.safe_extract``, sandboxed, images + the survey sheet
only) → ``inventory`` (the site's planned antennas from the site-inventory MCP
server, through governed tools) → ``inspect`` (each photo through the VLM
alias) → ``verdict`` (planned vs observed, a ``json`` artifact rendered as
``verdict.v1``) → an ``ApprovalGate`` for the ``site_approver`` role →
``record`` (the decision written back through MCP).

Headless (D15): a ``reject`` ends the run ``rejected`` in the gateway, and the
outbox tells the callback ``failed``; an approval completes it and the callback
hears ``completed``. Everything here uses vdreamer's public API only.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

import pydantic
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

import vdreamer
from vdreamer import ApprovalGate, ArtifactKind, BaseAgent, BuildContext, FileRef, Mode

#: Photos inspected per package (the VLM is the expensive step).
MAX_PHOTOS = 12

ALLOW = ["*.jpg", "*.jpeg", "*.png", "*.webp", "survey.json"]

INSPECT_PROMPT = """You inspect one photo from a telecom site survey.
Answer with JSON only: {"sector": <the sector letter or number written on the mount, or null>,
"antennas": <how many panel antennas are visible>, "issues": [<short strings: damage, corrosion,
loose cabling, obstruction ...>]}."""


class SiteInput(pydantic.BaseModel):
    """What the mail worker sends to ``/trigger``."""

    site_id: str = pydantic.Field(pattern=r"^[A-Z0-9-]{2,32}$", description="The site code.")
    package: FileRef = pydantic.Field(description="The survey ZIP under inputs/.")
    submitted_by: str | None = pydantic.Field(default=None, description="The surveyor.")


class VerdictEdit(pydantic.BaseModel):
    """What the approver may change before approving."""

    model_config = pydantic.ConfigDict(extra="forbid")

    status: Literal["pass", "fail", "needs_review"]
    note: str = pydantic.Field(default="", max_length=2000)


class SiteState(TypedDict, total=False):
    site_id: str
    package: dict[str, Any]
    submitted_by: str | None
    photos: list[dict[str, Any]]
    survey: dict[str, Any]
    skipped: list[str]
    planned: dict[str, Any]
    observations: list[dict[str, Any]]
    checks: list[dict[str, Any]]
    status: str
    note: str
    verdict_id: str
    review: dict[str, Any]
    recorded: dict[str, Any]


def parse_observation(text: str) -> dict[str, Any]:
    """The VLM's JSON answer, tolerant of prose or a code fence around it."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    antennas = data.get("antennas")
    return {
        "sector": None if data.get("sector") in (None, "") else str(data["sector"]).upper(),
        "antennas": antennas if isinstance(antennas, int) and antennas >= 0 else None,
        "issues": [str(issue) for issue in data.get("issues") or [] if str(issue).strip()],
    }


def compare(planned: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> tuple[
    str, list[dict[str, Any]]
]:
    """The verdict: every planned sector photographed with its antennas, and no issues."""
    checks: list[dict[str, Any]] = []
    by_sector: dict[str, int] = {}
    for item in observations:
        if item.get("sector") and item.get("antennas") is not None:
            sector = str(item["sector"])
            by_sector[sector] = max(by_sector.get(sector, 0), int(item["antennas"]))
    for sector in planned.get("sectors") or []:
        name = str(sector["sector"]).upper()
        expected = int(sector["antennas"])
        seen = by_sector.get(name)
        if seen is None:
            checks.append({"check": f"sector {name}", "result": "missing",
                           "detail": "no photo of this sector"})
        elif seen != expected:
            checks.append({"check": f"sector {name}", "result": "mismatch",
                           "detail": f"{seen} antenna(s) seen, {expected} planned"})
        else:
            checks.append({"check": f"sector {name}", "result": "ok",
                           "detail": f"{seen} antenna(s)"})
    issues = [issue for item in observations for issue in item.get("issues") or []]
    if issues:
        checks.append({"check": "condition", "result": "issues", "detail": "; ".join(issues)})
    results = {check["result"] for check in checks}
    if not checks:
        return "needs_review", [{"check": "inventory", "result": "missing",
                                 "detail": "the site has no planned sectors"}]
    if results <= {"ok"}:
        return "pass", checks
    if "missing" in results or "mismatch" in results:
        return "fail", checks
    return "needs_review", checks


def _tool(tools: Sequence[Any], suffix: str) -> Any:
    for tool in tools:
        if tool.name.endswith(suffix):
            return tool
    raise LookupError(f"the site-inventory MCP server has no {suffix!r} tool")


def _tool_json(result: Any) -> Any:
    """A tool result as data: MCP tools answer with text (JSON here) or content blocks."""
    if isinstance(result, tuple):  # (content, artifact) response format
        result = result[0]
    if isinstance(result, dict):
        return result
    if isinstance(result, list):  # content blocks: [{"type": "text", "text": "..."}]
        result = "".join(
            str(block.get("text", "")) if isinstance(block, Mapping) else str(block)
            for block in result
        )
    return json.loads(str(result))


class SiteValidationAgent(BaseAgent):
    """Headless: survey package → inventory + VLM inspection → reviewed verdict → callback."""

    name = "site-validation"
    mode = Mode.HEADLESS
    input_schema = SiteInput
    models = ["vlm"]
    resources: list[str] = []
    tool_policies = {
        "siteinv_get_site": vdreamer.Policy(timeout_s=20),
        # A write: runs under the run's idempotency key, so a retried run records once.
        "siteinv_record_validation": vdreamer.Policy(timeout_s=20, side_effects=True),
    }

    def build(self, ctx: BuildContext) -> Any:
        vlm = ctx.models["vlm"]
        tools = ctx.tools.mcp("siteinv")

        async def unpack(state: SiteState) -> dict[str, Any]:
            vdreamer.emit_progress("unpacking the survey", 5.0)
            files = vdreamer.current_context().files
            manifest = await files.asafe_extract(FileRef(**state["package"]), allow=ALLOW)
            photos = [entry.model_dump() for entry in manifest.files if entry.mime.startswith(
                "image/"
            )][:MAX_PHOTOS]
            survey: dict[str, Any] = {}
            for entry in manifest.files:
                if entry.path.rsplit("/", 1)[-1] == "survey.json":
                    survey = json.loads(await files.aread_bytes(entry))
            vdreamer.emit_log(
                f"{len(photos)} photo(s), {len(manifest.skipped)} file(s) skipped", level="info"
            )
            return {"photos": photos, "survey": survey,
                    "skipped": [item.path for item in manifest.skipped]}

        async def inventory(state: SiteState) -> dict[str, Any]:
            vdreamer.emit_progress("reading the site inventory", 20.0)
            result = await _tool(tools, "get_site").ainvoke({"site_id": state["site_id"]})
            return {"planned": _tool_json(result)}

        async def inspect(state: SiteState) -> dict[str, Any]:
            files = vdreamer.current_context().files
            observations = []
            total = max(len(state.get("photos") or []), 1)
            for index, photo in enumerate(state.get("photos") or []):
                vdreamer.emit_progress(f"inspecting {photo['path']}", 25.0 + 50.0 * index / total)
                data = base64.b64encode(await files.aread_bytes(photo["object_key"])).decode()
                message = HumanMessage(content=[
                    {"type": "text", "text": INSPECT_PROMPT},
                    {"type": "image", "base64": data, "mime_type": photo["mime"]},
                ])
                answer = await vlm.ainvoke([message])
                observation = parse_observation(str(answer.content))
                observations.append({"photo": photo["path"], **observation})
            return {"observations": observations}

        async def verdict(state: SiteState) -> dict[str, Any]:
            status, checks = compare(state.get("planned") or {}, state.get("observations") or [])
            document = {
                "site_id": state["site_id"],
                "status": status,
                "checks": checks,
                "observations": state.get("observations") or [],
                "planned": state.get("planned") or {},
                "submitted_by": state.get("submitted_by"),
            }
            ref = await vdreamer.aemit_artifact(
                f"verdict {state['site_id']}", ArtifactKind.JSON, document, renderer="verdict.v1"
            )
            vdreamer.emit_progress("waiting for the site approver", 80.0)
            return {"status": status, "checks": checks, "verdict_id": ref.artifact_id, "note": ""}

        async def record(state: SiteState) -> dict[str, Any]:
            review = state.get("review") or {}
            result = await _tool(tools, "record_validation").ainvoke({
                "site_id": state["site_id"],
                "status": state["status"],
                "note": state.get("note") or "",
                "approved_by": str(review.get("resolved_by") or "site_approver"),
            })
            vdreamer.emit_progress("recorded", 100.0)
            return {"recorded": _tool_json(result)}

        def payload(state: SiteState) -> dict[str, Any]:
            return {
                "site_id": state["site_id"],
                "status": state.get("status"),
                "checks": state.get("checks") or [],
                "verdict_artifact_id": state.get("verdict_id"),
                "photos": len(state.get("photos") or []),
            }

        gate = ApprovalGate(
            state_schema=SiteState,
            kind="site_verdict",
            payload=payload,
            execute=record,
            actions=("approve", "reject", "edit"),
            approver_role="site_approver",
            scope="site:verdict",
            edit_schema=VerdictEdit,
            decision_key="review",
            ttl_hours=72,
        )

        graph = StateGraph(SiteState)
        graph.add_node("unpack", unpack)
        graph.add_node("inventory", inventory)
        graph.add_node("inspect", inspect)
        graph.add_node("verdict", verdict)
        graph.add_node("review", gate.compile())
        graph.add_edge(START, "unpack")
        graph.add_edge("unpack", "inventory")
        graph.add_edge("inventory", "inspect")
        graph.add_edge("inspect", "verdict")
        graph.add_edge("verdict", "review")
        graph.add_edge("review", END)
        return graph


agent = SiteValidationAgent()
