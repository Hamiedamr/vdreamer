"""UC2 acceptance (PRD §7.3): headless site & antenna validation, end to end.

A service (the mail worker) uploads the survey ZIP and triggers the run with a
callback (``vdreamer run --service --trigger``). The run unpacks the ZIP in
the sandbox extractor, asks the real site-inventory MCP server (a stdio
subprocess) for the plan, inspects each photo with the VLM alias, emits a
``verdict.v1`` artifact and waits for a ``site_approver``. The approver
decides from the inbox (``vdreamer run --inbox``). Approve → the decision is
recorded through MCP and the callback hears ``completed``; reject → the run
ends ``rejected`` and the callback hears ``failed``.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from _uc_kit import ScriptedModel, scripted, serve, tiny_png, vdreamer_run, wait_status, zip_bytes
from site_validation import SiteValidationAgent

from vdreamer.outbox import OutboxStatus

CALLBACK = "https://ops.internal/hooks/sites"


def _observation(sector: str, antennas: int, issues: list[str] | None = None) -> str:
    return json.dumps({"sector": sector, "antennas": antennas, "issues": issues or []})


def _package(*sectors: str) -> bytes:
    files = {f"photos/sector_{s.lower()}.png": tiny_png() for s in sectors}
    files["survey.json"] = json.dumps({"surveyor": "M. Hassan", "date": "2026-10-10"}).encode()
    files["notes/readme.exe"] = b"MZ not allowed"  # skipped by the allow-list
    return zip_bytes(files)


@pytest.fixture
def inventory_log(tmp_path: Path) -> Path:
    return tmp_path / "recorded.jsonl"


def _stack(vlm: ScriptedModel, log: Path) -> Any:
    from _api import make_stack

    from vdreamer.mcp import open_mcp_registry, parse_server

    @asynccontextmanager
    async def backends() -> AsyncIterator[dict[str, Any]]:
        config = parse_server("siteinv", {
            "type": "mcp",
            "command": sys.executable,
            "args": ["-m", "site_validation.inventory_server"],
            "env": {"SITE_INVENTORY_LOG": str(log)},
        })
        async with open_mcp_registry([config], refresh=False) as registry:
            assert registry.servers["siteinv"].up, registry.servers["siteinv"].last_error
            yield {"models": {"vlm": vlm}, "mcp": registry}

    return make_stack(
        workers=True,
        extra_agents=[SiteValidationAgent()],
        worker_agents=("site-validation",),
        poll_interval_s=0.05,
        runner_options={"site-validation": backends},
    )


@pytest.fixture
def uc2(inventory_log: Path) -> Iterator[tuple[str, ScriptedModel, Any]]:
    vlm = scripted(_observation("A", 2), _observation("B", 2), _observation("C", 2))
    stack = _stack(vlm, inventory_log)
    try:
        with serve(stack) as url:
            yield url, vlm, stack
    finally:
        stack.close()


def _trigger(url: str, monkeypatch: pytest.MonkeyPatch, stack: Any, key: str,
             site: str = "CAI-0042", sectors: tuple[str, ...] = ("A", "B", "C")) -> tuple[
    str, str
]:
    object_key = f"inputs/surveys/{key}.zip"
    stack.artifacts.objects.put_bytes(object_key, _package(*sectors), "application/zip")
    result = vdreamer_run(
        url, monkeypatch, agent="site-validation", service=True, idempotency_key=key,
        callback=CALLBACK,
        input_json=json.dumps({"site_id": site, "package": {"object_key": object_key},
                               "submitted_by": "mail-worker"}),
    )
    assert result.code == 3, result.out  # the service cannot approve: it detaches
    assert f"Idempotency-Key {key}" in result.out
    run_id = re.search(r"\[triggered (run_[0-9a-f]+)", result.out)
    assert run_id, result.out
    return run_id.group(1), result.out


def _outbox(stack: Any, run_id: str) -> list[Any]:
    return list(asyncio.run(stack.outbox.entries(run_id=run_id)))


def test_an_approved_survey_is_recorded_and_called_back(
    uc2: tuple[str, ScriptedModel, Any], monkeypatch: pytest.MonkeyPatch, inventory_log: Path
) -> None:
    url, vlm, stack = uc2
    run_id, out = _trigger(url, monkeypatch, stack, "mail-0042")
    assert "-> siteinv_get_site(" in out and "[mcp:siteinv]" in out
    assert "[artifact] json 'verdict CAI-0042', renderer verdict.v1" in out
    assert "[log info] 3 photo(s), 1 file(s) skipped" in out
    assert len(vlm.seen) == 3  # one VLM call per photo, each with the image block
    image = vlm.seen[0][0].content[1]
    assert image["type"] == "image" and image["mime_type"] == "image/png"

    # A user without the role sees nothing; the site approver decides from the inbox.
    nobody = vdreamer_run(url, monkeypatch, agent="site-validation", user="eve", inbox=True)
    assert "[no pending interrupts" in nobody.out
    approver = vdreamer_run(url, monkeypatch, "approve", agent="site-validation",
                            user="sam", roles=("site_approver",), inbox=True)
    assert approver.code == 0, approver.out
    assert '"status": "pass"' in approver.out  # the payload the approver reviewed

    wait_status(url, run_id, "completed", service=True)
    receipt = json.loads(inventory_log.read_text(encoding="utf-8").splitlines()[-1])
    assert receipt == {"site_id": "CAI-0042", "status": "pass", "note": "",
                       "approved_by": "sam"}
    entries = _outbox(stack, run_id)
    kinds = {(e.channel, e.kind) for e in entries}
    assert ("callback", "interrupt") in kinds and ("callback", "completed") in kinds
    assert all(e.target == CALLBACK for e in entries if e.channel == "callback")
    assert all(e.status is OutboxStatus.PENDING for e in entries)  # delivery is M9's job


def test_a_mismatch_fails_and_a_reject_ends_the_run_rejected(
    inventory_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vlm = scripted(_observation("A", 2), _observation("B", 1, ["corroded mount"]))
    stack = _stack(vlm, inventory_log)
    try:
        with serve(stack) as url:
            run_id, out = _trigger(url, monkeypatch, stack, "mail-0043", sectors=("A", "B"))
            approver = vdreamer_run(url, monkeypatch, "reject", "sector C missing",
                                    agent="site-validation", user="sam",
                                    roles=("site_approver",), inbox=True)
            assert approver.code == 0, approver.out
            assert '"status": "fail"' in approver.out
            assert "sector C" in approver.out and "corroded mount" in approver.out
            wait_status(url, run_id, "rejected", service=True)
            entries = _outbox(stack, run_id)
    finally:
        stack.close()
    failed = [e for e in entries if e.kind == "failed"]
    assert len(failed) == 1 and failed[0].payload["status"] == "rejected"
    assert not inventory_log.exists()  # nothing was recorded


def test_the_approver_can_edit_the_verdict(
    uc2: tuple[str, ScriptedModel, Any], monkeypatch: pytest.MonkeyPatch, inventory_log: Path
) -> None:
    url, _, stack = uc2
    run_id, _ = _trigger(url, monkeypatch, stack, "mail-0044")
    approver = vdreamer_run(
        url, monkeypatch, "edit", '{"status": "needs_review", "note": "glare on sector B"}',
        agent="site-validation", user="sam", roles=("site_approver",), inbox=True,
    )
    assert approver.code == 0, approver.out
    wait_status(url, run_id, "completed", service=True)
    receipt = json.loads(inventory_log.read_text(encoding="utf-8").splitlines()[-1])
    assert receipt["status"] == "needs_review" and receipt["note"] == "glare on sector B"


def test_an_unsafe_package_fails_the_run(
    uc2: tuple[str, ScriptedModel, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, _, stack = uc2
    stack.artifacts.objects.put_bytes(
        "inputs/surveys/evil.zip", zip_bytes({"../../etc/passwd": b"x"}), "application/zip"
    )
    result = vdreamer_run(
        url, monkeypatch, agent="site-validation", service=True, idempotency_key="evil",
        input_json=json.dumps({"site_id": "CAI-0042",
                               "package": {"object_key": "inputs/surveys/evil.zip"}}),
    )
    assert result.code == 2, result.out
    assert "[run failed]" in result.out
