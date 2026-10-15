"""Live tier: UC1–UC3 with a real model, Ollama's ``gemma4:31b-cloud`` (opt-in).

The same stacks as the scripted acceptance tests, with each example's chat/VLM
alias built from its own ``config/models.ollama.yaml`` by ``vdreamer.models``.
Retrieval stays lexical (no fastembed download). Opt in with
``VDREAMER_TEST_OLLAMA=1`` and a running Ollama on ``127.0.0.1:11434``; the
``-cloud`` tag runs the model on ollama.com, so this tier causes egress and
never runs in CI. Assertions are about the workflow, not the model's wording.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from _uc_kit import EXAMPLES, FakeNeo4j, serve, vdreamer_run, wait_status

from vdreamer.models import build_models, load_models

pytestmark = pytest.mark.skipif(
    os.environ.get("VDREAMER_TEST_OLLAMA") != "1",
    reason="live Ollama tier: set VDREAMER_TEST_OLLAMA=1 (gemma4:31b-cloud, egress)",
)


def _alias(example: str, alias: str) -> Any:
    config = load_models(EXAMPLES / example / "config" / "models.ollama.yaml")
    return build_models(config, [alias])[alias]


def test_uc1_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_uc1_sql_analyst import _seed, _stack

    warehouse = _seed(tmp_path / "warehouse.db")
    stack = _stack(warehouse, _alias("uc1_sql_analyst", "analyst"), FakeNeo4j())
    try:
        with serve(stack) as url:
            result = vdreamer_run(url, monkeypatch, "approve", agent="sql-analyst",
                                  message="What is the total revenue per region?", once=True)
    finally:
        stack.close()
    print(result.out)
    assert result.code == 0, result.out
    assert "[interrupt sql_review]" in result.out
    assert "[artifact] table 'result'" in result.out and "[artifact] html 'report'" in result.out


def test_uc2_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_uc2_site_validation import _stack, _trigger

    log = tmp_path / "recorded.jsonl"
    stack = _stack(_alias("uc2_site_validation", "vlm"), log)
    try:
        with serve(stack) as url:
            run_id, out = _trigger(url, monkeypatch, stack, "live-0042")
            assert "[artifact] json 'verdict CAI-0042', renderer verdict.v1" in out
            approver = vdreamer_run(url, monkeypatch, "approve", agent="site-validation",
                                    user="sam", roles=("site_approver",), inbox=True)
            assert approver.code == 0, approver.out
            wait_status(url, run_id, "completed", service=True, timeout_s=60)
    finally:
        stack.close()
    print(out)
    receipt = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert receipt["approved_by"] == "sam" and receipt["status"] in {"pass", "fail",
                                                                      "needs_review"}


def test_uc3_live(monkeypatch: pytest.MonkeyPatch) -> None:
    from qdrant_client import AsyncQdrantClient
    from test_uc3_pdf_research import _ingest, _stack

    client = AsyncQdrantClient(":memory:")
    stack = _stack(_alias("uc3_pdf_research", "reasoner"), client)
    try:
        with serve(stack) as url:
            _ingest(url, monkeypatch, stack)
            english = vdreamer_run(url, monkeypatch, agent="pdf-research", once=True,
                                   message="How long are customer records retained?")
            arabic = vdreamer_run(url, monkeypatch, agent="pdf-research", once=True,
                                  message="ما مدة الاحتفاظ بسجلات العملاء؟")
    finally:
        stack.close()
        asyncio.run(client.close())
    print(english.out, arabic.out, sep="\n")
    for result in (english, arabic):
        assert result.code == 0, result.out
        assert "-> search_documents(" in result.out
        assert "(doc://policy_" in result.out  # at least one [n] became a citation
