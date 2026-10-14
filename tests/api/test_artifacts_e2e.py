"""Milestone 6 exit: artifacts render end-to-end through the gateway proxy.

A real run (the ``reports`` agent) emits every kind; the bytes go through the
store bound by the worker (D7) and come back through ``/api/v1/artifacts``:
paged and sorted table previews, CSV/XLSX/Parquet export, HTML with
``artifact://`` links rewritten under the CSP, granted-approver access, and
``410`` once the object is gone. Both store backends (SQLite is ``vdreamer
dev``: rows in the ``.db``, bytes on disk).
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from typing import Any

import pytest
from _api import (
    REPORT_ROWS,
    Stack,
    assert_error,
    events_of,
    headers,
    new_thread,
    read_sse,
    start_run,
    tiny_png,
    wait_for_status,
)
from fastapi.testclient import TestClient

from vdreamer import tables


def _report(live: TestClient) -> tuple[str, dict[str, dict[str, Any]]]:
    thread_id = new_thread(live, agent="reports")
    run_id = start_run(live, thread_id, {"input": {"text": "q3"}})
    wait_for_status(live, run_id, "completed")
    _, frames = read_sse(live, f"/api/v1/runs/{run_id}/events")
    created = [e["payload"] for e in events_of(frames) if e["type"] == "artifact.created"]
    return run_id, {payload["kind"]: payload for payload in created}


def test_every_kind_is_announced_with_the_full_field_set(live: TestClient) -> None:
    run_id, by_kind = _report(live)
    assert set(by_kind) == {"table", "image", "html", "json", "file"}
    for payload in by_kind.values():
        assert payload["question_id"] == run_id
        artifact_id = payload["artifact_id"]
        assert payload["view_url"] == f"/api/v1/artifacts/{artifact_id}/raw"
        assert payload["download_url"] == f"/api/v1/artifacts/{artifact_id}/download"
        meta = live.get(f"/api/v1/artifacts/{artifact_id}", headers=headers()).json()
        assert meta["preview"] == payload["preview"] and meta["size_bytes"] > 0

    table = by_kind["table"]
    assert table["mime_type"] == tables.PARQUET_MIME
    assert table["preview"]["row_count"] == REPORT_ROWS
    assert table["preview"]["schema"] == [
        {"name": "i", "type": "int64"},
        {"name": "city", "type": "string"},
        {"name": "score", "type": "int64"},
    ]
    assert table["preview"]["rows"][:2] == [
        {"i": 0, "city": "c0", "score": 0},
        {"i": 1, "city": "c1", "score": 37},
    ]
    assert len(table["preview"]["rows"]) == 10
    assert table["metadata"] == {"truncated": False}
    assert by_kind["image"]["preview"] == {"width": 3, "height": 2}
    assert by_kind["html"]["preview"] == {"title": "Q3 report", "summary": "Scores by city"}
    assert by_kind["json"]["preview"] == {"body": {"ok": True}}
    assert by_kind["json"]["renderer"] == "verdict.v1"
    assert by_kind["file"]["preview"] == {"filename": "notes.txt", "mime": "text/plain"}


def test_table_preview_pages_and_sorts(live: TestClient) -> None:
    _, by_kind = _report(live)
    base = f"/api/v1/artifacts/{by_kind['table']['artifact_id']}/preview"
    page = live.get(f"{base}?offset=2400&limit=1000", headers=headers()).json()
    assert page["row_count"] == REPORT_ROWS and len(page["rows"]) == 100
    assert page["rows"][0]["i"] == 2400 and page["rows"][-1]["i"] == REPORT_ROWS - 1
    assert page["columns"][0] == {"name": "i", "type": "int64"}

    top = live.get(f"{base}?limit=3&sort=-score,i", headers=headers()).json()
    assert [row["score"] for row in top["rows"]] == [100, 100, 100]
    assert top["rows"][0]["i"] < top["rows"][1]["i"]
    assert top["sort"] == "-score,i"
    assert_error(live.get(f"{base}?sort=nope", headers=headers()), 422, "invalid_request")


def test_table_downloads_convert_while_streaming(live: TestClient) -> None:
    _, by_kind = _report(live)
    base = f"/api/v1/artifacts/{by_kind['table']['artifact_id']}/download"
    csv = live.get(f"{base}?format=csv", headers=headers())
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    lines = csv.content.decode().splitlines()
    assert lines[0] == '"i","city","score"' and len(lines) == REPORT_ROWS + 1

    xlsx = live.get(f"{base}?format=xlsx", headers=headers())
    assert xlsx.status_code == 200
    with zipfile.ZipFile(io.BytesIO(xlsx.content)) as book:  # XLSX is a ZIP of XML parts
        assert "xl/worksheets/sheet1.xml" in book.namelist()
    assert 'filename="scores.xlsx"' in xlsx.headers["Content-Disposition"]

    parquet = live.get(base, headers=headers())
    assert parquet.status_code == 200 and parquet.content[:4] == b"PAR1"


def test_xlsx_beyond_the_sheet_limit_is_422(
    live: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, by_kind = _report(live)
    monkeypatch.setattr(tables, "XLSX_MAX_ROWS", REPORT_ROWS - 1)
    url = f"/api/v1/artifacts/{by_kind['table']['artifact_id']}/download?format=xlsx"
    error = assert_error(live.get(url, headers=headers()), 422, "unsupported_format")
    assert error["details"]["row_count"] == REPORT_ROWS


def test_html_links_are_rewritten_and_served_under_the_csp(live: TestClient) -> None:
    _, by_kind = _report(live)
    raw = live.get(by_kind["html"]["view_url"], headers=headers())
    assert raw.status_code == 200
    body = raw.text
    table_id, image_id = by_kind["table"]["artifact_id"], by_kind["image"]["artifact_id"]
    assert f'href="/api/v1/artifacts/{table_id}/raw"' in body
    assert f'src="/api/v1/artifacts/{image_id}/raw"' in body
    assert "artifact://" not in body
    csp = raw.headers["Content-Security-Policy"]
    assert "connect-src 'none'" in csp and "http" not in csp
    # The chart JS the HTML references is vendored, same-origin (A13).
    script = live.get("/static/vendor/plotly/plotly.min.js")
    assert script.status_code == 200 and b"plotly.js" in script.content[:200]

    image = live.get(by_kind["image"]["view_url"], headers=headers())
    assert image.content == tiny_png() and image.headers["content-type"] == "image/png"
    notes = live.get(by_kind["file"]["download_url"], headers=headers())
    assert notes.content == b"plain notes"
    assert 'filename="notes.txt"' in notes.headers["Content-Disposition"]


def test_access_owner_granted_approver_or_read_all(live: TestClient) -> None:
    thread_id = new_thread(live, agent="reviewed")
    run_id = start_run(live, thread_id, {"input": {"text": "q"}})
    wait_for_status(live, run_id, "awaiting_approval")
    # The run is paused (its stream stays open): the artifact id rides the interrupt.
    reviewer = headers(user="rita", roles=("reviewer",))
    (record,) = live.get("/api/v1/interrupts", headers=reviewer).json()["interrupts"]
    url = f"/api/v1/artifacts/{record['payload']['artifact']}"
    assert live.get(url, headers=headers()).status_code == 200  # the thread owner
    assert live.get(url, headers=headers(user="bob")).status_code == 404
    assert live.get(url, headers=reviewer).status_code == 200  # granted by the interrupt
    assert live.get(f"{url}/preview", headers=reviewer).json()["rows"] == [{"q": "SELECT 1"}]
    other_role = headers(user="olaf", roles=("auditor",))
    assert live.get(url, headers=other_role).status_code == 404
    read_all = headers(user="ann", roles=("artifact:read:all",))
    assert live.get(url, headers=read_all).status_code == 200


def test_a_vanished_object_is_410(live: TestClient, live_stack: Stack) -> None:
    _, by_kind = _report(live)
    artifact_id = by_kind["table"]["artifact_id"]
    record = asyncio.run(live_stack.artifacts.get(artifact_id))
    assert record is not None and record.object_key.startswith("artifacts/")
    assert record.object_key.endswith(f"/{record.run_id}/{artifact_id}.parquet")
    live_stack.artifacts.objects.delete(record.object_key)  # the 30-day lifecycle (A9)
    assert live.get(f"/api/v1/artifacts/{artifact_id}", headers=headers()).status_code == 200
    for suffix in ("/raw", "/download", "/download?format=csv", "/preview"):
        response = live.get(f"/api/v1/artifacts/{artifact_id}{suffix}", headers=headers())
        assert_error(response, 410, "artifact_expired")
