"""The artifact proxy routes (§2.5, §3.4, A8, A9) over the in-memory store."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from _api import Stack, assert_error, headers, new_thread
from fastapi.testclient import TestClient

from vdreamer.artifacts import ArtifactRecord


def _put(stack: Stack, thread_id: str, data: bytes | None, **overrides: Any) -> ArtifactRecord:
    fields: dict[str, Any] = {
        "artifact_id": "art_1",
        "run_id": "run_1",
        "thread_id": thread_id,
        "kind": "json",
        "name": "verdict",
        "mime_type": "application/json",
        "size_bytes": len(data) if data is not None else None,
        "renderer": "verdict.v1",
        "preview": {"ok": True},
        "metadata": {"source": "test"},
        "filename": "verdict.json",
    }
    fields.update(overrides)
    record = ArtifactRecord(**fields)
    asyncio.run(stack.artifacts.put(record, data))
    return record


def test_metadata_carries_the_artifact_created_field_set(client: TestClient, stack: Stack) -> None:
    thread_id = new_thread(client)
    _put(stack, thread_id, b'{"ok": true}')
    body = client.get("/api/v1/artifacts/art_1", headers=headers()).json()
    assert body == {
        "artifact_id": "art_1",
        "question_id": "run_1",
        "session_id": thread_id,
        "kind": "json",
        "renderer": "verdict.v1",
        "name": "verdict",
        "mime_type": "application/json",
        "size_bytes": 12,
        "preview": {"ok": True},
        "view_url": "/api/v1/artifacts/art_1/raw",
        "download_url": "/api/v1/artifacts/art_1/download",
        "metadata": {"source": "test"},
    }


def test_raw_and_download(client: TestClient, stack: Stack) -> None:
    _put(stack, new_thread(client), b'{"ok": true}', filename="vérdict 1.json")
    raw = client.get("/api/v1/artifacts/art_1/raw", headers=headers())
    assert raw.status_code == 200 and raw.content == b'{"ok": true}'
    assert raw.headers["content-type"].startswith("application/json")
    assert raw.headers["Content-Disposition"].startswith("inline;")
    download = client.get("/api/v1/artifacts/art_1/download", headers=headers())
    disposition = download.headers["Content-Disposition"]
    assert disposition.startswith('attachment; filename="v_rdict_1.json"')
    assert "filename*=UTF-8''v%C3%A9rdict%201.json" in disposition


def test_html_is_served_under_the_artifact_csp(client: TestClient, stack: Stack) -> None:
    _put(stack, new_thread(client), b"<h1>report</h1>", kind="html", mime_type="text/html")
    response = client.get("/api/v1/artifacts/art_1/raw", headers=headers())
    csp = response.headers["Content-Security-Policy"]
    assert "connect-src 'none'" in csp and "default-src 'none'" in csp
    assert "http" not in csp  # no external origin anywhere in the policy


def test_preview_bounds(client: TestClient, stack: Stack) -> None:
    _put(stack, new_thread(client), b"PAR1", kind="table", mime_type="application/x-parquet",
         preview={"columns": ["a"], "rows": [[1]], "row_count": 1})
    body = client.get("/api/v1/artifacts/art_1/preview?offset=5&limit=1000",
                      headers=headers()).json()
    assert (body["offset"], body["limit"]) == (5, 1000)
    assert body["preview"]["row_count"] == 1
    assert_error(
        client.get("/api/v1/artifacts/art_1/preview?limit=1001", headers=headers()),
        422,
        "invalid_request",
    )


def test_download_formats(client: TestClient, stack: Stack) -> None:
    _put(stack, new_thread(client), b"PAR1", kind="table", mime_type="application/x-parquet")
    assert client.get(
        "/api/v1/artifacts/art_1/download?format=parquet", headers=headers()
    ).status_code == 200
    error = assert_error(
        client.get("/api/v1/artifacts/art_1/download?format=csv", headers=headers()),
        422,
        "unsupported_format",
    )
    assert error["details"] == {"available": ["parquet"]}
    assert_error(
        client.get("/api/v1/artifacts/art_1/download?format=pdf", headers=headers()),
        422,
        "invalid_request",
    )


@pytest.mark.parametrize("suffix", ["/raw", "/download"])
def test_missing_bytes_are_410(client: TestClient, stack: Stack, suffix: str) -> None:
    _put(stack, new_thread(client), None)
    assert client.get("/api/v1/artifacts/art_1", headers=headers()).status_code == 200
    assert_error(
        client.get(f"/api/v1/artifacts/art_1{suffix}", headers=headers()), 410, "artifact_expired"
    )


@pytest.mark.parametrize("suffix", ["", "/preview", "/raw", "/download"])
def test_access_is_owner_or_read_all(client: TestClient, stack: Stack, suffix: str) -> None:
    _put(stack, new_thread(client), b"{}")
    url = f"/api/v1/artifacts/art_1{suffix}"
    assert_error(client.get(url, headers=headers(user="bob")), 404, "artifact_not_found")
    reader = client.get(url, headers=headers(user="bob", roles=("artifact:read:all",)))
    assert reader.status_code == 200
    assert_error(
        client.get(f"/api/v1/artifacts/ghost{suffix}", headers=headers()),
        404,
        "artifact_not_found",
    )
