"""Milestone 6 files through the REST surface: ``FileRef`` (A17) → ``safe_extract`` (A34).

The ``unpack`` agent (UC2's shape) receives a ``FileRef`` to a ZIP an upstream
system put under ``inputs/``, unpacks it with ``ctx.files.safe_extract`` in a
sandboxed child, and emits the manifest as a ``json`` artifact.
"""

from __future__ import annotations

import asyncio
import io
import zipfile

from _api import (
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


def _zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def test_a_file_ref_is_extracted_and_its_manifest_emitted(
    live: TestClient, live_stack: Stack
) -> None:
    package = _zip(
        {"site/notes.txt": b"tower 7", "site/photo.png": tiny_png(), "site/run.exe": b"MZ"}
    )
    live_stack.artifacts.objects.put_bytes("inputs/site-7.zip", package, "application/zip")
    thread_id = new_thread(live, agent="unpack")
    body = {"input": {"site_id": "s7", "package": {"object_key": "inputs/site-7.zip"}}}
    run_id = start_run(live, thread_id, body)
    wait_for_status(live, run_id, "completed", timeout=30.0)

    _, frames = read_sse(live, f"/api/v1/runs/{run_id}/events")
    (created,) = [e["payload"] for e in events_of(frames) if e["type"] == "artifact.created"]
    manifest = live.get(created["view_url"], headers=headers()).json()
    paths = [entry["path"] for entry in manifest["files"]]
    assert paths == ["site/notes.txt", "site/photo.png"]
    (skipped,) = manifest["skipped"]  # the mime of .exe differs between OS mime tables
    assert (skipped["path"], skipped["reason"]) == ("site/run.exe", "not_allowed")
    notes = manifest["files"][0]
    assert notes["size"] == 7 and len(notes["sha256"]) == 64
    assert notes["object_key"].startswith(f"extracted/{thread_id}/{run_id}/")
    assert created["metadata"]["first"] == "tower 7"  # read back via ctx.files
    stored = asyncio.run(live_stack.artifacts.objects.aget_bytes(notes["object_key"]))
    assert stored == b"tower 7"


def test_a_ref_outside_inputs_is_refused_by_the_gateway(client: TestClient) -> None:
    thread_id = new_thread(client, agent="unpack")
    for key in ("artifacts/t_x/run_x/art_1.json", "inputs/../artifacts/x", "/inputs/a", "inputs/"):
        response = client.post(
            f"/api/v1/threads/{thread_id}/runs",
            json={"input": {"site_id": "s", "package": {"object_key": key}}},
            headers=headers(),
        )
        error = assert_error(response, 422, "invalid_input")
        assert error["details"]["errors"][0]["loc"][:2] == ["package", "object_key"], key


def test_an_unsafe_package_fails_the_run_with_its_code(
    live: TestClient, live_stack: Stack
) -> None:
    live_stack.artifacts.objects.put_bytes(
        "inputs/evil.zip", _zip({"../../etc/cron.d/x": b"pwn"}), "application/zip"
    )
    thread_id = new_thread(live, agent="unpack")
    body = {"input": {"site_id": "s", "package": {"object_key": "inputs/evil.zip"}}}
    run_id = start_run(live, thread_id, body)
    wait_for_status(live, run_id, "failed", timeout=30.0)
    _, frames = read_sse(live, f"/api/v1/runs/{run_id}/events")
    final = events_of(frames)[-1]["payload"]
    assert final["status"] == "failed"
    assert "traversal" in final["error"]["message"]
