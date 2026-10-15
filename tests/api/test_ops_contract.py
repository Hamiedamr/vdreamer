"""Ops endpoints and the contract documents (§2.5, §5.2, §5.4, §6.1).

Milestone 4 exit: ``/api/v1/openapi.json`` is valid OpenAPI 3.1, ``/api/v1/docs``
renders from vendored assets, and the served HTML asserts CSP / no-CDN.
"""

from __future__ import annotations

import re
from typing import Any

import jsonschema
import pytest
from _api import Stack, assert_error, headers, new_thread, start_run
from fastapi.testclient import TestClient
from openapi_spec_validator import validate
from prometheus_client.parser import text_string_to_metric_families

from vdreamer.gateway import create_app
from vdreamer.gateway.backend import GatewayBackend
from vdreamer.vendored import STATIC_DIR, VENDORED_ASSETS, check_vendored_assets, external_urls

# -- health / readiness ---------------------------------------------------------------


def test_healthz(client: TestClient) -> None:
    assert client.get("/api/v1/healthz").json() == {"status": "ok"}


def test_readyz_reports_every_probe(stack: Stack) -> None:
    async def ok() -> None:
        return None

    async def down() -> None:
        raise ConnectionError("postgres is down")

    backend: GatewayBackend = stack.backend
    backend.probes = {"postgres": ok}
    with TestClient(stack.app) as client:
        assert client.get("/api/v1/readyz").json() == {
            "status": "ready",
            "checks": {"postgres": "ok"},
        }
        backend.probes = {"postgres": ok, "event_listener": down}
        error = assert_error(client.get("/api/v1/readyz"), 503, "not_ready")
    assert error["details"]["checks"] == {
        "postgres": "ok",
        "event_listener": "failed: ConnectionError",
    }
    assert "postgres is down" not in str(error)  # no internals in the body


# -- metrics -----------------------------------------------------------------------------


def test_metrics_exposes_every_prd_family(client: TestClient) -> None:
    start_run(client, new_thread(client), {"input": {}})
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    families = {family.name: family for family in text_string_to_metric_families(response.text)}
    for name in (
        "vdreamer_queue_depth",
        "vdreamer_run_duration_seconds",
        "vdreamer_sse_connections",
        "vdreamer_model_latency_seconds",
        "vdreamer_model_tokens",
        "vdreamer_tool_latency_seconds",
        "vdreamer_tool_errors",
        "vdreamer_mcp_server_up",
        "vdreamer_sandbox_timeouts",
        "vdreamer_outbox_lag_seconds",
    ):
        assert name in families, name
    depth = {
        sample.labels["status"]: sample.value for sample in families["vdreamer_queue_depth"].samples
    }
    assert depth["queued"] == 1.0 and depth["running"] == 0.0


# -- OpenAPI 3.1 ----------------------------------------------------------------------------


def _openapi(client: TestClient) -> dict[str, Any]:
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    return dict(response.json())


@pytest.mark.smoke  # floor/latest pins (A41): FastAPI + LangGraph together
def test_openapi_is_valid_3_1(client: TestClient) -> None:
    spec = _openapi(client)
    assert spec["openapi"].startswith("3.1.")
    validate(spec)  # raises on an invalid document


EXPECTED_OPERATIONS = {
    ("get", "/api/v1/agents"),
    ("get", "/api/v1/agents/{name}"),
    ("post", "/api/v1/agents/{name}/threads"),
    ("get", "/api/v1/threads"),
    ("get", "/api/v1/threads/{thread_id}"),
    ("delete", "/api/v1/threads/{thread_id}"),
    ("post", "/api/v1/threads/{thread_id}/runs"),
    ("post", "/api/v1/agents/{name}/trigger"),
    ("get", "/api/v1/runs/{run_id}"),
    ("post", "/api/v1/runs/{run_id}/cancel"),
    ("get", "/api/v1/runs/{run_id}/events"),
    ("get", "/api/v1/threads/{thread_id}/events"),
    ("get", "/api/v1/interrupts"),
    ("get", "/api/v1/interrupts/{interrupt_id}"),
    ("post", "/api/v1/interrupts/{interrupt_id}/resolve"),
    ("get", "/api/v1/artifacts/{artifact_id}"),
    ("get", "/api/v1/artifacts/{artifact_id}/preview"),
    ("get", "/api/v1/artifacts/{artifact_id}/raw"),
    ("get", "/api/v1/artifacts/{artifact_id}/download"),
    ("get", "/api/v1/protocol/schema.json"),
    ("get", "/api/v1/healthz"),
    ("get", "/api/v1/readyz"),
    ("get", "/api/v1/metrics"),
}


def test_openapi_covers_the_prd_route_table(client: TestClient) -> None:
    spec = _openapi(client)
    operations = {
        (method, path) for path, item in spec["paths"].items() for method in item
    }
    assert operations == EXPECTED_OPERATIONS


def test_openapi_documents_auth_and_errors(client: TestClient) -> None:
    spec = _openapi(client)
    assert spec["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
    runs = spec["paths"]["/api/v1/threads/{thread_id}/runs"]["post"]
    assert runs["security"] == [{"HTTPBearer": []}]
    params = {param["name"] for param in runs["parameters"]}
    assert {"X-User-Id", "X-User-Roles", "wait_for_idle"} <= params
    assert runs["responses"]["409"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ErrorResponse"
    )
    public = spec["paths"]["/api/v1/healthz"]["get"]
    assert "security" not in public


# -- protocol schema ------------------------------------------------------------------------


def test_protocol_schema_validates_real_events(live: TestClient) -> None:
    schema = live.get("/api/v1/protocol/schema.json").json()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    assert schema["properties"]["v"] == {"const": "1.0"}
    assert len(schema["x-event-types"]) == 12

    from _api import events_of, read_sse, wait_for_status

    run_id = start_run(live, new_thread(live), {"input": {"text": "hi"}})
    wait_for_status(live, run_id, "completed")
    _, frames = read_sse(live, f"/api/v1/runs/{run_id}/events")
    events = events_of(frames)
    assert events
    for event in events:
        validator.validate(event)

    bad = dict(events[0], payload={"attempt": 1})  # run.status without status
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad)
    unknown = dict(events[0], type="future.event", payload={"anything": True})
    validator.validate(unknown)  # clients ignore unknown types (§6.1)


# -- docs: vendored Swagger UI, CSP, no CDN ----------------------------------------------------


@pytest.mark.smoke  # floor/latest pins (A41): FastAPI + LangGraph together
def test_docs_render_from_vendored_assets_only(client: TestClient) -> None:
    response = client.get("/api/v1/docs")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert external_urls(html) == []  # no CDN, no absolute URL at all
    csp = response.headers["Content-Security-Policy"]
    directives = dict(part.strip().split(" ", 1) for part in csp.split(";"))
    assert directives["default-src"] == "'none'"
    assert directives["script-src"] == "'self'"  # no inline script
    assert directives["connect-src"] == "'self'"
    assert not re.search(r"https?:|//", csp)
    sources = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert sources and all(source.startswith("/static/") for source in sources)
    for source in sources:
        asset = client.get(source)
        assert asset.status_code == 200, source
    init = client.get("/static/vdreamer/swagger-init.js").text
    assert "validatorUrl: null" in init and '"/api/v1/openapi.json"' in init
    assert external_urls(init) == []
    assert "<script>" not in html  # every script is a same-origin file


def test_docs_follow_the_viewers_color_scheme(client: TestClient) -> None:
    # A dark browser must not show Swagger's light-theme text on a dark canvas.
    html = client.get("/api/v1/docs").text
    assert '<meta name="color-scheme" content="light dark">' in html
    assert "html.dark-mode" in html
    init = client.get("/static/vdreamer/swagger-init.js")
    assert "prefers-color-scheme: dark" in init.text and '"dark-mode"' in init.text
    # Revalidated, never reused stale across a vdreamer upgrade.
    assert init.headers["cache-control"] == "no-cache" and init.headers.get("etag")


def test_the_bare_origin_redirects_to_the_docs(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/api/v1/docs"


def test_fastapi_cdn_docs_are_disabled(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/docs/oauth2-redirect", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_vendored_assets_are_pinned_and_served(client: TestClient) -> None:
    assert check_vendored_assets() == []
    for relative in VENDORED_ASSETS:
        response = client.get(f"/static/vendor/{relative}")
        assert response.status_code == 200
        assert response.content == (STATIC_DIR / "vendor" / relative).read_bytes()


def test_vendored_asset_check_catches_tampering(tmp_path: Any) -> None:
    import shutil

    shutil.copytree(STATIC_DIR, tmp_path / "static")
    (tmp_path / "static" / "vendor" / "swagger-ui" / "swagger-ui.css").write_text("x")
    (tmp_path / "static" / "vdreamer" / "docs.html").write_text(
        '<script src="https://cdn.example.com/x.js"></script>'
    )
    problems = check_vendored_assets(tmp_path / "static")
    assert any("sha256 mismatch" in problem for problem in problems)
    assert any("external origins" in problem for problem in problems)


def test_create_app_does_not_leak_fastapi_into_vdreamer() -> None:
    import subprocess
    import sys

    code = "import sys, vdreamer; print('fastapi' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.stdout.strip() == "False", "D20: FastAPI stays out of the agent-author API"
    assert callable(create_app)


def test_tokens_never_appear_in_responses(client: TestClient) -> None:
    from _api import TOKEN

    for response in (
        client.get("/api/v1/agents", headers=headers()),
        client.get("/api/v1/agents", headers=headers(token="wrong-token-0123456789")),
        client.get("/api/v1/openapi.json"),
    ):
        assert TOKEN not in response.text
