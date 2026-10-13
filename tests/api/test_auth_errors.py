"""Internal auth (D13, A14), the protocol handshake (§6.1), and the one error body (§2.5)."""

from __future__ import annotations

import pytest
from _api import PREVIOUS_TOKEN, TOKEN, Stack, assert_error, headers
from fastapi.testclient import TestClient

from vdreamer.gateway import GatewayConfigError, GatewaySettings, tokens_from_env
from vdreamer.gateway.auth import parse_roles
from vdreamer.gateway.settings import validate_bind_host


def test_missing_token_is_401_with_bearer_challenge(client: TestClient) -> None:
    response = client.get("/api/v1/agents", headers=headers(token=None))
    assert_error(response, 401, "unauthorized")
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "authorization",
    ["Bearer wrong-token-0123456789abcdef", "Basic " + TOKEN, "Bearer", TOKEN, "Bearer  "],
)
def test_bad_token_is_401(client: TestClient, authorization: str) -> None:
    response = client.get(
        "/api/v1/agents", headers={"Authorization": authorization, "X-User-Id": "alice"}
    )
    assert_error(response, 401, "unauthorized")


def test_both_rotation_tokens_are_valid(client: TestClient) -> None:
    # A14: two tokens valid at once — the current and the rotated-out one.
    for token in (TOKEN, PREVIOUS_TOKEN):
        assert client.get("/api/v1/agents", headers=headers(token=token)).status_code == 200


def test_token_is_checked_before_identity_and_body(client: TestClient) -> None:
    # A14: token first — no identity, malformed JSON, unknown route: still 401.
    response = client.post(
        "/api/v1/threads/whatever/runs",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert_error(response, 401, "unauthorized")
    assert_error(client.get("/api/v1/no-such-route"), 401, "unauthorized")


def test_missing_identity_is_400(client: TestClient) -> None:
    assert_error(client.get("/api/v1/agents", headers=headers(user=None)), 400, "missing_identity")
    assert_error(client.get("/api/v1/agents", headers=headers(user="  ")), 400, "missing_identity")


def test_protocol_major_mismatch_is_400(client: TestClient) -> None:
    error = assert_error(
        client.get("/api/v1/agents", headers=headers(**{"X-Protocol-Version": "2"})),
        400,
        "protocol_version_mismatch",
    )
    assert error["details"]["server_version"] == "1.0"
    for version in ("1", "1.0", "1.7"):
        response = client.get("/api/v1/agents", headers=headers(**{"X-Protocol-Version": version}))
        assert response.status_code == 200


def test_every_response_carries_protocol_version_and_nosniff(client: TestClient) -> None:
    for response in (
        client.get("/api/v1/agents", headers=headers()),
        client.get("/api/v1/agents", headers=headers(token=None)),
        client.get("/api/v1/healthz"),
        client.get("/static/vendor/swagger-ui/LICENSE"),
    ):
        assert response.headers["X-Protocol-Version"] == "1.0"
        assert response.headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/healthz",
        "/api/v1/readyz",
        "/api/v1/metrics",
        "/api/v1/openapi.json",
        "/api/v1/protocol/schema.json",
        "/api/v1/docs",
    ],
)
def test_public_paths_need_no_token(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 200


def test_unknown_route_and_method_use_the_error_body(client: TestClient) -> None:
    assert_error(client.get("/api/v1/nope", headers=headers()), 404, "not_found")
    assert_error(client.put("/api/v1/agents", headers=headers()), 405, "method_not_allowed")


def test_request_validation_uses_the_error_body(client: TestClient) -> None:
    error = assert_error(
        client.get("/api/v1/threads?limit=0", headers=headers()), 422, "invalid_request"
    )
    assert error["details"]["errors"][0]["loc"] == ["query", "limit"]


def test_unhandled_exception_is_500_without_internals(stack: Stack) -> None:
    async def boom() -> None:
        raise RuntimeError("database password is hunter2")

    stack.app.add_api_route("/api/v1/boom", boom)
    with TestClient(stack.app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/boom", headers=headers())
    error = assert_error(response, 500, "internal_error")
    assert "hunter2" not in response.text
    assert "details" not in error


def test_parse_roles() -> None:
    assert parse_roles(None) == ()
    assert parse_roles(" a, b,,a ,c ") == ("a", "b", "c")


def test_settings_enforce_token_rules() -> None:
    with pytest.raises(GatewayConfigError):
        GatewaySettings(tokens=())
    with pytest.raises(GatewayConfigError):
        GatewaySettings(tokens=("short",))
    with pytest.raises(GatewayConfigError):
        GatewaySettings(tokens=(TOKEN, PREVIOUS_TOKEN, TOKEN + "x"))


def test_tokens_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VDREAMER_INTERNAL_TOKEN", raising=False)
    monkeypatch.delenv("VDREAMER_INTERNAL_TOKEN_PREVIOUS", raising=False)
    with pytest.raises(GatewayConfigError, match="VDREAMER_INTERNAL_TOKEN"):
        tokens_from_env()
    monkeypatch.setenv("VDREAMER_INTERNAL_TOKEN", TOKEN)
    assert tokens_from_env() == (TOKEN,)
    monkeypatch.setenv("VDREAMER_INTERNAL_TOKEN_PREVIOUS", PREVIOUS_TOKEN)
    assert tokens_from_env() == (TOKEN, PREVIOUS_TOKEN)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "*"])
def test_wildcard_bind_is_refused(host: str) -> None:
    # A14: the gateway binds to the internal interface only.
    with pytest.raises(GatewayConfigError, match="internal interface"):
        validate_bind_host(host)


def test_specific_bind_is_accepted() -> None:
    assert validate_bind_host("10.20.0.2") == "10.20.0.2"
    assert validate_bind_host("127.0.0.1") == "127.0.0.1"
