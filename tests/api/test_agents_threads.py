"""``/agents`` and ``/threads`` (§2.5): registry, role gating, owner scoping, soft delete."""

from __future__ import annotations

from _api import Stack, assert_error, headers, new_thread, start_run, wait_for_status
from fastapi.testclient import TestClient


def test_list_agents_hides_role_gated_agents(client: TestClient) -> None:
    names = [a["name"] for a in client.get("/api/v1/agents", headers=headers()).json()["agents"]]
    assert names == [
        "batch", "echo", "elicit", "form", "gate", "reports", "reviewed", "tooling", "unpack"
    ]
    with_role = client.get("/api/v1/agents", headers=headers(roles=("netops",))).json()
    assert "secret" in [agent["name"] for agent in with_role["agents"]]


def test_get_agent_returns_mode_and_input_schema(client: TestClient) -> None:
    body = client.get("/api/v1/agents/form", headers=headers()).json()
    assert body["name"] == "form"
    assert body["mode"] == "both"
    assert body["input_schema"]["properties"]["text"]["type"] == "string"
    assert body["input_schema"]["required"] == ["text"]
    assert client.get("/api/v1/agents/echo", headers=headers()).json()["input_schema"] is None


def test_get_agent_errors(client: TestClient) -> None:
    assert_error(client.get("/api/v1/agents/ghost", headers=headers()), 404, "agent_not_found")
    error = assert_error(
        client.get("/api/v1/agents/secret", headers=headers()), 403, "forbidden_role"
    )
    assert error["details"]["required_roles"] == ["netops"]


def test_create_thread_is_201_owned_by_the_caller(client: TestClient, stack: Stack) -> None:
    response = client.post(
        "/api/v1/agents/echo/threads", json={"metadata": {"topic": "x"}}, headers=headers()
    )
    assert response.status_code == 201
    thread_id = response.json()["thread_id"]
    assert response.headers["Location"] == f"/api/v1/threads/{thread_id}"
    body = client.get(f"/api/v1/threads/{thread_id}", headers=headers()).json()
    assert body["agent"] == "echo"
    assert body["metadata"] == {"topic": "x"}
    assert body["runs"] == [] and body["active_run_id"] is None and body["values"] is None


def test_create_thread_errors(client: TestClient) -> None:
    assert_error(
        client.post("/api/v1/agents/ghost/threads", headers=headers()), 404, "agent_not_found"
    )
    assert_error(
        client.post("/api/v1/agents/secret/threads", headers=headers()), 403, "forbidden_role"
    )
    assert client.post(
        "/api/v1/agents/secret/threads", headers=headers(roles=("netops",))
    ).status_code == 201
    assert_error(
        client.post("/api/v1/agents/batch/threads", headers=headers()), 400, "mode_not_supported"
    )
    assert_error(
        client.post("/api/v1/agents/echo/threads", json={"bogus": 1}, headers=headers()),
        422,
        "invalid_request",
    )


def test_threads_are_owner_scoped(client: TestClient) -> None:
    mine = [new_thread(client), new_thread(client, agent="form")]
    theirs = new_thread(client, user="bob")
    listed = client.get("/api/v1/threads", headers=headers()).json()
    assert [t["thread_id"] for t in listed["threads"]] == list(reversed(mine))  # newest first
    only_form = client.get("/api/v1/threads?agent=form", headers=headers()).json()["threads"]
    assert [t["thread_id"] for t in only_form] == [mine[1]]
    page = client.get("/api/v1/threads?limit=1&offset=1", headers=headers()).json()
    assert [t["thread_id"] for t in page["threads"]] == [mine[0]]
    assert (page["limit"], page["offset"]) == (1, 1)
    # Someone else's thread is indistinguishable from a missing one.
    assert_error(
        client.get(f"/api/v1/threads/{theirs}", headers=headers()), 404, "thread_not_found"
    )
    assert_error(
        client.delete(f"/api/v1/threads/{theirs}", headers=headers()), 404, "thread_not_found"
    )


def test_soft_delete_hides_the_thread(client: TestClient) -> None:
    thread_id = new_thread(client)
    assert client.delete(f"/api/v1/threads/{thread_id}", headers=headers()).status_code == 204
    assert_error(
        client.get(f"/api/v1/threads/{thread_id}", headers=headers()), 404, "thread_not_found"
    )
    assert client.get("/api/v1/threads", headers=headers()).json()["threads"] == []
    assert_error(
        client.delete(f"/api/v1/threads/{thread_id}", headers=headers()), 404, "thread_not_found"
    )


def test_delete_with_an_active_run_is_409(client: TestClient) -> None:
    thread_id = new_thread(client)
    run_id = start_run(client, thread_id, {"input": {"text": "hi"}})
    error = assert_error(
        client.delete(f"/api/v1/threads/{thread_id}", headers=headers()), 409, "run_conflict"
    )
    assert error["details"] == {"active_run_id": run_id, "reason": "thread_has_active_run"}


def test_thread_history_comes_from_the_checkpoint(live: TestClient) -> None:
    thread_id = new_thread(live)
    run_id = start_run(live, thread_id, {"input": {"text": "hi"}})
    wait_for_status(live, run_id, "completed")
    body = live.get(f"/api/v1/threads/{thread_id}", headers=headers()).json()
    assert body["values"] == {"text": "hi", "reply": "echo:hi"}
    assert body["checkpoint_id"]
    assert [run["run_id"] for run in body["runs"]] == [run_id]
    assert body["runs"][0]["status"] == "completed"
