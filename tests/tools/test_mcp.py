"""MCP servers as governed tool sources (PRD §2.7, A38–A40; Milestone 5 exit).

Discovery runs against a real FastMCP server (``mcp_fixture_server.py``) over
stdio and over Streamable HTTP with a bearer token; a failed optional server
is excluded and a failed required one fails readiness.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from _tools_kit import drive
from langchain.agents import create_agent

from vdreamer import BaseAgent, BuildContext
from vdreamer.mcp import (
    MCPConfigError,
    MCPRegistry,
    MCPUnavailable,
    NetworkAllowlist,
    mcp_servers_from_resources,
    open_mcp_registry,
    parse_server,
)
from vdreamer.resources import ResourcesError, parse_resources
from vdreamer.runtime import TOOL_SNAPSHOT_CONFIGURABLE_KEY, AgentRunner

SERVER_SCRIPT = str(Path(__file__).with_name("mcp_fixture_server.py").resolve())
PYTHON = sys.executable


def stdio_entry(**extra: Any) -> dict[str, Any]:
    return {"type": "mcp", "command": PYTHON, "args": [SERVER_SCRIPT], **extra}


# -- config (A23, A40) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "fragment"),
    [
        ({"type": "mcp"}, "exactly one of url"),
        ({"type": "mcp", "url": "https://h/mcp", "command": PYTHON}, "exactly one of url"),
        ({"type": "mcp", "command": "server"}, "absolute path"),
        ({"type": "mcp", "command": "/usr/bin/npx", "args": ["srv"]}, "fetches packages"),
        ({"type": "mcp", "command": "C:\\tools\\uvx.exe"}, "fetches packages"),
        ({"type": "mcp", "command": PYTHON, "args": ["-m", "pip", "x"]}, "fetches packages"),
        ({"type": "mcp", "command": str(Path(PYTHON).with_name("nope-bin"))}, "does not exist"),
        ({"type": "mcp", "url": "ftp://h/mcp"}, "http(s) URL"),
        ({"type": "mcp", "url": "https://u:p@h/mcp"}, "secret in YAML"),
        ({"type": "mcp", "url": "https://h/mcp", "identity": "forward"}, "P1"),
        ({"type": "mcp", "url": "https://h/mcp", "startup": "maybe"}, "startup"),
        ({"type": "mcp", "url": "https://h/mcp", "frobnicate": 1}, "unknown keys"),
        ({"type": "mcp", "url": "https://h/mcp", "args": ["x"]}, "stdio servers only"),
        ({"type": "mcp", "command": PYTHON, "bearer_env": "T"}, "HTTP servers only"),
    ],
)
def test_bad_entries_are_refused(entry: dict[str, Any], fragment: str) -> None:
    with pytest.raises(MCPConfigError, match=None) as caught:
        parse_server("srv", entry)
    assert fragment in str(caught.value)


def test_inline_secrets_are_refused_at_load() -> None:
    with pytest.raises(ResourcesError, match="A23"):
        parse_resources({"netinv": {"type": "mcp", "url": "https://h/mcp", "token": "abc"}})


def test_a_valid_resources_file() -> None:
    resources = parse_resources(
        {
            "netinv": {
                "type": "mcp",
                "url": "https://mcp.netinv.internal/mcp",
                "bearer_env": "NETINV_TOKEN",
                "startup": "required",
                "refresh_s": 300,
                "include": ["get_*", "search_*"],
                "identity": "service",
            },
            "pdfkit": stdio_entry(env={"LOG_LEVEL": "info"}, startup="optional"),
            "vectors": {"type": "qdrant", "url": "https://10.20.1.5:6333", "api_key_env": "Q"},
        }
    )
    netinv, pdfkit = mcp_servers_from_resources(resources)  # qdrant is M7's
    assert (netinv.transport, netinv.required, netinv.include) == (
        "http",
        True,
        ("get_*", "search_*"),
    )
    assert (pdfkit.transport, pdfkit.required, dict(pdfkit.env)) == (
        "stdio",
        False,
        {"LOG_LEVEL": "info"},
    )


def test_the_allowlist_at_config_load() -> None:
    allow = NetworkAllowlist.parse("*.internal, 10.20.0.0/16")
    allow.check_config("mcp.netinv.internal")
    allow.check_config("10.20.3.4")
    allow.check_config("other.example")  # a name: decided at connect, by resolution
    with pytest.raises(MCPConfigError, match="NETWORK_ALLOWLIST"):
        allow.check_config("192.168.1.1")
    with pytest.raises(MCPConfigError, match="NETWORK_ALLOWLIST"):
        NetworkAllowlist.parse("*.internal").check_config("mcp.example.com")
    with pytest.raises(MCPConfigError, match="NETWORK_ALLOWLIST"):
        parse_server(
            "srv",
            {"type": "mcp", "url": "https://evil.example/mcp"},
            allowlist=NetworkAllowlist.parse("*.internal"),
        )
    assert NetworkAllowlist.from_env({}) is None


async def test_the_allowlist_at_connect_resolves_names() -> None:
    await NetworkAllowlist.parse("127.0.0.0/8, ::1").check_connect("localhost", 80)
    with pytest.raises(MCPUnavailable, match="outside"):
        await NetworkAllowlist.parse("10.0.0.0/8").check_connect("localhost", 80)


# -- stdio ---------------------------------------------------------------------------------


@pytest.fixture
async def stdio_registry() -> AsyncIterator[MCPRegistry]:
    configs = [parse_server("fx", stdio_entry(env={"FIXTURE_FLAG": "on"}))]
    async with open_mcp_registry(configs, refresh=False) as registry:
        yield registry


async def test_stdio_tools_are_discovered_and_qualified(stdio_registry: MCPRegistry) -> None:
    server = stdio_registry.servers["fx"]
    assert server.up, server.last_error
    names = [tool.name for tool in stdio_registry.tools("fx")]
    assert names == ["fx_get_big", "fx_get_echo", "fx_get_env", "fx_search_fail"]
    assert [t.name for t in stdio_registry.tools("fx", include=["get_e*"])] == [
        "fx_get_echo",
        "fx_get_env",
    ]
    assert [t.name for t in stdio_registry.tools("fx", include=["fx_search_*"])] == [
        "fx_search_fail"
    ]
    with pytest.raises(LookupError, match="no MCP server 'nope'"):
        stdio_registry.tools("nope")
    await server.probe()


class McpAgent(BaseAgent):
    name = "mcp-agent"

    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self._calls = calls

    def build(self, ctx: BuildContext) -> Any:
        from _tools_kit import calling

        return create_agent(
            model=calling(*self._calls),
            tools=ctx.tools.mcp("fx", include=["get_*", "search_*"]),
            **ctx.agent_kwargs(),
        )


@pytest.mark.smoke  # floor/latest pins (A38): langchain.mcp + FastMCP over stdio
async def test_a_governed_mcp_run(stdio_registry: MCPRegistry) -> None:
    agent = McpAgent(
        [
            ("fx_get_echo", {"text": "hello"}),
            ("fx_get_big", {"n": 20000}),
            ("fx_search_fail", {"query": "q"}),
        ]
    )
    runner = AgentRunner(agent, checkpointer=None, mcp=stdio_registry)
    run = await drive(agent, runner=runner)
    assert run.error is None
    calls = {c["tool_call_id"]: c for c in run.of("tool.call")}
    assert {c["source"] for c in calls.values()} == {"mcp:fx"}
    results = {r["tool_call_id"]: r for r in run.of("tool.result")}
    assert results["call_1"]["status"] == "ok" and results["call_1"]["summary"] == "hello"
    # FastMCP returns the text block *and* structured content: each overflows
    # into its own artifact (file for the text, json for the structured).
    records = [await run.store.get(aid) for aid in results["call_2"]["artifact_ids"]]
    assert sorted(r.kind for r in records if r is not None) == ["file", "json"]
    text_record = next(r for r in records if r is not None and r.kind == "file")
    assert text_record.size_bytes == 20000
    failed = results["call_3"]
    assert failed["status"] == "error" and failed["error"]["code"] == "tool_error"
    assert "no results for q" in failed["error"]["message"]


@pytest.mark.skipif(sys.platform == "win32", reason="rlimits + env scrubbing are POSIX-only (D19)")
async def test_posix_stdio_servers_see_only_their_declared_env(
    stdio_registry: MCPRegistry,
) -> None:
    env_tool = next(t for t in stdio_registry.tools("fx") if t.name == "fx_get_env")
    assert "on" in str(await env_tool.ainvoke({"name": "FIXTURE_FLAG"}))
    assert "<unset>" in str(await env_tool.ainvoke({"name": "PATH"}))
    assert "<unset>" in str(await env_tool.ainvoke({"name": "HOME"}))


async def test_the_runner_rebuilds_on_a_new_snapshot_and_records_it(
    stdio_registry: MCPRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    builds: list[int] = []

    class Counting(McpAgent):
        def build(self, ctx: BuildContext) -> Any:
            builds.append(1)
            return super().build(ctx)

    agent = Counting([("fx_get_echo", {"text": "a"})])
    runner = AgentRunner(agent, checkpointer=InMemorySaver(), mcp=stdio_registry)
    runner.build()
    runner.build()
    assert len(builds) == 1  # cached while the snapshot holds
    monkeypatch.setattr(stdio_registry, "snapshot_hash", lambda: "changed")
    await drive(agent, runner=runner, thread_id="t-snap")
    assert len(builds) == 2
    state = await runner.aget_state({"configurable": {"thread_id": "t-snap"}})
    assert state.metadata[TOOL_SNAPSHOT_CONFIGURABLE_KEY] == "changed"


# -- Streamable HTTP -------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def http_server() -> Iterator[str]:
    port = _free_port()
    env = {**os.environ, "FIXTURE_REQUIRE_TOKEN": "1", "FASTMCP_CHECK_FOR_UPDATES": "off"}
    process = subprocess.Popen(
        [PYTHON, SERVER_SCRIPT, "http", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.1)
    else:
        process.kill()
        pytest.fail("the HTTP fixture server did not start")
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.kill()
        process.wait(10)


async def test_http_tools_are_discovered_with_the_bearer(
    http_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FX_TOKEN", "fixture-token")
    config = parse_server(
        "web",
        {"type": "mcp", "url": http_server, "bearer_env": "FX_TOKEN", "include": ["get_*"]},
        allowlist=NetworkAllowlist.parse("127.0.0.1"),
    )
    async with open_mcp_registry(
        [config], allowlist=NetworkAllowlist.parse("127.0.0.1"), refresh=False
    ) as registry:
        server = registry.servers["web"]
        assert server.up, server.last_error
        tools = registry.tools("web")
        assert [t.name for t in tools] == ["web_get_big", "web_get_echo", "web_get_env"]
        echo = next(t for t in tools if t.name == "web_get_echo")
        assert "over-http" in str(await echo.ainvoke({"text": "over-http"}))


async def test_a_wrong_or_missing_bearer_leaves_the_server_down(
    http_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FX_TOKEN", "wrong")
    configs = [
        parse_server("bad", {"type": "mcp", "url": http_server, "bearer_env": "FX_TOKEN"}),
        parse_server("unset", {"type": "mcp", "url": http_server, "bearer_env": "FX_UNSET"}),
    ]
    async with open_mcp_registry(configs, refresh=False) as registry:
        assert not registry.servers["bad"].up
        assert "FX_UNSET" in str(registry.servers["unset"].last_error)


async def test_a_down_optional_server_is_excluded_and_a_required_one_fails_readiness(
    caplog: pytest.LogCaptureFixture,
) -> None:
    closed = f"http://127.0.0.1:{_free_port()}/mcp"
    configs = [
        parse_server(
            "opt", {"type": "mcp", "url": closed, "startup": "optional", "connect_timeout_s": 5}
        ),
        parse_server("req", {"type": "mcp", "url": closed, "connect_timeout_s": 5}),
        parse_server("fx", stdio_entry()),
    ]
    async with open_mcp_registry(configs, refresh=False) as registry:
        assert registry.tools("opt") == []
        assert registry.servers["fx"].up
        probes = registry.probes()
        assert sorted(probes) == ["mcp:fx", "mcp:req"]  # optional: excluded, not a probe
        await probes["mcp:fx"]()
        with pytest.raises(MCPUnavailable, match="'req' is down"):
            await probes["mcp:req"]()
    assert any("optional MCP server 'opt'" in rec.message for rec in caplog.records)


async def test_refresh_brings_a_recovered_server_back(stdio_registry: MCPRegistry) -> None:
    server = stdio_registry.servers["fx"]
    before = stdio_registry.snapshot_hash()
    server._mark_down("simulated")  # noqa: SLF001 - drive the state machine directly
    assert not server.up
    await server.refresh()
    assert server.up and stdio_registry.snapshot_hash() == before
