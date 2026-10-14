"""``resources.yaml`` adapters: qdrant, neo4j, sql (PRD §3.7, A35, A23, A40).

Offline: SQL on SQLite (``sqlite+aiosqlite``), qdrant in its embedded
``:memory:`` mode, the neo4j read-only guard against a recording fake.
Live (gated): ``VDREAMER_TEST_QDRANT_URL``, ``VDREAMER_TEST_NEO4J_URI`` (+
``_USER``/``_PASSWORD``), ``VDREAMER_TEST_PG_DSN`` for PostgreSQL read-only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from vdreamer.adapters import (
    ADAPTERS,
    Neo4jAdapter,
    QdrantAdapter,
    ReadOnlyNeo4jDriver,
    ReadOnlyQdrant,
    ReadOnlyViolation,
    ResourceAdapter,
    ResourceUnavailable,
    SqlAdapter,
    open_resources,
    probe_resource,
    register_adapter,
    resource_adapters,
)
from vdreamer.network import NetworkAllowlist
from vdreamer.resources import ResourcesError, parse_resources

# -- config load -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"type": "qdrant"}, "exactly one of url, path, location"),
        ({"type": "qdrant", "url": "https://q:6333", "path": "/x"}, "exactly one of"),
        ({"type": "qdrant", "url": "ftp://q"}, "http/https URL"),
        ({"type": "qdrant", "location": "elsewhere"}, "location must be ':memory:'"),
        ({"type": "qdrant", "url": "https://q", "colour": "red"}, "unknown key"),
        ({"type": "neo4j"}, "needs uri"),
        ({"type": "neo4j", "uri": "http://n:7474"}, "bolt"),
        ({"type": "neo4j", "uri": "bolt://n", "user_env": "U"}, "go together"),
        ({"type": "neo4j", "uri": "bolt://u:p@n"}, "credentials in the URL"),
        ({"type": "sql"}, "exactly one of url_env"),
        ({"type": "sql", "url": "postgresql+asyncpg://u:pw@db/x"}, "password in url"),
        ({"type": "sql", "url": "sqlite+aiosqlite://", "read_only": "yes"}, "true or false"),
        (
            {"type": "sql", "url": "mssql+aioodbc://db/x", "read_only": True},
            "read_only is not enforceable for 'mssql'",
        ),
        ({"type": "sql", "url_env": "DWH", "pool_size": -1}, "non-negative integer"),
    ],
)
def test_bad_entries_fail_at_load(entry: dict[str, Any], message: str) -> None:
    with pytest.raises(ResourcesError, match=message):
        resource_adapters(parse_resources({"r": entry}))


def test_secrets_never_inline() -> None:
    with pytest.raises(ResourcesError, match="secret in YAML"):
        parse_resources({"r": {"type": "neo4j", "uri": "bolt://n", "password": "pw"}})


def test_unknown_types_name_the_known_ones() -> None:
    with pytest.raises(ResourcesError, match=r"no adapter for type 'redis'.*mcp.*neo4j"):
        resource_adapters(parse_resources({"cache": {"type": "redis", "url": "redis://x"}}))


def test_mcp_entries_are_not_resources() -> None:
    adapters = resource_adapters(
        parse_resources({"tools": {"type": "mcp", "url": "https://mcp.internal/mcp"}})
    )
    assert adapters == {}


def test_endpoints_are_checked_against_the_allowlist() -> None:
    allow = NetworkAllowlist.parse("*.internal")
    resource_adapters(
        parse_resources({"v": {"type": "qdrant", "url": "https://q.internal:6333"}}),
        allowlist=allow,
    )
    for entry in (
        {"type": "qdrant", "url": "https://q.example.com"},
        {"type": "neo4j", "uri": "bolt://graph.example.com"},
        {"type": "sql", "url": "postgresql+asyncpg://dwh.example.com/x"},
    ):
        with pytest.raises(ResourcesError, match="NETWORK_ALLOWLIST"):
            resource_adapters(parse_resources({"r": entry}), allowlist=allow)


# -- sql ---------------------------------------------------------------------------------


def _sql(tmp_path: Path, **extra: Any) -> SqlAdapter:
    url = f"sqlite+aiosqlite:///{(tmp_path / 'dwh.db').as_posix()}"
    (adapter,) = resource_adapters(
        parse_resources({"warehouse": {"type": "sql", "url": url, **extra}})
    ).values()
    assert isinstance(adapter, SqlAdapter)
    return adapter


async def test_sql_is_a_native_async_engine(tmp_path: Path) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncEngine

    adapter = _sql(tmp_path)
    async with open_resources({"warehouse": adapter}, ["warehouse"]) as clients:
        engine = clients["warehouse"]
        assert isinstance(engine, AsyncEngine)
        async with engine.begin() as conn:
            await conn.execute(text("CREATE TABLE sites (id INTEGER, name TEXT)"))
            await conn.execute(text("INSERT INTO sites VALUES (1, 'cairo')"))
        async with engine.connect() as conn:
            assert (await conn.execute(text("SELECT name FROM sites"))).scalar() == "cairo"
    assert adapter.client is None  # disposed on exit


async def test_sql_read_only_sessions_refuse_writes(tmp_path: Path) -> None:
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    writable = _sql(tmp_path)
    async with open_resources({"w": writable}, ["w"]) as clients:
        async with clients["w"].begin() as conn:
            await conn.execute(text("CREATE TABLE t (x INTEGER)"))
    reader = _sql(tmp_path, read_only=True)
    async with open_resources({"warehouse": reader}, ["warehouse"]) as clients:
        engine = clients["warehouse"]
        async with engine.connect() as conn:
            assert (await conn.execute(text("SELECT count(*) FROM t"))).scalar() == 0
        with pytest.raises(OperationalError, match="readonly|read-only|query_only"):
            async with engine.begin() as conn:
                await conn.execute(text("INSERT INTO t VALUES (1)"))


async def test_sql_url_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (adapter,) = resource_adapters(
        parse_resources({"warehouse": {"type": "sql", "url_env": "STUB_DWH_URL"}})
    ).values()
    monkeypatch.delenv("STUB_DWH_URL", raising=False)
    with pytest.raises(ResourceUnavailable, match="STUB_DWH_URL"):
        await probe_resource(adapter)
    monkeypatch.setenv("STUB_DWH_URL", f"sqlite+aiosqlite:///{(tmp_path / 'e.db').as_posix()}")
    assert await probe_resource(adapter) == "sql reachable"


async def test_a_sync_only_driver_is_reported(tmp_path: Path) -> None:
    (adapter,) = resource_adapters(
        parse_resources({"w": {"type": "sql", "url": f"sqlite:///{tmp_path / 's.db'}"}})
    ).values()
    with pytest.raises(ResourceUnavailable, match="resource 'w' \\(sql\\)"):
        await probe_resource(adapter)


# -- qdrant ------------------------------------------------------------------------------


async def test_qdrant_embedded_and_read_only() -> None:
    pytest.importorskip("qdrant_client")
    from qdrant_client import AsyncQdrantClient, models

    (adapter,) = resource_adapters(
        parse_resources({"vectors": {"type": "qdrant", "location": ":memory:"}})
    ).values()
    assert isinstance(adapter, QdrantAdapter)
    async with open_resources({"vectors": adapter}, ["vectors"]) as clients:
        client = clients["vectors"]
        assert isinstance(client, AsyncQdrantClient)
        await client.create_collection(
            "docs", vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT)
        )
        await client.upsert("docs", [models.PointStruct(id=1, vector=[1.0, 0.0])])
        guard = ReadOnlyQdrant(client, "vectors")
        hits = await guard.query_points("docs", query=[1.0, 0.0], limit=1)
        assert [p.id for p in hits.points] == [1]
        assert await guard.count("docs") == models.CountResult(count=1)
        assert await guard.collection_exists("docs")
        for write in ("upsert", "delete", "create_collection", "delete_collection", "set_payload"):
            with pytest.raises(ReadOnlyViolation, match=f"read_only: {write}"):
                getattr(guard, write)("docs")
        assert guard.native is client


# -- neo4j -------------------------------------------------------------------------------


class FakeSession:
    def __init__(self, log: list[Any], **kwargs: Any) -> None:
        self.log = log
        log.append(("session", kwargs))

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.log.append(("closed",))

    async def run(self, query: str) -> str:
        self.log.append(("run", query))
        return "result"

    async def execute_read(self, fn: Any) -> Any:
        return await fn("tx")


class FakeDriver:
    def __init__(self) -> None:
        self.log: list[Any] = []

    def session(self, **kwargs: Any) -> FakeSession:
        return FakeSession(self.log, **kwargs)

    async def execute_query(self, query: str, **kwargs: Any) -> str:
        self.log.append(("execute_query", query, kwargs))
        return "records"

    async def close(self) -> None:
        self.log.append(("driver.close",))


async def test_neo4j_read_only_driver() -> None:
    neo4j = pytest.importorskip("neo4j")
    native = FakeDriver()
    driver = ReadOnlyNeo4jDriver(native, "semantic")
    async with driver.session(database="graph") as session:
        assert await session.run("MATCH (n) RETURN n") == "result"
        assert await session.execute_read(lambda tx: _echo(tx)) == "tx"
        with pytest.raises(ReadOnlyViolation, match="execute_write"):
            await session.execute_write(lambda tx: None)
    assert native.log[0] == ("session", {"database": "graph",
                                         "default_access_mode": neo4j.READ_ACCESS})
    with pytest.raises(ReadOnlyViolation, match="default_access_mode"):
        driver.session(default_access_mode=neo4j.WRITE_ACCESS)
    assert await driver.execute_query("MATCH (n) RETURN n") == "records"
    assert native.log[-1][2]["routing_"] == neo4j.RoutingControl.READ
    with pytest.raises(ReadOnlyViolation, match="routing_"):
        await driver.execute_query("CREATE (n)", routing_=neo4j.RoutingControl.WRITE)
    await driver.close()  # forwarded to the native driver
    assert native.log[-1] == ("driver.close",)


async def _echo(tx: Any) -> Any:
    return tx


async def test_neo4j_connect_wraps_only_when_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    neo4j = pytest.importorskip("neo4j")
    created: list[Any] = []

    def fake_driver(uri: str, **kwargs: Any) -> FakeDriver:
        created.append((uri, kwargs))
        return FakeDriver()

    monkeypatch.setattr(neo4j.AsyncGraphDatabase, "driver", fake_driver)
    monkeypatch.setenv("STUB_NEO4J_USER", "reader")
    monkeypatch.setenv("STUB_NEO4J_PASS", "pw")
    entries = parse_resources(
        {
            "semantic": {"type": "neo4j", "uri": "bolt://graph.internal:7687",
                         "user_env": "STUB_NEO4J_USER", "password_env": "STUB_NEO4J_PASS",
                         "read_only": True, "max_pool_size": 5},
            "rw": {"type": "neo4j", "uri": "neo4j+s://graph.internal"},
        }
    )
    adapters = resource_adapters(entries)
    assert isinstance(adapters["semantic"], Neo4jAdapter)
    client = await adapters["semantic"].connect()
    assert isinstance(client, ReadOnlyNeo4jDriver)
    assert created[0] == ("bolt://graph.internal:7687",
                          {"auth": ("reader", "pw"), "max_connection_pool_size": 5})
    assert isinstance(await adapters["rw"].connect(), FakeDriver)


# -- the pool and custom types -----------------------------------------------------------


class EchoAdapter(ResourceAdapter):
    type = "echo"
    keys = frozenset({"value", "fail"})
    events: list[str] = []

    async def connect(self) -> Any:
        EchoAdapter.events.append(f"connect {self.alias}")
        return {"value": self.entry.get("value")}

    async def probe(self) -> None:
        if self.entry.get("fail"):
            raise ConnectionError("boom")

    async def close(self) -> None:
        EchoAdapter.events.append(f"close {self.alias}")


async def test_a_new_type_is_one_adapter_class() -> None:
    register_adapter(EchoAdapter)
    try:
        EchoAdapter.events = []
        adapters = resource_adapters(
            parse_resources(
                {"a": {"type": "echo", "value": 1}, "b": {"type": "echo", "fail": True},
                 "tools": {"type": "mcp", "url": "https://m.internal/mcp"}}
            )
        )
        async with open_resources(adapters, ["a", "tools"]) as clients:
            assert clients == {"a": {"value": 1}}
        assert EchoAdapter.events == ["connect a", "close a"]
        EchoAdapter.events = []
        with pytest.raises(ResourceUnavailable, match="resource 'b' \\(echo\\): ConnectionError"):
            async with open_resources(adapters, ["a", "b"]):
                pass
        # What was opened before the failure is closed again.
        assert EchoAdapter.events == ["connect a", "connect b", "close b", "close a"]
    finally:
        ADAPTERS.pop("echo", None)
    with pytest.raises(TypeError, match="reserved"):

        class Mcp(EchoAdapter):
            type = "mcp"

        register_adapter(Mcp)


# -- live --------------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("VDREAMER_TEST_QDRANT_URL"),
                    reason="set VDREAMER_TEST_QDRANT_URL for a live qdrant")
async def test_live_qdrant() -> None:
    from qdrant_client import models

    (adapter,) = resource_adapters(
        parse_resources({"v": {"type": "qdrant", "url": os.environ["VDREAMER_TEST_QDRANT_URL"],
                               "read_only": True}})
    ).values()
    assert await probe_resource(adapter) == "qdrant reachable (read_only)"
    async with open_resources({"v": adapter}, ["v"]) as clients:
        native = clients["v"].native
        if await native.collection_exists("vdreamer_live"):
            await native.delete_collection("vdreamer_live")
        params = models.VectorParams(size=2, distance=models.Distance.DOT)
        await native.create_collection("vdreamer_live", vectors_config=params)
        await native.upsert("vdreamer_live", [models.PointStruct(id=7, vector=[0.0, 1.0])])
        hits = await clients["v"].query_points("vdreamer_live", query=[0.0, 1.0], limit=1)
        assert [p.id for p in hits.points] == [7]
        with pytest.raises(ReadOnlyViolation):
            await clients["v"].delete_collection("vdreamer_live")
        await native.delete_collection("vdreamer_live")


@pytest.mark.skipif(not os.environ.get("VDREAMER_TEST_NEO4J_URI"),
                    reason="set VDREAMER_TEST_NEO4J_URI (+ _USER/_PASSWORD) for a live neo4j")
async def test_live_neo4j_read_only_is_enforced_by_the_server() -> None:
    import neo4j

    def entry(read_only: bool) -> dict[str, Any]:
        return {"type": "neo4j", "uri": os.environ["VDREAMER_TEST_NEO4J_URI"],
                "user_env": "VDREAMER_TEST_NEO4J_USER",
                "password_env": "VDREAMER_TEST_NEO4J_PASSWORD", "read_only": read_only}

    adapters = resource_adapters(parse_resources({"ro": entry(True), "rw": entry(False)}))
    async with open_resources(adapters, ["ro", "rw"]) as clients:
        await clients["rw"].execute_query("MERGE (:VdreamerLive {k: 1})")
        records, _, _ = await clients["ro"].execute_query(
            "MATCH (n:VdreamerLive) RETURN count(n) AS c"
        )
        assert records[0]["c"] >= 1
        # The guard opens READ sessions; the server rejects the write itself.
        async with clients["ro"].session() as session:
            with pytest.raises(neo4j.exceptions.ClientError):
                result = await session.run("CREATE (:VdreamerLive {k: 2})")
                await result.consume()
        await clients["rw"].execute_query("MATCH (n:VdreamerLive) DETACH DELETE n")


@pytest.mark.skipif(not os.environ.get("VDREAMER_TEST_PG_DSN"),
                    reason="set VDREAMER_TEST_PG_DSN for a live PostgreSQL")
async def test_live_postgres_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("psycopg")
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError

    dsn = os.environ["VDREAMER_TEST_PG_DSN"]
    url = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    monkeypatch.setenv("STUB_PG_URL", url)
    adapters = resource_adapters(
        parse_resources(
            {
                "dwh": {"type": "sql", "url_env": "STUB_PG_URL", "read_only": True},
                "admin": {"type": "sql", "url_env": "STUB_PG_URL"},
            }
        )
    )
    async with open_resources(adapters, ["admin"]) as clients:
        async with clients["admin"].begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS vdreamer_ro"))
    async with open_resources(adapters, ["dwh"]) as clients:
        async with clients["dwh"].connect() as conn:
            assert (await conn.execute(text("SELECT 1"))).scalar() == 1
        with pytest.raises(DBAPIError, match="read-only"):
            async with clients["dwh"].begin() as conn:
                await conn.execute(text("CREATE TABLE vdreamer_ro (x int)"))
        # Every pooled connection, not just the first, stays read-only.
        for _ in range(3):
            async with clients["dwh"].connect() as conn:
                mode = await conn.execute(text("SHOW default_transaction_read_only"))
                assert mode.scalar() == "on"
