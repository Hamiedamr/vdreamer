"""``resources.yaml`` adapters: qdrant, neo4j, sql (PRD §3.7, A35).

Workers open one pooled async client per declared resource at start and
inject it, native and unwrapped, as ``ctx.resources[alias]``: an
``AsyncQdrantClient``, a neo4j ``AsyncDriver``, a SQLAlchemy ``AsyncEngine``.
Query APIs are not wrapped. ``type: mcp`` entries are tools, not resources
(``vdreamer.mcp``, §2.7).

A new type is one :class:`ResourceAdapter` subclass — ``connect`` /
``probe`` / ``close`` — registered with :func:`register_adapter`; its
``__init__`` validates the entry at config load.

``read_only: true`` forces read sessions (A35):

- **sql:** every pooled connection is put in read-only mode as it opens
  (PostgreSQL ``SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY``,
  MySQL/MariaDB ``SET SESSION TRANSACTION READ ONLY``, SQLite
  ``PRAGMA query_only``); other dialects are refused. SQL can switch a
  session back, so read-only *credentials* stay the real control — this is
  the second line.
- **neo4j:** the driver comes back behind a guard that opens sessions in
  ``READ_ACCESS`` and routes ``execute_query`` to readers; the server then
  rejects writes. ``execute_write`` and an explicit write access mode raise.
- **qdrant:** the client comes back behind a guard that lets read methods
  through (search, query, scroll, count, retrieve, ``get_*``, …) and raises
  on every other method. Pair it with a read-only API key.

The guards forward everything else to the native object (``.native``).

Secrets never appear in YAML (A23): ``api_key_env``, ``user_env``,
``password_env``, ``url_env`` name environment variables. Endpoints must
match ``NETWORK_ALLOWLIST`` (A40) at load and at connect.
"""

from __future__ import annotations

import abc
import asyncio
import os
import urllib.parse
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, ClassVar, Final

from vdreamer.network import (
    EndpointError,
    EndpointUnavailable,
    NetworkAllowlist,
    check_url,
    default_port,
)
from vdreamer.resources import ResourcesError

#: Types handled elsewhere: their entries are not resources (§2.7).
NON_RESOURCE_TYPES: Final = frozenset({"mcp"})


class ResourceUnavailable(RuntimeError):
    """A resource could not be connected or failed its probe."""


class ReadOnlyViolation(PermissionError):
    """A write was attempted through a ``read_only: true`` resource."""


# ---------------------------------------------------------------------------
# The adapter contract
# ---------------------------------------------------------------------------


class ResourceAdapter(abc.ABC):
    """One ``resources.yaml`` type: validate, ``connect``, ``probe``, ``close``.

    ``__init__`` validates the entry (config load, no I/O) and raises
    :class:`~vdreamer.resources.ResourcesError`. ``connect`` returns the client
    agents receive; ``probe`` proves it works; ``close`` releases the pool.
    """

    type: ClassVar[str]
    #: Keys an entry may carry besides ``type`` and ``read_only``.
    keys: ClassVar[frozenset[str]] = frozenset()
    #: Keys that must be present.
    required: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        alias: str,
        entry: Mapping[str, Any],
        *,
        allowlist: NetworkAllowlist | None = None,
    ) -> None:
        self.alias = alias
        self.entry = dict(entry)
        self.allowlist = allowlist
        unknown = sorted(set(map(str, entry)) - self.keys - {"type", "read_only"})
        if unknown:
            raise self.error(f"unknown key(s) {unknown} (known: {sorted(self.keys)})")
        missing = sorted(key for key in self.required if not entry.get(key))
        if missing:
            raise self.error(f"needs {', '.join(missing)}")
        read_only = entry.get("read_only", False)
        if not isinstance(read_only, bool):
            raise self.error("read_only must be true or false")
        self.read_only: bool = read_only
        self.client: Any = None

    def error(self, message: str) -> ResourcesError:
        return ResourcesError(f"resource {self.alias!r} ({self.type}): {message}")

    def unavailable(self, message: str) -> ResourceUnavailable:
        return ResourceUnavailable(f"resource {self.alias!r} ({self.type}): {message}")

    def env(self, key: str) -> str | None:
        """The value of the environment variable an ``*_env`` key names."""
        name = self.entry.get(key)
        if name is None:
            return None
        value = os.environ.get(str(name), "")
        if not value:
            raise self.unavailable(f"environment variable {name} ({key}) is not set")
        return value

    def check_url(self, key: str, url: str, schemes: Sequence[str]) -> urllib.parse.SplitResult:
        try:
            return check_url(f"resource {self.alias!r} {key}", url, self.allowlist,
                             schemes=schemes)
        except EndpointError as exc:
            raise self.error(str(exc).split(": ", 1)[-1]) from exc

    async def check_connect(self, parsed: urllib.parse.SplitResult, port: int) -> None:
        if self.allowlist is None or not parsed.hostname:
            return
        try:
            await self.allowlist.check_connect(parsed.hostname, default_port(parsed, port))
        except EndpointUnavailable as exc:
            raise self.unavailable(str(exc)) from exc

    @abc.abstractmethod
    async def connect(self) -> Any:
        """Open the pooled client; returns what ``ctx.resources[alias]`` holds."""

    @abc.abstractmethod
    async def probe(self) -> None:
        """Prove the client works; raise on failure."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release the pool."""


ADAPTERS: dict[str, type[ResourceAdapter]] = {}


def register_adapter(cls: type[ResourceAdapter]) -> type[ResourceAdapter]:
    """Register a resource type (usable as a class decorator)."""
    if not getattr(cls, "type", None):
        raise TypeError(f"{cls.__name__} needs a 'type'")
    if cls.type in NON_RESOURCE_TYPES:
        raise TypeError(f"{cls.type!r} is reserved")
    ADAPTERS[cls.type] = cls
    return cls


def _bool(adapter: ResourceAdapter, key: str, default: bool) -> bool:
    value = adapter.entry.get(key, default)
    if not isinstance(value, bool):
        raise adapter.error(f"{key} must be true or false")
    return value


def _number(adapter: ResourceAdapter, key: str, default: float | None) -> float | None:
    value = adapter.entry.get(key, default)
    if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)
                              or value <= 0):
        raise adapter.error(f"{key} must be a positive number")
    return value


def _int(adapter: ResourceAdapter, key: str, default: int | None) -> int | None:
    value = adapter.entry.get(key, default)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise adapter.error(f"{key} must be a non-negative integer")
    return value


def _env_name(adapter: ResourceAdapter, key: str) -> None:
    value = adapter.entry.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise adapter.error(f"{key} must name an environment variable")


# ---------------------------------------------------------------------------
# Read-only guards
# ---------------------------------------------------------------------------


class _Guard:
    """Forwards to ``native``; subclasses intercept the write paths."""

    def __init__(self, native: Any, alias: str) -> None:
        object.__setattr__(self, "native", native)
        object.__setattr__(self, "_alias", alias)

    def _refuse(self, what: str) -> ReadOnlyViolation:
        return ReadOnlyViolation(f"resource {self._alias!r} is read_only: {what} is not allowed")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    def __repr__(self) -> str:
        return f"<read-only {self.native!r}>"


class ReadOnlyQdrant(_Guard):
    """An ``AsyncQdrantClient`` limited to its read methods."""

    READ_PREFIXES: ClassVar[tuple[str, ...]] = (
        "search", "query", "recommend", "discover", "scroll", "count", "retrieve",
        "get_", "list_", "collection_exists", "facet", "info",
    )
    ALWAYS: ClassVar[frozenset[str]] = frozenset({"close"})

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.native, name)
        if (
            callable(attr)
            and not name.startswith("_")
            and name not in self.ALWAYS
            and not name.startswith(self.READ_PREFIXES)
        ):
            def refused(*args: Any, **kwargs: Any) -> Any:
                raise self._refuse(f"{name}()")

            return refused
        return attr


class _ReadOnlyNeo4jSession(_Guard):
    async def __aenter__(self) -> _ReadOnlyNeo4jSession:
        await self.native.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self.native.__aexit__(*exc)

    async def execute_write(self, *args: Any, **kwargs: Any) -> Any:
        raise self._refuse("execute_write()")

    async def write_transaction(self, *args: Any, **kwargs: Any) -> Any:
        raise self._refuse("write_transaction()")


class ReadOnlyNeo4jDriver(_Guard):
    """A neo4j ``AsyncDriver`` whose sessions and queries run in read access mode."""

    def session(self, **kwargs: Any) -> _ReadOnlyNeo4jSession:
        import neo4j

        mode = kwargs.get("default_access_mode", neo4j.READ_ACCESS)
        if mode != neo4j.READ_ACCESS:
            raise self._refuse(f"default_access_mode={mode!r}")
        kwargs["default_access_mode"] = neo4j.READ_ACCESS
        return _ReadOnlyNeo4jSession(self.native.session(**kwargs), self._alias)

    async def execute_query(self, *args: Any, routing_: Any = None, **kwargs: Any) -> Any:
        import neo4j

        if routing_ is not None and routing_ != neo4j.RoutingControl.READ:
            raise self._refuse(f"routing_={routing_!r}")
        return await self.native.execute_query(
            *args, routing_=neo4j.RoutingControl.READ, **kwargs
        )


# ---------------------------------------------------------------------------
# qdrant
# ---------------------------------------------------------------------------


@register_adapter
class QdrantAdapter(ResourceAdapter):
    """``AsyncQdrantClient``: ``url`` (server), or ``path``/``location: ":memory:"``
    (qdrant's embedded local mode, for ``vdreamer dev`` and tests)."""

    type = "qdrant"
    keys = frozenset({"url", "path", "location", "api_key_env", "timeout_s", "prefer_grpc"})

    def __init__(self, alias: str, entry: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(alias, entry, **kwargs)
        targets = [key for key in ("url", "path", "location") if entry.get(key)]
        if len(targets) != 1:
            raise self.error("needs exactly one of url, path, location")
        self._parsed: urllib.parse.SplitResult | None = None
        if entry.get("url"):
            self._parsed = self.check_url("url", str(entry["url"]), ("http", "https"))
        if entry.get("location") not in (None, ":memory:"):
            raise self.error("location must be ':memory:' (use url for a server)")
        _env_name(self, "api_key_env")
        self.timeout_s = _number(self, "timeout_s", None)
        self.prefer_grpc = _bool(self, "prefer_grpc", False)

    async def connect(self) -> Any:
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError as exc:
            raise self.unavailable("needs the [qdrant] extra") from exc
        kwargs: dict[str, Any] = {}
        if self._parsed is not None:
            await self.check_connect(self._parsed, 6333)
            kwargs["url"] = self.entry["url"]
            kwargs["prefer_grpc"] = self.prefer_grpc
            api_key = self.env("api_key_env")
            if api_key:
                kwargs["api_key"] = api_key
        elif self.entry.get("path"):
            kwargs["path"] = str(self.entry["path"])
        else:
            kwargs["location"] = ":memory:"
        if self.timeout_s is not None:
            kwargs["timeout"] = int(self.timeout_s)
        self.client = AsyncQdrantClient(**kwargs)
        return ReadOnlyQdrant(self.client, self.alias) if self.read_only else self.client

    async def probe(self) -> None:
        await self.client.get_collections()

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None


# ---------------------------------------------------------------------------
# neo4j
# ---------------------------------------------------------------------------


@register_adapter
class Neo4jAdapter(ResourceAdapter):
    """A neo4j ``AsyncDriver`` (``bolt://``/``neo4j://`` and their ``+s``/``+ssc``)."""

    type = "neo4j"
    keys = frozenset(
        {"uri", "user_env", "password_env", "database", "max_pool_size", "connect_timeout_s"}
    )
    required = frozenset({"uri"})
    SCHEMES: ClassVar[tuple[str, ...]] = (
        "bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc",
    )

    def __init__(self, alias: str, entry: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(alias, entry, **kwargs)
        self._parsed = self.check_url("uri", str(entry["uri"]), self.SCHEMES)
        _env_name(self, "user_env")
        _env_name(self, "password_env")
        if bool(entry.get("user_env")) != bool(entry.get("password_env")):
            raise self.error("user_env and password_env go together")
        database = entry.get("database")
        if database is not None and (not isinstance(database, str) or not database):
            raise self.error("database must be a name")
        self.max_pool_size = _int(self, "max_pool_size", None)
        self.connect_timeout_s = _number(self, "connect_timeout_s", None)

    async def connect(self) -> Any:
        try:
            import neo4j
        except ImportError as exc:
            raise self.unavailable("needs the [neo4j] extra") from exc
        await self.check_connect(self._parsed, 7687)
        kwargs: dict[str, Any] = {}
        user = self.env("user_env")
        if user is not None:
            kwargs["auth"] = (user, self.env("password_env"))
        if self.max_pool_size is not None:
            kwargs["max_connection_pool_size"] = self.max_pool_size
        if self.connect_timeout_s is not None:
            kwargs["connection_timeout"] = self.connect_timeout_s
        self.client = neo4j.AsyncGraphDatabase.driver(str(self.entry["uri"]), **kwargs)
        return ReadOnlyNeo4jDriver(self.client, self.alias) if self.read_only else self.client

    async def probe(self) -> None:
        database = self.entry.get("database")
        await self.client.verify_connectivity(**({"database": database} if database else {}))

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None


# ---------------------------------------------------------------------------
# sql
# ---------------------------------------------------------------------------

#: Per-dialect statement that makes a whole session read-only.
READ_ONLY_SESSION_SQL: Final[Mapping[str, str]] = {
    "postgresql": "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY",
    "mysql": "SET SESSION TRANSACTION READ ONLY",
    "mariadb": "SET SESSION TRANSACTION READ ONLY",
    "sqlite": "PRAGMA query_only = ON",
}


@register_adapter
class SqlAdapter(ResourceAdapter):
    """A SQLAlchemy ``AsyncEngine`` (``url_env``, or a credential-free ``url``)."""

    type = "sql"
    keys = frozenset({"url", "url_env", "pool_size", "max_overflow", "pool_timeout_s"})

    def __init__(self, alias: str, entry: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(alias, entry, **kwargs)
        if bool(entry.get("url")) == bool(entry.get("url_env")):
            raise self.error("needs exactly one of url_env (credentials) or url")
        _env_name(self, "url_env")
        if entry.get("url"):
            self._check_sql_url(str(entry["url"]), config=True)
        self.pool_size = _int(self, "pool_size", None)
        self.max_overflow = _int(self, "max_overflow", None)
        self.pool_timeout_s = _number(self, "pool_timeout_s", None)

    def _check_sql_url(self, url: str, *, config: bool) -> Any:
        try:
            from sqlalchemy.engine import make_url
            from sqlalchemy.exc import ArgumentError
        except ImportError as exc:
            if config:
                return None  # checked again at connect, where the extra is required
            raise self.unavailable("needs the [sql] extra (SQLAlchemy)") from exc
        try:
            parsed = make_url(url)
        except ArgumentError as exc:
            raise self.error(f"cannot parse the database URL: {exc}") from exc
        if config and parsed.password:
            raise self.error("a password in url puts a secret in YAML (A23); use url_env")
        if self.read_only and parsed.get_backend_name() not in READ_ONLY_SESSION_SQL:
            raise self.error(
                f"read_only is not enforceable for {parsed.get_backend_name()!r} "
                f"(supported: {sorted(READ_ONLY_SESSION_SQL)}); connect with read-only "
                "credentials and drop read_only"
            )
        if self.allowlist is not None and parsed.host:
            try:
                self.allowlist.check_config(parsed.host)
            except EndpointError as exc:
                raise self.error(str(exc)) from exc
        return parsed

    async def connect(self) -> Any:
        try:
            from sqlalchemy import event
            from sqlalchemy.ext.asyncio import create_async_engine
        except ImportError as exc:
            raise self.unavailable("needs the [sql] extra (SQLAlchemy)") from exc
        url = str(self.entry.get("url") or self.env("url_env"))
        parsed = self._check_sql_url(url, config=False)
        if self.allowlist is not None and parsed.host:
            try:
                await self.allowlist.check_connect(parsed.host, parsed.port or 0)
            except EndpointUnavailable as exc:
                raise self.unavailable(str(exc)) from exc
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if self.pool_size is not None:
            kwargs["pool_size"] = self.pool_size
        if self.max_overflow is not None:
            kwargs["max_overflow"] = self.max_overflow
        if self.pool_timeout_s is not None:
            kwargs["pool_timeout"] = self.pool_timeout_s
        try:
            engine = create_async_engine(parsed, **kwargs)
        except Exception as exc:  # unknown or sync-only driver, bad pool args
            raise self.unavailable(f"{type(exc).__name__}: {exc}") from exc
        if self.read_only:
            statement = READ_ONLY_SESSION_SQL[engine.dialect.name]

            def read_only_session(dbapi_connection: Any, _record: Any) -> None:
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute(statement)
                finally:
                    cursor.close()
                # PostgreSQL rolls a SET back with its transaction, and the pool
                # rolls back every fresh connection: commit so the setting sticks.
                dbapi_connection.commit()

            event.listen(engine.sync_engine, "connect", read_only_session)
        self.client = engine
        return engine

    async def probe(self) -> None:
        from sqlalchemy import text

        probe = "SELECT 1 FROM DUAL" if self.client.dialect.name == "oracle" else "SELECT 1"
        async with self.client.connect() as connection:
            await connection.execute(text(probe))

    async def close(self) -> None:
        if self.client is not None:
            await self.client.dispose()
            self.client = None


# ---------------------------------------------------------------------------
# Loading and the worker's pool
# ---------------------------------------------------------------------------


def resource_adapters(
    resources: Mapping[str, Mapping[str, Any]],
    *,
    allowlist: NetworkAllowlist | None = None,
) -> dict[str, ResourceAdapter]:
    """Validate every non-MCP entry of a loaded ``resources.yaml`` (config load)."""
    adapters: dict[str, ResourceAdapter] = {}
    for alias, entry in resources.items():
        kind = str(entry.get("type"))
        if kind in NON_RESOURCE_TYPES:
            continue
        cls = ADAPTERS.get(kind)
        if cls is None:
            raise ResourcesError(
                f"resource {alias!r}: no adapter for type {kind!r} (known: "
                f"{sorted([*ADAPTERS, *NON_RESOURCE_TYPES])}; register_adapter adds one)"
            )
        adapters[alias] = cls(alias, entry, allowlist=allowlist)
    return adapters


def missing_resources(
    resources: Mapping[str, Mapping[str, Any]] | None, declared: Sequence[str]
) -> list[str]:
    known = resources or {}
    return [alias for alias in declared if alias not in known]


@asynccontextmanager
async def open_resources(
    adapters: Mapping[str, ResourceAdapter],
    declared: Sequence[str],
    *,
    probe: bool = True,
    on_open: Callable[[ResourceAdapter], None] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Connect (and probe) the declared resources for the block's lifetime.

    MCP aliases among ``declared`` are skipped: they reach the agent as tools.
    A failure closes what was already opened and raises
    :class:`ResourceUnavailable`.
    """
    opened: list[ResourceAdapter] = []
    clients: dict[str, Any] = {}
    try:
        for alias in declared:
            adapter = adapters.get(alias)
            if adapter is None:
                continue  # an MCP server; checked by the caller
            try:
                clients[alias] = await adapter.connect()
                opened.append(adapter)
                if probe:
                    await adapter.probe()
            except (ResourceUnavailable, ResourcesError):
                raise
            except Exception as exc:
                raise adapter.unavailable(f"{type(exc).__name__}: {exc}") from exc
            if on_open is not None:
                on_open(adapter)
        yield clients
    finally:
        for adapter in reversed(opened):
            try:
                await adapter.close()
            except Exception:  # closing is best effort; the process is going down
                pass


async def probe_resource(adapter: ResourceAdapter, *, timeout_s: float = 10.0) -> str:
    """Connect, probe, close: one resource's readiness (validate/doctor)."""
    try:
        await asyncio.wait_for(adapter.connect(), timeout_s)
        await asyncio.wait_for(adapter.probe(), timeout_s)
    except (ResourceUnavailable, ResourcesError):
        raise
    except TimeoutError as exc:
        raise adapter.unavailable(f"no answer within {timeout_s:g}s") from exc
    except Exception as exc:
        raise adapter.unavailable(f"{type(exc).__name__}: {exc}") from exc
    finally:
        try:
            await adapter.close()
        except Exception:
            pass
    mode = " (read_only)" if adapter.read_only else ""
    return f"{adapter.type} reachable{mode}"


__all__ = [
    "ADAPTERS",
    "READ_ONLY_SESSION_SQL",
    "Neo4jAdapter",
    "QdrantAdapter",
    "ReadOnlyNeo4jDriver",
    "ReadOnlyQdrant",
    "ReadOnlyViolation",
    "ResourceAdapter",
    "ResourceUnavailable",
    "SqlAdapter",
    "missing_resources",
    "open_resources",
    "probe_resource",
    "register_adapter",
    "resource_adapters",
]
