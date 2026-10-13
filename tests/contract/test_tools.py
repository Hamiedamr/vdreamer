"""Tests for vdreamer.tools — governed-tool primitives (PRD §2.7, A33, A37)."""

from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolCallRequest

from vdreamer.tools import (
    DEFAULT_REDACT_KEYS,
    VDREAMER_METADATA_KEY,
    ApprovalSpec,
    InterruptPolicy,
    Policy,
    VDreamerToolMiddleware,
    _govern_tool,
    redact_args,
    resolve_policy,
    tool,
)


class _FakeToolCallingModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel plus bind_tools (the venv fakes lack it)."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def _stub_request() -> ToolCallRequest:
    """A minimal, realistic ToolCallRequest (langgraph.prebuilt.tool_node)."""
    return ToolCallRequest(
        tool_call={"name": "echo", "args": {"text": "hi"}, "id": "call_1", "type": "tool_call"},
        tool=None,
        state={"messages": []},
        runtime=None,
    )


# ---------------------------------------------------------------- decorator


def test_bare_decorator_creates_governed_basetool() -> None:
    @tool
    def ping(value: str) -> str:
        """Ping the value."""
        return f"pong: {value}"

    assert isinstance(ping, BaseTool)
    assert ping.name == "ping"
    assert ping.description
    assert ping.invoke({"value": "hello"}) == "pong: hello"
    # Bare @tool still carries a (default) policy so downstream code never
    # has to special-case metadata-less tools.
    assert ping.metadata[VDREAMER_METADATA_KEY] == Policy().to_metadata()
    assert Policy.from_metadata(ping.metadata[VDREAMER_METADATA_KEY]) == Policy()


def test_configured_decorator_stores_policy_metadata() -> None:
    @tool(redact=["query"], timeout_s=30, side_effects=True)
    def run_query(query: str) -> str:
        """Run a read-only query."""
        return "ok"

    assert isinstance(run_query, BaseTool)
    assert run_query.name == "run_query"
    assert run_query.invoke({"query": "SELECT 1"}) == "ok"
    expected = Policy(redact=("query",), timeout_s=30.0, side_effects=True)
    assert run_query.metadata[VDREAMER_METADATA_KEY] == expected.to_metadata()
    assert Policy.from_metadata(run_query.metadata[VDREAMER_METADATA_KEY]) == expected


def test_policy_round_trips_through_metadata() -> None:
    spec = ApprovalSpec(
        kind="sql_review",
        approver_role="dba",
        scope="db:write:prod",
        ttl_hours=12,
    )
    policy = Policy(
        redact=("query", "dsn"),
        timeout_s=15.0,
        side_effects=True,
        approval=spec,
        max_result_chars=500,
        fail_run=True,
        scope="db:write",
        requires_roles=("dba", "analyst"),
    )
    meta = policy.to_metadata()
    assert meta == {
        "redact": ["query", "dsn"],
        "timeout_s": 15.0,
        "side_effects": True,
        "approval": {
            "kind": "sql_review",
            "approver_role": "dba",
            "scope": "db:write:prod",
            "ttl_hours": 12,
        },
        "max_result_chars": 500,
        "fail_run": True,
        "scope": "db:write",
        "requires_roles": ["dba", "analyst"],
    }
    assert json.loads(json.dumps(meta)) == meta  # JSON-safe
    assert Policy.from_metadata(meta) == policy  # round trip, approval included


def test_policy_kwarg_wins_over_individual_kwargs() -> None:
    @tool(policy=Policy(redact=("q",), timeout_s=5.0), redact=["other"], timeout_s=99.0)
    def search(q: str) -> str:
        """Search."""
        return "hit"

    expected = Policy(redact=("q",), timeout_s=5.0)
    assert search.metadata[VDREAMER_METADATA_KEY] == expected.to_metadata()


def test_tool_kwargs_name_and_description_pass_through() -> None:
    @tool(name="renamed_tool", description="A renamed tool.")
    def original(x: int) -> int:
        """Original docstring."""
        return x

    assert isinstance(original, BaseTool)
    assert original.name == "renamed_tool"
    assert original.description == "A renamed tool."
    assert original.invoke({"x": 3}) == 3
    # Positional name form is also a drop-in for langchain's @tool.

    def plain(x: int) -> int:
        """Plain."""
        return x

    positional = tool("positional_name", description="Via positional name.")(plain)
    assert isinstance(positional, BaseTool)
    assert positional.name == "positional_name"
    assert positional.description == "Via positional name."
    assert positional.invoke({"x": 3}) == 3


def test_governed_metadata_merge_preserves_unrelated_keys() -> None:
    def identity(x: int) -> int:
        """Identity."""
        return x

    base = tool(identity)
    assert isinstance(base, BaseTool)
    base.metadata = {"owner": "team-a"}  # set by whatever came before vdreamer
    governed = _govern_tool(base, Policy(timeout_s=12.0))
    assert governed.metadata["owner"] == "team-a"
    assert governed.metadata[VDREAMER_METADATA_KEY] == Policy(timeout_s=12.0).to_metadata()


# --------------------------------------------------------------- redact_args


def test_redact_args_nested_masking() -> None:
    args = {
        "a": {"api_key": "x", "inner": [{"token": "y"}, 1]},
        "password": "top",
        "keep": "visible",
    }
    redacted = redact_args(args)
    assert redacted == {
        "a": {"api_key": "***", "inner": [{"token": "***"}, 1]},
        "password": "***",
        "keep": "visible",
    }
    # Input untouched.
    assert args == {
        "a": {"api_key": "x", "inner": [{"token": "y"}, 1]},
        "password": "top",
        "keep": "visible",
    }


def test_redact_args_case_insensitive_and_defaults() -> None:
    assert redact_args({"API_KEY": "x", "Token": "y"}) == {"API_KEY": "***", "Token": "***"}
    # Every default key is masked even when keys is None.
    for key in DEFAULT_REDACT_KEYS:
        assert redact_args({key: "v"})[key] == "***"
    # Keys outside both sets survive.
    assert redact_args({"safe": "v"}) == {"safe": "v"}


def test_redact_args_custom_keys_still_include_defaults() -> None:
    redacted = redact_args({"dsn": "pg://", "api_key": "k"}, keys=["dsn"])
    assert redacted == {"dsn": "***", "api_key": "***"}


# ------------------------------------------------------------ resolve_policy


def test_resolve_policy_exact_and_glob_match() -> None:
    decorated = Policy(redact=("q",))
    policies = {
        "sql_db_query": Policy(timeout_s=5.0),
        "sql_db_*": Policy(side_effects=True),
    }
    # Exact key matches (and "sql_db_query" also matches the glob, so both
    # non-default fields apply).
    exact = resolve_policy("sql_db_query", decorated, policies)
    assert exact.timeout_s == 5.0
    assert exact.side_effects is True
    assert exact.redact == ("q",)
    # Glob-only match keeps the decorator's untouched fields.
    globbed = resolve_policy("sql_db_list", decorated, policies)
    assert globbed.side_effects is True
    assert globbed.timeout_s == 60.0  # class default: glob left it unspecified
    assert globbed.redact == ("q",)


def test_resolve_policy_overrides_only_non_default_fields() -> None:
    decorated = Policy(timeout_s=30.0, redact=("query",))
    # A field equal to the class default counts as "unspecified", so the
    # decorator's 30 survives Policy(timeout_s=60) (60 == the default 60.0).
    same_as_default = resolve_policy("sql_db_query", decorated, {"sql_db_*": Policy(timeout_s=60)})
    assert same_as_default.timeout_s == 30.0
    assert same_as_default.redact == ("query",)
    # A genuinely non-default field does override the decorator.
    overridden = resolve_policy("sql_db_query", decorated, {"sql_db_*": Policy(timeout_s=5.0)})
    assert overridden.timeout_s == 5.0
    assert overridden.redact == ("query",)
    # approval=None never overrides a decorator's approval.
    with_approval = Policy(approval=ApprovalSpec(kind="sql_review"))
    kept = resolve_policy("sql_db_query", with_approval, {"sql_db_*": Policy(timeout_s=5.0)})
    assert kept.approval == ApprovalSpec(kind="sql_review")
    # A glob's approval overrides only when set.
    replaced = resolve_policy(
        "sql_db_query",
        decorated,
        {"sql_db_*": Policy(approval=ApprovalSpec(kind="tool_review"))},
    )
    assert replaced.approval == ApprovalSpec(kind="tool_review")


def test_resolve_policy_no_match_and_no_decorated() -> None:
    decorated = Policy(timeout_s=30.0, redact=("q",))
    policies = {"sql_db_*": Policy(timeout_s=5.0)}
    # No match: decorator policy comes back verbatim.
    assert resolve_policy("web_search", decorated, policies) == decorated
    # No match and nothing decorated: all defaults.
    assert resolve_policy("web_search", None, policies) == Policy()


def test_resolve_policy_last_matching_key_wins() -> None:
    policies = {
        "sql_*": Policy(timeout_s=10.0, redact=("from_sql_star",)),
        "sql_db_*": Policy(timeout_s=20.0),
    }
    resolved = resolve_policy("sql_db_query", None, policies)
    assert resolved.timeout_s == 20.0  # later match wins per field
    assert resolved.redact == ("from_sql_star",)  # earlier match kept its field


def test_interrupt_policy_defaults() -> None:
    policy = InterruptPolicy()
    assert (policy.approver_role, policy.scope, policy.ttl_hours) == (None, None, None)
    full = InterruptPolicy(approver_role="site_approver", scope="db:write", ttl_hours=48)
    assert (full.approver_role, full.scope, full.ttl_hours) == ("site_approver", "db:write", 48)


# ------------------------------------------------------------- middleware


def test_middleware_sync_pass_through() -> None:
    middleware = VDreamerToolMiddleware()
    request = _stub_request()
    sentinel = ToolMessage(content="sentinel", tool_call_id="call_1")

    def handler(req: ToolCallRequest) -> ToolMessage:
        assert req is request
        return sentinel

    assert middleware.wrap_tool_call(request, handler) is sentinel


async def test_middleware_async_pass_through() -> None:
    middleware = VDreamerToolMiddleware()
    request = _stub_request()
    sentinel = ToolMessage(content="sentinel", tool_call_id="call_1")

    async def handler(req: ToolCallRequest) -> ToolMessage:
        assert req is request
        return sentinel

    assert await middleware.awrap_tool_call(request, handler) is sentinel


async def test_middleware_smoke_with_create_agent() -> None:
    model = _FakeToolCallingModel(
        responses=[
            AIMessage(
                "",
                tool_calls=[
                    {"name": "echo", "args": {"text": "hi"}, "id": "call_1", "type": "tool_call"}
                ],
            ),
            AIMessage("done"),
        ]
    )

    @tool
    def echo(text: str) -> str:
        """Echo the text back."""
        return f"echo: {text}"

    agent = create_agent(model=model, tools=[echo], middleware=[VDreamerToolMiddleware()])
    result = await agent.ainvoke({"messages": [HumanMessage(content="hi")]})
    messages = result["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    # The middleware wrapped the call without rewriting name or schema, and
    # the run continued to completion after the tool result.
    assert [m.content for m in tool_messages] == ["echo: hi"]
    assert messages[-1].content == "done"
