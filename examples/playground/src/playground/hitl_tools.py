"""The LangChain human-in-the-loop example, served by vdreamer.

Source: https://docs.langchain.com/oss/python/langchain/human-in-the-loop —
``create_agent`` with ``HumanInTheLoopMiddleware`` over ``write_file``,
``execute_sql`` and ``read_data``. Changes to serve it:

- ``model="gpt-5.5"`` becomes ``ctx.models["chat"]``.
- ``checkpointer=InMemorySaver()`` becomes ``**ctx.agent_kwargs()`` (vdreamer's
  durable checkpointer, the run context, and the governance middleware).
- The tools write to an in-memory dict: the docs leave them abstract.

The middleware's interrupt reaches the client as a ``tool_review`` interrupt
(PRD §4.5); ``vdreamer run`` offers its decisions inline::

    vdreamer dev --agent playground.hitl_tools:agent --models config/models.ollama.yaml
    vdreamer run hitl-tools --message "Save 'hello' to notes.txt, then delete all orders"
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool

from vdreamer import BaseAgent, BuildContext, Mode

FILES: dict[str, str] = {}
ROWS = [{"id": 1, "item": "widget"}, {"id": 2, "item": "gadget"}]


@tool
def write_file(path: str, content: str) -> str:
    """Write `content` to the file at `path`."""
    FILES[path] = content
    return f"wrote {len(content)} characters to {path}"


@tool
def execute_sql(query: str) -> str:
    """Execute a SQL statement against the orders database."""
    return f"executed: {query}"


@tool
def read_data(table: str) -> str:
    """Read every row of `table`."""
    return str(ROWS)


class HitlToolsAgent(BaseAgent):
    name = "hitl-tools"
    mode = Mode.CHAT
    models = ("chat",)

    def build(self, ctx: BuildContext) -> Any:
        kwargs = ctx.agent_kwargs()
        middleware = [
            *kwargs.pop("middleware"),
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "write_file": True,  # All decisions (approve, edit, reject, respond) allowed
                    # No editing allowed
                    "execute_sql": {"allowed_decisions": ["approve", "reject"]},
                    "read_data": False,  # Safe operation, no approval needed
                },
                description_prefix="Tool execution pending approval",
            ),
        ]
        return create_agent(
            ctx.models["chat"],
            tools=[write_file, execute_sql, read_data],
            middleware=middleware,
            **kwargs,
        )


agent = HitlToolsAgent()
