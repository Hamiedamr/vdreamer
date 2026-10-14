"""A real FastMCP server for the MCP tests: ``python mcp_fixture_server.py [stdio|http PORT]``.

Run as a subprocess — over stdio by the stdio tests, over Streamable HTTP on
127.0.0.1 by the HTTP ones. ``http`` mode requires ``Authorization: Bearer
fixture-token`` when ``FIXTURE_REQUIRE_TOKEN=1``.
"""

from __future__ import annotations

import os
import sys

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

TOKEN = "fixture-token"



def _auth() -> object | None:
    if os.environ.get("FIXTURE_REQUIRE_TOKEN") != "1":
        return None
    from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

    return StaticTokenVerifier(tokens={TOKEN: {"client_id": "test", "scopes": []}})


server: FastMCP = FastMCP("fixture", auth=_auth())  # type: ignore[arg-type]


@server.tool
def get_echo(text: str) -> str:
    """Echo the text back."""
    return text


@server.tool
def get_big(n: int) -> str:
    """Return n characters."""
    return "x" * n


@server.tool
def get_env(name: str) -> str:
    """Read one environment variable of the server process."""
    return os.environ.get(name, "<unset>")


@server.tool
def search_fail(query: str) -> str:
    """Always reports a tool error."""
    raise ToolError(f"no results for {query}")


def main() -> None:
    os.environ.setdefault("FASTMCP_CHECK_FOR_UPDATES", "off")
    mode = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if mode == "http":
        server.run(transport="http", host="127.0.0.1", port=int(sys.argv[2]), show_banner=False)
    else:
        server.run(show_banner=False)


if __name__ == "__main__":
    main()
