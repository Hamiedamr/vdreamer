"""The site-inventory MCP server for UC2 — a stand-in for the operator's internal system.

``python -m site_validation.inventory_server`` serves it over stdio (declare it
in ``resources.yaml`` as ``type: mcp`` with an absolute ``command``); with
``http PORT`` it serves Streamable HTTP on 127.0.0.1. Two tools:

- ``get_site(site_id)`` → the planned sectors and their antenna counts;
- ``record_validation(site_id, status, note, approved_by)`` → stores the
  approved verdict (idempotent per run: vdreamer passes its idempotency key).

The data lives in ``SITE_INVENTORY_FILE`` (default: the bundled
``inventory.json``); recorded validations append to ``SITE_INVENTORY_LOG``
when set.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

DATA = Path(__file__).with_name("inventory.json")

server: FastMCP = FastMCP("siteinv")


def _sites() -> dict[str, Any]:
    path = Path(os.environ.get("SITE_INVENTORY_FILE") or DATA)
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


@server.tool
def get_site(site_id: str) -> str:
    """The site's planned configuration: name, location, and antennas per sector (JSON)."""
    site = _sites().get(site_id)
    if site is None:
        raise ToolError(f"unknown site {site_id!r}")
    return json.dumps({"site_id": site_id, **site})


@server.tool
def record_validation(site_id: str, status: str, note: str = "", approved_by: str = "") -> str:
    """Record an approved validation verdict for the site (JSON receipt)."""
    if site_id not in _sites():
        raise ToolError(f"unknown site {site_id!r}")
    receipt = {"site_id": site_id, "status": status, "note": note, "approved_by": approved_by}
    log = os.environ.get("SITE_INVENTORY_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt) + "\n")
    return json.dumps({"recorded": True, **receipt})


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "http":
        server.run(transport="http", host="127.0.0.1", port=int(sys.argv[2]))
    else:
        server.run()


if __name__ == "__main__":
    main()
