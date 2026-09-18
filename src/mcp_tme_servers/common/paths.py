"""Default runtime-data location for a server, overridable via env var.

Unlike a dev checkout where "runs/" can live next to the source, an
installed package (pip/uvx) has no fixed project directory to write into, so
each server defaults to a per-server directory under the user's home instead.
Override with MCP_TME_SERVERS_HOME to point all servers at a shared location
(e.g. for a specific analysis project).
"""

from __future__ import annotations

import os
from pathlib import Path


def server_home(server_name: str) -> Path:
    base = os.environ.get("MCP_TME_SERVERS_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".mcp-tme-servers"
    return root / server_name
