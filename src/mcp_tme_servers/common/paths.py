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


def bioinfotools_script() -> Path:
    """Location of Bioinfo-tools' functions.r, overridable via env var.

    Unlike the other three servers, Bioinfo-tools isn't an installed R
    package -- it's a single sourced script (no DESCRIPTION/NAMESPACE) -- so
    there's no system R library for Rscript to resolve it from. Override with
    BIOINFOTOOLS_SCRIPT if the checkout doesn't live at the default path.
    """
    base = os.environ.get("BIOINFOTOOLS_SCRIPT")
    return Path(base).expanduser() if base else Path.home() / "Documents" / "Bioinfo-tools" / "functions.r"
