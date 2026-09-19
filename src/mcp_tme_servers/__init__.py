"""mcp-tme-servers: five MCP servers for tumor-microenvironment analysis of bulk RNAseq.

Each server has its own console entry point (see pyproject.toml
[project.scripts]) and is meant to be run standalone, not imported as a
combined app -- see mcp_tme_servers.multideconv.server,
mcp_tme_servers.pipeml.server, mcp_tme_servers.celltfusion.server,
mcp_tme_servers.racing.server, and mcp_tme_servers.bioinfotools.server.
"""
