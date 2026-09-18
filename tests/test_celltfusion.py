"""In-process regression test for the CellTFusion server.

run_celltfusion is heavy (WGCNA + GSVA + NMF, ~60-90s even on a small
subset) -- this one test doubles as both the functional smoke test and the
session-lifecycle test to avoid paying that cost twice.
"""

from pathlib import Path

import pytest
from mcp import Client

from mcp_tme_servers.celltfusion.server import mcp

DATA = Path(__file__).parent / "data"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client():
    async with Client(mcp, raise_exceptions=True) as c:
        yield c


@pytest.mark.anyio
async def test_run_celltfusion_and_session_lifecycle(client: Client) -> None:
    started = await client.call_tool("start_workflow_session", {"cohort_id": "pytest_celltfusion"})
    sid = started.structured_content["session_id"]
    assert started.structured_content["export_ready"] is False

    result = await client.call_tool(
        "run_celltfusion",
        {
            "raw_counts_csv": str(DATA / "celltfusion_raw_counts_small.csv"),
            "deconv_methods": ["Epidish"],
            "min_targets_size": 5,
            "min_module_size": 5,
            "cancer_type": "skcm",
            "verbose": False,
            "session_id": sid,
        },
    )
    assert not result.is_error
    sc = result.structured_content
    assert "TFs_matrix" in sc["summary"]["names"]
    assert "TFs_matrix" in sc["field_glossary"]
    assert sc["session_id"] == sid

    status = await client.call_tool("get_session_status", {"session_id": sid})
    assert status.structured_content["export_ready"] is True
