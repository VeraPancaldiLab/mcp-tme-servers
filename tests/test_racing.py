"""In-process regression tests for the RaCInG server.

run_racing_montecarlo is exercised manually rather than here (very slow even
at reduced settings) -- see README.md.
"""

from pathlib import Path

import pytest
from mcp import Client

from mcp_tme_servers.racing.server import mcp

DATA = Path(__file__).parent / "data"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client():
    async with Client(mcp, raise_exceptions=True) as c:
        yield c


@pytest.mark.anyio
async def test_prepare_inputs_and_kernel_chain_with_session(client: Client) -> None:
    started = await client.call_tool("start_workflow_session", {"cohort_id": "pytest_racing"})
    sid = started.structured_content["session_id"]
    assert started.structured_content["export_ready"] is False

    prep = await client.call_tool(
        "prepare_racing_inputs",
        {"counts_csv": str(DATA / "racing_counts_small.csv"), "session_id": sid},
    )
    assert not prep.is_error
    prep_sc = prep.structured_content
    assert "Lmatrix" in prep_sc["field_glossary"]
    input_rds = prep_sc["run_dir"] + "/result.rds"

    kernel = await client.call_tool(
        "run_racing_kernel",
        {"input_data_rds": input_rds, "communication_type": ["D", "W"], "session_id": sid},
    )
    assert not kernel.is_error
    kernel_sc = kernel.structured_content
    assert "features" in kernel_sc["summary"]["names"]
    assert "features" in kernel_sc["field_glossary"]

    status = await client.call_tool("get_session_status", {"session_id": sid})
    assert status.structured_content["export_ready"] is True


@pytest.mark.anyio
async def test_cross_session_chaining_is_refused(client: Client) -> None:
    s1 = await client.call_tool("start_workflow_session", {"cohort_id": "cohort_A"})
    sid1 = s1.structured_content["session_id"]
    prep = await client.call_tool(
        "prepare_racing_inputs",
        {"counts_csv": str(DATA / "racing_counts_small.csv"), "session_id": sid1},
    )
    input_rds = prep.structured_content["run_dir"] + "/result.rds"

    s2 = await client.call_tool("start_workflow_session", {"cohort_id": "cohort_B"})
    sid2 = s2.structured_content["session_id"]

    result = await client.call_tool(
        "run_racing_kernel",
        {"input_data_rds": input_rds, "session_id": sid2},
    )
    assert result.is_error
    assert "belongs to session" in result.content[0].text
