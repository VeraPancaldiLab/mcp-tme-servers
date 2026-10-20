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
        {"input_data_rds": input_rds, "session_id": sid},
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


@pytest.mark.anyio
async def test_explain_lr_contributions(client: Client) -> None:
    prep = await client.call_tool("prepare_racing_inputs", {"counts_csv": str(DATA / "racing_counts_small.csv")})
    assert not prep.is_error
    input_rds = prep.structured_content["run_dir"] + "/result.rds"
    cells = prep.structured_content["summary"]["elements"]["celltypes"]["preview"]
    assert len(cells) >= 2

    direct = await client.call_tool(
        "explain_lr_contributions",
        {"input_data_rds": input_rds, "sender": cells[0], "receiver": cells[1], "top_n": 5},
    )
    assert not direct.is_error, direct.content[0].text
    summary = direct.structured_content["summary"]
    if summary["class"] != "NULL":
        assert {"Patient", "Ligand", "Receptor", "Contribution", "PctOfTotal"} <= set(summary["colnames"])

    unknown = await client.call_tool(
        "explain_lr_contributions", {"input_data_rds": input_rds, "sender": "NotACellType", "receiver": cells[1]}
    )
    assert unknown.is_error


@pytest.mark.anyio
async def test_montecarlo_direct_and_gscc(client: Client) -> None:
    # Monte Carlo now computes only 'D' and/or 'GSCC' (wedges and triangles were removed upstream)
    prep = await client.call_tool("prepare_racing_inputs", {"counts_csv": str(DATA / "racing_counts_small.csv")})
    assert not prep.is_error
    result = await client.call_tool(
        "run_racing_montecarlo",
        {
            "input_data_rds": prep.structured_content["run_dir"] + "/result.rds",
            "communication_type": ["D", "GSCC"],
            "n_patients": 2,
            "n_cells": 100,
            "n_graphs": 2,
            "n_degree": 3,
        },
    )
    assert not result.is_error, result.content[0].text
    sc = result.structured_content
    assert {"input", "output"} <= set(sc["summary"]["names"])

    # A removed motif type is rejected by the tool schema
    try:
        rejected = await client.call_tool(
            "run_racing_montecarlo", {"input_data_rds": prep.structured_content["run_dir"] + "/result.rds", "communication_type": ["W"]}
        )
    except Exception:
        return
    assert rejected.is_error


@pytest.mark.anyio
async def test_kernel_from_counts_writes_inside_run_dir(client: Client) -> None:
    # The intermediate input files must stay inside the call's run_dir.
    result = await client.call_tool("run_racing_kernel", {"counts_csv": str(DATA / "racing_counts_small.csv")})
    assert not result.is_error, result.content[0].text
    sc = result.structured_content
    assert any(f.startswith("Results/Lmatrix_") for f in sc["output_files"])
    # The kernel computes direct communication only: Dir_<A>_<B> columns
    features = sc["summary"]["elements"]["features"]
    assert features["colnames"] and all(c.startswith("Dir_") for c in features["colnames"])
