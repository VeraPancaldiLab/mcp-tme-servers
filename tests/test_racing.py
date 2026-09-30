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


@pytest.mark.anyio
async def test_explain_lr_contributions(client: Client) -> None:
    prep = await client.call_tool("prepare_racing_inputs", {"counts_csv": str(DATA / "racing_counts_small.csv")})
    assert not prep.is_error
    input_rds = prep.structured_content["run_dir"] + "/result.rds"
    cells = prep.structured_content["summary"]["elements"]["celltypes"]["preview"]
    assert len(cells) >= 3

    direct = await client.call_tool(
        "explain_lr_contributions",
        {"input_data_rds": input_rds, "motif": "direct", "cell_types": cells[:2], "top_n": 5},
    )
    assert not direct.is_error, direct.content[0].text
    summary = direct.structured_content["summary"]
    if summary["class"] != "NULL":
        assert {"Patient", "Ligand", "Receptor", "Contribution", "PctOfTotal"} <= set(summary["colnames"])
        # Patients are labelled with the real sample names, not Patient_1, Patient_2, ...
        assert not any(str(p).startswith("Patient_") for p in summary["rownames"] or [])

    wedge = await client.call_tool(
        "explain_lr_contributions", {"input_data_rds": input_rds, "motif": "wedge", "cell_types": cells[:3]}
    )
    assert not wedge.is_error, wedge.content[0].text

    wrong = await client.call_tool(
        "explain_lr_contributions", {"input_data_rds": input_rds, "motif": "wedge", "cell_types": cells[:2]}
    )
    assert wrong.is_error
    assert "needs exactly 3" in wrong.content[0].text


@pytest.mark.anyio
async def test_kernel_from_counts_writes_inside_run_dir(client: Client) -> None:
    # compute_racing_kernel's own output_folder default is a hardcoded ~/Documents/racing/vignettes/;
    # the server must keep the intermediate input files in the call's run_dir.
    result = await client.call_tool(
        "run_racing_kernel", {"counts_csv": str(DATA / "racing_counts_small.csv"), "communication_type": ["D"]}
    )
    assert not result.is_error, result.content[0].text
    assert any(f.startswith("Results/Lmatrix_") for f in result.structured_content["output_files"])
