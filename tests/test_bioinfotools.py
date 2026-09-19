"""In-process regression tests for the Bioinfo-tools server.

compute_tf_activity/compute_pathway_activity aren't exercised here (they fetch
a TF/pathway collection from OmnipathR over the network on first use) -- see
README.md for how the other four servers' tests handle the same tradeoff for
their own network-fetching tools.
"""

from pathlib import Path

import pytest
from mcp import Client

from mcp_tme_servers.bioinfotools.server import mcp

DATA = Path(__file__).parent / "data"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client():
    async with Client(mcp, raise_exceptions=True) as c:
        yield c


@pytest.mark.anyio
async def test_compute_pca_analysis(client: Client) -> None:
    result = await client.call_tool(
        "compute_pca_analysis",
        {"data_csv": str(DATA / "bioinfotools_scores_small.csv"), "file_name": "pytest_pca"},
    )
    assert not result.is_error
    sc = result.structured_content
    assert sc["summary"]["class"] == "list"
    assert "coords" in sc["summary"]["names"]
    assert "var_explained" in sc["field_glossary"]


@pytest.mark.anyio
async def test_scores_pearson_test_and_session_lifecycle(client: Client) -> None:
    started = await client.call_tool("start_workflow_session", {"cohort_id": "pytest_bioinfotools"})
    sid = started.structured_content["session_id"]
    assert started.structured_content["export_ready"] is False

    result = await client.call_tool(
        "scores_pearson_test",
        {
            "scores_csv": str(DATA / "bioinfotools_scores_small.csv"),
            "coldata_csv": str(DATA / "bioinfotools_coldata.csv"),
            "trait": "trait",
            "session_id": sid,
        },
    )
    assert not result.is_error
    sc = result.structured_content
    assert sc["session_id"] == sid
    # Feature_correlated (~2x trait + small noise) should survive as the
    # significant-subset data frame at scores[[1]] (summary.elements[0]).
    assert sc["summary"]["elements"][0]["colnames"] == "Feature_correlated"

    status = await client.call_tool("get_session_status", {"session_id": sid})
    assert "trait_association" in status.structured_content["stages_completed"]
    assert status.structured_content["export_ready"] is True


@pytest.mark.anyio
async def test_scores_stat_analysis_no_significant_features_warns(client: Client) -> None:
    result = await client.call_tool(
        "scores_stat_analysis",
        {
            "scores_csv": str(DATA / "bioinfotools_scores_noise.csv"),
            "coldata_csv": str(DATA / "bioinfotools_coldata.csv"),
            "trait": "trait",
            "method": "pearson",
        },
    )
    assert not result.is_error
    sc = result.structured_content
    assert sc["summary"]["class"] == "NULL"
    assert any("No feature reached" in w for w in sc["warnings"])


@pytest.mark.anyio
async def test_missing_csv_is_a_tool_error(client: Client) -> None:
    result = await client.call_tool(
        "compute_pca_analysis",
        {"data_csv": str(DATA / "does_not_exist.csv")},
    )
    assert result.is_error
    assert "does_not_exist.csv" in result.content[0].text
