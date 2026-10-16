"""In-process regression test for the CellTFusion server.

run_celltfusion is heavy (WGCNA + GSVA + NMF, ~60-90s even on a small
subset) -- this one test doubles as both the functional smoke test and the
session-lifecycle test to avoid paying that cost twice.
"""

from pathlib import Path

import pytest
from mcp import Client

from mcp_tme_servers.celltfusion.server import mcp
from mcp_tme_servers.multideconv.server import mcp as multideconv_mcp

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
            "verbose": False,
            "session_id": sid,
        },
    )
    assert not result.is_error
    sc = result.structured_content
    assert "TFs_matrix" in sc["summary"]["names"]
    assert "TFs_matrix" in sc["field_glossary"]
    assert sc["session_id"] == sid
    # CellTFusion no longer maps factors onto TCGA meta-programs: 8 elements
    assert set(sc["summary"]["names"]) == set(sc["field_glossary"])
    assert "TME_states" not in sc["summary"]["names"] and "Latent_spaces" in sc["summary"]["names"]

    status = await client.call_tool("get_session_status", {"session_id": sid})
    assert status.structured_content["export_ready"] is True


@pytest.mark.anyio
async def test_input_validation_is_upfront(client: Client) -> None:
    counts = str(DATA / "celltfusion_raw_counts_small.csv")
    cases = [
        ({"batch_column": "cohort"}, "batch_column needs coldata_csv"),
        ({}, "cbsx_mail/cbsx_token"),  # CBSX is in the default deconv_methods
        ({"deconv_methods": ["Epidish", "CBSX"]}, "cbsx_mail/cbsx_token"),
        ({"deconv_methods": ["Epidish"], "tf_collection": "ARACNE"}, "needs aracne_network_file"),
        (
            {"deconv_methods": ["Epidish"], "tf_collection": "ARACNE", "aracne_network_file": str(DATA / "missing_network.txt")},
            "not found",
        ),
    ]
    for extra, message in cases:
        result = await client.call_tool("run_celltfusion", {"raw_counts_csv": counts, **extra})
        assert result.is_error, extra
        assert message in result.content[0].text, (extra, result.content[0].text)


@pytest.mark.anyio
async def test_prepare_celltfusion_folds_per_fold(client: Client) -> None:
    # One fold: CellTFusion learns the cell groups on 16 training samples and scores the other 8.
    result = await client.call_tool(
        "prepare_celltfusion_folds",
        {
            "counts_csv": str(DATA / "celltfusion_raw_counts_small.csv"),
            "deconvolution_csv": str(DATA / "celltfusion_deconv_small.csv"),
            "target_csv": str(DATA / "celltfusion_coldata_small.csv"),  # samples in reverse order
            "target_column": "target",
            "compute_full_dataset": False,
            "folds": {"Fold1": list(range(1, 17))},
            "min_targets_size": 5,
            "min_module_size": 5,
        },
    )
    assert not result.is_error, result.content[0].text
    sc = result.structured_content
    assert sc["summary"]["class"] == "list"
    assert sc["summary"]["names"] in ("Fold1", ["Fold1"])  # jsonlite unboxes a length-1 vector
    assert any(f.endswith("Results/fold_Fold1.rds") for f in sc["output_files"])


@pytest.mark.anyio
async def test_prepare_celltfusion_folds_validation(client: Client) -> None:
    base = {
        "counts_csv": str(DATA / "celltfusion_raw_counts_small.csv"),
        "deconvolution_csv": str(DATA / "celltfusion_deconv_small.csv"),
        "target_csv": str(DATA / "celltfusion_coldata_small.csv"),
    }
    cases = [
        ({}, "either target_column"),
        ({"target_column": "target", "time_column": "time"}, "either target_column"),
        ({"time_column": "time"}, "both time_column and event_column"),
        ({"target_column": "target", "compute_full_dataset": False}, "folds is required"),
        ({"target_column": "target", "batch_column": "cohort"}, "batch_column needs coldata_csv"),
    ]
    for extra, message in cases:
        result = await client.call_tool("prepare_celltfusion_folds", {**base, **extra})
        assert result.is_error, extra
        assert message in result.content[0].text, (extra, result.content[0].text)

    for tool, data in (
        ("score_celltfusion_test_set", {"deconvolution_test_csv": base["deconvolution_csv"]}),
        ("project_celltfusion_factors", {"scores_csv": base["deconvolution_csv"]}),
    ):
        for extra in ({}, {"model_rds": "a.rds", "celltfusion_rds": "b.rds"}):
            result = await client.call_tool(tool, {**data, **extra})
            assert result.is_error, (tool, extra)
            assert "exactly one of model_rds" in result.content[0].text


@pytest.mark.anyio
async def test_run_celltfusion_with_precomputed_subgroups(client: Client) -> None:
    # Subgroups harmonized by multideconv-server (0.2.0 naming) passed as dt_rds: CellTFusion skips
    # its own deconvolution analysis.
    async with Client(multideconv_mcp, raise_exceptions=True) as md:
        analyzed = await md.call_tool(
            "analyze_deconvolution", {"deconvolution_csv": str(DATA / "celltfusion_deconv_small.csv")}
        )
    assert not analyzed.is_error
    result = await client.call_tool(
        "run_celltfusion",
        {
            "raw_counts_csv": str(DATA / "celltfusion_raw_counts_small.csv"),
            "dt_rds": str(Path(analyzed.structured_content["run_dir"]) / "result.rds"),
            "min_targets_size": 5,
            "min_module_size": 5,
            "verbose": False,
        },
    )
    assert not result.is_error, result.content[0].text
    sc = result.structured_content
    assert "Latent_spaces" in sc["summary"]["names"]
    celltfusion_rds = str(Path(sc["run_dir"]) / "result.rds")

    # A new cohort (here the same samples) scored on these cell groups, then placed on the factors
    scored = await client.call_tool(
        "score_celltfusion_test_set",
        {"celltfusion_rds": celltfusion_rds, "deconvolution_test_csv": str(DATA / "celltfusion_deconv_small.csv")},
    )
    assert not scored.is_error, scored.content[0].text
    ssc = scored.structured_content
    assert ssc["summary"]["dim"][0] == 24
    projected = await client.call_tool(
        "project_celltfusion_factors",
        {"celltfusion_rds": celltfusion_rds, "scores_csv": str(Path(ssc["run_dir"]) / ssc["csv_outputs"]["result"])},
    )
    assert not projected.is_error, projected.content[0].text
    assert projected.structured_content["summary"]["dim"][0] == 24
