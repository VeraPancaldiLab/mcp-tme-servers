"""In-process regression tests for the multideconv server."""

import csv
from pathlib import Path

import pytest
from mcp import Client

from mcp_tme_servers.multideconv.server import mcp

DATA = Path(__file__).parent / "data"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client():
    async with Client(mcp, raise_exceptions=True) as c:
        yield c


@pytest.mark.anyio
async def test_analyze_deconvolution(client: Client) -> None:
    result = await client.call_tool(
        "analyze_deconvolution",
        {"deconvolution_csv": str(DATA / "deconvolution.csv"), "corr": 0.7, "file_name": "pytest_analyze"},
    )
    assert not result.is_error
    sc = result.structured_content
    assert sc["summary"]["class"] == "list"
    assert "Deconvolution matrix" in sc["summary"]["names"]
    assert "Deconvolution matrix" in sc["field_glossary"]


@pytest.mark.anyio
async def test_benchmark_deconvolution(client: Client) -> None:
    result = await client.call_tool(
        "benchmark_deconvolution",
        {
            "deconvolution_csv": str(DATA / "deconvolution.csv"),
            "groundtruth_csv": str(DATA / "cells_groundtruth.csv"),
            "cells_extra": ["Myeloid.cells"],
            "scatter": False,
        },
    )
    assert not result.is_error
    sc = result.structured_content
    assert sc["summary"]["dim"][0] == 15
    assert "numeric_summary" in sc["summary"]


@pytest.mark.anyio
async def test_missing_csv_is_a_tool_error(client: Client) -> None:
    result = await client.call_tool(
        "benchmark_deconvolution",
        {"deconvolution_csv": str(DATA / "does_not_exist.csv"), "groundtruth_csv": str(DATA / "cells_groundtruth.csv")},
    )
    assert result.is_error
    assert "does_not_exist.csv" in result.content[0].text


@pytest.mark.anyio
async def test_prepare_deconvolution_folds_full_dataset(client: Client) -> None:
    result = await client.call_tool(
        "prepare_deconvolution_folds",
        {
            "deconvolution_csv": str(DATA / "deconvolution_with_target.csv"),
            "compute_full_dataset": True,
        },
    )
    assert not result.is_error
    sc = result.structured_content
    assert sc["summary"]["class"] == "list"


@pytest.mark.anyio
async def test_prepare_deconvolution_folds_per_fold(client: Client) -> None:
    # compute_full_dataset=False -> the per-fold branch pipeML's fold_construction_fun drives.
    # It used to save folds but return NULL (missing return()); fixed upstream, it now returns
    # the named fold list.
    result = await client.call_tool(
        "prepare_deconvolution_folds",
        {
            "deconvolution_csv": str(DATA / "deconvolution_with_target.csv"),
            "compute_full_dataset": False,
            "folds": {"Fold1": list(range(1, 11)), "Fold2": list(range(6, 16))},
            "ncores": 1,
        },
    )
    assert not result.is_error
    sc = result.structured_content
    assert sc["summary"]["class"] == "list"
    assert sc["summary"]["names"] == ["Fold1", "Fold2"]
    assert any("fold_Fold1.rds" in f for f in sc["output_files"])
    assert any("fold_Fold2.rds" in f for f in sc["output_files"])


@pytest.mark.anyio
async def test_prepare_deconvolution_folds_requires_folds(client: Client) -> None:
    result = await client.call_tool(
        "prepare_deconvolution_folds",
        {"deconvolution_csv": str(DATA / "deconvolution_with_target.csv"), "compute_full_dataset": False},
    )
    assert result.is_error
    assert "folds is required" in result.content[0].text


@pytest.mark.anyio
async def test_replicate_subgroups(client: Client) -> None:
    trained = await client.call_tool("analyze_deconvolution", {"deconvolution_csv": str(DATA / "deconvolution.csv")})
    assert not trained.is_error
    result = await client.call_tool(
        "replicate_subgroups",
        {
            "deconv_res_rds": str(Path(trained.structured_content["run_dir"]) / "result.rds"),
            "deconvolution_test_csv": str(DATA / "deconvolution.csv"),
        },
    )
    assert not result.is_error
    # Replicating onto the training cohort itself must give the same features, with the original
    # (non-syntactic) names kept now that upstream uses check.names = FALSE
    def header(sc: dict, element: str) -> set[str]:
        with open(Path(sc["run_dir"]) / sc["csv_outputs"][element], newline="") as fh:
            return set(next(csv.reader(fh))[1:])

    trained_cols = header(trained.structured_content, "Deconvolution matrix")
    assert header(result.structured_content, "result") == trained_cols
    assert any("_Subgroup." in c for c in trained_cols)


@pytest.mark.anyio
async def test_session_lifecycle(client: Client) -> None:
    started = await client.call_tool("start_workflow_session", {"cohort_id": "pytest_multideconv"})
    sid = started.structured_content["session_id"]
    assert started.structured_content["export_ready"] is False

    deconv = await client.call_tool(
        "analyze_deconvolution",
        {"deconvolution_csv": str(DATA / "deconvolution.csv"), "session_id": sid},
    )
    assert not deconv.is_error
    assert deconv.structured_content["session_id"] == sid

    status = await client.call_tool("get_session_status", {"session_id": sid})
    assert "harmonization" in status.structured_content["stages_completed"]


@pytest.mark.anyio
async def test_prepare_deconvolution_folds_survival(client: Client) -> None:
    # Survival outcome now travels as 'time'/'event' columns in the data itself (as pipeML passes
    # it); they are kept out of the harmonization and added back to train/test data.
    result = await client.call_tool(
        "prepare_deconvolution_folds",
        {
            "deconvolution_csv": str(DATA / "deconvolution_with_survival.csv"),
            "compute_full_dataset": False,
            "folds": {"Fold1": list(range(1, 11)), "Fold2": list(range(6, 16))},
            "ncores": 1,
        },
    )
    assert not result.is_error, result.content[0].text
    run_dir = Path(result.structured_content["run_dir"])
    assert result.structured_content["summary"]["names"] == ["Fold1", "Fold2"]
    assert (run_dir / "Results" / "fold_Fold1.rds").is_file()

    full = await client.call_tool(
        "prepare_deconvolution_folds",
        {"deconvolution_csv": str(DATA / "deconvolution_with_survival.csv"), "compute_full_dataset": True},
    )
    assert not full.is_error, full.content[0].text
    # (1) harmonized features + outcome, (2) the analysis result, (3) bestune
    features = full.structured_content["summary"]["elements"][0] if isinstance(
        full.structured_content["summary"]["elements"], list) else None
    assert full.structured_content["summary"]["class"] == "list"
    if features is not None:
        assert features["dim"][0] == 15  # all samples, outcome columns re-attached at the end
