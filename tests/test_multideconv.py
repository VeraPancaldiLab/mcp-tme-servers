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


@pytest.mark.anyio
async def test_analyze_deconvolution_v020_output(client: Client) -> None:
    # multideconv 0.2.0: complete-linkage subgroups named <CellType>_Subgroup.<i> (no .Iteration.),
    # CV-based low-variance filter (cv_thr), no pruning step -> 6 output elements.
    result = await client.call_tool(
        "analyze_deconvolution",
        {
            "deconvolution_csv": str(DATA / "deconvolution.csv"),
            "cv_thr": 0.2,
            # cohort coded 1/2/3 and listed in reverse order: matched by sample ID, treated as categorical
            "batch_csv": str(DATA / "deconvolution_batch.csv"),
            "batch_column": "cohort",
        },
    )
    assert not result.is_error, result.content[0].text
    sc = result.structured_content
    assert sc["summary"]["names"] == [
        "Deconvolution matrix",
        "Deconvolution subgroups per cell types",
        "Deconvolution subgroups composition",
        "Discarded features with high number of zeros",
        "Discarded features with low variance",
        "Discarded cell types",
    ]
    assert set(sc["summary"]["names"]) == set(sc["field_glossary"])
    with open(Path(sc["run_dir"]) / sc["csv_outputs"]["Deconvolution matrix"], newline="") as fh:
        columns = next(csv.reader(fh))[1:]
    subgroups = [c for c in columns if "_Subgroup." in c]
    assert subgroups and not any("Iteration" in c for c in columns)
    # Per-cell-type feature tables are written too (a plain list of tables)
    assert any(key.startswith("Deconvolution subgroups per cell types/") for key in sc["csv_outputs"])


@pytest.mark.anyio
async def test_relate_subgroups_to_pathways(client: Client) -> None:
    analyzed = await client.call_tool("analyze_deconvolution", {"deconvolution_csv": str(DATA / "deconvolution.csv")})
    assert not analyzed.is_error
    subgroups_rds = str(Path(analyzed.structured_content["run_dir"]) / "result.rds")

    result = await client.call_tool(
        "relate_subgroups_to_pathways",
        {"subgroups_rds": subgroups_rds, "pathways_csv": str(DATA / "pathways_small.csv"), "corr_type": "spearman"},
    )
    assert not result.is_error, result.content[0].text
    sc = result.structured_content
    # One element per compared cell type, each with correlations + pvalues tables
    assert sc["summary"]["class"] == "list"
    keys = sc["csv_outputs"]
    assert any(k.endswith("/correlations") for k in keys) and any(k.endswith("/pvalues") for k in keys)
    assert any(f.endswith(".pdf") for f in sc["output_files"])

    # Sample names that don't match the deconvolution are an error, not an empty result
    mismatched = await client.call_tool(
        "relate_subgroups_to_pathways",
        {"subgroups_rds": subgroups_rds, "pathways_csv": str(DATA / "ml_features_test.csv")},
    )
    assert mismatched.is_error
    assert "Fewer than 5 samples in common" in mismatched.content[0].text


@pytest.mark.anyio
async def test_aggregate_cell_groups_then_analyze(client: Client) -> None:
    groups = {
        "Myeloid.cells": ["Macrophages.cells", "Macrophages.M0", "Macrophages.M1", "Macrophages.M2", "Monocytes", "Dendritic.cells"],
        "Lymphocytes": ["B.cells", "CD4.cells", "CD8.cells", "NK.cells"],
    }
    aggregated = await client.call_tool(
        "aggregate_cell_groups", {"deconvolution_csv": str(DATA / "deconvolution.csv"), "cell_groups": groups}
    )
    assert not aggregated.is_error, aggregated.content[0].text
    sc = aggregated.structured_content
    assert any("cells_extra=['Myeloid.cells', 'Lymphocytes']" in w for w in sc["warnings"])
    aggregated_csv = Path(sc["run_dir"]) / sc["csv_outputs"]["result"]
    with open(aggregated_csv, newline="") as fh:
        columns = next(csv.reader(fh))[1:]
    with open(DATA / "deconvolution.csv", newline="") as fh:
        original = next(csv.reader(fh))[1:]
    added = [c for c in columns if c not in original]
    assert set(original) <= set(columns)  # original features are kept
    assert any(c.endswith("_Lymphocytes") for c in added)

    # A new group name survives the harmonization only when listed in cells_extra
    def lymphocyte_features(result) -> list[str]:
        out = result.structured_content
        with open(Path(out["run_dir"]) / out["csv_outputs"]["Deconvolution matrix"], newline="") as fh:
            return [c for c in next(csv.reader(fh))[1:] if "Lymphocytes" in c]

    kept = await client.call_tool(
        "analyze_deconvolution", {"deconvolution_csv": str(aggregated_csv), "cells_extra": ["Lymphocytes"]}
    )
    assert not kept.is_error, kept.content[0].text
    assert lymphocyte_features(kept)
    dropped = await client.call_tool("analyze_deconvolution", {"deconvolution_csv": str(aggregated_csv)})
    assert not dropped.is_error
    assert not lymphocyte_features(dropped)


@pytest.mark.anyio
async def test_aggregate_cell_groups_rejects_ambiguous_group_name(client: Client) -> None:
    result = await client.call_tool(
        "aggregate_cell_groups",
        {"deconvolution_csv": str(DATA / "deconvolution.csv"), "cell_groups": {"All_T": ["CD4.cells", "CD8.cells"]}},
    )
    assert result.is_error
    assert "must not contain '_'" in result.content[0].text
