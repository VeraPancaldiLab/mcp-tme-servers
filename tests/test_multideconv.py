"""In-process regression tests for the multideconv server."""

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
async def test_prepare_deconvolution_folds_per_fold_surfaces_known_upstream_bug(client: Client) -> None:
    # compute_full_dataset=False -> multideconv::prepare_multideconv_folds()'s fold-construction
    # branch (bestune=NULL) computes and saves each fold to Results/fold_*.rds but never
    # return()s processed_folds -- it falls through and returns NULL. See README's "Known
    # upstream issue" for multideconv. TEMP: once the missing return() is fixed upstream, this
    # should start returning the fold list and can be simplified to assert on its contents
    # instead of documenting the NULL result.
    result = await client.call_tool(
        "prepare_deconvolution_folds",
        {
            "deconvolution_csv": str(DATA / "deconvolution_with_target.csv"),
            "compute_full_dataset": False,
            "folds": {"Fold1": list(range(1, 11))},
        },
    )
    assert not result.is_error
    sc = result.structured_content
    assert sc["summary"]["class"] == "NULL"
    assert any("fold_Fold1.rds" in f for f in sc["output_files"])


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
