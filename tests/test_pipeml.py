"""In-process regression tests for the pipeML server."""

from pathlib import Path

import pytest
from mcp import Client

from mcp_tme_servers.pipeml.server import mcp

DATA = Path(__file__).parent / "data"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client():
    async with Client(mcp, raise_exceptions=True) as c:
        yield c


@pytest.mark.anyio
async def test_train_ml_model(client: Client) -> None:
    result = await client.call_tool(
        "train_ml_model",
        {
            "features_train_csv": str(DATA / "ml_features_train.csv"),
            "task_type": "classification",
            "target_csv": str(DATA / "ml_coldata_train.csv"),
            "target_column": "target",
            "trait_positive": "1",
            "metric": "AUROC",
            "k_folds": 3,
            "n_rep": 1,
        },
    )
    assert not result.is_error
    sc = result.structured_content
    assert "AUROC_median" in sc["summary"]["names"]
    assert "AUROC_median" in sc["field_glossary"]


@pytest.mark.anyio
async def test_default_metric_surfaces_known_upstream_bug(client: Client) -> None:
    # metric left unset -> pipeML's own default path -- currently broken upstream
    # ("argument is of length zero"; metric="Accuracy" explicitly instead hits a
    # related but distinct bug, "object 'hyperparams' not found" -- see README).
    # Once fixed upstream and this repo's pinned pipeML version is bumped, this
    # should start passing and can be simplified to a plain success assertion.
    result = await client.call_tool(
        "train_ml_model",
        {
            "features_train_csv": str(DATA / "ml_features_train.csv"),
            "task_type": "classification",
            "target_csv": str(DATA / "ml_coldata_train.csv"),
            "target_column": "target",
            "trait_positive": "1",
            "k_folds": 3,
            "n_rep": 1,
        },
    )
    assert result.is_error
    assert "argument is of length zero" in result.content[0].text


@pytest.mark.anyio
async def test_session_lifecycle(client: Client) -> None:
    started = await client.call_tool("start_workflow_session", {"cohort_id": "pytest_pipeml"})
    sid = started.structured_content["session_id"]
    assert started.structured_content["export_ready"] is False

    result = await client.call_tool(
        "train_ml_model",
        {
            "features_train_csv": str(DATA / "ml_features_train.csv"),
            "task_type": "classification",
            "target_csv": str(DATA / "ml_coldata_train.csv"),
            "target_column": "target",
            "trait_positive": "1",
            "metric": "AUROC",
            "k_folds": 3,
            "n_rep": 1,
            "session_id": sid,
        },
    )
    assert not result.is_error

    status = await client.call_tool("get_session_status", {"session_id": sid})
    assert status.structured_content["export_ready"] is True
