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


TRAIN_ARGS = {
    "features_train_csv": str(DATA / "ml_features_train.csv"),
    "task_type": "classification",
    "target_csv": str(DATA / "ml_coldata_train.csv"),
    "target_column": "target",
    "trait_positive": "1",
    "metric": "AUROC",
    "k_folds": 3,
    "n_rep": 1,
}


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
async def test_predict_ml_model(client: Client) -> None:
    training = await client.call_tool(
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
    assert not training.is_error
    model_rds = str(Path(training.structured_content["run_dir"]) / "result.rds")

    # ml_coldata.csv holds the whole 60-sample cohort, not just the 20 test samples: the
    # target must be matched to test_data_csv by sample ID (current pipeML errors on a
    # length mismatch, and older versions silently used misaligned labels).
    result = await client.call_tool(
        "predict_ml_model",
        {
            "model_rds": model_rds,
            "test_data_csv": str(DATA / "ml_features_test.csv"),
            "task_type": "classification",
            "target_csv": str(DATA / "ml_coldata.csv"),
            "target_column": "target",
            "trait_positive": "1",
        },
    )
    assert not result.is_error
    sc = result.structured_content
    assert "AUC" in sc["summary"]["names"]
    assert "AUC" in sc["field_glossary"]
    assert "Curve_bands" in sc["summary"]["names"]


@pytest.mark.anyio
async def test_compute_shap_values(client: Client) -> None:
    training = await client.call_tool("train_ml_model", TRAIN_ARGS)
    assert not training.is_error
    model_rds = str(Path(training.structured_content["run_dir"]) / "result.rds")

    # pipeML now explains the final model and takes everything from it: only model + task + seed.
    result = await client.call_tool("compute_shap_values", {"model_rds": model_rds, "task_type": "classification"})
    assert not result.is_error
    sc = result.structured_content
    assert sc["summary"]["class"] == "data.frame"
    assert sc["summary"]["dim"][0] == 40  # one row per training sample
    # The model's average prediction, surfaced from attr(, "baseline")
    assert 0 <= sc["summary"]["attributes"]["baseline"] <= 1

@pytest.mark.anyio
async def test_default_metric_now_works(client: Client) -> None:
    # metric left unset -> pipeML's own default, "AUROC" (it used to crash on NULL).
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
    assert not result.is_error
    sc = result.structured_content
    assert "AUROC_median" in sc["summary"]["names"]


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


@pytest.mark.anyio
async def test_seed_makes_training_reproducible(client: Client) -> None:
    runs = []
    for _ in range(2):
        result = await client.call_tool("train_ml_model", {**TRAIN_ARGS, "seed": 7})
        assert not result.is_error
        runs.append((Path(result.structured_content["run_dir"]) / "result_AUROC_median.csv").read_text())
    assert runs[0] == runs[1]


@pytest.mark.anyio
async def test_misaligned_target_csv_is_a_tool_error(client: Client) -> None:
    # ml_coldata.csv (60 rows) lacks most of deconvolution.csv's samples -> neither
    # row-name nor positional matching is possible, so fail instead of mislabeling.
    result = await client.call_tool(
        "train_ml_model",
        {**TRAIN_ARGS, "features_train_csv": str(DATA / "deconvolution.csv"), "target_csv": str(DATA / "ml_coldata.csv")},
    )
    assert result.is_error
    assert "cannot be matched" in result.content[0].text


@pytest.mark.anyio
async def test_custom_folds_train_then_shap(client: Client) -> None:
    # Leakage-free path end to end: multideconv::prepare_multideconv_folds harmonizes each
    # training fold, then SHAP explains the final model on the subgroups built on all samples.
    training = await client.call_tool(
        "train_ml_model",
        {
            "features_train_csv": str(DATA / "deconvolution.csv"),
            "task_type": "classification",
            "target_csv": str(DATA / "deconvolution_with_target.csv"),
            "target_column": "target",
            "trait_positive": "1",
            "metric": "AUROC",
            "k_folds": 3,
            "n_rep": 1,
            "ncores": 2,
            "fold_construction_fun": "prepare_multideconv_folds",
            "fold_construction_args_fixed": {"ncores": 2},
        },
    )
    assert not training.is_error, training.content[0].text
    sc = training.structured_content
    assert "Custom_output" in sc["summary"]["names"]
    assert "Custom_output" in sc["field_glossary"]
    run_dir = Path(sc["run_dir"])

    shap = await client.call_tool(
        "compute_shap_values",
        {"model_rds": str(run_dir / "result.rds"), "task_type": "classification"},
    )
    assert not shap.is_error, shap.content[0].text
    assert shap.structured_content["summary"]["class"] == "data.frame"
    assert shap.structured_content["summary"]["dim"][0] == 15  # every training sample


@pytest.mark.anyio
async def test_accuracy_metric_is_no_longer_accepted(client: Client) -> None:
    # pipeML dropped "Accuracy" (only AUROC/AUPRC select classification models now).
    try:
        result = await client.call_tool("train_ml_model", {**TRAIN_ARGS, "metric": "Accuracy"})
    except Exception:
        return
    assert result.is_error


@pytest.mark.anyio
async def test_survival_train_predict_shap(client: Client) -> None:
    survival = {
        "task_type": "survival",
        "target_csv": str(DATA / "ml_survival.csv"),
        "time_column": "time",
        "event_column": "status",
    }
    training = await client.call_tool(
        "train_ml_model",
        {"features_train_csv": str(DATA / "ml_survival_features.csv"), **survival, "k_folds": 3, "n_rep": 1, "ncores": 2},
    )
    assert not training.is_error, training.content[0].text
    sc = training.structured_content
    assert "C_index_median" in sc["summary"]["names"]
    assert "C_index_median" in sc["field_glossary"]
    model_rds = str(Path(sc["run_dir"]) / "result.rds")

    predicted = await client.call_tool(
        "predict_ml_model", {"model_rds": model_rds, "test_data_csv": str(DATA / "ml_survival_features.csv"), **survival}
    )
    assert not predicted.is_error, predicted.content[0].text
    assert "c_index" in predicted.structured_content["summary"]["names"]

    shap = await client.call_tool("compute_shap_values", {"model_rds": model_rds, "task_type": "survival"})
    assert not shap.is_error, shap.content[0].text
    assert shap.structured_content["summary"]["dim"] == [80, 8]
