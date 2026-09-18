"""pipeML-server: MCP tools for leakage-free ML/survival modeling

Wraps two entry points from the pipeML R package: training a classification or survival model with repeated/LODO cross-validation (train_ml_model), and the combined train+evaluate workflow against a held-out test set (train_and_evaluate_ml_model).

Each tool shells out to system R via the shared common/r_bridge.py dispatcher
(common/r/run_tool.R). pipeML must already be installed in the system R
library `Rscript` (on PATH) resolves to.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import Field

from mcp_tme_servers.common.paths import server_home
from mcp_tme_servers.common.r_bridge import args_from, csv_arg, csv_column_arg, lit, r_function_arg, rds_arg
from mcp_tme_servers.common.session import SessionConfig
from mcp_tme_servers.common.tools import (
    NETWORK_ANNOTATIONS,
    READ_ANNOTATIONS,
    OutputDir,
    RunResult,
    SessionIdParam,
    make_run,
    register_session_tools,
)

logger = logging.getLogger(__name__)

mcp = MCPServer(
    "pipeML-server",
    instructions=(
        "Tools for leakage-free classification/survival modeling on precomputed feature tables (typically cell-type deconvolution, TF-module, or network features from this package's multideconv, CellTFusion, or RaCInG servers). Use train_ml_model to train and cross-validate only; use train_and_evaluate_ml_model to also score a held-out test set in the same call. Both support classification (AUROC/AUPRC/Accuracy) and survival (C-index, Kaplan-Meier) tasks, repeated k-fold or leave-one-dataset-out (LODO) cross-validation for testing cross-cohort generalization."
    ),
)

SESSION_CONFIG = SessionConfig(
    sessions_root=server_home("pipeml") / "runs" / "sessions",
    stage_by_tool={'train_ml_model': 'model_training', 'train_and_evaluate_ml_model': 'model_training_and_evaluation'},
    export_requires=[frozenset({'model_training', 'model_training_and_evaluation'})],
    stage_explanations={'model_training': 'no model has been trained yet (train_ml_model or train_and_evaluate_ml_model) -- there is nothing to export.', 'model_training_and_evaluation': 'no model has been trained yet (train_ml_model or train_and_evaluate_ml_model) -- there is nothing to export.'},
)

_run = make_run(runs_root=server_home("pipeml") / "runs")

register_session_tools(mcp, SESSION_CONFIG, domain="ML/survival modeling")


# --------------------------------------------------------------------------
# pipeML tools
# --------------------------------------------------------------------------

MLMetric = Literal["Accuracy", "AUROC", "AUPRC", "C-index"]

_TRAINING_GLOSSARY = {
    "Model": "The trained model object(s) -- or the meta-learner plus base models if stack=True.",
    "Features": "The exact feature set used for training after pipeML's internal preprocessing (near-zero-variance removal, collinearity filtering, optional Boruta selection).",
    "CV_Results": "Cross-validated performance (metric per fold/repetition) from the repeated or LODO cross-validation -- this is the honest, leakage-free estimate of how the model would perform on unseen data from this cohort, not a training-set fit.",
    "AUROC_median": "Median AUROC across CV folds/repetitions -- discrimination between classes irrespective of a decision threshold. 0.5 = no better than chance, 1.0 = perfect separation.",
    "AUPRC_median": "Median AUPRC (area under precision-recall curve) across CV folds/repetitions -- more informative than AUROC when the positive class is rare, since it isn't inflated by a large pool of easy true negatives.",
    "ML_Models": "The per-algorithm trained models compared during selection (before/alongside picking the best one).",
}

_EVAL_GLOSSARY = {
    "Model": "The trained model (classification) or refitted best model (survival) evaluated on the test set.",
    "Metrics": "Performance metrics computed on features_test -- the actual held-out evaluation, not a CV estimate.",
    "AUC": "For classification: AUROC and AUPRC computed on the test set.",
    "Prediction": "Predicted class probabilities (classification) or risk scores (survival) for each test-set sample.",
    "CV_Results": "Cross-validation results from training (median/MAD of the tuning metric) -- the model-selection evidence, separate from the test-set evaluation above.",
    "Test_CINDEX": "Concordance index on the test set (survival only) -- the fraction of comparable patient pairs correctly ordered by predicted risk; 0.5 = random, 1.0 = perfect ranking.",
    "KM_Plot": "Kaplan-Meier plot object comparing survival curves across predicted risk strata (Low/Medium/High), if return=True.",
}


def _cv_diagnostics(summary: dict) -> list[str]:
    warnings: list[str] = []
    elements = summary.get("elements", {})
    for key in ("AUROC_median", "AUPRC_median"):
        el = elements.get(key, {})
        preview = el.get("preview")
        if preview and isinstance(preview, list) and preview and isinstance(preview[0], (int, float)):
            val = preview[0]
            if val < 0.55:
                warnings.append(
                    f"{key.replace('_', ' ')} is close to chance-level ({val:.2f}). This usually "
                    "means the features don't separate the target well with this cohort's sample "
                    "size, the target itself may not be predictable from these features, or the "
                    "positive class is small enough that CV folds are noisy -- consider more "
                    "repetitions (n_rep), a simpler feature set, or checking class balance before "
                    "trusting this model."
                )
    return warnings


@mcp.tool(annotations=READ_ANNOTATIONS)
async def train_ml_model(
    features_train_csv: Annotated[str, Field(description="Training features CSV (samples as rows, features as columns) -- typically deconvolution or TF-module features from multideconv-server/CellTFusion-server, NOT including the target column itself.")],
    task_type: Annotated[
        Literal["classification", "survival"],
        Field(
            description=(
                "'classification' predicts a categorical outcome (e.g. treatment responder vs "
                "non-responder) via AUROC/AUPRC/Accuracy. 'survival' predicts time-to-event risk "
                "(e.g. progression-free survival) via a Cox/AFT/survival-forest-family model "
                "evaluated by concordance index (C-index)."
            )
        ),
    ],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    target_csv: Annotated[str | None, Field(description="CSV with a target column, row-aligned to (same rows/order as) features_train_csv -- required for classification.")] = None,
    target_column: Annotated[str | None, Field(description="Column in target_csv to use as the classification target.")] = None,
    trait_positive: Annotated[str | None, Field(description="Value in the target column that represents the positive class (e.g. 'Responder') -- determines which direction AUROC/AUPRC are computed in.")] = None,
    time_column: Annotated[str | None, Field(description="Column in target_csv with survival time (survival tasks).")] = None,
    event_column: Annotated[str | None, Field(description="Column in target_csv with the event indicator: 1 = event occurred (e.g. death, progression), 0 = censored (survival tasks).")] = None,
    metric: Annotated[MLMetric | None, Field(description="Metric to optimize hyperparameter selection against. AUROC/AUPRC for imbalanced classification, Accuracy for balanced classification, C-index for survival.")] = None,
    stack: Annotated[bool, Field(description="Ensemble-stack multiple base algorithms via a meta-learner instead of picking one best algorithm -- usually more robust, at the cost of interpretability and runtime.")] = False,
    k_folds: Annotated[int, Field(ge=2, description="Number of cross-validation folds.")] = 10,
    n_rep: Annotated[int, Field(ge=1, description="Number of repeated CV repetitions -- more repetitions give a more stable performance estimate at proportionally more compute.")] = 5,
    lodo: Annotated[bool, Field(description="Leave-one-dataset-out CV (stratified by batch_column) instead of random k-fold -- tests whether the model generalizes across cohorts/sequencing batches, a stronger check than random-split k-fold when you have multiple source cohorts.")] = False,
    batch_csv: Annotated[str | None, Field(description="CSV with a cohort/batch column, required if lodo=True.")] = None,
    batch_column: Annotated[str | None, Field(description="Column in batch_csv identifying each sample's cohort, required if lodo=True.")] = None,
    file_name: Annotated[str | None, Field(description="File name prefix for saved performance plots.")] = None,
    ncores: Annotated[int | None, Field(description="CPU cores for parallel CV; default detectCores()-1.")] = None,
) -> RunResult:
    """Train a leakage-free classification/survival model with repeated or LODO CV
    (pipeML::compute_features.training.ML).

    WHAT: pipeML's "leakage-free" claim matters specifically when features depend on the whole
    dataset's context (e.g. enrichment scores, cross-sample correlations, or network features
    computed relative to a full cohort) -- computing those once on the full data before
    cross-validating lets the model implicitly see test-fold information, inflating performance.
    This tool trains and cross-validates only (no held-out test-set evaluation) -- use
    train_and_evaluate_ml_model instead if you also have a separate test set to score.

    WHEN: after you have a feature table (from multideconv-server, CellTFusion-server, or
    RaCInG-server) and want an honest estimate of how well it predicts an outcome, without yet
    committing to a specific held-out test split.

    Default k-fold/LODO fold construction only -- pipeML's custom `fold_construction_fun` hook
    (for recomputing context-dependent features independently per fold) isn't exposed here since
    it takes an R function object, not serializable data. This can be a slow call with many
    folds/repetitions or stacking enabled; progress is reported periodically.

    Known upstream issue: pass metric explicitly as "AUROC" or "AUPRC" for classification tasks.
    Leaving metric unset (or "Accuracy") currently crashes inside pipeML itself with
    `object 'hyperparams' not found` -- a bug in compute_features.training.ML, not in this
    wrapper (see this server's README).
    """
    args = args_from(
        **{
            "features_train": csv_arg(features_train_csv, row_names=1),
            "task_type": lit(task_type),
            "target_var": csv_column_arg(target_csv, target_column) if (target_csv and target_column) else None,
            "trait.positive": lit(trait_positive) if trait_positive else None,
            "time_var": csv_column_arg(target_csv, time_column) if (target_csv and time_column) else None,
            "event_var": csv_column_arg(target_csv, event_column) if (target_csv and event_column) else None,
            "metric": lit(metric) if metric else None,
            "stack": lit(stack),
            "k_folds": lit(k_folds),
            "n_rep": lit(n_rep),
            "LODO": lit(lodo),
            "batch_var": csv_column_arg(batch_csv, batch_column) if (batch_csv and batch_column) else None,
            "file_name": lit(file_name) if file_name else None,
            "ncores": lit(ncores) if ncores is not None else None,
            "return": lit(True),
        }
    )
    return await _run(
        package="pipeML", function="compute_features.training.ML", args=args, tool_name="train_ml_model",
        ctx=ctx, timeout=3600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary=_TRAINING_GLOSSARY, diagnostics=_cv_diagnostics,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def train_and_evaluate_ml_model(
    features_train_csv: Annotated[str, Field(description="Training features CSV (samples as rows, features as columns).")],
    features_test_csv: Annotated[str, Field(description="Held-out test features CSV, same columns as features_train_csv -- samples the model never sees during training or CV-based selection.")],
    coldata_csv: Annotated[str, Field(description="Metadata CSV (rows = train+test samples) with the trait/time/event/batch columns referenced below.")],
    task_type: Annotated[Literal["classification", "survival"], Field(description="'classification' for a categorical outcome, 'survival' for time-to-event risk.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    trait_column: Annotated[str | None, Field(description="Column in coldata_csv with the classification target (classification tasks).")] = None,
    trait_positive: Annotated[str | None, Field(description="Value in trait_column representing the positive class.")] = None,
    time_column: Annotated[str | None, Field(description="Column in coldata_csv with survival time (survival tasks).")] = None,
    event_column: Annotated[str | None, Field(description="Column in coldata_csv with the event indicator (survival tasks).")] = None,
    metric: MLMetric = "Accuracy",
    stack: Annotated[bool, Field(description="Ensemble-stack multiple base algorithms instead of picking one best.")] = False,
    k_folds: Annotated[int, Field(ge=2)] = 10,
    n_rep: Annotated[int, Field(ge=1)] = 5,
    lodo: Annotated[bool, Field(description="Leave-one-dataset-out CV instead of k-fold, for cross-cohort generalization testing.")] = False,
    batch_column: Annotated[str | None, Field(description="Column in coldata_csv identifying each sample's cohort, required if lodo=True.")] = None,
    file_name: Annotated[str | None, Field(description="File name prefix for saved performance plots.")] = None,
    ncores: Annotated[int | None, Field(description="CPU cores for parallel CV.")] = None,
) -> RunResult:
    """Train on features_train and evaluate on a held-out features_test in one call
    (pipeML::compute_features.ML).

    WHAT: combines training (with the same leakage-free repeated/LODO CV as train_ml_model, used
    for model selection) with a final evaluation on features_test -- samples the model never
    influenced during training or hyperparameter selection. For classification this reports
    test-set AUROC/AUPRC/accuracy; for survival it fits a Cox/AFT/survival-forest-family model,
    reports test-set C-index, and plots Kaplan-Meier curves across quantile-based risk strata
    (Low/Medium/High) with a log-rank test.

    WHEN: use this once you have a genuine held-out test split (a separate cohort, a temporal
    split, or a reserved subset) and want a single number for "how well does this model do on data
    it hasn't seen at all" -- not just the CV estimate train_ml_model gives you.

    Can be slow with stacking or many folds/repetitions; progress is reported periodically.

    Known upstream issue: pass metric="AUROC" or "AUPRC" explicitly for classification tasks --
    the default "Accuracy" currently crashes inside pipeML itself with
    `object 'hyperparams' not found`, a bug in compute_features.ML, not in this wrapper (see this
    server's README).
    """
    args = args_from(
        **{
            "features_train": csv_arg(features_train_csv, row_names=1),
            "features_test": csv_arg(features_test_csv, row_names=1),
            "coldata": csv_arg(coldata_csv, row_names=1),
            "task_type": lit(task_type),
            "trait": lit(trait_column) if trait_column else None,
            "trait.positive": lit(trait_positive) if trait_positive else None,
            "time_var": lit(time_column) if time_column else None,
            "event_var": lit(event_column) if event_column else None,
            "metric": lit(metric),
            "stack": lit(stack),
            "k_folds": lit(k_folds),
            "n_rep": lit(n_rep),
            "LODO": lit(lodo),
            "batch_id": lit(batch_column) if batch_column else None,
            "file_name": lit(file_name) if file_name else None,
            "ncores": lit(ncores) if ncores is not None else None,
            "return": lit(True),
        }
    )
    return await _run(
        package="pipeML", function="compute_features.ML", args=args, tool_name="train_and_evaluate_ml_model",
        ctx=ctx, timeout=3600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary=_EVAL_GLOSSARY,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
