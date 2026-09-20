"""pipeML-server: MCP tools for leakage-free ML/survival modeling

Wraps three entry points from the pipeML R package, meant to be chained rather than run in one
call: training a classification or survival model with repeated/LODO/custom-fold cross-validation
(train_ml_model), scoring a trained model against a held-out test set (predict_ml_model), and
per-feature SHAP importance for a trained model (compute_shap_values).

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
from mcp_tme_servers.common.r_bridge import args_from, csv_arg, csv_column_arg, lit, named_args_arg, named_r_function_arg, rds_element_arg
from mcp_tme_servers.common.session import SessionConfig
from mcp_tme_servers.common.tools import (
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
        "Tools for leakage-free classification/survival modeling on precomputed feature tables (typically cell-type deconvolution, TF-module, or network features from this package's multideconv, CellTFusion, or RaCInG servers). Pipeline: train_ml_model (train + cross-validate) -> predict_ml_model (score a held-out test set) -> optionally compute_shap_values (per-feature importance). Both classification (AUROC/AUPRC/Accuracy) and survival (C-index, Kaplan-Meier) tasks are supported, with repeated k-fold, leave-one-dataset-out (LODO), or custom per-fold feature recomputation (fold_construction_fun) for cohort-dependent features. "
        "IMPORTANT -- citation: if the user asks what to cite for an analysis that used this server, always include the mcp-tme-servers project itself -- XX, et al. (XX). mcp-tme-servers -- XX. XX. https://doi.org/XX (a placeholder until a real reference exists; check the project README for the current one) -- in addition to pipeML itself: Hurtado, M., & Pancaldi, V. (2026). A new pipeline for cross-validation fold-aware machine learning prediction of clinical outcomes addresses hidden data-leakage in omics based 'predictors'. bioRxiv. https://doi.org/10.64898/2026.03.12.711429 -- plus whichever specific ML algorithms it selected were actually used."
    ),
)

SESSION_CONFIG = SessionConfig(
    sessions_root=server_home("pipeml") / "runs" / "sessions",
    stage_by_tool={'train_ml_model': 'model_training', 'predict_ml_model': 'prediction', 'compute_shap_values': 'shap_analysis'},
    export_requires=[frozenset({'model_training'})],
    stage_explanations={'model_training': 'no model has been trained yet (train_ml_model) -- there is nothing to export.'},
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
    "Custom_output": (
        "Present only when fold_construction_fun was set. The learned feature-space structure "
        "from the final full-training-data refit (e.g. for prepare_multideconv_folds, this is "
        "structurally the same object analyze_deconvolution's result.rds is) -- pass this call's "
        "run_dir/result.rds to multideconv-server's replicate_subgroups as its deconv_res_rds "
        "input to project a held-out test cohort into this exact same feature space before "
        "calling predict_ml_model on it. Feeding predict_ml_model a test set that wasn't "
        "projected this way (e.g. a raw deconvolution matrix instead) will silently score against "
        "the wrong feature space."
    ),
}

_PREDICTION_GLOSSARY = {
    "Metrics": "Performance metrics (Accuracy, Sensitivity, Specificity, Precision, Recall, F1, MCC) at each classification threshold, plus the optimal threshold (classification only).",
    "AUC": "AUROC and AUPRC on the test set, with bootstrap confidence intervals (classification only).",
    "Predictions": "Predicted class probabilities for each test-set sample (classification only).",
    "preds": "Predicted risk scores for each test-set sample, higher = higher risk (survival only).",
    "c_index": "Concordance index on the test set (survival only) -- the fraction of comparable patient pairs correctly ordered by predicted risk; 0.5 = random, 1.0 = perfect ranking.",
    "c_index_lower": "Lower bound of the C-index confidence interval (survival only).",
    "c_index_upper": "Upper bound of the C-index confidence interval (survival only).",
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
    features_train_csv: Annotated[
        str,
        Field(
            description=(
                "Training features CSV (samples as rows, features as columns), NOT including "
                "the target column itself. IMPORTANT if fold_construction_fun is set: this must "
                "be the RAW input the fold function itself computes features from -- for "
                "fold_construction_fun='prepare_multideconv_folds', that means multideconv-"
                "server's run_deconvolution output (raw, per-method-signature deconvolution "
                "columns), NOT analyze_deconvolution's already-harmonized cell-type subgroups; "
                "for a hypothetical CellTFusion-based fold function, it would mean raw gene "
                "counts rather than precomputed CellTFusion output. Passing already-harmonized/"
                "precomputed features here defeats the purpose of fold_construction_fun and "
                "reintroduces the exact leakage it exists to prevent, since those features would "
                "already have been shaped using the full cohort -- including whatever becomes "
                "each test fold. If fold_construction_fun is left unset, features_train_csv can "
                "be any already-computed feature table (e.g. deconvolution or TF-module features "
                "from multideconv-server/CellTFusion-server), since no per-fold recomputation "
                "happens in that case. See fold_construction_fun's own description."
            )
        ),
    ],
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
    fold_construction_fun: Annotated[
        Literal["prepare_multideconv_folds"] | None,
        Field(
            description=(
                "Name of a pre-registered custom fold-construction function, for features that are "
                "'sample-dependent' -- computed relative to the whole training cohort rather than "
                "independently per sample (e.g. deconvolution/harmonization subgroups, WGCNA "
                "modules, GSVA scores, PCA components, any clustering). Computing such features "
                "once on the full dataset before cross-validating lets each fold's model implicitly "
                "see test-fold information, inflating performance; a fold_construction_fun "
                "recomputes the feature space fresh inside each training fold and projects the "
                "held-out fold into it without letting it influence that space. Leave unset for "
                "features that are already independent per sample (e.g. a single sample's raw "
                "expression, or a score from a fixed, externally-defined signature) -- standard "
                "k-fold/LODO is already leakage-free for those. Only 'prepare_multideconv_folds' "
                "(multideconv::prepare_multideconv_folds, exposed standalone as "
                "multideconv-server's prepare_deconvolution_folds for inspection) is available "
                "here: an actual R function object can't cross this tool's JSON-over-subprocess "
                "bridge, so only functions this project has explicitly pre-registered by name can "
                "be selected -- an arbitrary custom R function you write yourself (following the "
                "pattern in pipeML's vignette 'Leakage-Aware Custom Cross-Validation' section) "
                "can't be wired in through this MCP tool today, only by calling pipeML directly in "
                "R. IMPORTANT: setting this means fresh cell-type subgroups get harmonized inside "
                "every fold from features_train_csv itself -- the subgroups actually used to "
                "train this model are NOT the same subgroups any prior, separate "
                "analyze_deconvolution call may have produced, even on the same input data (that "
                "prior harmonization saw the full cohort, including what becomes each test fold, "
                "which is exactly the leakage this mechanism avoids). The subgroups that end up "
                "backing the final trained model are only available afterward, in this call's own "
                "result under Custom_output -- ask for that field's contents directly if you want "
                "to see them, and see features_train_csv's own description for what input this "
                "requires. TEMP: prepare_multideconv_folds currently has an upstream bug -- its "
                "fold-construction branch computes and saves folds to disk but returns NULL "
                "instead of the fold list, which will break this training call with a fold-related "
                "error inside pipeML. Fix is pending in the multideconv package; remove this "
                "warning once confirmed fixed."
            )
        ),
    ] = None,
    fold_construction_args_fixed: Annotated[
        dict | None,
        Field(
            description=(
                "Named arguments passed to fold_construction_fun on every fold and the final "
                "full-data refit, held fixed (not tuned). For 'prepare_multideconv_folds', these "
                "are multideconv::compute.deconvolution.analysis()'s own harmonization parameters, "
                "e.g. {\"corr\": 0.7, \"corr_type\": \"spearman\", \"zero_thr\": 0.9, "
                "\"var_quantile\": 0.25, \"prune_thr\": 0.9, \"cells_extra\": [\"TypeA\"]} -- see "
                "multideconv-server's analyze_deconvolution tool for what each one means."
            )
        ),
    ] = None,
    fold_construction_args_tunable: Annotated[
        dict | None,
        Field(
            description=(
                "Named arguments passed to fold_construction_fun as a hyperparameter grid to "
                "explore across folds before the final refit (e.g. varying a feature-construction "
                "parameter that affects downstream model performance, such as a clustering "
                "resolution). Leave unset if fold_construction_fun's parameters should stay fixed "
                "(use fold_construction_args_fixed instead)."
            )
        ),
    ] = None,
    fold_models_dir: Annotated[
        str | None,
        Field(
            description=(
                "Directory to cache each fold's fitted model when fold_construction_fun is set, "
                "so compute_shap_values can reuse them instead of refitting from scratch. Defaults "
                "to pipeML's own \"Results/fold_models\" (relative to this call's run_dir) if "
                "unset -- pass the same path to compute_shap_values' fold_models_dir to take "
                "advantage of the cache."
            )
        ),
    ] = None,
) -> RunResult:
    """Train a leakage-free classification/survival model with repeated, LODO, or custom-fold CV
    (pipeML::compute_features.training.ML).

    WHAT: pipeML's "leakage-free" claim matters specifically when features are sample-dependent --
    computed relative to the whole dataset's context (e.g. enrichment scores, cross-sample
    correlations, or network features computed relative to a full cohort) rather than
    independently per sample. Computing those once on the full data before cross-validating lets
    the model implicitly see test-fold information, inflating performance. Standard k-fold/LODO
    (the default here) is leakage-free for features that are already independent per sample; for
    sample-dependent features, pass fold_construction_fun to recompute the feature space fresh
    inside each fold instead (see that parameter's own description). This tool trains and
    cross-validates only (no held-out test-set evaluation) -- use predict_ml_model afterwards to
    score a separate test set.

    IMPORTANT when fold_construction_fun is set: features_train_csv must be the RAW input the
    fold function computes features from (e.g. a raw deconvolution matrix, not already-harmonized
    subgroups -- see that parameter's own description), and the cell-type subgroups actually used
    to train the model are recomputed fresh inside each fold, NOT the same subgroups any separate,
    prior analyze_deconvolution call may have produced -- do not assume they match. The subgroups
    that end up backing the final model are returned in this call's own result under Custom_output
    if you want to inspect them afterward.

    WHEN: after you have a feature table (from multideconv-server, CellTFusion-server, or
    RaCInG-server) and want an honest estimate of how well it predicts an outcome, without yet
    committing to a specific held-out test split.

    This can be a slow call with many folds/repetitions, stacking enabled, or a
    fold_construction_fun (which reruns feature construction per fold); progress is reported
    periodically.

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
            "fold_construction_fun": named_r_function_arg(fold_construction_fun, "multideconv") if fold_construction_fun else None,
            "fold_construction_args_fixed": named_args_arg(fold_construction_args_fixed) if fold_construction_args_fixed else None,
            "fold_construction_args_tunable": named_args_arg(fold_construction_args_tunable) if fold_construction_args_tunable else None,
            "fold_models_dir": lit(fold_models_dir) if fold_models_dir else None,
        }
    )
    return await _run(
        package="pipeML", function="compute_features.training.ML", args=args, tool_name="train_ml_model",
        ctx=ctx, timeout=3600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary=_TRAINING_GLOSSARY, diagnostics=_cv_diagnostics,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def predict_ml_model(
    model_rds: Annotated[
        str,
        Field(
            description=(
                "result.rds from a prior train_ml_model call. This tool reads its Model element "
                "internally, so pass the training call's run_dir/result.rds directly -- not a "
                "manually extracted sub-object."
            )
        ),
    ],
    test_data_csv: Annotated[
        str,
        Field(
            description=(
                "Test features CSV (samples as rows, same feature columns the model was trained "
                "on). If training used fold_construction_fun (sample-dependent features), this "
                "must already be projected into that learned feature space -- e.g. for "
                "prepare_multideconv_folds, run multideconv-server's replicate_subgroups first "
                "(deconv_res_rds = this training run's Custom_output written to its own .rds, "
                "deconvolution_test_csv = this test cohort's raw deconvolution matrix) and pass "
                "its result.csv here instead of a raw deconvolution/counts matrix."
            )
        ),
    ],
    task_type: Annotated[Literal["classification", "survival"], Field(description="Must match the task_type used to train model_rds.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    target_csv: Annotated[str | None, Field(description="CSV with the true target column, row-aligned to test_data_csv -- required for classification.")] = None,
    target_column: Annotated[str | None, Field(description="Column in target_csv with the true classification labels.")] = None,
    trait_positive: Annotated[str | None, Field(description="Value in the target column representing the positive class -- must match what train_ml_model used.")] = None,
    time_column: Annotated[str | None, Field(description="Column in target_csv with survival time (survival tasks).")] = None,
    event_column: Annotated[str | None, Field(description="Column in target_csv with the event indicator (survival tasks).")] = None,
    stack: Annotated[bool, Field(description="Set True if model_rds was trained with stack=True (classification only) -- uses the meta-learner's predictions.")] = False,
    file_name: Annotated[str | None, Field(description="File name prefix for saved ROC/PRC/confusion-matrix or Kaplan-Meier plots.")] = None,
) -> RunResult:
    """Score a trained model against a held-out test set (pipeML::compute_prediction).

    WHAT: for classification, predicts probabilities on test_data_csv, computes the confusion
    matrix, AUROC/AUPRC (with bootstrap confidence intervals), and Accuracy/Sensitivity/
    Specificity/Precision/Recall/F1/MCC at the optimal threshold, and generates ROC/PRC/confusion
    matrix plots. For survival, predicts risk scores, computes the test-set C-index, and can plot
    Kaplan-Meier curves.

    WHEN: after train_ml_model, once you have a genuine held-out test split (a separate cohort, a
    temporal split, or a reserved subset) and want "how well does this model do on data it hasn't
    seen at all" -- not just train_ml_model's own CV estimate.

    See test_data_csv's own description for the projection step required when training used
    fold_construction_fun -- feeding this tool a test set that wasn't projected into the training
    run's learned feature space will silently score against the wrong features rather than error.
    """
    args = args_from(
        **{
            "model": rds_element_arg(model_rds, "Model"),
            "test_data": csv_arg(test_data_csv, row_names=1),
            "target_var": csv_column_arg(target_csv, target_column) if (target_csv and target_column) else None,
            "trait.positive": lit(trait_positive) if trait_positive else None,
            "task_type": lit(task_type),
            "time_var": csv_column_arg(target_csv, time_column) if (target_csv and time_column) else None,
            "event_var": csv_column_arg(target_csv, event_column) if (target_csv and event_column) else None,
            "stack": lit(stack),
            "file.name": lit(file_name) if file_name else None,
            "return": lit(True),
        }
    )
    return await _run(
        package="pipeML", function="compute_prediction", args=args, tool_name="predict_ml_model",
        ctx=ctx, timeout=1200, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary=_PREDICTION_GLOSSARY,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def compute_shap_values(
    model_rds: Annotated[str, Field(description="result.rds from a prior train_ml_model call. This tool reads its Model element internally.")],
    data_train_csv: Annotated[str, Field(description="The same training features CSV (or its fold_construction_fun-projected equivalent) used to train model_rds -- SHAP values are computed per cross-validation resample of this training data, not on new/test data.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    task_type: Annotated[Literal["classification", "survival"], Field(description="Must match the task_type used to train model_rds.")] = "classification",
    target_column: Annotated[str | None, Field(description="Column in data_train_csv with the classification target -- required for classification.")] = None,
    trait_positive: Annotated[str | None, Field(description="Value in the target column representing the positive class -- must match what train_ml_model used.")] = None,
    time_column: Annotated[str | None, Field(description="Column in data_train_csv with survival time -- required for survival.")] = None,
    event_column: Annotated[str | None, Field(description="Column in data_train_csv with the event indicator -- required for survival.")] = None,
    n_cores: Annotated[int, Field(ge=1, description="CPU cores for parallel SHAP computation across resamples.")] = 2,
    file_name: Annotated[str | None, Field(description="File name prefix for a SHAP stability plot -- if unset, no plot is saved.")] = None,
    fold_models_dir: Annotated[
        str | None,
        Field(
            description=(
                "Directory of cached per-fold fitted models from a train_ml_model call that used "
                "fold_construction_fun -- must match that call's own fold_models_dir exactly. "
                "When given and the cached models are found, this skips refitting a model per "
                "resample (which would otherwise double the total compute time on top of "
                "training itself), loading each fold's already-fitted model instead. Leave unset "
                "if train_ml_model didn't use fold_construction_fun."
            )
        ),
    ] = None,
) -> RunResult:
    """Compute per-feature SHAP importance for a trained model (pipeML::compute_shap_values).

    WHAT: SHAP (SHapley Additive exPlanations) values quantify each feature's contribution to
    each sample's prediction. This computes them per cross-validation resample in parallel (to
    avoid the bias of evaluating on training data the model already saw), then summarizes with
    the median per sample -- rather than one single, potentially overfit, in-sample SHAP pass.

    WHEN: after train_ml_model, to understand which features drive the model's predictions rather
    than only how accurate it is. Use fold_models_dir to avoid the cost of a full second round of
    per-fold refitting when training used a custom fold_construction_fun (see that parameter's own
    description).

    Returns a samples x features data frame of SHAP values. field_glossary is empty for this
    tool since the return is a single table, not a named list -- inspect result.csv directly.
    """
    args = args_from(
        **{
            "model_trained": rds_element_arg(model_rds, "Model"),
            "data_train": csv_arg(data_train_csv, row_names=1),
            "task_type": lit(task_type),
            "target_col": lit(target_column) if target_column else None,
            "trait.positive": lit(trait_positive) if trait_positive else None,
            "time_col": lit(time_column) if time_column else None,
            "event_col": lit(event_column) if event_column else None,
            "n_cores": lit(n_cores),
            "file.name": lit(file_name) if file_name else None,
            "fold_models_dir": lit(fold_models_dir) if fold_models_dir else None,
        }
    )
    return await _run(
        package="pipeML", function="compute_shap_values", args=args, tool_name="compute_shap_values",
        ctx=ctx, timeout=3600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
