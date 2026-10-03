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
from mcp.server.mcpserver.exceptions import ToolError
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
        "Tools for leakage-free classification/survival modeling on precomputed feature tables (typically cell-type deconvolution, TF-module, or network features from this package's multideconv, CellTFusion, or RaCInG servers). Pipeline: train_ml_model (train + cross-validate) -> predict_ml_model (score a held-out test set) -> optionally compute_shap_values (per-feature importance). Both classification (AUROC/AUPRC) and survival (C-index, Kaplan-Meier) tasks are supported, with repeated k-fold, leave-one-dataset-out (LODO), or custom per-fold feature recomputation (fold_construction_fun) for cohort-dependent features. "
        "IMPORTANT -- citation: if the user asks what to cite for an analysis that used this server, always include the mcp-tme-servers project itself -- XX, et al. (XX). mcp-tme-servers -- XX. XX. https://doi.org/XX (a placeholder until a real reference exists; check the project README for the current one) -- in addition to pipeML itself: Hurtado, M., & Pancaldi, V. (2026). A new pipeline for cross-validation fold-aware machine learning prediction of clinical outcomes addresses hidden data-leakage in omics based 'predictors'. bioRxiv. https://doi.org/10.64898/2026.03.12.711429 -- plus whichever specific ML algorithms it selected were actually used. "
        "IMPORTANT -- output location: once any tool call on this server finishes, always tell the user the run_dir path from that result (where result.rds/CSVs/plots/run.log were saved) -- do not just use it silently to chain into the next call without also surfacing it to the user."
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

MLMetric = Literal["AUROC", "AUPRC"]

_TRAINING_GLOSSARY = {
    "Model": (
        "The selected (best cross-validated) model, trained on all training samples with its "
        "tuned hyperparameters. Classification: a caret train object (tuned hyperparameters in "
        "$bestTune, performance per CV resample in $resample -- the honest, leakage-free "
        "estimate of how it would perform on unseen data from this cohort, not a training-set "
        "fit). Survival: a list with the fitted workflow ($Model_object), $bestTune, the C-index "
        "per resample ($Resample_matrix, whose `model` column names the selected algorithm) and "
        "the training data ($trainingData). predict_ml_model and compute_shap_values read this "
        "element."
    ),
    "ML_Models": "Every algorithm trained and cross-validated during selection (the best one is also returned as Model), with each one's per-fold CV results -- useful to compare how close the runners-up were. Classification models that predict the same value for every training sample are left out.",
    "AUROC_median": "Median AUROC across CV folds/repetitions, per algorithm (classification) -- discrimination between classes irrespective of a decision threshold. 0.5 = no better than chance, 1.0 = perfect separation. Median and MAD per algorithm.",
    "AUPRC_median": "Median AUPRC (area under precision-recall curve) across CV folds/repetitions, per algorithm (classification) -- more informative than AUROC when the positive class is rare, since it isn't inflated by a large pool of easy true negatives. Median and MAD per algorithm.",
    "C_index_median": "Median cross-validated concordance index of the selected survival model (survival only) -- the fraction of comparable patient pairs correctly ordered by predicted risk; 0.5 = random, 1.0 = perfect ranking.",
    "Custom_output": (
        "Present only when fold_construction_fun was set. The learned feature-space structure "
        "from the final full-training-data refit, needed to project a held-out test cohort into "
        "this exact same feature space before calling predict_ml_model on it. For "
        "prepare_multideconv_folds it is structurally the same object analyze_deconvolution's "
        "result.rds is -- project with multideconv-server's replicate_subgroups (deconv_res_rds). "
        "For prepare_celltfusion_folds it is a full run_celltfusion-style result (cell groups, "
        "NMF latent factors, TME states) learned on all training samples -- project with "
        "CellTFusion-server's project_celltfusion_factors, passing this call's run_dir/result.rds "
        "as its model_rds. Feeding predict_ml_model a test set that wasn't projected this way (e.g. "
        "a raw deconvolution or counts matrix instead) will silently score against the wrong "
        "feature space. The selected feature-construction parameters (when "
        "fold_construction_args_tunable was used) are in its $Parameters."
    ),
}

# Pre-registered fold construction functions -> the R package exporting each one
_FOLD_FUNCTION_PACKAGES = {
    "prepare_multideconv_folds": "multideconv",
    "prepare_celltfusion_folds": "CellTFusion",
}

# Arguments prepare_celltfusion_folds gets from this tool's own parameters, or can't take at all
_CELLTFUSION_FOLD_RESERVED = {
    "deconv": "fold_deconv_csv",
    "raw.counts": "features_train_csv (it is passed on automatically)",
    "coldata": "fold_coldata_csv",
    "dt": None,
    "tfs": None,
    "pathways": None,
}

_PREDICTION_GLOSSARY = {
    "Metrics": "Performance metrics (Accuracy, Sensitivity, Specificity, Precision, Recall, F1, MCC) at each classification threshold: one row per test sample, sorted by decreasing predicted probability (classification only).",
    "AUC": "AUROC and AUPRC on the test set, each with its estimate and the lower/upper bounds of a 95% bootstrap confidence interval (1000 resamples stratified by class) (classification only).",
    "Predictions": "Predicted class probabilities (columns no/yes) for each test-set sample, in test_data_csv's row order (classification only).",
    "Curve_bands": "Pointwise 95% bootstrap confidence bands around the ROC (fpr, lower, upper) and precision-recall (recall, lower, upper) curves, from 1000 bootstrap resamples of the test set (classification only) -- the shaded bands in the saved ROC/PRC plots.",
    "preds": "Predicted risk scores for each test-set sample, higher = higher risk (survival only).",
    "c_index": "Concordance index on the test set (survival only) -- the fraction of comparable patient pairs correctly ordered by predicted risk; 0.5 = random, 1.0 = perfect ranking.",
    "c_index_lower": "Lower bound of the C-index 95% confidence interval, from 1000 bootstrap resamples stratified by event (survival only).",
    "c_index_upper": "Upper bound of the C-index confidence interval (survival only).",
}



def _cv_diagnostics(summary: dict) -> list[str]:
    warnings: list[str] = []
    elements = summary.get("elements", {})
    for key in ("AUROC_median", "AUPRC_median", "C_index_median"):
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
                "Training features CSV (samples as rows, features as columns -- or the reverse "
                "with features_as_rows=True), NOT including the target column itself. IMPORTANT "
                "if fold_construction_fun is set: this must be the RAW input the fold function "
                "itself computes features from -- for fold_construction_fun="
                "'prepare_multideconv_folds', that means multideconv-server's run_deconvolution "
                "output (raw, per-method-signature deconvolution columns), NOT "
                "analyze_deconvolution's already-harmonized cell-type subgroups; for "
                "'prepare_celltfusion_folds', it means the RAW gene counts (typically the usual "
                "genes x samples counts CSV, with features_as_rows=True), NOT a precomputed "
                "run_celltfusion output (TF activities, cell groups, latent factors). Passing already-harmonized/"
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
                "non-responder) via AUROC/AUPRC. 'survival' predicts time-to-event risk "
                "(e.g. progression-free survival) via a Cox/AFT/survival-forest-family model "
                "evaluated by concordance index (C-index)."
            )
        ),
    ],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    target_csv: Annotated[str | None, Field(description="CSV with the target column (or survival time/event columns), sample IDs as its first column -- required. Its values are matched to features_train_csv's samples by those IDs, so it may hold extra samples or a different order; only when IDs don't match does it fall back to row position, which then requires the same number of rows.")] = None,
    target_column: Annotated[str | None, Field(description="Column in target_csv to use as the classification target. It must have no missing values for the training samples (drop those samples first) and contain samples both with and without trait_positive -- pipeML stops otherwise.")] = None,
    trait_positive: Annotated[str | None, Field(description="Value in the target column that represents the positive class (e.g. 'Responder') -- determines which direction AUROC/AUPRC are computed in.")] = None,
    time_column: Annotated[str | None, Field(description="Column in target_csv with survival time (survival tasks).")] = None,
    event_column: Annotated[str | None, Field(description="Column in target_csv with the event indicator, coded exactly 1 = event occurred (e.g. death, progression) and 0 = censored, without missing values (survival tasks) -- pipeML stops on any other coding (e.g. 1/2 or text labels), so recode it first.")] = None,
    metric: Annotated[MLMetric | None, Field(description="Metric to select hyperparameters and the best algorithm by (classification only): 'AUROC' (used when unset) or 'AUPRC', which is more informative when the positive class is rare. Ignored for survival, which always selects by C-index.")] = None,
    k_folds: Annotated[int, Field(ge=2, description="Number of cross-validation folds.")] = 10,
    n_rep: Annotated[int, Field(ge=1, description="Number of repeated CV repetitions -- more repetitions give a more stable performance estimate at proportionally more compute.")] = 5,
    lodo: Annotated[bool, Field(description="Leave-one-dataset-out CV (stratified by batch_column) instead of random k-fold -- tests whether the model generalizes across cohorts/sequencing batches, a stronger check than random-split k-fold when you have multiple source cohorts. Folds are stratified by cohort and outcome (the cohort itself is never used as a predictor), and every cohort needs at least k_folds samples.")] = False,
    batch_csv: Annotated[str | None, Field(description="CSV with a cohort/batch column, required if lodo=True.")] = None,
    batch_column: Annotated[str | None, Field(description="Column in batch_csv identifying each sample's cohort, required if lodo=True.")] = None,
    file_name: Annotated[str | None, Field(description="File name prefix for saved performance plots.")] = None,
    ncores: Annotated[int | None, Field(ge=1, description="CPU cores to process cross-validation folds in parallel (classification and survival; results don't depend on it). If unset, everything runs sequentially. With fold_construction_fun, classification models are trained sequentially whatever this is, and survival only uses it for the parameter combinations of fold_construction_args_tunable -- the fold function's own parallelism is its \"ncores\" in fold_construction_args_fixed.")] = None,
    seed: Annotated[int, Field(description="Random seed fixing the fold assignment and the randomness in model fitting (random forest, bagging, boosting), also when running in parallel -- the same seed and inputs give the same model and CV results. Randomness inside fold_construction_fun's own parallel workers is not covered.")] = 123,
    features_as_rows: Annotated[
        bool,
        Field(
            description=(
                "Set True when features_train_csv has features as rows and samples as columns -- "
                "e.g. the usual genes x samples counts CSV, which is what "
                "fold_construction_fun='prepare_celltfusion_folds' needs. It is transposed for "
                "pipeML, and target_csv/batch_csv/fold_deconv_csv/fold_coldata_csv are then "
                "matched to its column names (sample IDs)."
            )
        ),
    ] = False,
    preprocess: Annotated[
        bool,
        Field(
            description=(
                "If True (default), near-constant features are removed and, for each pair of "
                "features with |correlation| > 0.9, one of the two -- the outcome is not used, and "
                "the features must be numeric. Without fold_construction_fun this is done once on "
                "all training samples before cross-validation; with it, inside each fold on the "
                "features built from the training part (and again for the final model). "
                "predict_ml_model keeps only the features the final model retained. Set False to "
                "train on the features exactly as given (e.g. a small, hand-picked feature set "
                "where correlated features should all stay)."
            )
        ),
    ] = True,
    fold_construction_fun: Annotated[
        Literal["prepare_multideconv_folds", "prepare_celltfusion_folds"] | None,
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
                "k-fold/LODO is already leakage-free for those. Two functions are available: "
                "'prepare_multideconv_folds' (multideconv::prepare_multideconv_folds, exposed "
                "standalone as multideconv-server's prepare_deconvolution_folds for inspection) "
                "harmonizes deconvolution cell-type subgroups per fold; 'prepare_celltfusion_folds' "
                "(CellTFusion::prepare_celltfusion_folds, exposed standalone as CellTFusion-server's "
                "prepare_celltfusion_folds for inspection) runs the whole CellTFusion pipeline per "
                "fold -- TF activity, TF modules, cell groups and NMF latent factors are learned on "
                "the fold's training samples only, and the held-out samples are projected onto those "
                "factors; the model is trained on the latent factors. It needs raw counts as "
                "features_train_csv (features_as_rows=True for a genes x samples CSV) plus "
                "fold_deconv_csv, and is much heavier (one full CellTFusion run per fold, plus one "
                "on all samples). Only these two are available "
                "here: an actual R function object can't cross this tool's JSON-over-subprocess "
                "bridge, so only functions this project has explicitly pre-registered by name can "
                "be selected -- an arbitrary custom R function you write yourself (following the "
                "pattern in pipeML's vignette 'Leakage-Aware Custom Cross-Validation' section) "
                "can't be wired in through this MCP tool today, only by calling pipeML directly in "
                "R. IMPORTANT: setting this means the feature space is rebuilt inside every fold "
                "from features_train_csv itself -- the cell-type subgroups "
                "(prepare_multideconv_folds) or CellTFusion latent factors "
                "(prepare_celltfusion_folds) actually used to train this model are NOT the same as "
                "any prior, separate analyze_deconvolution/run_celltfusion call may have produced, "
                "even on the same input data (that prior run saw the full cohort, including what "
                "becomes each test fold, which is exactly the leakage this mechanism avoids). The "
                "feature space that ends up backing the final trained model is only available "
                "afterward, in this call's own result under Custom_output -- ask for that field's "
                "contents directly if you want to see them, and see features_train_csv's own "
                "description for what input this requires. Works for classification and survival: "
                "pipeML hands the fold function the features plus the outcome columns (target, or "
                "time and event), which it keeps out of the feature construction."
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
                "\"cv_thr\": 0.1, \"cells_extra\": [\"TypeA\"]} -- see "
                "multideconv-server's analyze_deconvolution tool for what each one means. For "
                "'prepare_celltfusion_folds', these are CellTFusion() arguments under their R names "
                "(e.g. {\"normalized\": true, \"TF.collection\": \"CollecTRI\", "
                "\"min_targets_size\": 5, \"minMod\": 5, \"corr_mod\": 0.9, \"corr\": 0.7, "
                "\"pval\": 0.05, \"cancer_type\": \"skcm\"} -- see CellTFusion-server's run_celltfusion "
                "for what each one means), plus \"ncores\" (folds computed in parallel, default 1), "
                "\"batch\"/\"batch_id\" (multi-cohort mode, needs fold_coldata_csv) and "
                "\"verbose\". deconv, raw.counts and coldata come from this tool's own parameters; "
                "dt, tfs and pathways can't be given (they must be computed per fold); "
                "TF.collection='ARACNE' isn't supported here."
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
                "(use fold_construction_args_fixed instead). Not supported by "
                "'prepare_celltfusion_folds'."
            )
        ),
    ] = None,
    fold_deconv_csv: Annotated[
        str | None,
        Field(
            description=(
                "fold_construction_fun='prepare_celltfusion_folds' only, and required there: the "
                "RAW deconvolution matrix of all training samples (samples as rows), e.g. "
                "multideconv-server's run_deconvolution result.csv -- not analyze_deconvolution's "
                "harmonized subgroups. Deconvolution is computed independently per sample, so "
                "computing it once for the whole cohort doesn't leak between folds; the "
                "cross-method subgroup analysis on top of it is redone per fold. Matched to the "
                "training samples by sample ID."
            )
        ),
    ] = None,
    fold_coldata_csv: Annotated[
        str | None,
        Field(
            description=(
                "fold_construction_fun='prepare_celltfusion_folds' only: sample metadata CSV "
                "(samples as rows), required when fold_construction_args_fixed sets \"batch\": true "
                "(multi-cohort mode), to read its \"batch_id\" column. Matched to the training "
                "samples by sample ID."
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
    fold function computes features from (a raw deconvolution matrix for
    prepare_multideconv_folds, raw gene counts plus fold_deconv_csv for prepare_celltfusion_folds
    -- see that parameter's own description), and the features actually used to train the model
    (cell-type subgroups, or CellTFusion latent factors) are recomputed fresh inside each fold, NOT
    the same ones any separate, prior analyze_deconvolution/run_celltfusion call may have produced
    -- do not assume they match. The feature space that ends up backing the final model is
    returned in this call's own result under Custom_output if you want to inspect it afterward,
    and a test cohort must be projected into it before predict_ml_model (see Custom_output in
    field_glossary).

    WHEN: after you have a feature table (from multideconv-server, CellTFusion-server, or
    RaCInG-server) and want an honest estimate of how well it predicts an outcome, without yet
    committing to a specific held-out test split.

    pipeML stops (rather than returning nothing) when no model can be selected -- every
    algorithm predicting the same value for all training samples, i.e. uninformative features.

    This can be a slow call with many folds/repetitions or a fold_construction_fun (which reruns
    feature construction per fold -- for prepare_celltfusion_folds, a full CellTFusion run per
    fold, so keep k_folds x n_rep modest and consider its "ncores"); progress is reported
    periodically.
    """
    celltfusion_folds = fold_construction_fun == "prepare_celltfusion_folds"
    if (fold_deconv_csv or fold_coldata_csv) and not celltfusion_folds:
        raise ToolError("fold_deconv_csv/fold_coldata_csv are only used with fold_construction_fun='prepare_celltfusion_folds'.")
    fixed_args: dict = dict(fold_construction_args_fixed or {})
    if celltfusion_folds:
        if not fold_deconv_csv:
            raise ToolError("fold_construction_fun='prepare_celltfusion_folds' needs fold_deconv_csv (the raw deconvolution of all training samples).")
        if fold_construction_args_tunable:
            raise ToolError("prepare_celltfusion_folds doesn't support fold_construction_args_tunable; pass its arguments via fold_construction_args_fixed.")
        for key, instead in _CELLTFUSION_FOLD_RESERVED.items():
            if key in fixed_args:
                hint = f"pass it via {instead}" if instead else "it must be computed within each fold"
                raise ToolError(f"'{key}' can't be set in fold_construction_args_fixed for prepare_celltfusion_folds: {hint}.")
        if fixed_args.get("TF.collection") == "ARACNE":
            raise ToolError("TF.collection='ARACNE' isn't supported by train_ml_model's prepare_celltfusion_folds; use CollecTRI or Dorothea.")
        if fixed_args.get("batch") and not (fold_coldata_csv and fixed_args.get("batch_id")):
            raise ToolError("\"batch\": true needs fold_coldata_csv and a \"batch_id\" column name in fold_construction_args_fixed.")
        samples = {"align_to": features_train_csv, "align_to_columns": features_as_rows}
        fixed_args["deconv"] = csv_arg(fold_deconv_csv, row_names=1, **samples)
        # Original gene symbols (pipeML's make.names() would turn e.g. HLA-A into HLA.A), genes x samples
        fixed_args["raw.counts"] = csv_arg(features_train_csv, row_names=1, as_matrix=True, transpose=not features_as_rows)
        if fold_coldata_csv:
            fixed_args["coldata"] = csv_arg(fold_coldata_csv, row_names=1, **samples)

    # pipeML warns whenever metric is set for survival (which is always selected by C-index)
    if task_type == "survival":
        metric = None
    if lodo and not (batch_csv and batch_column):
        raise ToolError("lodo=True needs batch_csv and batch_column (the cohort of each sample).")

    def target_col(column: str | None) -> dict | None:
        if not (target_csv and column):
            return None
        return csv_column_arg(target_csv, column, align_to=features_train_csv, align_to_columns=features_as_rows)

    args = args_from(
        **{
            "features_train": csv_arg(features_train_csv, row_names=1, transpose=features_as_rows),
            "task_type": lit(task_type),
            "target_var": target_col(target_column),
            "trait.positive": lit(trait_positive) if trait_positive else None,
            "time_var": target_col(time_column),
            "event_var": target_col(event_column),
            "metric": lit(metric) if metric else None,
            "k_folds": lit(k_folds),
            "n_rep": lit(n_rep),
            "LODO": lit(lodo),
            "batch_var": csv_column_arg(batch_csv, batch_column, align_to=features_train_csv, align_to_columns=features_as_rows) if (batch_csv and batch_column) else None,
            "file_name": lit(file_name) if file_name else None,
            "ncores": lit(ncores) if ncores is not None else None,
            "return": lit(True),
            "fold_construction_fun": named_r_function_arg(fold_construction_fun, _FOLD_FUNCTION_PACKAGES[fold_construction_fun]) if fold_construction_fun else None,
            "fold_construction_args_fixed": named_args_arg(fixed_args) if fixed_args else None,
            "fold_construction_args_tunable": named_args_arg(fold_construction_args_tunable) if fold_construction_args_tunable else None,
            "seed": lit(seed),
            "preprocess": lit(preprocess),
        }
    )
    return await _run(
        package="pipeML", function="compute_features.training.ML", args=args, tool_name="train_ml_model",
        # Custom folds rerun feature construction per fold (a full CellTFusion run each, for CellTFusion)
        ctx=ctx, timeout=28800 if fold_construction_fun else 3600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
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
                "its result.csv here instead of a raw deconvolution/counts matrix; for "
                "prepare_celltfusion_folds, run CellTFusion-server's project_celltfusion_factors "
                "(model_rds = the training run's result.rds, deconvolution_test_csv = this test "
                "cohort's raw deconvolution matrix, with the same methods/signatures as "
                "fold_deconv_csv) and pass its result.csv here."
            )
        ),
    ],
    task_type: Annotated[Literal["classification", "survival"], Field(description="Must match the task_type used to train model_rds.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    target_csv: Annotated[str | None, Field(description="CSV with the true target column (or survival time/event columns), sample IDs as its first column -- required. Its values are matched to test_data_csv's samples by those IDs, so it may hold extra samples (e.g. a whole-cohort coldata) or a different order; only when IDs don't match does it fall back to row position, which then requires the same number of rows.")] = None,
    target_column: Annotated[str | None, Field(description="Column in target_csv with the true classification labels.")] = None,
    trait_positive: Annotated[str | None, Field(description="Value in the target column representing the positive class -- must match what train_ml_model used.")] = None,
    time_column: Annotated[str | None, Field(description="Column in target_csv with survival time (survival tasks).")] = None,
    event_column: Annotated[str | None, Field(description="Column in target_csv with the event indicator (survival tasks).")] = None,
    file_name: Annotated[str | None, Field(description="File name prefix for saved ROC/PRC/confusion-matrix or Kaplan-Meier plots.")] = None,
    n_groups: Annotated[int, Field(ge=2, description="Survival only: number of predicted-risk groups in the saved Kaplan-Meier plot (2 = high vs low risk, 3 = low/medium/high, ...).")] = 2,
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
    def target_col(column: str | None) -> dict | None:
        return csv_column_arg(target_csv, column, align_to=test_data_csv) if (target_csv and column) else None

    args = args_from(
        **{
            "model": rds_element_arg(model_rds, "Model"),
            "test_data": csv_arg(test_data_csv, row_names=1),
            "target_var": target_col(target_column),
            "trait.positive": lit(trait_positive) if trait_positive else None,
            "task_type": lit(task_type),
            "time_var": target_col(time_column),
            "event_var": target_col(event_column),
            "file.name": lit(file_name) if file_name else None,
            "return": lit(True),
            "n_groups": lit(n_groups) if task_type == "survival" else None,
        }
    )
    return await _run(
        package="pipeML", function="compute_prediction", args=args, tool_name="predict_ml_model",
        ctx=ctx, timeout=1200, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary=_PREDICTION_GLOSSARY,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def compute_shap_values(
    model_rds: Annotated[
        str,
        Field(
            description=(
                "result.rds from a prior train_ml_model call. This tool reads its Model element "
                "internally; the training samples and features are taken from that model, so no "
                "data CSV is passed."
            )
        ),
    ],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    task_type: Annotated[Literal["classification", "survival"], Field(description="Must match the task_type used to train model_rds.")] = "classification",
    seed: Annotated[int, Field(description="Random seed: SHAP values are Monte Carlo estimates, so the same seed gives the same values.")] = 123,
) -> RunResult:
    """Compute per-feature SHAP importance for a trained model (pipeML::compute_shap_values).

    WHAT: SHAP (SHapley Additive exPlanations) values quantify each feature's contribution to each
    training sample's prediction, for the final model train_ml_model selected (trained on all
    training samples) -- so every feature has one fixed definition, including features built by a
    fold_construction_fun, which are computed once on all training samples. Values are in the units
    of the model output: probability of the positive class for classification, risk score for
    survival. For each sample, the baseline (the model's average prediction, in
    summary.attributes.baseline) plus the sum of its SHAP values equals its prediction.

    WHEN: after train_ml_model, to understand which features drive the model's predictions rather
    than only how accurate it is. Global importance = mean absolute SHAP value per feature
    (column).

    CAVEAT: the final model was trained on the samples it explains, so SHAP describes how the model
    uses each feature on its training data. With flexible models and small cohorts, an overfit
    model can assign importance to features that don't generalize -- compare the model's
    cross-validation performance (train_ml_model) before interpreting the values.

    Returns a samples x features data frame of SHAP values; NULL (with a warning) if the model
    predicts the same value for every sample.
    """
    args = args_from(
        **{
            "model_trained": rds_element_arg(model_rds, "Model"),
            "task_type": lit(task_type),
            "seed": lit(seed),
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
