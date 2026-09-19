"""Bioinfo-tools-server: MCP tools for general-purpose bulk RNAseq analysis

Wraps every function in Bioinfo-tools' functions.r: TF-activity inference (VIPER),
PROGENy/GSVA pathway activity, edgeR/limma-voom differential expression, Kaplan-Meier
survival analysis, TPM/z-score normalization, ComBat-seq batch correction, PCA, and
correlation/trait-association testing (Pearson, Spearman, Kendall, linear regression,
Fisher's exact) against a continuous clinical/experimental variable.

Unlike the other three servers in this package, Bioinfo-tools is not an installed R
package (no DESCRIPTION/NAMESPACE) -- it's a single script, functions.r, meant to be
source()'d. Each tool here still shells out to system R via the shared
common/r_bridge.py dispatcher (common/r/run_tool.R), but in "source" mode rather than
"package" mode: the script is source()'d fresh into its own environment for every call
rather than looked up in an installed package's namespace. See
common/paths.py:bioinfotools_script() for how its location is resolved (default
~/Documents/Bioinfo-tools/functions.r, override with BIOINFOTOOLS_SCRIPT). All of
functions.r's own R package dependencies (decoupleR, GSVA, edgeR, limma, survival,
survminer, WGCNA, ADImpute, sva, ...) must already be installed in the system R library
`Rscript` (on PATH) resolves to.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import Field

from mcp_tme_servers.common.paths import bioinfotools_script, server_home
from mcp_tme_servers.common.r_bridge import args_from, csv_arg, csv_column_arg, csv_list_arg, lit, rds_arg
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
    "Bioinfo-tools-server",
    instructions=(
        "General-purpose R functions for bulk RNAseq analysis, gathered in a single functions.r "
        "file: transcription factor activity (VIPER/CollecTRI/Dorothea/ARACNE), PROGENy/GSVA "
        "pathway activity, edgeR/limma-voom differential expression, Kaplan-Meier survival "
        "analysis, TPM/z-score normalization, ComBat-seq batch correction, PCA, and "
        "correlation/trait-association testing (Pearson, Spearman, Kendall, linear regression, "
        "Fisher's exact) between a score matrix and a continuous clinical/experimental trait. "
        "These are standalone utilities rather than one fixed pipeline -- reach for whichever "
        "function fits the current step. Common chains: run compute_normalization on raw counts "
        "before compute_tf_activity/compute_pathway_activity (both expect normalized "
        "expression); then relate the resulting scores to a clinical trait with "
        "scores_stat_analysis (or one of the individual scores_*_test tools) or to survival with "
        "compute_survival_analysis. Score matrices from this server, or from multideconv-server/ "
        "CellTFusion-server/RaCInG-server in this same package, are typical inputs to the "
        "trait-association and survival tools here."
    ),
)

SESSION_CONFIG = SessionConfig(
    sessions_root=server_home("bioinfotools") / "runs" / "sessions",
    stage_by_tool={
        "compute_tf_activity": "tf_activity",
        "compute_pathway_activity": "pathway_activity",
        "run_deg_analysis": "differential_expression",
        "compute_survival_analysis": "survival_analysis",
        "compute_normalization": "normalization",
        "compute_batch_correction": "batch_correction",
        "compute_pca_analysis": "pca",
        "compute_correlation": "correlation",
        "scores_pearson_test": "trait_association",
        "scores_spearman_test": "trait_association",
        "scores_kendall_test": "trait_association",
        "scores_lm_test": "trait_association",
        "scores_fisher_test_continuous": "trait_association",
        "scores_stat_analysis": "trait_association",
    },
    export_requires=[
        frozenset(
            {
                "tf_activity",
                "pathway_activity",
                "differential_expression",
                "survival_analysis",
                "normalization",
                "batch_correction",
                "pca",
                "correlation",
                "trait_association",
            }
        )
    ],
    stage_explanations={
        "tf_activity": "no analysis tool on this server has been run yet for this session -- there is nothing to export.",
        "pathway_activity": "no analysis tool on this server has been run yet for this session -- there is nothing to export.",
        "differential_expression": "no analysis tool on this server has been run yet for this session -- there is nothing to export.",
        "survival_analysis": "no analysis tool on this server has been run yet for this session -- there is nothing to export.",
        "normalization": "no analysis tool on this server has been run yet for this session -- there is nothing to export.",
        "batch_correction": "no analysis tool on this server has been run yet for this session -- there is nothing to export.",
        "pca": "no analysis tool on this server has been run yet for this session -- there is nothing to export.",
        "correlation": "no analysis tool on this server has been run yet for this session -- there is nothing to export.",
        "trait_association": "no analysis tool on this server has been run yet for this session -- there is nothing to export.",
    },
)

_run = make_run(runs_root=server_home("bioinfotools") / "runs")
_SOURCE = str(bioinfotools_script())

register_session_tools(mcp, SESSION_CONFIG, domain="general-purpose bulk RNAseq analysis")


# --------------------------------------------------------------------------
# Bioinfo-tools tools
# --------------------------------------------------------------------------

ScoresCsv = Annotated[
    str | None,
    Field(
        description=(
            "Samples x features score matrix CSV (e.g. TF activity from compute_tf_activity, "
            "pathway activity from compute_pathway_activity, a deconvolution matrix from "
            "multideconv-server, or Cell_groups/Latent_spaces$Z from CellTFusion-server). "
            "Mutually exclusive with scores_rds; exactly one is required."
        )
    ),
]
ScoresRds = Annotated[
    str | None,
    Field(
        description=(
            "A previously saved .rds list whose first element ([[1]]) is the samples x features "
            "score matrix to test (e.g. chained from another tool's run_dir/result.rds that "
            "already has this shape). Mutually exclusive with scores_csv; exactly one is required."
        )
    ),
]
TraitParam = Annotated[str, Field(description="Name of the (numeric) column in coldata_csv holding the continuous clinical/experimental variable to test against, e.g. age, tumor purity, or a continuous score.")]
PvalParam = Annotated[float, Field(ge=0, le=1, description="P-value threshold below which a feature is kept as significant.")]


def _no_significant_features_diagnostics(method_label: str):
    def _diagnostics(summary: dict) -> list[str]:
        if summary.get("class") == "NULL":
            return [
                f"No feature reached the p-value threshold after the {method_label} test -- the "
                "call succeeded but nothing was significant. Try a higher pval threshold, check "
                "that trait has enough non-missing numeric values, or confirm the score matrix "
                "and coldata share the same sample identifiers (a silent row-name mismatch looks "
                "identical to 'nothing significant')."
            ]
        return []

    return _diagnostics


@mcp.tool(annotations=READ_ANNOTATIONS)
async def compute_tf_activity(
    rna_counts_csv: Annotated[str, Field(description="Normalized gene expression matrix CSV (e.g. log-TPM from compute_normalization), genes as rows, samples as columns.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    tf_collection: Annotated[
        Literal["CollecTRI", "Dorothea", "ARACNE"],
        Field(
            description=(
                "Source of the TF-target regulon network. 'CollecTRI' (default) and 'Dorothea' "
                "are prebuilt, literature-curated collections fetched from OmnipathR (and cached "
                "to Results/TF_target_collection.csv under run_dir for reuse within the same "
                "output_dir) -- generic, not tissue/cancer-specific. 'ARACNE' instead correlates "
                "each TF against its targets in a user-supplied, data-driven network (regulator/"
                "target triples, e.g. from ARACNe-AP) -- pick this for a cancer-type/cohort-"
                "specific regulatory network; requires cancer_type (or a single network.txt "
                "auto-discovered under input/ARACNE/) and a network file at "
                "input/ARACNE/<cancer_type>/network/network.txt relative to output_dir."
            )
        ),
    ] = "CollecTRI",
    min_targets_size: Annotated[int, Field(ge=1, description="Minimum number of a TF's target genes that must be present in rna_counts_csv for that TF to be scored. Lower it if too few TFs get activity scores (see warnings); raise it to require more evidence per TF.")] = 5,
    universe_csv: Annotated[str | None, Field(description="Optional user-supplied TF-target network CSV (source/target columns, i.e. regulator/target gene symbols) to use instead of fetching tf_collection.")] = None,
    cancer_type: Annotated[str | None, Field(description="Cancer type label, used both to cache/name the fetched TF collection and, when tf_collection='ARACNE', to locate that cancer type's network file.")] = None,
    cores: Annotated[int, Field(ge=1, description="Cores used by VIPER inference.")] = 3,
    scale: Annotated[bool, Field(description="Z-score scale the TF activity matrix across samples after inference (default True).")] = True,
    file_name: Annotated[str | None, Field(description="File name suffix for the TF activity matrix CSV written to run_dir/Results.")] = None,
) -> RunResult:
    """Infer transcription factor activity from expression via VIPER
    (Bioinfo-tools::compute.TFs.activity).

    WHAT: infers each TF's regulatory activity (not its own expression) from how coordinately
    up/down its known target genes move in each sample, using the VIPER algorithm against a
    TF-target regulon network. Higher activity means the TF's targets show more coordinated
    up-regulation in that sample -- a functional readout of regulatory state that raw TF gene
    expression alone doesn't capture.

    WHEN: run on already-normalized expression (e.g. compute_normalization's output, or any
    log-TPM/log2CPM matrix) -- not raw counts. The resulting samples x TFs matrix is a typical
    input to compute_survival_analysis (via its features parameter), scores_stat_analysis, or
    compute_correlation.

    Returns a samples x TFs data frame of (by default, z-scored) TF activity.
    """
    args = args_from(
        **{
            "RNA.counts": csv_arg(rna_counts_csv, row_names=1, as_matrix=True),
            "TF.collection": lit(tf_collection),
            "min_targets_size": lit(min_targets_size),
            "universe": csv_arg(universe_csv, row_names=None) if universe_csv else None,
            "cancer.type": lit(cancer_type) if cancer_type else None,
            "cores": lit(cores),
            "scale": lit(scale),
            "return": lit(True),
            "file.name": lit(file_name) if file_name else None,
        }
    )
    return await _run(
        source=_SOURCE, function="compute.TFs.activity", args=args, tool_name="compute_tf_activity",
        ctx=ctx, timeout=900, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_tf_activity_diagnostics,
    )


def _tf_activity_diagnostics(summary: dict) -> list[str]:
    warnings: list[str] = []
    dim = summary.get("dim")
    if dim and len(dim) == 2 and dim[1] < 5:
        warnings.append(
            f"Only {dim[1]} TF(s) had inferable activity -- unusually few. This often means "
            "min_targets_size is too strict for how many of the regulon's target genes are "
            "present in this dataset (try lowering it), or tf_collection doesn't cover this gene "
            "panel well."
        )
    return warnings


@mcp.tool(annotations=READ_ANNOTATIONS)
async def compute_pathway_activity(
    rna_tpm_csv: Annotated[str, Field(description="Normalized gene expression matrix CSV (e.g. log-TPM from compute_normalization), genes as rows, samples as columns.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    gene_sets_rds: Annotated[
        str | None,
        Field(
            description=(
                "Optional .rds holding a named list of gene sets (character vectors of gene "
                "symbols), e.g. MSigDB Hallmark signatures. If given, GSVA scores are also "
                "computed for these sets alongside PROGENy, and the result becomes a two-element "
                "list instead of a single matrix (see field_glossary)."
            )
        ),
    ] = None,
    paths_csv: Annotated[str | None, Field(description="Optional PROGENy pathway-gene weight table CSV (source/target/weight columns) to use instead of fetching the bundled human PROGENy resource (top 500 genes, cached to Results/Pathways_collection_PROGENy.csv under run_dir).")] = None,
    file_name: Annotated[str | None, Field(description="File name suffix for the pathway activity matrix CSV(s) written to run_dir/Results.")] = None,
) -> RunResult:
    """Compute PROGENy (and optionally GSVA) pathway activity scores
    (Bioinfo-tools::compute.pathway.activity).

    WHAT: scores each sample's pathway activity two ways: PROGENy (always computed) uses a
    multivariate linear model against a curated, weighted pathway-responsive gene footprint
    (~14 core signaling pathways); GSVA (only if gene_sets_rds is given) instead scores arbitrary
    gene sets (e.g. Hallmark) by how enriched their genes are within each sample's expression
    ranking. Both are z-scored across samples.

    WHEN: run on already-normalized expression (e.g. compute_normalization's output). Useful on
    its own, or as the pathways input to multideconv-server's relate_subgroups_to_pathways.

    Returns a samples x pathways matrix if gene_sets_rds is omitted, or a list with PROGENy and
    GSVA elements if given (see field_glossary).
    """
    args = args_from(
        **{
            "RNA.tpm": csv_arg(rna_tpm_csv, row_names=1, as_matrix=True),
            "gene_sets": rds_arg(gene_sets_rds) if gene_sets_rds else None,
            "paths": csv_arg(paths_csv, row_names=None) if paths_csv else None,
            "return": lit(True),
            "file.name": lit(file_name) if file_name else None,
        }
    )
    return await _run(
        source=_SOURCE, function="compute.pathway.activity", args=args, tool_name="compute_pathway_activity",
        ctx=ctx, timeout=900, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary={
            "PROGENy": "Z-scored PROGENy pathway activity scores (samples x ~14 core signaling pathways), always computed.",
            "GSVA": "Z-scored GSVA enrichment scores (samples x gene_sets_rds' sets), only present when gene_sets_rds was given.",
        },
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def run_deg_analysis(
    counts_csv: Annotated[str, Field(description="Raw RNAseq counts CSV (not normalized): genes as rows, samples as columns.")],
    coldata_csv: Annotated[str, Field(description="Sample metadata CSV, row names matching counts_csv's sample columns.")],
    group_col: Annotated[str, Field(description="Column in coldata_csv giving the two-level grouping factor to compare (e.g. Responder/Non-responder, Tumor/Normal).")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    ref_level: Annotated[str | None, Field(description="Reference ('baseline') level of group_col -- the other level's logFC is expressed relative to this one. Uses R's default factor ordering (alphabetical) if omitted.")] = None,
) -> RunResult:
    """Differential expression analysis with edgeR/limma-voom (Bioinfo-tools::run_deg_analysis).

    WHAT: filters low-expression genes (edgeR::filterByExpr), TMM-normalizes, applies the voom
    mean-variance transformation, and fits a linear model contrasting group_col's two levels --
    the standard edgeR/limma-voom workflow for RNAseq count data. The significance threshold
    (adj.P.Val < 0.05) is fixed internally, not a parameter of this tool.

    WHEN: use on raw counts (not normalized expression) for a simple two-group comparison. For
    more than two groups or a covariate-adjusted design, this function's fixed one-factor
    ~ group_col model isn't sufficient -- build a custom edgeR/limma script instead.

    Returns a data frame of genes with adj.P.Val < 0.05 (logFC, AveExpr, t, P.Value, adj.P.Val, B),
    ranked by significance. An empty result (0 rows) means no gene passed the threshold (see
    warnings), not a tool failure.
    """
    args = args_from(
        **{
            "counts": csv_arg(counts_csv, row_names=1, as_matrix=True),
            "coldata": csv_arg(coldata_csv, row_names=1),
            "group_col": lit(group_col),
            "ref_level": lit(ref_level) if ref_level else None,
        }
    )
    return await _run(
        source=_SOURCE, function="run_deg_analysis", args=args, tool_name="run_deg_analysis",
        ctx=ctx, timeout=300, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_deg_diagnostics,
    )


def _deg_diagnostics(summary: dict) -> list[str]:
    dim = summary.get("dim")
    if dim and len(dim) == 2 and dim[0] == 0:
        return [
            "No gene reached adj.P.Val < 0.05 -- this usually means too few samples per group, a "
            "genuinely weak effect, or that group_col/ref_level don't reflect the intended "
            "contrast (double-check the factor levels)."
        ]
    return []


@mcp.tool(annotations=READ_ANNOTATIONS)
async def compute_survival_analysis(
    survival_data_csv: Annotated[str, Field(description="Clinical/survival metadata CSV, samples as rows.")],
    pfs_column: Annotated[str, Field(description="Column in survival_data_csv holding the survival/follow-up time.")],
    pfs_event_column: Annotated[str, Field(description="Column in survival_data_csv holding the event indicator (1 = event occurred, 0 = censored).")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    group_column: Annotated[str | None, Field(description="Column in survival_data_csv defining a predefined categorical grouping (e.g. a clinical risk group or cluster label) to test with a single Kaplan-Meier curve. Mutually exclusive with features_csv -- exactly one is required.")] = None,
    features_csv: Annotated[
        str | None,
        Field(
            description=(
                "Samples x features numeric matrix CSV (e.g. TF activity from compute_tf_activity, "
                "pathway activity from compute_pathway_activity, or a deconvolution matrix from "
                "multideconv-server), row names matching survival_data_csv. Each feature is "
                "quantile-split into High/Low groups and tested individually; only features with "
                "a significant log-rank test are returned. Mutually exclusive with group_column "
                "-- exactly one is required."
            )
        ),
    ] = None,
    p_value: Annotated[float, Field(ge=0, le=1, description="Log-rank test p-value threshold for keeping a feature as significant, when features_csv is used.")] = 0.05,
    thres: Annotated[float, Field(ge=0, le=1, description="Quantile cutoff used to split each feature into High/Low groups, when features_csv is used. 0.5 (default) is a median split.")] = 0.5,
    file_name: Annotated[str | None, Field(description="File name suffix used when saving Kaplan-Meier plots to run_dir/Results (as SVG, one per significant group/feature).")] = None,
) -> RunResult:
    """Kaplan-Meier survival analysis and log-rank testing
    (Bioinfo-tools::compute.survival.analysis).

    WHAT: fits Kaplan-Meier curves and a log-rank test, either (group_column) for one predefined
    categorical grouping, or (features_csv) automatically screened across every column of a
    feature matrix -- splitting each feature into High/Low by quantile and keeping only those
    where survival differs significantly between the two groups.

    WHEN: use group_column for a known clinical/cluster grouping you want to test directly; use
    features_csv to screen a whole score matrix (e.g. TF activity, pathway activity, or
    deconvolution output) for which individual features are prognostic, without picking one in
    advance.

    Returns (group_column case) a list with the KM plot, median PFS per group, log-rank p-value,
    and median follow-up (see field_glossary); or (features_csv case) a named list of significant
    Surv() ~ feature formulas, one per significant feature -- NULL if none are significant (see
    warnings). A Kaplan-Meier plot (with risk table) is saved as an SVG per significant result.
    """
    args = args_from(
        **{
            "survival.data": csv_arg(survival_data_csv, row_names=1),
            "PFS": lit(pfs_column),
            "PFS_event": lit(pfs_event_column),
            "file_name": lit(file_name) if file_name else None,
            "features": csv_arg(features_csv, row_names=1) if (features_csv and not group_column) else None,
            "p.value": lit(p_value),
            "thres": lit(thres),
            "group_column": lit(group_column) if group_column else None,
        }
    )
    return await _run(
        source=_SOURCE, function="compute.survival.analysis", args=args, tool_name="compute_survival_analysis",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary={
            "km_plot": "The survminer::ggsurvplot object for the group_column comparison.",
            "median_PFS": "Median survival time per group (group_column case).",
            "p_value": "Log-rank test p-value across groups (group_column case).",
            "median_follow_up": "Median follow-up time via reverse Kaplan-Meier (group_column case).",
        },
        diagnostics=_survival_diagnostics,
    )


def _survival_diagnostics(summary: dict) -> list[str]:
    if summary.get("class") == "NULL":
        return [
            "No feature reached the log-rank p-value threshold -- the call succeeded but nothing "
            "was prognostic at this cutoff. Try a higher p_value, a different thres split, or "
            "confirm survival_data_csv and features_csv share the same sample identifiers."
        ]
    return []


@mcp.tool(annotations=READ_ANNOTATIONS)
async def compute_normalization(
    raw_counts_csv: Annotated[str, Field(description="Raw gene expression count matrix CSV, genes as rows, samples as columns.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    log: Annotated[bool, Field(description="Log-transform the TPM values (ADImpute::NormalizeTPM).")] = True,
    zscore: Annotated[bool, Field(description="Z-score scale each gene (row) across samples after TPM/log normalization.")] = True,
    tr_length_csv: Annotated[str | None, Field(description="Optional CSV with a transcript-length column, row names matching raw_counts_csv's genes, passed through to ADImpute::NormalizeTPM(). Requires tr_length_column.")] = None,
    tr_length_column: Annotated[str | None, Field(description="Column in tr_length_csv giving each gene's transcript length.")] = None,
) -> RunResult:
    """Normalize raw counts to log-TPM and optionally z-score scale
    (Bioinfo-tools::compute.normalization).

    WHAT: converts raw counts to TPM via ADImpute::NormalizeTPM (optionally log-transformed),
    then optionally centers/scales each gene to mean 0, unit variance across samples.

    WHEN: run this first on raw counts before compute_tf_activity, compute_pathway_activity, or
    any tool here documented as expecting normalized expression -- they do not normalize
    internally.

    Returns a genes x samples data frame of normalized expression.
    """
    args = args_from(
        **{
            "raw.counts": csv_arg(raw_counts_csv, row_names=1, as_matrix=True),
            "log": lit(log),
            "zscore": lit(zscore),
            "tr_length": csv_column_arg(tr_length_csv, tr_length_column, row_names=1) if (tr_length_csv and tr_length_column) else None,
        }
    )
    return await _run(
        source=_SOURCE, function="compute.normalization", args=args, tool_name="compute_normalization",
        ctx=ctx, timeout=300, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def compute_batch_correction(
    raw_counts_csv: Annotated[str, Field(description="Raw gene expression count matrix CSV, genes as rows, samples as columns.")],
    batch_csv: Annotated[str, Field(description="CSV with a batch/cohort covariate column, row names matching raw_counts_csv's samples.")],
    batch_column: Annotated[str, Field(description="Column in batch_csv giving each sample's batch/cohort label.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    group_csv: Annotated[str | None, Field(description="Optional CSV with the biological condition of interest (e.g. treatment/response), row names matching raw_counts_csv's samples, so sva::ComBat_seq preserves this variation instead of also removing it as if it were batch. Requires group_column.")] = None,
    group_column: Annotated[str | None, Field(description="Column in group_csv giving each sample's biological condition of interest.")] = None,
    covar_mod_csv: Annotated[str | None, Field(description="Optional model matrix CSV of additional covariates to preserve during correction, passed through to sva::ComBat_seq().")] = None,
    normalize: Annotated[bool, Field(description="If True (default), the batch-corrected counts are also run through compute_normalization (log-TPM + z-score) before being returned, matching the ComBat_seq -> NormalizeTPM pattern used elsewhere in this toolkit's Shiny apps/scripts. If False, returns raw corrected counts.")] = True,
    log: Annotated[bool, Field(description="Passed to compute_normalization when normalize=True.")] = True,
    zscore: Annotated[bool, Field(description="Passed to compute_normalization when normalize=True.")] = True,
) -> RunResult:
    """Remove batch effects from raw counts with ComBat-seq (Bioinfo-tools::compute.batch.correction).

    WHAT: corrects raw RNAseq counts across batches/cohorts with sva::ComBat_seq() -- a
    negative-binomial model appropriate for raw counts (unlike limma::removeBatchEffect, which
    expects already-normalized/log-scale data) -- then optionally normalizes the corrected counts.

    WHEN: use before pooling samples from multiple batches/cohorts into one downstream analysis
    (deconvolution, TF/pathway activity, DEG, survival) when a known technical batch variable
    would otherwise confound the biology. Always pass group_csv/group_column when a biological
    condition of interest is confounded or correlated with batch, so ComBat_seq doesn't remove
    that signal along with the batch effect.

    Returns a genes x samples data frame: batch-corrected raw counts if normalize=False, or
    batch-corrected log-TPM (optionally z-scored) expression if normalize=True (default).
    """
    args = args_from(
        **{
            "raw.counts": csv_arg(raw_counts_csv, row_names=1, as_matrix=True),
            "batch": csv_column_arg(batch_csv, batch_column, row_names=1),
            "group": csv_column_arg(group_csv, group_column, row_names=1) if (group_csv and group_column) else None,
            "covar_mod": csv_arg(covar_mod_csv, row_names=1, as_matrix=True) if covar_mod_csv else None,
            "normalize": lit(normalize),
            "log": lit(log),
            "zscore": lit(zscore),
        }
    )
    return await _run(
        source=_SOURCE, function="compute.batch.correction", args=args, tool_name="compute_batch_correction",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def compute_pca_analysis(
    data_csv: Annotated[str, Field(description="Samples x features numeric matrix CSV (e.g. deconvolution fractions, TF activity, or pathway activity scores). Zero-variance and all-NA columns are dropped automatically.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    coldata_csv: Annotated[str | None, Field(description="Optional sample-level annotation CSV, row names matching data_csv, used to color samples in the PCA plot. Requires group_col.")] = None,
    group_col: Annotated[str | None, Field(description="Column in coldata_csv used to color/group samples in the PCA plot.")] = None,
    center: Annotated[bool, Field(description="Center features before PCA.")] = True,
    scale: Annotated[bool, Field(description="Scale features to unit variance before PCA.")] = True,
    file_name: Annotated[str | None, Field(description="File name suffix for the PCA plot PDF saved to run_dir/Results.")] = None,
) -> RunResult:
    """PCA on a samples x features matrix, with an optional PC1/PC2 plot
    (Bioinfo-tools::compute.pca.analysis).

    WHAT: runs stats::prcomp on any samples x features numeric matrix and saves a PC1 vs. PC2
    scatter plot, optionally colored by a coldata_csv grouping column.

    WHEN: a quick sanity check on any score matrix from this server or elsewhere in this
    package -- e.g. does the deconvolution or TF activity structure separate by cohort/batch or
    by a clinical group -- before committing it to a downstream statistical test.

    Returns a list with the prcomp object, PC1/PC2 coordinates + Sample/Group columns, percent
    variance explained by PC1/PC2, and the ggplot2 plot object (see field_glossary).
    """
    args = args_from(
        **{
            "data": csv_arg(data_csv, row_names=1),
            "coldata": csv_arg(coldata_csv, row_names=1) if coldata_csv else None,
            "group_col": lit(group_col) if group_col else None,
            "center": lit(center),
            "scale": lit(scale),
            "file.name": lit(file_name) if file_name else None,
            "return": lit(True),
        }
    )
    return await _run(
        source=_SOURCE, function="compute.pca.analysis", args=args, tool_name="compute_pca_analysis",
        ctx=ctx, timeout=300, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary={
            "pca": "The stats::prcomp object -- rotation/loadings and full component scores, for anything beyond PC1/PC2.",
            "coords": "PC1/PC2 coordinates per sample, with Sample and Group (from group_col, or 'Samples' if none given) columns.",
            "var_explained": "Percentage of total variance explained by PC1 and PC2.",
            "plot": "The ggplot2 PC1 vs. PC2 scatter plot object (also saved as a PDF under run_dir/Results).",
        },
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def compute_correlation(
    x_csv: Annotated[str, Field(description="CSV holding x: a samples x features matrix (if x_column is omitted) or, together with x_column, the source of a single numeric vector (if pairing with a y vector).")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    x_column: Annotated[str | None, Field(description="If given (together with y_csv/y_column), x is the named column of x_csv rather than the whole matrix -- selects the two-vector case (Pearson + Spearman between x and y).")] = None,
    y_csv: Annotated[str | None, Field(description="CSV holding y: a numeric vector column (with y_column, two-vector case) or a samples x features matrix (without y_column, matrix-vs-matrix case). Omit entirely for the score-matrix-vs-trait case (with coldata_csv/trait).")] = None,
    y_column: Annotated[str | None, Field(description="If given (with y_csv and x_column), y is that named column of y_csv -- pairs with x_column for the two-vector case.")] = None,
    coldata_csv: Annotated[str | None, Field(description="Sample annotation CSV including the continuous trait, row names matching x_csv. Together with trait, selects the score-matrix-vs-trait case: every column of x_csv is tested against trait by both Pearson and Spearman.")] = None,
    trait: Annotated[str | None, Field(description="Column in coldata_csv holding the continuous trait to correlate x_csv's columns against (score-matrix-vs-trait case).")] = None,
    file_name: Annotated[str | None, Field(description="Base file name for correlation heatmap PDFs saved to run_dir/Results (matrix-vs-matrix case only).")] = None,
    pval: Annotated[float, Field(ge=0, le=1, description="P-value threshold for significance (score-matrix-vs-trait case).")] = 0.05,
) -> RunResult:
    """Pearson + Spearman correlation, for vectors, a score matrix vs. a trait, or two matrices
    (Bioinfo-tools::compute.correlation).

    WHAT: reports both Pearson (linear) and Spearman (monotonic, rank-based) correlation together
    rather than picking one, in whichever of three shapes the inputs describe:
    1. Two vectors (x_csv+x_column, y_csv+y_column) -- one row per method with estimate/p-value.
    2. A score matrix vs. a continuous trait (x_csv, coldata_csv+trait) -- tests every column of
       x_csv against trait by both methods, keeps the union of significant features.
    3. Two matrices (x_csv, y_csv without y_column, same samples/order) -- every column of x_csv
       correlated against every column of y_csv by both methods, with an optional heatmap PDF per
       method saved to run_dir/Results if file_name is given.

    WHEN: case 1 for a quick two-variable check; case 2 in place of running scores_pearson_test
    and scores_spearman_test separately when you want both methods' verdicts at once; case 3 to
    relate two whole score matrices to each other (e.g. cell-type deconvolution vs. pathway
    activity) rather than one matrix to a single trait column.

    Returns a data frame (case 1), a list with the significant score subset plus a combined
    per-feature $stats table (case 2, NULL if nothing is significant), or a list with $pearson and
    $spearman correlation/p-value matrices (case 3).
    """
    args = args_from(
        **{
            "x": csv_column_arg(x_csv, x_column, row_names=1) if x_column else csv_arg(x_csv, row_names=1),
            "y": (
                csv_column_arg(y_csv, y_column, row_names=1)
                if (y_csv and y_column)
                else (csv_arg(y_csv, row_names=1) if y_csv else None)
            ),
            "coldata": csv_arg(coldata_csv, row_names=1) if coldata_csv else None,
            "trait": lit(trait) if trait else None,
            "file_name": lit(file_name) if file_name else None,
            "pval": lit(pval),
        }
    )
    return await _run(
        source=_SOURCE, function="compute.correlation", args=args, tool_name="compute_correlation",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary={
            "pearson": "Case 3 only: $cor and $pval matrices (features of x x features of y) for Pearson correlation.",
            "spearman": "Case 3 only: $cor and $pval matrices (features of x x features of y) for Spearman correlation.",
            "stats": "Case 2 only: per-feature Pearson r / Spearman rho and p-values for every column of x (not just the significant subset).",
        },
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def scores_pearson_test(
    coldata_csv: Annotated[str, Field(description="Sample annotation CSV including the trait, row names matching the score matrix.")],
    trait: TraitParam,
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    scores_csv: ScoresCsv = None,
    scores_rds: ScoresRds = None,
    pval: PvalParam = 0.05,
) -> RunResult:
    """Pearson correlation test between each score column and a continuous trait
    (Bioinfo-tools::scores.pearson.test).

    WHAT: for every column of the score matrix, tests linear correlation with trait via
    stats::cor.test(method="pearson"). Appropriate when a linear relationship is expected;
    prefer scores_spearman_test or scores_kendall_test if the relationship is monotonic but not
    linear, or the trait/scores are skewed with outliers.

    WHEN: a targeted single-method test. For all five methods (pearson/spearman/kendall/lm/fisher)
    behind one interface, use scores_stat_analysis instead, which also self-wraps a plain score
    matrix/data.frame (this tool requires the scores_csv/scores_rds framing below).

    Returns the significant-feature subset of the score matrix (NULL if none significant, see
    warnings). Each significant feature's scatter plot (with fitted regression line) is saved as a
    PDF under run_dir/Results.
    """
    args = args_from(
        **{
            "scores": csv_list_arg(scores_csv, row_names=1) if scores_csv else rds_arg(scores_rds),
            "coldata": csv_arg(coldata_csv, row_names=1),
            "trait": lit(trait),
            "pval": lit(pval),
        }
    )
    return await _run(
        source=_SOURCE, function="scores.pearson.test", args=args, tool_name="scores_pearson_test",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_no_significant_features_diagnostics("Pearson"),
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def scores_spearman_test(
    coldata_csv: Annotated[str, Field(description="Sample annotation CSV including the trait, row names matching the score matrix.")],
    trait: TraitParam,
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    scores_csv: ScoresCsv = None,
    scores_rds: ScoresRds = None,
    pval: PvalParam = 0.05,
) -> RunResult:
    """Spearman rank correlation test between each score column and a continuous trait
    (Bioinfo-tools::scores.spearman.test).

    WHAT: for every column of the score matrix, tests monotonic (rank-based) correlation with
    trait via stats::cor.test(method="spearman"). More robust than scores_pearson_test to a
    non-linear-but-monotonic relationship or outliers.

    WHEN: default choice over scores_pearson_test when the score's relationship to trait isn't
    expected to be strictly linear, or values are right-skewed (as e.g. deconvolution proportions
    often are).

    Returns the significant-feature subset of the score matrix (NULL if none significant, see
    warnings). Each significant feature's scatter plot is saved as a PDF under run_dir/Results.
    """
    args = args_from(
        **{
            "scores": csv_list_arg(scores_csv, row_names=1) if scores_csv else rds_arg(scores_rds),
            "coldata": csv_arg(coldata_csv, row_names=1),
            "trait": lit(trait),
            "pval": lit(pval),
        }
    )
    return await _run(
        source=_SOURCE, function="scores.spearman.test", args=args, tool_name="scores_spearman_test",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_no_significant_features_diagnostics("Spearman"),
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def scores_kendall_test(
    coldata_csv: Annotated[str, Field(description="Sample annotation CSV including the trait, row names matching the score matrix.")],
    trait: TraitParam,
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    scores_csv: ScoresCsv = None,
    scores_rds: ScoresRds = None,
    pval: PvalParam = 0.05,
) -> RunResult:
    """Kendall's tau correlation test between each score column and a continuous trait
    (Bioinfo-tools::scores.kendall.test).

    WHAT: for every column of the score matrix, tests rank correlation with trait via
    stats::cor.test(method="kendall"). More robust to outliers/ties than Spearman and generally
    preferred for small sample sizes.

    WHEN: reach for this over scores_spearman_test specifically for a small cohort or a score/
    trait with many tied values.

    Returns the significant-feature subset of the score matrix (NULL if none significant, see
    warnings). Each significant feature's scatter plot is saved as a PDF under run_dir/Results.
    """
    args = args_from(
        **{
            "scores": csv_list_arg(scores_csv, row_names=1) if scores_csv else rds_arg(scores_rds),
            "coldata": csv_arg(coldata_csv, row_names=1),
            "trait": lit(trait),
            "pval": lit(pval),
        }
    )
    return await _run(
        source=_SOURCE, function="scores.kendall.test", args=args, tool_name="scores_kendall_test",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_no_significant_features_diagnostics("Kendall"),
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def scores_lm_test(
    coldata_csv: Annotated[str, Field(description="Sample annotation CSV including the trait, row names matching the score matrix.")],
    trait: TraitParam,
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    scores_csv: ScoresCsv = None,
    scores_rds: ScoresRds = None,
    pval: PvalParam = 0.05,
) -> RunResult:
    """Linear regression F-test between each score column and a continuous trait
    (Bioinfo-tools::scores.lm.test).

    WHAT: fits Value ~ Trait for every column of the score matrix and tests whether trait
    significantly explains variation in the score via the model's F-test (stats::anova on an
    stats::lm fit) -- model-based, parametric, closely related to scores_pearson_test but
    reported as a regression rather than a correlation coefficient.

    WHEN: when you want the regression framing (fitted equation, adjusted R-squared annotated on
    the plot) rather than a bare correlation coefficient; otherwise equivalent in spirit to
    scores_pearson_test for a linear relationship.

    Returns the significant-feature subset of the score matrix (NULL if none significant, see
    warnings). Each significant feature's scatter plot (with fitted line, equation, and adjusted
    R-squared) is saved as a PDF under run_dir/Results.
    """
    args = args_from(
        **{
            "scores": csv_list_arg(scores_csv, row_names=1) if scores_csv else rds_arg(scores_rds),
            "coldata": csv_arg(coldata_csv, row_names=1),
            "trait": lit(trait),
            "pval": lit(pval),
        }
    )
    return await _run(
        source=_SOURCE, function="scores.lm.test", args=args, tool_name="scores_lm_test",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_no_significant_features_diagnostics("linear regression"),
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def scores_fisher_test_continuous(
    coldata_csv: Annotated[str, Field(description="Sample annotation CSV including the trait, row names matching the score matrix.")],
    trait: TraitParam,
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    scores_csv: ScoresCsv = None,
    scores_rds: ScoresRds = None,
    pval: PvalParam = 0.05,
) -> RunResult:
    """Fisher's exact test between each score column and a continuous trait, both median-split
    (Bioinfo-tools::scores.fisher.test.continuous).

    WHAT: binarizes both trait and each score column at their own median into High/Low groups,
    then applies Fisher's exact test to the resulting 2x2 contingency table -- a non-parametric
    test with no linearity/monotonicity assumption at all, at the cost of discarding within-group
    magnitude information that the other scores_*_test tools use.

    WHEN: when neither a linear (scores_pearson_test/scores_lm_test) nor a monotonic
    (scores_spearman_test/scores_kendall_test) relationship is a safe assumption, or you
    specifically want a High/Low-group framing (e.g. to match how compute_survival_analysis
    splits features).

    Returns the significant-feature subset of the score matrix (NULL if none significant, see
    warnings). Each significant feature's High/Low bar chart is saved as a PDF under
    run_dir/Results.
    """
    args = args_from(
        **{
            "scores": csv_list_arg(scores_csv, row_names=1) if scores_csv else rds_arg(scores_rds),
            "coldata": csv_arg(coldata_csv, row_names=1),
            "trait": lit(trait),
            "pval": lit(pval),
        }
    )
    return await _run(
        source=_SOURCE, function="scores.fisher.test.continuous", args=args, tool_name="scores_fisher_test_continuous",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_no_significant_features_diagnostics("Fisher's exact"),
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def scores_stat_analysis(
    coldata_csv: Annotated[str, Field(description="Sample annotation CSV including the trait, row names matching the score matrix.")],
    trait: TraitParam,
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    scores_csv: Annotated[
        str | None,
        Field(
            description=(
                "Samples x features score matrix CSV. Unlike the individual scores_*_test tools, "
                "this is passed as a plain matrix/data.frame -- the underlying R function "
                "self-wraps it (and also accepts an NMF-style object with a $Z element via "
                "scores_rds). Mutually exclusive with scores_rds; exactly one is required."
            )
        ),
    ] = None,
    scores_rds: Annotated[
        str | None,
        Field(
            description=(
                "A previously saved .rds object: either an NMF-style list with a $Z samples x "
                "features element (its $Z is used), or a samples x features matrix/data.frame "
                "directly. Mutually exclusive with scores_csv; exactly one is required."
            )
        ),
    ] = None,
    method: Annotated[
        Literal["pearson", "spearman", "kendall", "lm", "fisher"],
        Field(
            description=(
                "Which single test to run: 'pearson' (linear correlation), 'spearman' (monotonic "
                "rank correlation), 'kendall' (rank correlation, robust to outliers/ties/small n), "
                "'lm' (linear regression F-test), or 'fisher' (both variables median-split, "
                "Fisher's exact test on the 2x2 table). See scores_pearson_test/scores_spearman_test/"
                "scores_kendall_test/scores_lm_test/scores_fisher_test_continuous's own docstrings "
                "for when to prefer each."
            )
        ),
    ] = "pearson",
    pval: PvalParam = 0.05,
) -> RunResult:
    """Unified dispatcher for the scores_*_test family against a continuous trait
    (Bioinfo-tools::scores.stat.analysis.continuous).

    WHAT: the same five tests as scores_pearson_test/scores_spearman_test/scores_kendall_test/
    scores_lm_test/scores_fisher_test_continuous, selected via method, behind one interface that
    also self-wraps a plain score matrix or an NMF-style $Z object -- convenient when method is
    chosen dynamically or you'd rather not call five separate tools.

    WHEN: prefer this over calling the individual scores_*_test tools when you already know which
    single method you want (set method accordingly), or want to pass a raw score matrix/
    data.frame without the list-wrapping the individual tools require via scores_csv/scores_rds.

    Returns the significant-feature subset of the score matrix (NULL if none significant, see
    warnings), with per-feature plots saved as PDFs under run_dir/Results (see the corresponding
    scores_*_test tool's docstring for the plot type each method produces).
    """
    args = args_from(
        **{
            "scores": csv_arg(scores_csv, row_names=1) if scores_csv else rds_arg(scores_rds),
            "coldata": csv_arg(coldata_csv, row_names=1),
            "trait": lit(trait),
            "method": lit(method),
            "pval": lit(pval),
        }
    )
    return await _run(
        source=_SOURCE, function="scores.stat.analysis.continuous", args=args, tool_name="scores_stat_analysis",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_no_significant_features_diagnostics("selected"),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
