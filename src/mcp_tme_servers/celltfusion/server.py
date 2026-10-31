"""CellTFusion-server: MCP tools for TF-activity-based TME characterization

Wraps CellTFusion's one-step pipeline: deconvolution, transcription-factor activity inference, WGCNA-based TF-module detection, pathway scoring, cell-group construction, and NMF latent-factor extraction, combined into one call (run_celltfusion); the same pipeline packaged for pipeML's leakage-free custom cross-validation (prepare_celltfusion_folds); and scoring a new cohort on trained cell groups (score_celltfusion_test_set) and latent factors (project_celltfusion_factors).

Each tool shells out to system R via the shared common/r_bridge.py dispatcher
(common/r/run_tool.R). CellTFusion must already be installed in the system R
library `Rscript` (on PATH) resolves to.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from mcp_tme_servers.common.paths import server_home
from mcp_tme_servers.common.r_bridge import (
    args_from,
    csv_arg,
    csv_column_arg,
    lit,
    named_args_arg,
    rds_arg,
    rds_element_arg,
)
from mcp_tme_servers.common.session import SessionConfig
from mcp_tme_servers.common.tools import (
    DEFAULTS_INSTRUCTIONS,
    SCOPE_INSTRUCTIONS,
    EXPRESSION_SCALE_INSTRUCTIONS,
    READ_ANNOTATIONS,
    OutputDir,
    RunResult,
    SessionIdParam,
    make_run,
    register_session_tools,
    resolve_output_dir,
)

logger = logging.getLogger(__name__)

mcp = MCPServer(
    "CellTFusion-server",
    instructions=(
        "Tools for characterizing tumor-microenvironment cell states from bulk RNAseq by combining cell-type deconvolution with transcription-factor activity and co-regulation. run_celltfusion is a single, heavy call (WGCNA + GSVA + NMF) that returns cell-type proportions, TF activity scores, TF co-activity modules, pathway activity, composite cell-group scores, and NMF-derived latent factors (recurring TME programs) described by their top cell groups (cell niches) -- a broader state characterization than deconvolution alone. It can optionally take any of a precomputed raw deconvolution matrix, an already-harmonized cell-subgroup object, a TF activity matrix, or a pathway activity matrix (e.g. from this package's multideconv-server/bioinfotools-server, or a prior run_celltfusion call) instead of recomputing the corresponding step internally. "
        "IMPORTANT -- this server is NOT a downstream step of multideconv-server and does not require it to have been run first: run_celltfusion computes its own deconvolution, TF activity, and pathway activity internally by default. multideconv-server and this server answer different questions -- multideconv-server answers \"what cell types, and how much of each, are in this sample\" (composition/proportions only); this server answers \"what regulatory/functional TME state is this sample in\" (composition combined with TF activity and co-regulation -- TME niches, functional polarization, etc.). IMPORTANT -- immune infiltration is a composition/proportions question (how much immune cell content is present), so it belongs to multideconv-server, NOT here -- do not treat \"is this patient immune infiltrated\" as a reason to reach for this server. This server is for a different, narrower question that goes beyond raw infiltration: the functional/regulatory state of the TME once composition is known (e.g. immune-active vs -suppressed polarization, co-regulation modules, TME programs) -- \"state\" here means regulatory activity, not the presence/amount of cell types. Pick the server based on which question the user is actually asking, and do not assume the user wants both run in sequence just because they asked for one. "
        "IMPORTANT -- check for existing inputs before recomputing: before running run_celltfusion, proactively check whether any of its expensive internal steps (deconvolution, harmonized deconvolution subgroups, TF activity, pathway activity) have already been computed for this same cohort -- e.g. a deconvolution/harmonized-subgroup result from multideconv-server's run_deconvolution/analyze_deconvolution, a TF or pathway activity matrix from bioinfotools-server, or the output of a prior run_celltfusion call. Look for these via that server's get_session_status/prior sessions for the same input data, or by asking the user whether they already have one. If you locate a candidate, do not pass it in silently -- ask the user to confirm it's the right one for this cohort before using it via deconv_csv/dt_rds/tfs_csv/pathways_csv. If the user doesn't confirm it (or none is found), recompute that step from scratch internally rather than guessing. This reuse is a real efficiency option, not something to avoid, it's just never a prerequisite you need to go create first, and never something to assume without the user's confirmation. "
        "IMPORTANT -- CBSX: when run_celltfusion runs its own internal deconvolution (deconv_csv/dt_rds not supplied), CIBERSORTx (CBSX) IS part of the default deconv_methods, as in multideconv-server's run_deconvolution, but it needs a CIBERSORTx account (cbsx_mail + cbsx_token) that nothing in this pipeline can supply on its own. Before calling run_celltfusion with CBSX included, ask the user for their CIBERSORTx email and API token (free registration at https://cibersortx.stanford.edu). If they have them, pass them and keep CBSX. If they don't, tell them the two options -- run now without CBSX (remove it from deconv_methods), or wait until they have the credentials -- and let them choose. Don't drop CBSX silently and don't call without asking: CellTFusion itself would just skip CBSX without the credentials, so the call is rejected up front to make sure the user decides. "
        "Predicting outcomes from CellTFusion states: CellTFusion's cell groups (and latent factors) are learned from the whole cohort, so computing them once and then cross-validating a model on them leaks test-fold information. For leakage-free modeling, use pipeML-server's train_ml_model with fold_construction_fun='prepare_celltfusion_folds' (raw counts + raw deconvolution in, CellTFusion rerun per fold; the model is trained on the cell group scores), inspect that per-fold construction standalone with prepare_celltfusion_folds, and score a held-out cohort on the trained cell groups with score_celltfusion_test_set before predict_ml_model (project_celltfusion_factors then places it on the latent factors, for characterization). "
        + EXPRESSION_SCALE_INSTRUCTIONS
        + 'On this server: run_celltfusion and prepare_celltfusion_folds take raw counts (normalized=True, the default) or linear TPM (normalized=False) and apply both rules themselves -- TPM for the internal deconvolution, log2(TPM + 1) for TF activity and pathway activity -- so never give them log-transformed values. A precomputed deconv_csv/dt_rds must also come from linear TPM, and precomputed tfs_csv/pathways_csv from log2(TPM + 1). '
        + "IMPORTANT -- citation: if the user asks what to cite for an analysis that used this server, always include the mcp-tme-servers project itself -- XX, et al. (XX). mcp-tme-servers -- XX. XX. https://doi.org/XX (a placeholder until a real reference exists; check the project README for the current one) -- in addition to CellTFusion itself: Hurtado, M., & Pancaldi, V. (2026). CellTFusion: A transcriptional regulatory network framework for the identification of functional multicellular states from bulk RNA-seq data. bioRxiv. https://doi.org/10.64898/2026.06.30.735682 -- plus whichever other specific R packages/methods (e.g. WGCNA, the TF-target network used) were actually used. "
        "IMPORTANT -- output location: once any tool call on this server finishes, always tell the user the run_dir path from that result (where result.rds/CSVs/plots/run.log were saved) -- do not just use it silently to chain into the next call without also surfacing it to the user."
        + SCOPE_INSTRUCTIONS
        + DEFAULTS_INSTRUCTIONS
    ),
)

SESSION_CONFIG = SessionConfig(
    sessions_root=server_home("celltfusion") / "runs" / "sessions",
    stage_by_tool={'run_celltfusion': 'tme_characterization', 'prepare_celltfusion_folds': 'fold_preparation', 'score_celltfusion_test_set': 'test_set_scoring', 'project_celltfusion_factors': 'factor_projection'},
    export_requires=[frozenset({'tme_characterization'})],
    stage_explanations={'tme_characterization': "run_celltfusion hasn't completed yet -- there is no TME characterization (TF activity, cell groups, latent factors) in this session to export."},
)

_run = make_run(runs_root=server_home("celltfusion") / "runs")

register_session_tools(mcp, SESSION_CONFIG, domain="TF-activity-based TME characterization")


# --------------------------------------------------------------------------
# CellTFusion tools
# --------------------------------------------------------------------------

CellTypeList = Annotated[
    list[str] | None,
    Field(description="Extra, non-standard cell type names to keep in the internal deconvolution analysis instead of discarding (see multideconv's nomenclature). This includes group names created with multideconv-server's aggregate_cell_groups on the deconvolution passed in (e.g. 'Lymphocytes'): list them here or they are discarded."),
]

_CELLTFUSION_GLOSSARY = {
    "Deconvolution": (
        "Cell-type proportions (samples x cell types) from the internal deconvolution step -- "
        "the same kind of output multideconv-server's run_deconvolution produces, computed here "
        "unless you passed deconv_csv (or dt_rds, in which case this is simply echoed from "
        "deconv_csv if that was also given, or NULL otherwise)."
    ),
    "TFs_matrix": (
        "Transcription factor activity scores (samples x TFs), inferred from expression via the "
        "chosen TF.collection regulon network and decoupleR statistic (method). Higher = that "
        "TF's target genes show more coordinated up-regulation in that sample, i.e. the TF is "
        "inferred to be more active, not necessarily more highly expressed itself. Single cohort: "
        "raw scores (not scaled), so they don't depend on the other samples. Batch mode: a list "
        "of matrices, one per cohort, z-scored within each cohort."
    ),
    "TF_network": (
        "The WGCNA TF co-activity module structure: which TFs cluster into the same module "
        "(correlated activity across samples) and the module scores (in raw_counts_csv's sample "
        "order). A module is a set of TFs whose activity rises and falls together across samples "
        "-- often reflecting a shared regulatory program rather than coincidence."
    ),
    "Pathways_scores": (
        "Pathway activity scores per sample (PROGENy via the decoupleR statistic, plus any custom "
        "gene sets via GSVA): raw in single-cohort mode, z-scored within each cohort in batch "
        "mode. Usable as multideconv-server's relate_subgroups_to_pathways input."
    ),
    "Processed_deconvolution": (
        "The harmonized deconvolution result after multideconv's cross-method subgroup analysis "
        "(same structure as multideconv-server's analyze_deconvolution output)."
    ),
    "Cell_groups": (
        "CellTFusion's core construct, a named list: Cell_groups (samples x cell groups scores), "
        "Composition (the deconvolution features of each cell group) and Weights (the CCA "
        "projection parameters used to score new samples). A cell group is a set of deconvolution "
        "features related to the same TF module to the same degree; its score (a CCA with the "
        "module) is positive when the sample is on the active side of that TF module and negative "
        "on the inactive side -- 'how much of these cells' combined with 'how active is their "
        "regulatory program'. These scores are the features pipeML-server trains on with "
        "fold_construction_fun='prepare_celltfusion_folds'."
    ),
    "Latent_spaces": (
        "NMF latent factors of the (positive part of the) cell group scores: Z (samples x "
        "factors), W (cell groups x factors), the bootstrap stability of each factor and, when the "
        "rank was chosen automatically, the rank selection table. A small number of recurring "
        "TME programs instead of one score per cell group. NULL with compute_factors=False."
    ),
    "Cells_niches": (
        "For each latent factor, its top_n most contributing cell groups with their TF module, NMF "
        "weight, deconvolution features and CCA coefficients (red/positive = the feature is higher "
        "where the cell group is active) -- what each factor is made of. NULL with "
        "compute_factors=False."
    ),
}


def _celltfusion_diagnostics(summary: dict) -> list[str]:
    warnings: list[str] = []
    elements = summary.get("elements", {})
    tf_matrix = elements.get("TFs_matrix", {})
    dim = tf_matrix.get("dim")
    if dim and len(dim) == 2 and dim[1] < 5:
        warnings.append(
            f"Only {dim[1]} TF(s) had inferable activity -- unusually few. This often means "
            "min_targets_size is too strict for how many of the regulon's target genes are "
            "present/expressed in this dataset (try lowering it), or the chosen TF.collection "
            "doesn't cover this gene panel well."
        )
    return warnings


DecoupleRMethod = Annotated[
    Literal["ulm", "mlm", "wsum", "consensus"],
    Field(
        description=(
            "decoupleR statistic used to score the TFs and the PROGENy pathways: 'ulm' (univariate "
            "linear model, default), 'mlm' (multivariate linear model), 'wsum' (weighted sum) or "
            "'consensus' (consensus of the three)."
        )
    ),
]


def _celltfusion_settings(
    *,
    normalized: bool,
    tf_collection: str,
    method: str,
    min_targets_size: int,
    min_module_size: int,
    corr_mod: float,
    soft_power: list[float] | None,
    consensus_quantile: float,
    clustering_method: str,
    corr: float,
    corr_type: str,
    cells_extra: list[str] | None,
    pval: float,
    padj: bool,
    min_cluster_size: int,
    seed: int,
) -> dict:
    """CellTFusion() settings shared by run_celltfusion and prepare_celltfusion_folds (R names)."""
    return {
        "normalized": lit(normalized),
        "TF.collection": lit(tf_collection),
        "method": lit(method),
        "min_targets_size": lit(min_targets_size),
        "min_module_size": lit(min_module_size),
        "corr_mod": lit(corr_mod),
        "soft_power": lit(soft_power) if soft_power else None,
        "consensus_quantile": lit(consensus_quantile),
        "clustering_method": lit(clustering_method),
        "corr": lit(corr),
        "corr_type": lit(corr_type),
        "cells_extra": lit(cells_extra) if cells_extra else None,
        "pval": lit(pval),
        "padj": lit(padj),
        "min_cluster_size": lit(min_cluster_size),
        "seed": lit(seed),
    }


_SOFT_POWER_DOC = (
    "Soft-thresholding power used to build the TF network (WGCNA): one value, or one per cohort "
    "(in the order cohorts first appear) in batch mode. Chosen automatically if unset."
)
_CONSENSUS_QUANTILE_DOC = (
    "Batch mode only: quantile of each TF pair's topological overlaps across cohorts used as their "
    "consensus. 0 (default, the minimum) = two TFs are close only if they are close in every "
    "cohort; higher values are less strict."
)
_CLUSTERING_DOC = (
    "Linkage method (stats::hclust) of the clustering of the TFs into modules and of each "
    "module's deconvolution features into cell groups."
)


@mcp.tool(annotations=READ_ANNOTATIONS)
async def run_celltfusion(
    raw_counts_csv: Annotated[str, Field(description="CSV of RNAseq expression: genes as rows (first column = gene symbol), samples as columns. Either raw counts (normalized=True, the default) or LINEAR TPM (normalized=False). Never log-transformed values: convert log2(TPM + 1) back to TPM (2^x - 1) first. Verify the scale before calling.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    deconv_csv: Annotated[
        str | None,
        Field(
            description=(
                "Optional precomputed RAW (unharmonized) deconvolution matrix CSV (e.g. from "
                "multideconv-server's run_deconvolution, optionally with aggregate_cell_groups) -- "
                "skips the internal deconvolution step only; harmonization (cross-method subgroup "
                "analysis) still runs on it. Computed internally via multideconv if omitted, which "
                "is slower but self-contained. For an ALREADY-harmonized cell-subgroup object that "
                "skips both steps, use dt_rds instead."
            )
        ),
    ] = None,
    dt_rds: Annotated[
        str | None,
        Field(
            description=(
                "Optional precomputed, ALREADY-HARMONIZED cell-subgroup object -- typically "
                "multideconv-server's analyze_deconvolution result.rds, or a prior "
                "run_celltfusion call's own Processed_deconvolution element written to its own "
                ".rds. Must have the same samples as raw_counts_csv. Skips BOTH the internal "
                "deconvolution AND harmonization steps entirely, going straight to cell-group "
                "construction. If both dt_rds and deconv_csv are given, only dt_rds is actually "
                "used for computation -- deconv_csv is merely echoed into the result's own "
                "Deconvolution field, not recomputed or reharmonized."
            )
        ),
    ] = None,
    tfs_csv: Annotated[
        str | None,
        Field(
            description=(
                "Optional precomputed TF activity matrix CSV (samples as rows, TFs as columns) -- "
                "typically a prior run_celltfusion call's TFs_matrix, or bioinfotools-server's "
                "compute_tf_activity output. Skips the internal TF activity inference step. Not "
                "supported in batch mode (CellTFusion then needs one matrix per cohort)."
            )
        ),
    ] = None,
    pathways_csv: Annotated[
        str | None,
        Field(
            description=(
                "Optional precomputed pathway activity matrix CSV (samples as rows, pathways as "
                "columns) -- typically a prior run_celltfusion call's Pathways_scores, or "
                "bioinfotools-server's compute_pathway_activity output. Skips the internal "
                "pathway activity inference step."
            )
        ),
    ] = None,
    universe_csv: Annotated[
        str | None,
        Field(description="Optional user-supplied TF-target network CSV (columns source, target and mor -- the mode of regulation, 1 activation / -1 repression -- all three required) for the internal TF activity step, instead of fetching tf_collection. Ignored if tfs_csv is given, and when tf_collection='ARACNE' (use aracne_network_file for that).")
    ] = None,
    paths_csv: Annotated[
        str | None,
        Field(description="Optional PROGENy pathway-gene weight table CSV (source/target/weight columns) for the internal pathway activity step, instead of fetching the bundled human PROGENy resource. Ignored if pathways_csv is given.")
    ] = None,
    gene_sets_rds: Annotated[
        str | None,
        Field(description="Optional .rds holding a named list of custom gene sets (character vectors of gene symbols) additionally scored via GSVA during the internal pathway activity step. Ignored if pathways_csv is given.")
    ] = None,
    normalized: Annotated[bool, Field(description="True (default): raw_counts_csv holds raw counts; they are normalized to TPM for the deconvolution and to log2(TPM + 1) for TF activity and pathway activity. Set False only if raw_counts_csv already holds LINEAR TPM: the TPM values are used as they are for the deconvolution and log2(TPM + 1) is computed internally for the other steps. Do not pass log-transformed values (they would be log-transformed a second time; CellTFusion warns when the values don't look like TPM).")] = True,
    coldata_csv: Annotated[str | None, Field(description="Sample metadata CSV, samples as rows (matched to raw_counts_csv's samples by ID). Only used for batch_column.")] = None,
    batch_column: Annotated[
        str | None,
        Field(
            description=(
                "Column in coldata_csv with each sample's cohort. Setting it runs the multi-cohort "
                "mode: TF and pathway activities are computed per cohort, TF modules are built from "
                "a consensus network across cohorts (see consensus_quantile), and the cohort is "
                "controlled for in the deconvolution analysis and cell-group construction. The "
                "column is always treated as categorical; a single cohort is the same as no batch."
            )
        ),
    ] = None,
    deconv_methods: Annotated[
        list[str],
        Field(
            description=(
                "Deconvolution methods to run internally if deconv_csv/dt_rds are not supplied. "
                "CIBERSORTx (CBSX) IS part of the default set, but it needs separate CIBERSORTx "
                "credentials (cbsx_mail + cbsx_token) and the call is rejected without them: ask "
                "the user for them before calling, and remove 'CBSX' only if the user chooses to "
                "run without it."
            )
        ),
    ] = ["Quantiseq", "Epidish", "DeconRNASeq", "DWLS", "CBSX"],
    cbsx_mail: Annotated[str | None, Field(description="CIBERSORTx account email, required (with cbsx_token) when 'CBSX' is in deconv_methods (it is by default). Register for free at https://cibersortx.stanford.edu if the user doesn't already have an account.")] = None,
    cbsx_token: Annotated[str | None, Field(description="CIBERSORTx API token, required (with cbsx_mail) when 'CBSX' is in deconv_methods (it is by default). Get it from the account's settings at https://cibersortx.stanford.edu.")] = None,
    do_parallel: Annotated[bool, Field(description="Run the deconvolution methods that support it (CIBERSORTx and DWLS) in parallel, one signature per worker. Only for the internal deconvolution.")] = False,
    workers: Annotated[int | None, Field(ge=1, description="Parallel workers when do_parallel=True; all cores but one if unset.")] = None,
    file_name: Annotated[str | None, Field(description="File name used in the outputs (plots, module matrices) saved under run_dir/Results ('CellTFusion' if unset).")] = None,
    tf_collection: Annotated[
        Literal["CollecTRI", "Dorothea", "ARACNE"],
        Field(
            description=(
                "Source of the TF-target regulon network used to infer TF activity. 'CollecTRI' "
                "(from OmnipathR) and 'Dorothea' (from the dorothea package) are prebuilt, "
                "literature-curated collections -- generic, not tissue/cancer-specific, and a "
                "reasonable default for most analyses. 'ARACNE' instead uses a user-supplied, "
                "data-driven network (e.g. from ARACNe-AP run on matched expression data) -- pick "
                "this when you have a cancer-type/cohort-specific regulatory network rather than "
                "a generic literature-derived one. Requires aracne_network_file."
            )
        ),
    ] = "CollecTRI",
    aracne_network_file: Annotated[
        str | None,
        Field(
            description=(
                "Tab-separated ARACNe network file with 'Regulator' and 'Target' columns and, "
                "ideally, a 'mor' column with each edge's sign (1 activation, -1 repression; "
                "without it CellTFusion warns and uses the sign of the TF-target Spearman "
                "correlation) -- required when tf_collection='ARACNE', ignored otherwise. "
                "CellTFusion only reads it from input/ARACNE/<cancer_type>/network/network.txt "
                "relative to the run directory, so it is copied there before the R call (under "
                "'custom' when cancer_type is unset)."
            )
        ),
    ] = None,
    method: DecoupleRMethod = "ulm",
    min_targets_size: Annotated[int, Field(ge=1, description="Minimum number of target genes a TF's regulon must have present/expressed in this dataset to be scored. Lower it if too few TFs get activity scores (see warnings); raise it to require more evidence per TF.")] = 5,
    min_module_size: Annotated[int, Field(ge=1, description="Minimum number of TFs per WGCNA co-activity module. Lower for more, finer/smaller modules; raise for fewer, larger, more robust modules.")] = 5,
    corr_mod: Annotated[float, Field(ge=0, le=1, description="Correlation threshold for merging similar TF modules together. Lower merges more aggressively into fewer, broader modules; higher keeps modules more finely split.")] = 0.9,
    soft_power: Annotated[list[float] | None, Field(description=_SOFT_POWER_DOC)] = None,
    consensus_quantile: Annotated[float, Field(ge=0, le=1, description=_CONSENSUS_QUANTILE_DOC)] = 0,
    clustering_method: Annotated[Literal["ward.D2", "ward.D", "complete", "average", "single", "mcquitty", "median", "centroid"], Field(description=_CLUSTERING_DOC)] = "ward.D2",
    corr: Annotated[float, Field(ge=0, le=1, description="Correlation threshold used in the internal deconvolution subgroup analysis (same meaning as multideconv-server's analyze_deconvolution corr parameter).")] = 0.7,
    corr_type: Annotated[Literal["spearman", "pearson"], Field(description="Correlation used in every step that computes one: deconvolution subgroups, TF network and module merging, and TF module vs deconvolution feature correlations.")] = "spearman",
    cells_extra: CellTypeList = None,
    pval: Annotated[float, Field(ge=0, le=1, description="P-value threshold for selecting the deconvolution features significantly correlated with each TF module when building cell groups (adjusted p-value with padj=True).")] = 0.05,
    padj: Annotated[bool, Field(description="Correct those p-values for multiple testing (Benjamini-Hochberg, across the deconvolution features of each TF module) before comparing them with pval.")] = False,
    min_cluster_size: Annotated[int, Field(ge=1, description="Minimum number of deconvolution features in a cell group (minimum cluster size of the dynamic tree cut).")] = 3,
    compute_factors: Annotated[bool, Field(description="Compute the NMF latent factors and cell niches from the cell group scores (Latent_spaces/Cells_niches). Set False to stop at the cell groups (faster; the factors then are NULL).")] = True,
    top_n: Annotated[int, Field(ge=1, description="Number of top-contributing cell groups per latent factor described in Cells_niches.")] = 5,
    cancer_type: Annotated[
        str | None,
        Field(
            description=(
                "Only with tf_collection='ARACNE': the cancer-type folder name for the ARACNe "
                "network (input/ARACNE/<cancer_type>/...). Has no other effect -- CellTFusion no "
                "longer maps latent factors onto TCGA meta-programs."
            )
        ),
    ] = None,
    seed: Annotated[int, Field(description="Random seed of the pipeline (deconvolution methods with random steps and NMF) -- the same seed and inputs give the same results.")] = 123,
    verbose: bool = True,
) -> RunResult:
    """Run the full one-step CellTFusion pipeline: deconvolution -> TF activity ->
    TF co-activity modules -> pathway scoring -> cell groups -> NMF latent factors and cell
    niches (CellTFusion::CellTFusion).

    WHAT: goes beyond cell-type abundance alone by asking which transcription factors' activity
    programs (WGCNA modules) associate with which cell-type deconvolution subgroups (cell groups,
    scored per sample by CCA with their module), then summarizes that combined cell-type x
    regulatory-state space into a handful of NMF latent factors, each described by its top cell
    groups (Cells_niches). Where deconvolution alone tells you "how much of each cell type," this
    adds "and what regulatory/functional state those cells are likely in."

    WHEN: use this when you want a broader tumor-microenvironment state characterization than
    deconvolution alone -- e.g. distinguishing an immune-enriched-but-suppressed TME from a
    genuinely immune-active one, which raw cell-type fractions can't do by themselves. Any of
    deconv_csv/dt_rds/tfs_csv/pathways_csv can be passed to skip the corresponding internal step
    and reuse a trusted precomputed result instead (e.g. from multideconv-server or
    bioinfotools-server) -- otherwise each step is recomputed internally, which is slower but
    self-contained. To score a new cohort on this run's cell groups, use
    score_celltfusion_test_set (celltfusion_rds = this run's result.rds), then
    project_celltfusion_factors to place it on the latent factors.

    CIBERSORTx (CBSX): part of the default deconv_methods. It's only relevant when this tool
    runs its own deconvolution (no deconv_csv/dt_rds). Before calling with CBSX included, ask the
    user for their CIBERSORTx email and API token (cbsx_mail/cbsx_token; free registration at
    https://cibersortx.stanford.edu). If they don't have them, let them choose between running
    now without CBSX (remove it from deconv_methods) and waiting for the credentials -- don't
    drop CBSX silently. CellTFusion itself would just skip CBSX without credentials, so this tool
    rejects that call up front to make sure the user decides.

    Returns a list with 8 elements (see field_glossary). This is the heaviest tool on this server
    (WGCNA + GSVA + NMF with bootstrap rank selection); progress is reported periodically.
    """
    if batch_column and not coldata_csv:
        raise ToolError("batch_column needs coldata_csv (the sample metadata CSV holding that column).")
    if batch_column and tfs_csv:
        raise ToolError("tfs_csv isn't supported in batch mode (CellTFusion needs one TF matrix per cohort): let it compute TF activity.")
    if "CBSX" in deconv_methods and not (deconv_csv or dt_rds) and not (cbsx_mail and cbsx_token):
        raise ToolError(
            "'CBSX' is in deconv_methods (it is by default) but cbsx_mail/cbsx_token weren't given -- "
            "CellTFusion would silently skip CIBERSORTx. Ask the user for the credentials; remove 'CBSX' "
            "from deconv_methods only if the user chooses to run without CIBERSORTx."
        )
    if tf_collection == "ARACNE" and not tfs_csv:
        if not aracne_network_file:
            raise ToolError("tf_collection='ARACNE' needs aracne_network_file (tab-separated, Regulator/Target(/mor) columns).")
        src = Path(aracne_network_file).expanduser().resolve()
        if not src.is_file():
            raise ToolError(f"aracne_network_file not found: {src}")
        output_dir = resolve_output_dir(
            runs_root=server_home("celltfusion") / "runs", tool_name="run_celltfusion",
            output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        )
        dest = Path(output_dir) / "input" / "ARACNE" / (cancer_type or "custom") / "network"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / "network.txt")

    args = args_from(
        **{
            "raw_counts": csv_arg(raw_counts_csv, row_names=1, as_matrix=True),
            "deconv": csv_arg(deconv_csv, row_names=1) if deconv_csv else None,
            "dt": rds_arg(dt_rds) if dt_rds else None,
            "tfs": csv_arg(tfs_csv, row_names=1) if tfs_csv else None,
            "pathways": csv_arg(pathways_csv, row_names=1) if pathways_csv else None,
            "universe": csv_arg(universe_csv, row_names=None) if universe_csv else None,
            "paths": csv_arg(paths_csv, row_names=None) if paths_csv else None,
            "gene_sets": rds_arg(gene_sets_rds) if gene_sets_rds else None,
            "batch": csv_column_arg(coldata_csv, batch_column, align_to=raw_counts_csv, align_to_columns=True, as_character=True) if batch_column else None,
            "deconv_methods": lit(deconv_methods),
            "cbsx_mail": lit(cbsx_mail) if cbsx_mail else None,
            "cbsx_token": lit(cbsx_token) if cbsx_token else None,
            "do_parallel": lit(do_parallel),
            "workers": lit(workers) if workers is not None else None,
            "file_name": lit(file_name) if file_name else None,
            **_celltfusion_settings(
                normalized=normalized, tf_collection=tf_collection, method=method,
                min_targets_size=min_targets_size, min_module_size=min_module_size, corr_mod=corr_mod,
                soft_power=soft_power, consensus_quantile=consensus_quantile,
                clustering_method=clustering_method, corr=corr, corr_type=corr_type,
                cells_extra=cells_extra, pval=pval, padj=padj, min_cluster_size=min_cluster_size, seed=seed,
            ),
            "compute_factors": lit(compute_factors),
            "top_n": lit(top_n),
            "cancer_type": lit(cancer_type) if cancer_type else None,
            "return": lit(True),
            "verbose": lit(verbose),
        }
    )
    return await _run(
        package="CellTFusion", function="CellTFusion", args=args, tool_name="run_celltfusion",
        ctx=ctx, timeout=28800, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary=_CELLTFUSION_GLOSSARY, diagnostics=_celltfusion_diagnostics,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def prepare_celltfusion_folds(
    counts_csv: Annotated[
        str,
        Field(
            description=(
                "CSV of RNAseq expression: genes as rows (first column = gene symbol), samples as "
                "columns -- raw counts (normalized=True) or linear TPM (normalized=False), never "
                "log-transformed. The same RAW input pipeML-server's train_ml_model takes as "
                "features_train_csv (with features_as_rows=True) when driving this function."
            )
        ),
    ],
    deconvolution_csv: Annotated[
        str,
        Field(
            description=(
                "RAW deconvolution matrix of all samples (samples as rows), e.g. "
                "multideconv-server's run_deconvolution result.csv -- not analyze_deconvolution's "
                "harmonized subgroups (the subgroup analysis is redone per fold). Matched to "
                "counts_csv's samples by sample ID."
            )
        ),
    ],
    target_csv: Annotated[
        str,
        Field(
            description=(
                "CSV with the outcome, sample IDs as its first column (matched to counts_csv's "
                "samples by ID): a class-label column (target_column) for classification, or "
                "time_column + event_column for survival. The outcome is kept out of the feature "
                "construction and added back to the returned train/test data, exactly as pipeML "
                "hands it to this function."
            )
        ),
    ],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    target_column: Annotated[str | None, Field(description="Column in target_csv with each sample's class label -- classification.")] = None,
    time_column: Annotated[str | None, Field(description="Column in target_csv with survival time -- survival, with event_column.")] = None,
    event_column: Annotated[str | None, Field(description="Column in target_csv with the event indicator (1 = event occurred, 0 = censored) -- survival.")] = None,
    compute_full_dataset: Annotated[
        bool,
        Field(
            description=(
                "If True (default), runs CellTFusion once on all samples and returns their cell "
                "group scores with the outcome attached -- equivalent to what pipeML's "
                "fold_construction_fun contract does on its own final full-data refit after "
                "cross-validation. If False, instead builds per-fold training/test sets using "
                "folds (required, see that parameter) -- the code path pipeML-server's "
                "train_ml_model(fold_construction_fun='prepare_celltfusion_folds') drives per fold: "
                "CellTFusion runs on each fold's training samples only (without latent factors), and "
                "its held-out samples are scored on that fold's cell groups (via "
                "score_celltfusion_test_set's logic)."
            )
        ),
    ] = True,
    folds: Annotated[
        dict[str, list[int]] | None,
        Field(
            description=(
                "Required if compute_full_dataset=False: fold name -> 1-indexed sample positions "
                "(counts_csv's column order) to use as that fold's TRAINING set (the test set is "
                "every other sample, implicitly). This mirrors what train_ml_model constructs "
                "internally via caret when driving this function -- building it by hand here is "
                "mainly for inspecting or debugging one specific fold's output, e.g. "
                "{\"Fold1\": [1, 2, ..., 16]}. Each fold is a full CellTFusion run, so keep training "
                "sets big enough for WGCNA (a few dozen samples in practice). A fold whose "
                "CellTFusion run fails (e.g. no cell groups found) is skipped with a warning; the "
                "call fails only if every fold fails."
            )
        ),
    ] = None,
    do_parallel: Annotated[bool, Field(description="Compute the folds in parallel when compute_full_dataset=False (the first fold always runs first, so TF/pathway collections are downloaded once).")] = False,
    workers: Annotated[int | None, Field(ge=1, description="Parallel workers when do_parallel=True; all cores but one if unset.")] = None,
    coldata_csv: Annotated[str | None, Field(description="Sample metadata CSV (samples as rows), matched to counts_csv's samples by ID -- only used for batch_column.")] = None,
    batch_column: Annotated[str | None, Field(description="Column in coldata_csv with each sample's cohort: runs CellTFusion's multi-cohort mode (see run_celltfusion's batch_column); held-out samples of each cohort are then scored separately, z-scored on their own means and SDs (unreliable with fewer than ~10 test samples per cohort and fold -- leave-one-cohort-out folds avoid it).")] = None,
    normalized: Annotated[bool, Field(description="Same as run_celltfusion's normalized.")] = True,
    tf_collection: Annotated[Literal["CollecTRI", "Dorothea"], Field(description="Same as run_celltfusion's tf_collection (ARACNE isn't supported here).")] = "CollecTRI",
    method: DecoupleRMethod = "ulm",
    min_targets_size: Annotated[int, Field(ge=1, description="Same as run_celltfusion's min_targets_size.")] = 5,
    min_module_size: Annotated[int, Field(ge=1, description="Same as run_celltfusion's min_module_size.")] = 5,
    corr_mod: Annotated[float, Field(ge=0, le=1, description="Same as run_celltfusion's corr_mod.")] = 0.9,
    soft_power: Annotated[list[float] | None, Field(description=_SOFT_POWER_DOC)] = None,
    consensus_quantile: Annotated[float, Field(ge=0, le=1, description=_CONSENSUS_QUANTILE_DOC)] = 0,
    clustering_method: Annotated[Literal["ward.D2", "ward.D", "complete", "average", "single", "mcquitty", "median", "centroid"], Field(description=_CLUSTERING_DOC)] = "ward.D2",
    corr: Annotated[float, Field(ge=0, le=1, description="Same as run_celltfusion's corr.")] = 0.7,
    corr_type: Literal["spearman", "pearson"] = "spearman",
    cells_extra: CellTypeList = None,
    pval: Annotated[float, Field(ge=0, le=1, description="Same as run_celltfusion's pval.")] = 0.05,
    padj: Annotated[bool, Field(description="Same as run_celltfusion's padj.")] = False,
    min_cluster_size: Annotated[int, Field(ge=1, description="Same as run_celltfusion's min_cluster_size.")] = 3,
    seed: Annotated[int, Field(description="Same as run_celltfusion's seed.")] = 123,
    verbose: bool = False,
) -> RunResult:
    """Inspect what pipeML's CellTFusion custom fold-construction integration would train on
    (CellTFusion::prepare_celltfusion_folds).

    WHAT: the same CellTFusion pipeline run_celltfusion performs (TF activity, TF modules, cell
    groups), but packaged to match pipeML's fold_construction_fun contract -- it's what
    pipeML-server's train_ml_model calls internally, per fold, when given
    fold_construction_fun="prepare_celltfusion_folds". The model is trained on the CELL GROUP
    SCORES; learning the cell groups on each fold's training samples only (and scoring the
    held-out samples on them) is what keeps the cross-validation leakage-free. The NMF latent
    factors are not used as features: they're only computed in the final full-data run, to
    characterize the final model. Calling this standalone lets you inspect those features before
    committing to a full, very slow train_ml_model run, or reproduce/debug one specific fold's
    scored test features.

    WHEN: before train_ml_model with fold_construction_fun="prepare_celltfusion_folds", to sanity
    check what CellTFusion would actually produce for this cohort -- or afterwards, to inspect a
    specific fold's output directly. Heavy: one full CellTFusion run per fold (and one on all
    samples with compute_full_dataset=True).

    Returns, with compute_full_dataset=True, an unnamed list of (1) the cell group scores of all
    samples with the outcome columns attached, (2) the full run_celltfusion-style result on all
    samples (what train_ml_model returns as Custom_output), and (3) the bestune marker. With
    compute_full_dataset=False, a list named by fold, each with train_data (cell group scores of
    the training samples + the outcome columns), test_data (held-out samples scored on that
    fold's cell groups, plus time/event for survival), obs_test (class labels, or a time/event
    data frame for survival), rowIndex, and fold_name -- each fold is also saved to
    run_dir/Results/fold_<name>.rds (stale fold_*.rds from earlier runs in the same run_dir are
    removed first).
    """
    if bool(target_column) == bool(time_column or event_column):
        raise ToolError("Give either target_column (classification) or time_column + event_column (survival).")
    if not target_column and not (time_column and event_column):
        raise ToolError("Survival needs both time_column and event_column.")
    if not compute_full_dataset and not folds:
        raise ToolError("folds is required when compute_full_dataset=False (fold name -> 1-indexed training samples).")
    if batch_column and not coldata_csv:
        raise ToolError("batch_column needs coldata_csv (the sample metadata CSV holding that column).")

    def outcome(column: str) -> dict:
        return csv_column_arg(target_csv, column, align_to=counts_csv, align_to_columns=True)

    outcome_columns = {"target": outcome(target_column)} if target_column else {
        "time": outcome(time_column), "event": outcome(event_column)
    }
    samples = {"align_to": counts_csv, "align_to_columns": True}
    args = args_from(
        **{
            # pipeML's data: samples x genes plus the outcome columns
            "data": csv_arg(counts_csv, row_names=1, transpose=True, add_columns=outcome_columns),
            "folds": named_args_arg(folds) if (folds and not compute_full_dataset) else None,
            "bestune": lit(True) if compute_full_dataset else None,
            "deconv": csv_arg(deconvolution_csv, row_names=1, **samples),
            "raw_counts": csv_arg(counts_csv, row_names=1, as_matrix=True),
            "batch": csv_column_arg(coldata_csv, batch_column, as_character=True, **samples) if batch_column else None,
            "do_parallel": lit(do_parallel),
            "workers": lit(workers) if workers is not None else None,
            "return": lit(True),
            "verbose": lit(verbose),
            **_celltfusion_settings(
                normalized=normalized, tf_collection=tf_collection, method=method,
                min_targets_size=min_targets_size, min_module_size=min_module_size, corr_mod=corr_mod,
                soft_power=soft_power, consensus_quantile=consensus_quantile,
                clustering_method=clustering_method, corr=corr, corr_type=corr_type,
                cells_extra=cells_extra, pval=pval, padj=padj, min_cluster_size=min_cluster_size, seed=seed,
            ),
        }
    )
    return await _run(
        package="CellTFusion", function="prepare_celltfusion_folds", args=args, tool_name="prepare_celltfusion_folds",
        ctx=ctx, timeout=28800, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


def _celltfusion_result(model_rds: str | None, celltfusion_rds: str | None, element: str) -> dict:
    """One element of a CellTFusion result: from a train_ml_model result.rds (its Custom_output) or
    a run_celltfusion result.rds."""
    if bool(model_rds) == bool(celltfusion_rds):
        raise ToolError("Give exactly one of model_rds (a train_ml_model result) or celltfusion_rds (a run_celltfusion result).")
    if model_rds:
        return rds_element_arg(model_rds, ["Custom_output", element])
    return rds_element_arg(celltfusion_rds, element)


_MODEL_RDS_DOC = (
    "result.rds of a pipeML-server train_ml_model call that used "
    "fold_construction_fun='prepare_celltfusion_folds' -- its Custom_output (the CellTFusion result "
    "learned on all training samples) is used. Give this or celltfusion_rds."
)
_CELLTFUSION_RDS_DOC = "result.rds of a prior run_celltfusion call. Give this or model_rds."


@mcp.tool(annotations=READ_ANNOTATIONS)
async def score_celltfusion_test_set(
    deconvolution_test_csv: Annotated[
        str,
        Field(
            description=(
                "RAW deconvolution matrix of the new/held-out cohort (samples as rows), with the "
                "same method/signature columns as the deconvolution the cell groups were learned "
                "from (e.g. multideconv-server's run_deconvolution with the same methods, and the "
                "same aggregate_cell_groups if any were used)."
            )
        ),
    ],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    model_rds: Annotated[str | None, Field(description=_MODEL_RDS_DOC)] = None,
    celltfusion_rds: Annotated[str | None, Field(description=_CELLTFUSION_RDS_DOC)] = None,
    group_names: Annotated[list[str] | None, Field(description="Score only these cell groups; all of them if unset.")] = None,
) -> RunResult:
    """Score a new cohort on cell groups learned by CellTFusion on another cohort
    (CellTFusion::compute.test.set).

    WHAT: CellTFusion's cell groups are data-dependent -- re-running it on a different cohort
    would give different groups. This instead (1) rebuilds the training deconvolution subgroups in
    the new cohort, (2) scales each cell group's features with the training means and SDs and
    (3) projects them with the training CCA weights, so the new samples get scores on the same
    scale and with the same columns as the training cell group scores. Each sample is scored on
    its own (except with cell groups built in batch mode: the test samples are then z-scored on
    their own means and SDs, so project one cohort with several samples per call).

    WHEN: before predict_ml_model on a held-out cohort, when the model was trained with
    fold_construction_fun='prepare_celltfusion_folds' -- pass this call's result.csv as
    predict_ml_model's test_data_csv. Also to score a validation cohort consistently with a
    discovery cohort's run_celltfusion cell groups, and as the input of
    project_celltfusion_factors.

    Returns a samples x cell groups data frame (result.csv). A cell group is scored only if all
    its features are in the test deconvolution without missing values; the others are skipped
    with an R warning (see log_tail), and the call fails if none can be scored.
    """
    args = args_from(
        **{
            "dt": _celltfusion_result(model_rds, celltfusion_rds, "Processed_deconvolution"),
            "cell_groups": _celltfusion_result(model_rds, celltfusion_rds, "Cell_groups"),
            "group_names": lit(group_names) if group_names else None,
            "deconv_test": csv_arg(deconvolution_test_csv, row_names=1),
        }
    )
    return await _run(
        package="CellTFusion", function="compute.test.set", args=args, tool_name="score_celltfusion_test_set",
        ctx=ctx, timeout=1800, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def project_celltfusion_factors(
    scores_csv: Annotated[
        str,
        Field(
            description=(
                "Cell group scores of the new cohort (samples as rows, cell groups as columns) -- "
                "score_celltfusion_test_set's result.csv, computed from the same model_rds/"
                "celltfusion_rds."
            )
        ),
    ],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    model_rds: Annotated[str | None, Field(description=_MODEL_RDS_DOC + " Its latent factors exist only if train_ml_model's final run computed them (the default).")] = None,
    celltfusion_rds: Annotated[str | None, Field(description=_CELLTFUSION_RDS_DOC + " It must have been run with compute_factors=True (the default).")] = None,
) -> RunResult:
    """Project a new cohort onto NMF latent factors learned by CellTFusion on another cohort
    (CellTFusion::project_factors).

    WHAT: keeps the positive part of the new samples' cell group scores (as in training) and
    projects them onto the trained NMF latent space (pseudoinverse of the scaled basis W), so
    both cohorts are described by the same Factor1..FactorK columns (non-negative). Cell groups
    missing from scores_csv are tolerated with a warning naming them.

    WHEN: after score_celltfusion_test_set, to place a validation/held-out cohort on the
    discovery cohort's TME programs (latent factors) -- e.g. to compare factor levels across
    cohorts. Not needed for predict_ml_model: CellTFusion-fold models are trained on the cell
    group scores themselves.

    Returns a samples x factors matrix (result.csv).
    """
    args = args_from(
        **{
            "latent_factors": _celltfusion_result(model_rds, celltfusion_rds, "Latent_spaces"),
            "scores": csv_arg(scores_csv, row_names=1, as_matrix=True),
        }
    )
    return await _run(
        package="CellTFusion", function="project_factors", args=args, tool_name="project_celltfusion_factors",
        ctx=ctx, timeout=1800, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
