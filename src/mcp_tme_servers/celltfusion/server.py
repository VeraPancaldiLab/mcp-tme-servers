"""CellTFusion-server: MCP tools for TF-activity-based TME characterization

Wraps CellTFusion's one-step pipeline: deconvolution, transcription-factor activity inference, WGCNA-based TF-module detection, pathway scoring, cell-group construction, and NMF latent-factor extraction, combined into one call (run_celltfusion); the same pipeline packaged for pipeML's leakage-free custom cross-validation (prepare_celltfusion_folds); and the projection of a new cohort onto trained latent factors (project_celltfusion_factors).

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
        "Tools for characterizing tumor-microenvironment cell states from bulk RNAseq by combining cell-type deconvolution with transcription-factor activity and co-regulation. run_celltfusion is a single, heavy call (WGCNA + GSVA + NMF) that returns cell-type proportions, TF activity scores, TF co-activity modules, pathway activity, composite cell-group scores, and NMF-derived latent factors annotated against TCGA meta-programs and Bagaev et al. TME (MFP) subtypes -- a broader state characterization than deconvolution alone. It can optionally take any of a precomputed raw deconvolution matrix, an already-harmonized cell-subgroup object, a TF activity matrix, or a pathway activity matrix (e.g. from this package's multideconv-server/bioinfotools-server, or a prior run_celltfusion call) instead of recomputing the corresponding step internally. "
        "IMPORTANT -- this server is NOT a downstream step of multideconv-server and does not require it to have been run first: run_celltfusion computes its own deconvolution, TF activity, and pathway activity internally by default. multideconv-server and this server answer different questions -- multideconv-server answers \"what cell types, and how much of each, are in this sample\" (composition/proportions only); this server answers \"what regulatory/functional TME state is this sample in\" (composition combined with TF activity and co-regulation -- TME niches, functional polarization, etc.). IMPORTANT -- immune infiltration is a composition/proportions question (how much immune cell content is present), so it belongs to multideconv-server, NOT here -- do not treat \"is this patient immune infiltrated\" as a reason to reach for this server. This server is for a different, narrower question that goes beyond raw infiltration: the functional/regulatory state of the TME once composition is known (e.g. immune-active vs -suppressed polarization, co-regulation modules, MFP subtypes) -- \"state\" here means regulatory activity, not the presence/amount of cell types. Pick the server based on which question the user is actually asking, and do not assume the user wants both run in sequence just because they asked for one. "
        "IMPORTANT -- check for existing inputs before recomputing: before running run_celltfusion, proactively check whether any of its expensive internal steps (deconvolution, harmonized deconvolution subgroups, TF activity, pathway activity) have already been computed for this same cohort -- e.g. a deconvolution/harmonized-subgroup result from multideconv-server's run_deconvolution/analyze_deconvolution, a TF or pathway activity matrix from bioinfotools-server, or the output of a prior run_celltfusion call. Look for these via that server's get_session_status/prior sessions for the same input data, or by asking the user whether they already have one. If you locate a candidate, do not pass it in silently -- ask the user to confirm it's the right one for this cohort before using it via deconv_csv/dt_rds/tfs_csv/pathways_csv. If the user doesn't confirm it (or none is found), recompute that step from scratch internally rather than guessing. This reuse is a real efficiency option, not something to avoid, it's just never a prerequisite you need to go create first, and never something to assume without the user's confirmation. "
        "IMPORTANT -- CBSX: when run_celltfusion runs its own internal deconvolution (deconv_csv/dt_rds not supplied), CIBERSORTx (CBSX) is NOT in the default deconv_methods (matching CellTFusion itself, unlike multideconv-server's run_deconvolution). If the user wants CBSX included, ask for their CIBERSORTx credentials (cbsx_mail/cbsx_token; free registration at https://cibersortx.stanford.edu) and add 'CBSX' to deconv_methods -- CellTFusion stops without credentials, so the call is rejected up front. "
        "Predicting outcomes from CellTFusion states: CellTFusion's latent factors are learned from the whole cohort, so computing them once and then cross-validating a model on them leaks test-fold information. For leakage-free modeling, use pipeML-server's train_ml_model with fold_construction_fun='prepare_celltfusion_folds' (raw counts + raw deconvolution in, CellTFusion rerun per fold), inspect that per-fold construction standalone with prepare_celltfusion_folds, and project a held-out cohort onto the trained factors with project_celltfusion_factors before predict_ml_model. "
        "IMPORTANT -- citation: if the user asks what to cite for an analysis that used this server, always include the mcp-tme-servers project itself -- XX, et al. (XX). mcp-tme-servers -- XX. XX. https://doi.org/XX (a placeholder until a real reference exists; check the project README for the current one) -- in addition to CellTFusion itself: Hurtado, M., & Pancaldi, V. (2026). CellTFusion: A transcriptional regulatory network framework for the identification of functional multicellular states from bulk RNA-seq data. bioRxiv. https://doi.org/10.64898/2026.06.30.735682 -- plus whichever other specific R packages/methods (e.g. WGCNA, the TF-target network used) were actually used. "
        "IMPORTANT -- output location: once any tool call on this server finishes, always tell the user the run_dir path from that result (where result.rds/CSVs/plots/run.log were saved) -- do not just use it silently to chain into the next call without also surfacing it to the user."
    ),
)

SESSION_CONFIG = SessionConfig(
    sessions_root=server_home("celltfusion") / "runs" / "sessions",
    stage_by_tool={'run_celltfusion': 'tme_characterization', 'prepare_celltfusion_folds': 'fold_preparation', 'project_celltfusion_factors': 'factor_projection'},
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
        "chosen TF.collection regulon network (decoupleR consensus methods). Higher = that TF's "
        "target genes show more coordinated up-regulation in that sample, i.e. the TF is inferred "
        "to be more active, not necessarily more highly expressed itself. In batch mode, a list "
        "of such matrices, one per cohort."
    ),
    "TF_network": (
        "The WGCNA TF co-activity module structure: which TFs cluster into the same module "
        "(correlated activity across samples) and the module eigengene scores. A module is a set "
        "of TFs whose activity rises and falls together across samples -- often reflecting a "
        "shared regulatory program rather than coincidence."
    ),
    "Pathways_scores": (
        "Pathway activity scores per sample (e.g. PROGENy or custom gene sets via GSVA) -- usable "
        "as multideconv-server's relate_subgroups_to_pathways input."
    ),
    "Processed_deconvolution": (
        "The harmonized deconvolution result after multideconv's cross-method subgroup analysis "
        "(same structure as multideconv-server's analyze_deconvolution output)."
    ),
    "Cell_groups": (
        "Composite scores per sample linking TF modules to deconvolution-derived cell subgroups "
        "(via CCA) -- CellTFusion's core construct: a cell-state score that combines 'how much of "
        "this cell type' with 'how active is its associated regulatory program', not abundance "
        "alone."
    ),
    "Latent_spaces": (
        "NMF-derived latent factors summarizing recurring co-occurrence patterns across "
        "Cell_groups -- a small number of factors instead of one score per cell group, "
        "each often corresponding to a recognizable TME archetype (see TME_states)."
    ),
    "Cells_niches": (
        "Cell-type composition enriched within each latent factor -- which cell types define/"
        "characterize each NMF factor."
    ),
    "TME_states": (
        "Each latent factor annotated against Bagaev et al. (2021) Molecular Functional Portrait "
        "(MFP) TME subtypes (e.g. immune-enriched, fibrotic, desert) and/or derived meta-programs "
        "-- the closest this pipeline comes to a single biological label per factor. NULL when "
        "cancer_type wasn't given (the meta-program mapping step is skipped)."
    ),
    "Metaprograms_reference": (
        "The TCGA-derived meta-program reference used for TME_states annotation, for the given "
        "cancer_type (NULL without one)."
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


@mcp.tool(annotations=READ_ANNOTATIONS)
async def run_celltfusion(
    raw_counts_csv: Annotated[str, Field(description="CSV of RNAseq counts: genes as rows (first column = gene symbol), samples as columns.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    deconv_csv: Annotated[
        str | None,
        Field(
            description=(
                "Optional precomputed RAW (unharmonized) deconvolution matrix CSV (e.g. from "
                "multideconv-server's run_deconvolution) -- skips the internal deconvolution step "
                "only; harmonization (cross-method subgroup analysis) still runs on it. Computed "
                "internally via multideconv if omitted, which is slower but self-contained. For "
                "an ALREADY-harmonized cell-subgroup object that skips both steps, use dt_rds "
                "instead."
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
                ".rds. Skips BOTH the internal deconvolution AND harmonization steps entirely, "
                "going straight to cell-group construction. If both dt_rds and deconv_csv are "
                "given, only dt_rds is actually used for computation -- deconv_csv is merely "
                "echoed into the result's own Deconvolution field, not recomputed or reharmonized."
            )
        ),
    ] = None,
    tfs_csv: Annotated[
        str | None,
        Field(
            description=(
                "Optional precomputed TF activity matrix CSV (samples as rows, TFs as columns) -- "
                "typically a prior run_celltfusion call's TFs_matrix, or bioinfotools-server's "
                "compute_tf_activity output. Skips the internal TF activity inference step."
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
        Field(description="Optional user-supplied TF-target network CSV (source/target/mor columns) for the internal TF activity step, instead of fetching tf_collection. Ignored if tfs_csv is given, and when tf_collection='ARACNE' (use aracne_network_file for that).")
    ] = None,
    paths_csv: Annotated[
        str | None,
        Field(description="Optional PROGENy pathway-gene weight table CSV (source/target/weight columns) for the internal pathway activity step, instead of fetching the bundled human PROGENy resource. Ignored if pathways_csv is given.")
    ] = None,
    gene_sets_rds: Annotated[
        str | None,
        Field(description="Optional .rds holding a named list of custom gene sets (character vectors of gene symbols) additionally scored via GSVA during the internal pathway activity step. Ignored if pathways_csv is given.")
    ] = None,
    normalized: Annotated[bool, Field(description="Set False if raw_counts_csv already holds log-TPM normalized values instead of raw counts.")] = True,
    coldata_csv: Annotated[str | None, Field(description="Optional sample metadata CSV, samples as rows. Only used in batch mode, to read batch_column -- required then.")] = None,
    batch: Annotated[bool, Field(description="Run in multi-cohort/batch mode: TF activity is inferred per cohort, TF modules are built with consensus WGCNA across cohorts, and the cohort is controlled for in the deconvolution analysis and cell-group construction, rather than treating all samples as one uniform batch. Requires coldata_csv and batch_column.")] = False,
    batch_column: Annotated[str | None, Field(description="Column in coldata_csv identifying each sample's cohort, if batch=True.")] = None,
    deconv_methods: Annotated[
        list[str],
        Field(
            description=(
                "Deconvolution methods to run internally if deconv_csv/dt_rds are not supplied. "
                "Add 'CBSX' to also run CIBERSORTx -- not in the default set (as in CellTFusion "
                "itself), since it needs separate CIBERSORTx credentials (cbsx_mail + cbsx_token) "
                "and the call is rejected without them."
            )
        ),
    ] = ["Quantiseq", "Epidish", "DeconRNASeq", "DWLS"],
    cbsx_mail: Annotated[str | None, Field(description="CIBERSORTx account email, required (with cbsx_token) when 'CBSX' is in deconv_methods. Register for free at https://cibersortx.stanford.edu if the user doesn't already have an account.")] = None,
    cbsx_token: Annotated[str | None, Field(description="CIBERSORTx API token, required (with cbsx_mail) when 'CBSX' is in deconv_methods. Get it from the account's settings at https://cibersortx.stanford.edu.")] = None,
    file_name: Annotated[str | None, Field(description="File name prefix for outputs (plots, module matrices) saved under run_dir/Results.")] = None,
    tf_collection: Annotated[
        Literal["CollecTRI", "Dorothea", "ARACNE"],
        Field(
            description=(
                "Source of the TF-target regulon network used to infer TF activity. 'CollecTRI' "
                "(from OmnipathR) and 'Dorothea' (confidence A and B, from the dorothea package) "
                "are prebuilt, literature-curated collections -- "
                "generic, not tissue/cancer-specific, and a reasonable default for most analyses. "
                "'ARACNE' instead uses a user-supplied, data-driven network (e.g. from ARACNe-AP "
                "run on matched expression data; each edge's sign is the Spearman correlation "
                "between TF and target) -- pick this when you have a cancer-type/cohort-specific "
                "regulatory network rather than a generic literature-derived one. Requires "
                "aracne_network_file."
            )
        ),
    ] = "CollecTRI",
    aracne_network_file: Annotated[
        str | None,
        Field(
            description=(
                "Tab-separated ARACNe network file with 'Regulator' and 'Target' columns (e.g. "
                "ARACNe-AP's network.txt) -- required when tf_collection='ARACNE', ignored "
                "otherwise. CellTFusion only reads it from input/ARACNE/<cancer_type>/network/"
                "network.txt relative to the run directory, so it is copied there before the R "
                "call (under 'custom' instead of the cancer type when cancer_type is unset)."
            )
        ),
    ] = None,
    min_targets_size: Annotated[int, Field(ge=1, description="Minimum number of target genes a TF's regulon must have present/expressed in this dataset to be scored. Lower it if too few TFs get activity scores (see warnings); raise it to require more evidence per TF.")] = 3,
    min_module_size: Annotated[int, Field(ge=1, description="Minimum number of TFs per WGCNA co-activity module. Lower for more, finer/smaller modules; raise for fewer, larger, more robust modules.")] = 3,
    corr_mod: Annotated[float, Field(ge=0, le=1, description="Correlation threshold for merging similar TF modules together. Lower merges more aggressively into fewer, broader modules; higher keeps modules more finely split.")] = 0.9,
    corr: Annotated[float, Field(ge=0, le=1, description="Correlation threshold used in the internal deconvolution subgroup analysis (same meaning as multideconv-server's analyze_deconvolution corr parameter).")] = 0.7,
    corr_type: Literal["spearman", "pearson"] = "spearman",
    cells_extra: CellTypeList = None,
    pval: Annotated[float, Field(ge=0, le=1, description="P-value threshold for statistical tests (metadata/trait association, module-pathway relationships).")] = 0.05,
    enrich_thresh: Annotated[float, Field(ge=0, description="Minimum enrichment ratio (foreground/background cell-type frequency) required to include a cell type in a latent factor's niche description.")] = 1.5,
    quantile_cutoff: Annotated[float, Field(ge=0, le=1, description="Quantile threshold for selecting the top-contributing cell groups per NMF factor when describing what each factor represents.")] = 0.7,
    cancer_type: Annotated[
        Literal["blca", "luad", "skcm"] | None,
        Field(
            description=(
                "TCGA cancer type with a bundled meta-program reference ('blca', 'luad' or "
                "'skcm'), used to map the NMF latent factors onto TCGA meta-programs (TME_states). "
                "If unset, that mapping step is skipped and TME_states/Metaprograms_reference are "
                "NULL -- the rest of the pipeline is unaffected."
            )
        ),
    ] = None,
    verbose: bool = True,
) -> RunResult:
    """Run the full one-step CellTFusion pipeline: deconvolution -> TF activity ->
    TF co-activity modules -> pathway scoring -> cell-group / latent-factor / TME-state
    characterization (CellTFusion::CellTFusion).

    WHAT: goes beyond cell-type abundance alone by asking which transcription factors' activity
    programs (WGCNA modules) associate with which cell-type deconvolution subgroups, then
    summarizes that combined cell-type x regulatory-state space into a handful of NMF latent
    factors, each annotated against known TME archetypes (Bagaev et al. MFP subtypes / TCGA
    meta-programs). Where deconvolution alone tells you "how much of each cell type," this adds
    "and what regulatory/functional state those cells are likely in."

    WHEN: use this when you want a broader tumor-microenvironment state characterization than
    deconvolution alone -- e.g. distinguishing an immune-enriched-but-suppressed TME from a
    genuinely immune-active one, which raw cell-type fractions can't do by themselves. Any of
    deconv_csv/dt_rds/tfs_csv/pathways_csv can be passed to skip the corresponding internal step
    and reuse a trusted precomputed result instead (e.g. from multideconv-server or
    bioinfotools-server) -- otherwise each step is recomputed internally, which is slower but
    self-contained.

    CIBERSORTx (CBSX): not in the default deconv_methods (matching CellTFusion itself). It's only
    relevant when this tool runs its own deconvolution (no deconv_csv/dt_rds). If the user wants
    CBSX included, ask for their CIBERSORTx email and API token (cbsx_mail/cbsx_token; free
    registration at https://cibersortx.stanford.edu) -- CellTFusion stops without them, so the
    call is rejected up front.

    Returns a list with 10 elements (see field_glossary). This is the heaviest tool on this
    server (WGCNA + GSVA + NMF); progress is reported periodically. cancer_type is required for
    the TME_states element to populate.
    """
    if batch and not (coldata_csv and batch_column):
        raise ToolError("batch=True needs coldata_csv and batch_column (the column with each sample's cohort).")
    if "CBSX" in deconv_methods and not (deconv_csv or dt_rds) and not (cbsx_mail and cbsx_token):
        raise ToolError(
            "'CBSX' is in deconv_methods but cbsx_mail/cbsx_token weren't given -- CellTFusion stops "
            "without CIBERSORTx credentials. Pass them, or drop 'CBSX' from deconv_methods."
        )
    if tf_collection == "ARACNE" and not tfs_csv:
        if not aracne_network_file:
            raise ToolError("tf_collection='ARACNE' needs aracne_network_file (tab-separated, Regulator/Target columns).")
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
            "raw.counts": csv_arg(raw_counts_csv, row_names=1, as_matrix=True),
            "deconv": csv_arg(deconv_csv, row_names=1) if deconv_csv else None,
            "dt": rds_arg(dt_rds) if dt_rds else None,
            "tfs": csv_arg(tfs_csv, row_names=1) if tfs_csv else None,
            "pathways": csv_arg(pathways_csv, row_names=1) if pathways_csv else None,
            "universe": csv_arg(universe_csv, row_names=None) if universe_csv else None,
            "paths": csv_arg(paths_csv, row_names=None) if paths_csv else None,
            "gene_sets": rds_arg(gene_sets_rds) if gene_sets_rds else None,
            "normalized": lit(normalized),
            "coldata": csv_arg(coldata_csv, row_names=1) if coldata_csv else None,
            "batch": lit(batch),
            "batch_id": lit(batch_column) if batch_column else None,
            "deconv_methods": lit(deconv_methods),
            "cbsx.mail": lit(cbsx_mail) if cbsx_mail else None,
            "cbsx.token": lit(cbsx_token) if cbsx_token else None,
            "file_name": lit(file_name) if file_name else None,
            "TF.collection": lit(tf_collection),
            "min_targets_size": lit(min_targets_size),
            "minMod": lit(min_module_size),
            "corr_mod": lit(corr_mod),
            "corr": lit(corr),
            "corr_type": lit(corr_type),
            "cells_extra": lit(cells_extra) if cells_extra else None,
            "pval": lit(pval),
            "enrich_thresh": lit(enrich_thresh),
            "quantile_cutoff": lit(quantile_cutoff),
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
                "CSV of RNAseq counts: genes as rows (first column = gene symbol), samples as "
                "columns -- the same RAW input pipeML-server's train_ml_model takes as "
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
                "If True (default), runs CellTFusion once on all samples and returns their latent "
                "factors with the outcome attached -- equivalent to what pipeML's "
                "fold_construction_fun contract does on its own final full-data refit after "
                "cross-validation. If False, instead builds per-fold training/test sets using "
                "folds (required, see that parameter) -- the code path pipeML-server's "
                "train_ml_model(fold_construction_fun='prepare_celltfusion_folds') drives per fold: "
                "CellTFusion runs on each fold's training samples only, and its held-out samples "
                "are projected onto that fold's latent factors (via project_celltfusion_factors' "
                "logic)."
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
                "sets big enough for WGCNA/NMF (a few dozen samples in practice)."
            )
        ),
    ] = None,
    ncores: Annotated[int, Field(ge=1, description="Folds computed in parallel when compute_full_dataset=False (the first fold always runs first, so TF/pathway collections are downloaded once).")] = 1,
    coldata_csv: Annotated[str | None, Field(description="Sample metadata CSV (samples as rows), matched to counts_csv's samples by ID -- required with batch=True, to read batch_column.")] = None,
    batch: Annotated[bool, Field(description="Multi-cohort mode (see run_celltfusion's batch); held-out samples of each cohort are then projected separately, centred on their own means. Requires coldata_csv and batch_column.")] = False,
    batch_column: Annotated[str | None, Field(description="Column in coldata_csv identifying each sample's cohort, if batch=True.")] = None,
    normalized: Annotated[bool, Field(description="Same as run_celltfusion's normalized.")] = True,
    tf_collection: Annotated[Literal["CollecTRI", "Dorothea"], Field(description="Same as run_celltfusion's tf_collection (ARACNE isn't supported here).")] = "CollecTRI",
    min_targets_size: Annotated[int, Field(ge=1, description="Same as run_celltfusion's min_targets_size.")] = 3,
    min_module_size: Annotated[int, Field(ge=1, description="Same as run_celltfusion's min_module_size.")] = 3,
    corr_mod: Annotated[float, Field(ge=0, le=1, description="Same as run_celltfusion's corr_mod.")] = 0.9,
    corr: Annotated[float, Field(ge=0, le=1, description="Same as run_celltfusion's corr.")] = 0.7,
    corr_type: Literal["spearman", "pearson"] = "spearman",
    cells_extra: CellTypeList = None,
    pval: Annotated[float, Field(ge=0, le=1, description="Same as run_celltfusion's pval.")] = 0.05,
    enrich_thresh: Annotated[float, Field(ge=0, description="Same as run_celltfusion's enrich_thresh.")] = 1.5,
    quantile_cutoff: Annotated[float, Field(ge=0, le=1, description="Same as run_celltfusion's quantile_cutoff.")] = 0.7,
    cancer_type: Annotated[Literal["blca", "luad", "skcm"] | None, Field(description="Same as run_celltfusion's cancer_type (TME_states mapping; skipped if unset).")] = None,
    verbose: bool = False,
) -> RunResult:
    """Inspect what pipeML's CellTFusion custom fold-construction integration would train on
    (CellTFusion::prepare_celltfusion_folds).

    WHAT: the same CellTFusion pipeline run_celltfusion performs (TF activity, TF modules, cell
    groups, NMF latent factors), but packaged to match pipeML's fold_construction_fun contract --
    it's what pipeML-server's train_ml_model calls internally, per fold, when given
    fold_construction_fun="prepare_celltfusion_folds". The model is trained on the NMF latent
    factors; learning them on each fold's training samples only (and projecting the held-out
    samples onto them) is what keeps the cross-validation leakage-free. Calling it here standalone
    lets you inspect those features before committing to a full, very slow train_ml_model run, or
    reproduce/debug one specific fold's projected test features.

    WHEN: before train_ml_model with fold_construction_fun="prepare_celltfusion_folds", to sanity
    check what CellTFusion would actually produce for this cohort -- or afterwards, to inspect a
    specific fold's output directly. Heavy: one full CellTFusion run per fold (and one on all
    samples with compute_full_dataset=True).

    Returns, with compute_full_dataset=True, an unnamed list of (1) the latent factors of all
    samples with the outcome columns attached, (2) the full run_celltfusion-style result on all
    samples (what train_ml_model returns as Custom_output), and (3) the bestune marker. With
    compute_full_dataset=False, a list named by fold, each with train_data (latent factors of the
    training samples + the outcome columns), test_data (held-out samples projected onto that
    fold's factors, plus time/event for survival), obs_test (class labels, or a time/event data
    frame for survival), rowIndex, and fold_name -- each fold is also saved to
    run_dir/Results/fold_<name>.rds (stale fold_*.rds from earlier runs in the same run_dir are
    removed first).
    """
    if bool(target_column) == bool(time_column or event_column):
        raise ToolError("Give either target_column (classification) or time_column + event_column (survival).")
    if not target_column and not (time_column and event_column):
        raise ToolError("Survival needs both time_column and event_column.")
    if not compute_full_dataset and not folds:
        raise ToolError("folds is required when compute_full_dataset=False (fold name -> 1-indexed training samples).")
    if batch and not (coldata_csv and batch_column):
        raise ToolError("batch=True needs coldata_csv and batch_column (the column with each sample's cohort).")

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
            "raw.counts": csv_arg(counts_csv, row_names=1, as_matrix=True),
            "coldata": csv_arg(coldata_csv, row_names=1, **samples) if coldata_csv else None,
            "batch": lit(batch),
            "batch_id": lit(batch_column) if batch_column else None,
            "ncores": lit(ncores),
            "verbose": lit(verbose),
            "normalized": lit(normalized),
            "TF.collection": lit(tf_collection),
            "min_targets_size": lit(min_targets_size),
            "minMod": lit(min_module_size),
            "corr_mod": lit(corr_mod),
            "corr": lit(corr),
            "corr_type": lit(corr_type),
            "cells_extra": lit(cells_extra) if cells_extra else None,
            "pval": lit(pval),
            "enrich_thresh": lit(enrich_thresh),
            "quantile_cutoff": lit(quantile_cutoff),
            "cancer_type": lit(cancer_type) if cancer_type else None,
        }
    )
    return await _run(
        package="CellTFusion", function="prepare_celltfusion_folds", args=args, tool_name="prepare_celltfusion_folds",
        ctx=ctx, timeout=28800, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def project_celltfusion_factors(
    deconvolution_test_csv: Annotated[
        str,
        Field(
            description=(
                "RAW deconvolution matrix of the new/held-out cohort (samples as rows), with the "
                "same method/signature columns as the deconvolution the model was trained with "
                "(e.g. multideconv-server's run_deconvolution with the same methods)."
            )
        ),
    ],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    model_rds: Annotated[
        str | None,
        Field(
            description=(
                "result.rds of a pipeML-server train_ml_model call that used "
                "fold_construction_fun='prepare_celltfusion_folds' -- its Custom_output (the "
                "CellTFusion result learned on all training samples) is used. Give this or "
                "celltfusion_rds."
            )
        ),
    ] = None,
    celltfusion_rds: Annotated[
        str | None,
        Field(description="result.rds of a prior run_celltfusion call, to project onto its latent factors instead. Give this or model_rds.")
    ] = None,
) -> RunResult:
    """Project a new cohort onto latent factors learned by CellTFusion on another cohort
    (CellTFusion::project_test_factors).

    WHAT: CellTFusion's cell groups and NMF latent factors are data-dependent -- re-running it on
    a different cohort would give different factors. This instead rebuilds the new cohort's
    cell-group scores with the trained CCA loadings and projects them onto the trained NMF
    factors, so both cohorts are described by the same Factor1..FactorK columns (non-negative
    scores).

    WHEN: before predict_ml_model on a held-out cohort, when the model was trained with
    fold_construction_fun='prepare_celltfusion_folds' -- pass this call's result.csv as
    predict_ml_model's test_data_csv. Also to score a validation cohort consistently with a
    discovery cohort's run_celltfusion factors. If the training used batch mode, the test samples
    are centred on their own means, so project one cohort per call.

    Returns a samples x factors matrix (result.csv).
    """
    if bool(model_rds) == bool(celltfusion_rds):
        raise ToolError("Give exactly one of model_rds (a train_ml_model result) or celltfusion_rds (a run_celltfusion result).")
    args = args_from(
        **{
            "train_processed": rds_element_arg(model_rds, "Custom_output") if model_rds else rds_arg(celltfusion_rds),
            "test_deconv": csv_arg(deconvolution_test_csv, row_names=1),
        }
    )
    return await _run(
        package="CellTFusion", function="project_test_factors", args=args, tool_name="project_celltfusion_factors",
        ctx=ctx, timeout=1800, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
