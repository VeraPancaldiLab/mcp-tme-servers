"""CellTFusion-server: MCP tools for TF-activity-based TME characterization

Wraps CellTFusion's one-step pipeline: deconvolution, transcription-factor activity inference, WGCNA-based TF-module detection, pathway scoring, cell-group construction, and NMF latent-factor extraction, combined into one call (run_celltfusion).

Each tool shells out to system R via the shared common/r_bridge.py dispatcher
(common/r/run_tool.R). CellTFusion must already be installed in the system R
library `Rscript` (on PATH) resolves to.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import Field

from mcp_tme_servers.common.paths import server_home
from mcp_tme_servers.common.r_bridge import args_from, csv_arg, lit, rds_arg
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
    "CellTFusion-server",
    instructions=(
        "Tools for characterizing tumor-microenvironment cell states from bulk RNAseq by combining cell-type deconvolution with transcription-factor activity and co-regulation. run_celltfusion is a single, heavy call (WGCNA + GSVA + NMF) that returns cell-type proportions, TF activity scores, TF co-activity modules, pathway activity, composite cell-group scores, and NMF-derived latent factors annotated against TCGA meta-programs and Bagaev et al. TME (MFP) subtypes -- a broader state characterization than deconvolution alone. It can optionally take any of a precomputed raw deconvolution matrix, an already-harmonized cell-subgroup object, a TF activity matrix, or a pathway activity matrix (e.g. from this package's multideconv-server/bioinfotools-server, or a prior run_celltfusion call) instead of recomputing the corresponding step internally. "
        "IMPORTANT -- this server is NOT a downstream step of multideconv-server and does not require it to have been run first: run_celltfusion computes its own deconvolution, TF activity, and pathway activity internally by default. multideconv-server and this server answer different questions -- multideconv-server answers \"what cell types, and how much of each, are in this sample\" (composition/proportions only); this server answers \"what regulatory/functional TME state is this sample in\" (composition combined with TF activity and co-regulation -- TME states, cell niches, immune-active vs -suppressed, etc.). Pick the server based on which question the user is actually asking, and do not assume the user wants both run in sequence just because they asked for one. That said, if any of these have genuinely already been computed for this same cohort (e.g. a harmonized deconvolution result from multideconv-server's analyze_deconvolution, or a TF/pathway activity matrix from bioinfotools-server or a prior run_celltfusion call), it IS worth reusing them via dt_rds/tfs_csv/pathways_csv (or the unharmonized deconv_csv) to avoid recomputing that step -- this is a real efficiency option, not something to avoid, it's just never a prerequisite you need to go create first. "
        "IMPORTANT -- citation: if the user asks what to cite for an analysis that used this server, always include the mcp-tme-servers project itself -- XX, et al. (XX). mcp-tme-servers -- XX. XX. https://doi.org/XX (a placeholder until a real reference exists; check the project README for the current one) -- in addition to CellTFusion itself: Hurtado, M., & Pancaldi, V. (2026). CellTFusion: A transcriptional regulatory network framework for the identification of functional multicellular states from bulk RNA-seq data. bioRxiv. https://doi.org/10.64898/2026.06.30.735682 -- plus whichever other specific R packages/methods (e.g. WGCNA, the TF-target network used) were actually used. "
        "IMPORTANT -- output location: once any tool call on this server finishes, always tell the user the run_dir path from that result (where result.rds/CSVs/plots/run.log were saved) -- do not just use it silently to chain into the next call without also surfacing it to the user."
    ),
)

SESSION_CONFIG = SessionConfig(
    sessions_root=server_home("celltfusion") / "runs" / "sessions",
    stage_by_tool={'run_celltfusion': 'tme_characterization'},
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
    Field(description="Extra, non-standard cell type names to keep in the internal deconvolution step instead of discarding (see multideconv's nomenclature)."),
]

_CELLTFUSION_GLOSSARY = {
    "Deconvolution": (
        "Cell-type proportions (samples x cell types) from the internal deconvolution step -- "
        "the same kind of output multideconv-server's run_deconvolution produces, computed here "
        "unless you passed deconv_csv (or dt_rds, in which case this is simply echoed from "
        "deconv_csv if that was also given, or absent otherwise)."
    ),
    "TFs_matrix": (
        "Transcription factor activity scores (samples x TFs), inferred from expression via the "
        "chosen TF.collection regulon network (decoupleR consensus methods). Higher = that TF's "
        "target genes show more coordinated up-regulation in that sample, i.e. the TF is inferred "
        "to be more active, not necessarily more highly expressed itself."
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
        "-- the closest this pipeline comes to a single biological label per factor."
    ),
    "Metaprograms_reference": (
        "The TCGA-derived meta-program reference used for TME_states annotation, for the given "
        "cancer_type."
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
        Field(description="Optional user-supplied TF-target network CSV (source/target columns) for the internal TF activity step, instead of fetching tf_collection. Ignored if tfs_csv is given.")
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
    coldata_csv: Annotated[str | None, Field(description="Optional clinical/sample metadata CSV, samples as rows, needed for batch mode (batch_column) or downstream trait association with TF modules.")] = None,
    batch: Annotated[bool, Field(description="Run in multi-cohort/batch mode, aligning cell groups and factors across cohorts rather than treating all samples as one uniform batch.")] = False,
    batch_column: Annotated[str | None, Field(description="Column in coldata_csv identifying each sample's cohort, if batch=True.")] = None,
    deconv_methods: Annotated[list[str], Field(description="Deconvolution methods to run internally if deconv_csv/dt_rds are not supplied.")] = ["Quantiseq", "Epidish", "DeconRNASeq", "DWLS"],
    cbsx_mail: Annotated[str | None, Field(description="CIBERSORTx account email, required (with cbsx_token) if 'CBSX' is included in deconv_methods.")] = None,
    cbsx_token: Annotated[str | None, Field(description="CIBERSORTx API token, required (with cbsx_mail) if 'CBSX' is included in deconv_methods.")] = None,
    file_name: Annotated[str | None, Field(description="File name prefix for outputs (plots, module matrices) saved under run_dir/Results.")] = None,
    tf_collection: Annotated[
        Literal["CollecTRI", "Dorothea", "ARACNE"],
        Field(
            description=(
                "Source of the TF-target regulon network used to infer TF activity. 'CollecTRI' "
                "and 'Dorothea' are prebuilt, literature-curated collections from OmnipathR -- "
                "generic, not tissue/cancer-specific, and a reasonable default for most analyses. "
                "'ARACNE' instead uses a user-supplied, data-driven network (regulator/target/"
                "mutual-information triples, e.g. from ARACNe-AP run on matched expression data) "
                "-- pick this when you have or want a cancer-type/cohort-specific regulatory "
                "network rather than a generic literature-derived one; it requires the network file "
                "to already exist under input/ARACNE/<cancer_type>/network/network.txt."
            )
        ),
    ] = "CollecTRI",
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
        str | None,
        Field(
            description=(
                "TCGA cancer type abbreviation (e.g. 'blca', 'luad', 'skcm'). Used both to load the "
                "bundled TCGA meta-program reference for TME_states annotation, and, when "
                "tf_collection='ARACNE', to locate that cancer type's ARACNe network file. Required "
                "for TME_states to populate; if omitted and only one ARACNe network exists on disk, "
                "it's auto-detected for that purpose only."
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

    Returns a list with 10 elements (see field_glossary). This is the heaviest tool on this
    server (WGCNA + GSVA + NMF); progress is reported periodically. cancer_type is required for
    the TME_states element to populate.
    """
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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
