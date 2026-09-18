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
    "CellTFusion-server",
    instructions=(
        "Tools for characterizing tumor-microenvironment cell states from bulk RNAseq by combining cell-type deconvolution with transcription-factor activity and co-regulation. run_celltfusion is a single, heavy call (WGCNA + GSVA + NMF) that returns cell-type proportions, TF activity scores, TF co-activity modules, pathway activity, composite cell-group scores, and NMF-derived latent factors annotated against TCGA meta-programs and Bagaev et al. TME (MFP) subtypes -- a broader state characterization than deconvolution alone. It can optionally take a precomputed deconvolution matrix (e.g. from this package's multideconv server) instead of recomputing it internally."
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
        "unless you passed deconv_csv."
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
    deconv_csv: Annotated[str | None, Field(description="Optional precomputed deconvolution matrix CSV (e.g. from multideconv-server's run_deconvolution); computed internally via multideconv if omitted, which is slower but self-contained.")] = None,
    normalized: Annotated[bool, Field(description="Set False if raw_counts_csv already holds log-TPM normalized values instead of raw counts.")] = True,
    coldata_csv: Annotated[str | None, Field(description="Optional clinical/sample metadata CSV, samples as rows, needed for supervised DEG analysis (contrast_column) or downstream trait/survival association.")] = None,
    batch: Annotated[bool, Field(description="Run in multi-cohort/batch mode, aligning cell groups and factors across cohorts rather than treating all samples as one uniform batch.")] = False,
    batch_column: Annotated[str | None, Field(description="Column in coldata_csv identifying each sample's cohort, if batch=True.")] = None,
    deconv_methods: Annotated[list[str], Field(description="Deconvolution methods to run internally if deconv_csv is not supplied.")] = ["Quantiseq", "Epidish", "DeconRNASeq", "DWLS"],
    file_name: Annotated[str | None, Field(description="File name prefix for outputs (plots, module matrices) saved under run_dir/Results.")] = None,
    task: Annotated[Literal["supervised", "unsupervised"], Field(description="'unsupervised' (default) derives cell states purely from data structure. 'supervised' additionally runs differential-expression analysis contrasting groups defined by contrast_column/ref_level.")] = "unsupervised",
    contrast_column: Annotated[str | None, Field(description="Column in coldata_csv defining condition groups, for supervised DEG analysis.")] = None,
    ref_level: Annotated[str | None, Field(description="Reference level of contrast_column for supervised DEG analysis (the 'baseline' group DEG is computed against).")] = None,
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
    genuinely immune-active one, which raw cell-type fractions can't do by themselves. Pass
    deconv_csv (e.g. from multideconv-server's run_deconvolution) if you already have a trusted,
    harmonized deconvolution result -- otherwise this recomputes deconvolution internally.

    Returns a list with 10 elements (see field_glossary). This is the heaviest tool on this
    server (WGCNA + GSVA + NMF); progress is reported periodically. cancer_type is required for
    the TME_states element to populate.
    """
    args = args_from(
        **{
            "raw.counts": csv_arg(raw_counts_csv, row_names=1, as_matrix=True),
            "deconv": csv_arg(deconv_csv, row_names=1) if deconv_csv else None,
            "normalized": lit(normalized),
            "coldata": csv_arg(coldata_csv, row_names=1) if coldata_csv else None,
            "batch": lit(batch),
            "batch_id": lit(batch_column) if batch_column else None,
            "deconv_methods": lit(deconv_methods),
            "file_name": lit(file_name) if file_name else None,
            "task": lit(task),
            "contrast": lit(contrast_column) if contrast_column else None,
            "ref_level": lit(ref_level) if ref_level else None,
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
        ctx=ctx, timeout=3600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary=_CELLTFUSION_GLOSSARY, diagnostics=_celltfusion_diagnostics,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
