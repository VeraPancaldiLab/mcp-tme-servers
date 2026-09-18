"""multideconv-server: MCP tools for bulk RNAseq cell-type deconvolution

Wraps five entry points from the multideconv R package: running multiple first- and second-generation deconvolution methods on bulk RNAseq, harmonizing their output across methods/signatures into cross-method-agreed cell subgroups, benchmarking against ground-truth proportions, relating subgroups to pathway activity, and replicating learned subgroups onto a new cohort.

Each tool shells out to system R via the shared common/r_bridge.py dispatcher
(common/r/run_tool.R). multideconv must already be installed in the system R
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
    "multideconv-server",
    instructions=(
        'Tools for bulk RNAseq cell-type deconvolution: estimating the relative abundance of immune/stromal/tumor cell types from bulk transcriptomics, then reconciling disagreement across methods and signatures into a smaller set of trustworthy cell subgroups. The pipeline order is run_deconvolution -> analyze_deconvolution, then optionally benchmark_deconvolution (validate against known cell fractions), relate_subgroups_to_pathways (correlate with pathway activity), or replicate_subgroups (apply the same subgroup definitions to a new cohort). Deconvolution output from this server is the typical input to the pipeML, CellTFusion, and RaCInG servers in this same package.'
    ),
)

SESSION_CONFIG = SessionConfig(
    sessions_root=server_home("multideconv") / "runs" / "sessions",
    stage_by_tool={'run_deconvolution': 'deconvolution', 'analyze_deconvolution': 'harmonization', 'benchmark_deconvolution': 'benchmarking', 'relate_subgroups_to_pathways': 'pathway_relation', 'replicate_subgroups': 'subgroup_replication'},
    export_requires=[frozenset({'deconvolution'}), frozenset({'harmonization'})],
    stage_explanations={'deconvolution': 'no cell-type deconvolution has been run yet (run_deconvolution) -- there is no cell-type abundance data in this session to export.', 'harmonization': "deconvolution results haven't been harmonized across methods/signatures yet (analyze_deconvolution) -- exporting now would give raw, per-method-and-signature columns full of redundant or disagreeing features instead of the reduced, cross-method-agreed cell subgroups that downstream analyses are meant to consume."},
)

_run = make_run(runs_root=server_home("multideconv") / "runs")

register_session_tools(mcp, SESSION_CONFIG, domain="cell-type deconvolution")


# --------------------------------------------------------------------------
# multideconv tools
# --------------------------------------------------------------------------

CellTypeList = Annotated[
    list[str] | None,
    Field(
        description=(
            "Extra, non-standard cell type names to keep instead of discarding. multideconv "
            "recognizes a fixed vocabulary of cell type names (immune/stromal/tumor lineages); "
            "any deconvolution column whose cell type isn't in that vocabulary is dropped as "
            "'Discarded cell types' unless listed here."
        )
    ),
]


def _analyze_deconvolution_glossary() -> dict[str, str]:
    return {
        "Deconvolution matrix": (
            "The harmonized cell-type proportion matrix after merging agreeing "
            "method/signature estimates into subgroups and removing zero-heavy/low-variance "
            "features -- this is the main output to carry into downstream analysis "
            "(pipeML-server, CellTFusion-server, RaCInG-server)."
        ),
        "Deconvolution subgroups per cell types": (
            "For each canonical cell type, which method/signature combinations were judged "
            "to agree closely enough (>= corr) to be treated as estimating the same thing."
        ),
        "Deconvolution subgroups composition": (
            "The subgroup membership structure itself. This is the object replicate_subgroups "
            "and relate_subgroups_to_pathways expect as their subgroups/deconv_res input -- "
            "pass this call's run_dir/result.rds to those tools' *_rds parameter."
        ),
        "Discarded groups with equal method": (
            "Subgroups dropped because every member came from the same deconvolution method "
            "across different signatures rather than genuine cross-method agreement -- "
            "agreement between two signatures of the same method isn't evidence the cell type "
            "estimate itself is robust."
        ),
        "Discarded features with high number of zeros": (
            "Features dropped for exceeding zero_thr fraction of samples at zero -- usually a "
            "cell type this cohort's data can't reliably distinguish from absent."
        ),
        "Discarded features with low variance": (
            "Features dropped for near-constant values across samples (below var_quantile) -- "
            "uninformative for distinguishing samples from each other."
        ),
        "Discarded cell types": (
            "Cell type labels found in the input that aren't part of multideconv's standard "
            "nomenclature and weren't listed in cells_extra -- pass them via cells_extra if "
            "they're real, meaningful cell types you want kept."
        ),
        "High correlated deconvolution groups (>0.9) per cell type": (
            "Which specific feature pairs exceeded prune_thr and were pruned/merged together "
            "-- useful for auditing exactly what got collapsed."
        ),
    }


def _analyze_deconvolution_diagnostics(summary: dict) -> list[str]:
    warnings: list[str] = []
    elements = summary.get("elements", {})
    deconv = elements.get("Deconvolution matrix", {})
    dim = deconv.get("dim")
    if dim and len(dim) == 2:
        n_features = dim[1]
        if n_features == 0:
            warnings.append(
                "No cell-type subgroups survived filtering -- every feature was discarded as "
                "zero-heavy, low-variance, or non-standard. This usually means the input has "
                "very few samples, uses cell type names multideconv doesn't recognize (pass "
                "cells_extra), or the deconvolution itself produced near-degenerate values. "
                "Re-run with verbose=True to see what's being dropped and why."
            )
        elif n_features < 3:
            warnings.append(
                f"Only {n_features} cell-type subgroup(s) survived filtering -- unusually few. "
                "If you expected more cell types, this usually means the deconvolution methods/"
                "signatures disagreed substantially with each other (try a lower corr to merge "
                "more aggressively) or too many features were discarded as zero-heavy/"
                "low-variance for this cohort's sample size (try raising zero_thr or lowering "
                "var_quantile)."
            )
    discarded_types = elements.get("Discarded cell types", {})
    if discarded_types.get("length", 0) > 0:
        warnings.append(
            "Some cell type labels in the input weren't recognized by multideconv's standard "
            "nomenclature and were discarded. If these are real, non-standard cell types "
            "(e.g. from a custom signature) that should be kept, re-run with their names "
            "passed via cells_extra."
        )
    return warnings


def _benchmark_diagnostics(summary: dict) -> list[str]:
    warnings: list[str] = []
    stats = summary.get("numeric_summary")
    if stats:
        mean_corr = stats.get("mean")
        if mean_corr is not None and mean_corr < 0.3:
            warnings.append(
                f"Mean correlation across all deconvolution-vs-groundtruth pairs is low "
                f"({mean_corr:.2f}). This usually means either the deconvolution run and the "
                "ground truth disagree substantially (check cell type name alignment between "
                "the two matrices -- a silent mismatch looks like poor correlation, not an "
                "error), or the deconvolution itself is unreliable for this cohort. Check the "
                "per-cell-type values in result.csv before trusting a specific method/signature."
            )
    dim = summary.get("dim")
    if dim and len(dim) == 2 and dim[0] == 0:
        warnings.append(
            "The benchmark matrix has zero rows -- no ground-truth cell types matched any "
            "deconvolution feature after cell type name alignment. Check that groundtruth_csv's "
            "column names match multideconv's cell type nomenclature (or are listed in "
            "cells_extra)."
        )
    return warnings


@mcp.tool(annotations=READ_ANNOTATIONS)
async def run_deconvolution(
    raw_counts_csv: Annotated[
        str,
        Field(
            description=(
                "CSV of raw RNAseq counts: genes as rows (first column = gene symbol), samples "
                "as columns. Raw (un-normalized) integer-like counts are expected by default; "
                "set normalized=False if you're instead passing already-normalized expression."
            )
        ),
    ],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    methods: Annotated[
        list[Literal["Quantiseq", "CBSX", "Epidish", "DeconRNASeq", "DWLS"]],
        Field(
            description=(
                "First-generation (signature-based) deconvolution methods to run. These estimate "
                "cell-type fractions directly from bulk expression using a fixed, precomputed "
                "signature matrix -- fast and don't need matched single-cell data, but are only "
                "as good as how well the signature matches this cohort's tissue/technology. "
                "Running several methods (the default) is the point: agreement across methods "
                "is what analyze_deconvolution uses to identify trustworthy cell subgroups "
                "versus method-specific artifacts. CIBERSORTx (CBSX) is omitted from the default "
                "set as it needs separate Docker credentials."
            )
        ),
    ] = ["Quantiseq", "Epidish", "DeconRNASeq", "DWLS"],
    signatures_select: Annotated[
        list[str] | None,
        Field(description="Signature matrix names to run (e.g. tissue- or platform-specific references); all bundled signatures are used if omitted."),
    ] = None,
    normalized: Annotated[bool, Field(description="Set False only if raw_counts_csv already holds normalized (not raw count) expression values.")] = True,
    do_parallel: Annotated[bool, Field(description="Parallelize CBSX/DWLS across signatures (faster with many signatures, more memory).")] = False,
    workers: Annotated[int | None, Field(description="Worker processes if do_parallel=True; default is detectCores()-1.")] = None,
    create_signature: Annotated[bool, Field(description="Build custom signature matrices from the provided single-cell reference (requires the sc_* arguments) instead of using bundled signatures.")] = False,
    cells_extra: CellTypeList = None,
    file_name: Annotated[str | None, Field(description="File name prefix for CSV/plots written under run_dir/Results.")] = None,
    sc_deconv: Annotated[
        bool,
        Field(
            description=(
                "Also run second-generation, single-cell-reference-based methods. Unlike the "
                "first-generation methods above, these build a signature tailored to this cohort's "
                "own matched scRNAseq reference rather than using a generic precomputed one -- "
                "more tailored, but only as good as how representative and well-annotated the "
                "single-cell reference is, and requires sc_matrix_csv/sc_metadata_csv."
            )
        ),
    ] = False,
    sc_matrix_csv: Annotated[str | None, Field(description="Single-cell count matrix (genes x cells) CSV, required if sc_deconv=True.")] = None,
    sc_metadata_csv: Annotated[str | None, Field(description="Single-cell metadata CSV with a cell-type-label column and a sample-of-origin column, required if sc_deconv=True.")] = None,
    cell_label_column: Annotated[str | None, Field(description="Column in sc_metadata_csv giving each cell's type label.")] = None,
    sample_label_column: Annotated[str | None, Field(description="Column in sc_metadata_csv giving each cell's sample of origin.")] = None,
    methods_sc: Annotated[list[str], Field(description="Second-generation methods to run if sc_deconv=True.")] = ["Autogenes", "BayesPrism", "Bisque", "CPM", "MuSic", "SCDC"],
) -> RunResult:
    """Estimate cell-type abundance from bulk RNAseq (multideconv::compute.deconvolution).

    WHAT: runs one or more deconvolution methods (and optionally signatures) against a bulk
    expression matrix, producing one abundance/proportion column per method-signature-cell-type
    combination -- e.g. `Epidish_TIL10_CD8.cells` is EpiDISH's estimate of CD8 T-cell fraction
    using the TIL10 signature. Nothing is merged or reconciled at this stage; that happens next.

    WHEN: this is the first step of every pipeline on this server. Its output feeds
    analyze_deconvolution to reconcile disagreement across methods, and from there into
    pipeML-server, CellTFusion-server, or RaCInG-server.

    Returns a samples x features matrix of cell-type proportions.
    """
    args = args_from(
        **{
            "raw.counts": csv_arg(raw_counts_csv, row_names=1, as_matrix=True),
            "methods": lit(methods),
            "signatures_select": lit(signatures_select) if signatures_select else None,
            "normalized": lit(normalized),
            "doParallel": lit(do_parallel),
            "workers": lit(workers) if workers is not None else None,
            "return": lit(True),
            "create_signature": lit(create_signature),
            "cells_extra": lit(cells_extra) if cells_extra else None,
            "file_name": lit(file_name) if file_name else None,
            "sc_deconv": lit(sc_deconv),
            "sc_matrix": csv_arg(sc_matrix_csv, row_names=1, as_matrix=True) if sc_matrix_csv else None,
            "sc_metadata": csv_arg(sc_metadata_csv, row_names=1) if sc_metadata_csv else None,
            "cell_label": csv_column_arg(sc_metadata_csv, cell_label_column) if (sc_metadata_csv and cell_label_column) else None,
            "sample_label": csv_column_arg(sc_metadata_csv, sample_label_column) if (sc_metadata_csv and sample_label_column) else None,
            "methods_sc": lit(methods_sc) if sc_deconv else None,
        }
    )
    return await _run(
        package="multideconv", function="compute.deconvolution", args=args, tool_name="run_deconvolution",
        ctx=ctx, timeout=1800, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def analyze_deconvolution(
    deconvolution_csv: Annotated[str, Field(description="Deconvolution matrix CSV (samples as rows, features as columns) from run_deconvolution's result.csv.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    corr: Annotated[
        float,
        Field(
            ge=0, le=1,
            description=(
                "Minimum correlation for grouping different methods'/signatures' estimates of "
                "(presumably) the same cell type into one merged subgroup. Higher (e.g. 0.9) is "
                "stricter: only near-identical estimates merge, so you keep more separate "
                "features, some of which may just be noisy or redundant restatements of each "
                "other. Lower (e.g. 0.5) merges more aggressively into fewer, more robust "
                "cross-method-agreed cell types, at the risk of conflating estimates that only "
                "partially agree. 0.7 (default) is a moderate middle ground; lower it if "
                "downstream analyses end up with implausibly many near-duplicate cell-type "
                "columns, raise it if subgroups seem to be merging cell types you'd expect to "
                "stay distinct."
            ),
        ),
    ] = 0.7,
    corr_type: Annotated[Literal["spearman", "pearson"], Field(description="Correlation type for subgroup clustering. Spearman (rank-based, default) is more robust to the non-normal, right-skewed distributions typical of deconvolution proportions.")] = "spearman",
    zero_thr: Annotated[float, Field(ge=0, le=1, description="Maximum fraction of samples allowed to be zero for a feature before it's discarded as unreliably estimated in this cohort.")] = 0.9,
    var_quantile: Annotated[float, Field(ge=0, le=1, description="Quantile threshold below which near-constant (uninformative) features are removed.")] = 0.25,
    prune_thr: Annotated[float, Field(ge=0, le=1, description="Pearson correlation above which within-subgroup feature pairs are pruned as redundant duplicates rather than kept separately.")] = 0.9,
    seed: Annotated[int | None, Field(description="Seed for reproducible tie-breaking when choosing which of several highly correlated features to keep.")] = None,
    batch_csv: Annotated[
        str | None,
        Field(
            description=(
                "Optional CSV with a batch/cohort covariate column, row-aligned to "
                "deconvolution_csv. When given (with batch_column), correlations are computed as "
                "*partial* correlations controlling for batch -- use this for multi-cohort data "
                "so that cohort-driven co-variation isn't mistaken for genuine cross-method "
                "agreement about a cell type."
            )
        ),
    ] = None,
    batch_column: Annotated[str | None, Field(description="Column in batch_csv identifying each sample's batch/cohort.")] = None,
    cells_extra: CellTypeList = None,
    file_name: Annotated[str | None, Field(description="File name prefix for the subgroups CSV written under run_dir/Results.")] = None,
    verbose: Annotated[bool, Field(description="Print details of what's discarded and why at each filtering step -- useful when few subgroups survive and you need to see why.")] = False,
) -> RunResult:
    """Harmonize deconvolution output across methods into robust cell subgroups
    (multideconv::compute.deconvolution.analysis).

    WHAT: different deconvolution methods and signatures rarely agree exactly on a cell type's
    abundance. This step (1) discards features that are mostly zero or near-constant across
    samples (unreliable/uninformative for this cohort), then (2) clusters the remaining
    method/signature estimates of presumably-the-same cell type by pairwise correlation, merging
    ones that agree (>= corr) into a single subgroup and pruning near-duplicates within a
    subgroup (> prune_thr). The result is a smaller set of cell-type features you can trust
    because multiple independent methods converged on them, rather than one method's idiosyncratic
    estimate.

    WHEN: always run this right after run_deconvolution before using the deconvolution output for
    anything else -- benchmarking, ML features, pathway relation, or feeding into RaCInG/
    CellTFusion. Skipping it means downstream tools see raw, redundant, disagreement-laden
    columns instead of the reconciled cell types.

    Returns a list (see field_glossary); run_dir/result.rds is the `subgroups`/`deconv_res` input
    relate_subgroups_to_pathways and replicate_subgroups expect.
    """
    args = args_from(
        **{
            "deconvolution": csv_arg(deconvolution_csv, row_names=1),
            "corr": lit(corr),
            "corr_type": lit(corr_type),
            "zero_thr": lit(zero_thr),
            "var_quantile": lit(var_quantile),
            "prune_thr": lit(prune_thr),
            "seed": lit(seed) if seed is not None else None,
            "batch": csv_column_arg(batch_csv, batch_column) if (batch_csv and batch_column) else None,
            "cells_extra": lit(cells_extra) if cells_extra else None,
            "file_name": lit(file_name) if file_name else None,
            "return": lit(True),
            "verbose": lit(verbose),
        }
    )
    return await _run(
        package="multideconv", function="compute.deconvolution.analysis", args=args, tool_name="analyze_deconvolution",
        ctx=ctx, timeout=900, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary=_analyze_deconvolution_glossary(), diagnostics=_analyze_deconvolution_diagnostics,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def benchmark_deconvolution(
    deconvolution_csv: Annotated[str, Field(description="Deconvolution matrix CSV to benchmark (samples as rows).")],
    groundtruth_csv: Annotated[str, Field(description="True cell type proportions CSV (e.g. from flow cytometry/FACS or a known mixture), samples as rows, cell types as columns.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    cells_extra: CellTypeList = None,
    corr_type: Annotated[Literal["spearman", "pearson"], Field(description="Correlation type for the benchmark. Spearman (default) is robust to the non-linear relationship deconvolution estimates often have with true proportions.")] = "spearman",
    scatter: Annotated[bool, Field(description="Return per-cell-type scatter plots of estimated vs true proportion.")] = True,
    plot: Annotated[bool, Field(description="Save the benchmark heatmap/plot to run_dir/Results.")] = False,
    pval: Annotated[float, Field(ge=0, le=1, description="P-value threshold below which a correlation is flagged as statistically significant in the plot.")] = 0.05,
    file_name: Annotated[str | None, Field(description="File name for the saved plot.")] = None,
    width: float = 16,
    height: float = 8,
) -> RunResult:
    """Validate deconvolution output against known ground-truth cell proportions
    (multideconv::compute.benchmark).

    WHAT: correlates every deconvolution feature (each method-signature-cell-type combination)
    against the matching ground-truth cell type, so you can see which methods/signatures perform
    best on data where you actually know the answer (e.g. a spike-in mixture, FACS-sorted
    reference, or a cohort with orthogonal cell-count measurements).

    WHEN: use this to choose which method(s)/signature(s) to trust for a given tissue/technology
    before committing to them for a cohort where the truth is unknown -- not part of the main
    run_deconvolution -> analyze_deconvolution -> ... pipeline itself.

    Returns a correlation matrix; low correlations can mean either genuinely poor deconvolution
    performance or a silent cell-type name mismatch between the two inputs (see warnings).
    """
    args = args_from(
        **{
            "deconvolution": csv_arg(deconvolution_csv, row_names=1),
            "groundtruth": csv_arg(groundtruth_csv, row_names=1),
            "cells_extra": lit(cells_extra) if cells_extra else None,
            "corr_type": lit(corr_type),
            "scatter": lit(scatter),
            "plot": lit(plot),
            "pval": lit(pval),
            "file_name": lit(file_name) if file_name else None,
            "width": lit(width),
            "height": lit(height),
        }
    )
    return await _run(
        package="multideconv", function="compute.benchmark", args=args, tool_name="benchmark_deconvolution",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_benchmark_diagnostics,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def relate_subgroups_to_pathways(
    subgroups_rds: Annotated[str, Field(description="result.rds from a prior analyze_deconvolution call.")],
    pathways_csv: Annotated[str, Field(description="Pathway activity matrix CSV (samples as rows, pathways as columns), e.g. CellTFusion-server's run_celltfusion Pathways_scores output, or a GSVA/PROGENy score matrix computed elsewhere.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    file_name: Annotated[str, Field(description="File name prefix for the output heatmap PDFs.")] = "multideconv-server",
    height: float = 6,
    width: float = 12,
    pval: Annotated[float, Field(ge=0, le=1, description="P-value above which a correlation is left unstarred on the heatmap.")] = 0.05,
) -> RunResult:
    """Correlate cell-type subgroups with pathway activity, per cell type
    (multideconv::compute.subgroup.pathways).

    WHAT: for each harmonized cell-type subgroup from analyze_deconvolution, correlates its
    abundance against each pathway's activity score across samples -- e.g. does CD8 T-cell
    abundance track with interferon-gamma pathway activity in this cohort. Saves one heatmap PDF
    per cell type.

    WHEN: after analyze_deconvolution, once you have (or compute separately, e.g. via
    CellTFusion-server) a pathway activity matrix you want to relate cell composition to.

    Side-effect only: check output_files for the generated heatmap PDF paths under run_dir/Results.
    """
    args = args_from(
        **{
            "subgroups": rds_arg(subgroups_rds),
            "pathways": csv_arg(pathways_csv, row_names=1),
            "file_name": lit(file_name),
            "height": lit(height),
            "width": lit(width),
            "pval": lit(pval),
        }
    )
    return await _run(
        package="multideconv", function="compute.subgroup.pathways", args=args, tool_name="relate_subgroups_to_pathways",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def replicate_subgroups(
    deconv_res_rds: Annotated[str, Field(description="result.rds from a prior analyze_deconvolution call, defining the subgroup structure to replicate.")],
    deconvolution_test_csv: Annotated[str, Field(description="Deconvolution matrix CSV from a new/held-out cohort to project onto the trained subgroup definitions.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
) -> RunResult:
    """Apply subgroup definitions learned on one cohort to a new cohort
    (multideconv::replicate_deconvolution_subgroups).

    WHAT: analyze_deconvolution's subgroup clustering is data-dependent -- re-running it on a
    different cohort could group features differently. This instead reuses the exact subgroup
    membership already learned on a training cohort and scores a new cohort's raw deconvolution
    output against those same definitions, so the two cohorts' harmonized features are directly
    comparable (same columns, same meaning).

    WHEN: for a validation/replication cohort you want to score consistently with a discovery
    cohort's already-established subgroups -- e.g. before pooling both into a joint downstream
    analysis, or validating a model trained on the discovery cohort's harmonized features.

    Returns a data.frame with the projected subgroup feature proportions for the new cohort.
    """
    args = args_from(
        **{
            "deconv_res": rds_arg(deconv_res_rds),
            "deconvolution_test": csv_arg(deconvolution_test_csv, row_names=1),
        }
    )
    return await _run(
        package="multideconv", function="replicate_deconvolution_subgroups", args=args, tool_name="replicate_subgroups",
        ctx=ctx, timeout=300, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
