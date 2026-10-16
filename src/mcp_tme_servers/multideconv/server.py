"""multideconv-server: MCP tools for bulk RNAseq cell-type deconvolution

Wraps the multideconv R package's entry points: running multiple first- and second-generation deconvolution methods on bulk RNAseq, summing cell types into coarser user-defined groups, harmonizing their output across methods/signatures into cross-method-agreed cell subgroups, benchmarking against ground-truth proportions, relating subgroups to pathway activity, and replicating learned subgroups onto a new cohort.

Each tool shells out to system R via the shared common/r_bridge.py dispatcher
(common/r/run_tool.R). multideconv must already be installed in the system R
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
from mcp_tme_servers.common.r_bridge import args_from, csv_arg, csv_column_arg, lit, named_args_arg, rds_arg
from mcp_tme_servers.common.session import SessionConfig
from mcp_tme_servers.common.tools import (
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
    "multideconv-server",
    instructions=(
        'Tools for bulk RNAseq cell-type deconvolution: estimating the relative abundance of immune/stromal/tumor cell types from bulk transcriptomics, then reconciling disagreement across methods and signatures into a smaller set of cell subgroups. The pipeline order within THIS server is run_deconvolution -> (optionally aggregate_cell_groups, to add coarser groups such as total myeloid or lymphoid cells) -> analyze_deconvolution, then optionally benchmark_deconvolution (validate against known cell fractions), relate_subgroups_to_pathways (correlate with pathway activity), or replicate_subgroups (apply the same subgroup definitions to a new cohort). '
        "IMPORTANT -- this server answers a specific question (\"what cell types, and how much of each, are in this sample\") and is not a required first step for the other servers in this package. pipeML-server can train on this server's harmonized output as one possible feature table among others; RaCInG-server needs cell-type abundance as a genuine input to build its interaction networks. CellTFusion-server is different: it answers a completely separate question (regulatory/TME-state characterization -- cell composition PLUS transcription-factor activity and co-regulation, not proportions alone) and computes its own deconvolution, TF activity, and pathway activity internally by default, so it does NOT need this server to have been run first. Never assume the user wants both servers run in sequence just because they asked for one of them -- pick the server based on which question was actually asked (composition/proportions -> this server; TME states/cell niches/regulatory activity -> CellTFusion-server). IMPORTANT -- immune infiltration IS a composition/proportions question: \"is this sample/patient immune infiltrated\" is answered from this server's cell-type (sub)group proportions (e.g. total or per-cell-type immune fraction from analyze_deconvolution), not from CellTFusion-server -- do not route plain infiltration questions there. Reserve CellTFusion-server for when the question goes beyond how much immune content is present, into the functional/regulatory state of that content (e.g. immune-active vs -suppressed, inflamed vs excluded, TME niches). That said, if this server's output already exists for the cohort in question (e.g. from analyze_deconvolution), it IS worth passing into CellTFusion-server's run_celltfusion (via deconv_csv/dt_rds) to avoid recomputing that step -- a genuine efficiency option when available, just never a prerequisite to go create first. "
        + EXPRESSION_SCALE_INSTRUCTIONS
        + 'On this server: run_deconvolution takes raw counts (normalized=True, the default) or linear TPM (normalized=False); it has no option for log-transformed input. '
        + 'IMPORTANT -- citation: if the user asks what to cite for an analysis that used this server, always include the mcp-tme-servers project itself -- XX, et al. (XX). mcp-tme-servers -- XX. XX. https://doi.org/XX (a placeholder until a real reference exists; check the project README for the current one) -- in addition to multideconv itself: Hurtado, M., Khajavi, L., Essabbar, A., & Pancaldi, V. (2025). multideconv -- Integrative pipeline for cell type deconvolution from bulk RNAseq using first and second generation methods. bioRxiv. https://doi.org/10.1101/2025.04.29.651220 -- plus whichever individual deconvolution methods/signatures were actually used. '
        'IMPORTANT -- output location: once any tool call on this server finishes, always tell the user the run_dir path from that result (where result.rds/CSVs/plots/run.log were saved) -- do not just use it silently to chain into the next call without also surfacing it to the user.'
        'IMPORTANT -- visualizing cell-type proportions: when showing cell-type composition/proportions from this server\'s output (e.g. analyze_deconvolution\'s harmonized subgroups), always plot every subgroup per cell type as its own bar -- never average or collapse subgroups within a cell type into a single value/bar. A cell type\'s subgroups (different method/signature combinations judged to agree) can still legitimately differ in their exact proportion estimates, and averaging them away hides that heterogeneity, which is itself informative (e.g. about how much the surviving methods/signatures actually agree on the number, not just the direction). Bar plots are the right chart for this -- one bar per subgroup, grouped/faceted by cell type (and by sample/patient when comparing across samples) -- so the spread across subgroups stays visible rather than being flattened into a single summary bar.'
    ),
)

SESSION_CONFIG = SessionConfig(
    sessions_root=server_home("multideconv") / "runs" / "sessions",
    stage_by_tool={'run_deconvolution': 'deconvolution', 'analyze_deconvolution': 'harmonization', 'benchmark_deconvolution': 'benchmarking', 'relate_subgroups_to_pathways': 'pathway_relation', 'replicate_subgroups': 'subgroup_replication', 'prepare_deconvolution_folds': 'fold_preparation', 'aggregate_cell_groups': 'cell_group_aggregation'},
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
            "recognizes a fixed vocabulary of cell type names (immune/stromal/tumor lineages -- "
            "B, T/CD4/CD8, NK, monocyte/macrophage, dendritic incl. plasmacytoid, myeloid, "
            "neutrophil/eosinophil/basophil/mast, plasma, cancer, epithelial, endothelial, "
            "fibroblast/CAF, pericyte/mural, myocyte); any deconvolution column whose cell type "
            "isn't in that vocabulary is dropped as 'Discarded cell types' unless listed here. "
            "This includes groups created with aggregate_cell_groups under a new name (e.g. "
            "Lymphocytes): list them here or they are discarded."
        )
    ),
]


def _analyze_deconvolution_glossary() -> dict[str, str]:
    return {
        "Deconvolution matrix": (
            "The harmonized cell-type proportion matrix after merging agreeing "
            "method/signature estimates into subgroups and removing zero-heavy/low-variance "
            "features -- this is the main output to carry into downstream analysis "
            "(for example pipeML-server, CellTFusion-server, RaCInG-server)."
        ),
        "Deconvolution subgroups per cell types": (
            "For each canonical cell type, its final features: one column per subgroup "
            "(<CellType>_Subgroup.<i>, the median of its member features) plus the "
            "method/signature features that didn't group with any other."
        ),
        "Deconvolution subgroups composition": (
            "The subgroup membership structure itself: per cell type, which method/signature "
            "features make up each <CellType>_Subgroup.<i> (every pair of them correlates >= "
            "corr). This is the object replicate_subgroups and relate_subgroups_to_pathways "
            "expect as their subgroups/deconv_res input -- pass this call's run_dir/result.rds "
            "to those tools' *_rds parameter."
        ),
        "Discarded features with high number of zeros": (
            "Features dropped for exceeding zero_thr fraction of samples at zero -- usually a "
            "cell type this cohort's data can't reliably distinguish from absent."
        ),
        "Discarded features with low variance": (
            "Features dropped for near-constant values across samples (coefficient of "
            "variation, sd/mean, below cv_thr) -- uninformative for distinguishing samples from "
            "each other."
        ),
        "Discarded cell types": (
            "Cell type labels found in the input that aren't part of multideconv's standard "
            "nomenclature and weren't listed in cells_extra -- pass them via cells_extra if "
            "they're real, meaningful cell types you want kept."
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
                "signatures were discarded as zero-heavy/low-variance for this cohort's sample "
                "size (try raising zero_thr or lowering cv_thr) -- see the two 'Discarded "
                "features' elements for what was dropped."
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
                "CSV of RNAseq expression: genes as rows (first column = gene symbol), samples "
                "as columns. Either raw (un-normalized, integer-like) counts, which is the default "
                "(normalized=True: TPM normalization is applied internally), or LINEAR TPM with "
                "normalized=False. Never log-transformed values: convert log2(TPM + 1) back to "
                "TPM (2^x - 1) first. Verify the scale before calling."
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
                "versus method-specific artifacts. CIBERSORTx (CBSX) IS part of the default set "
                "-- but it needs separate Docker credentials (credentials_mail + "
                "credentials_token) to actually run, which cannot be supplied automatically. "
                "See this tool's docstring for the credential-gathering step required before "
                "calling with CBSX included."
            )
        ),
    ] = ["Quantiseq", "Epidish", "DeconRNASeq", "DWLS", "CBSX"],
    credentials_mail: Annotated[
        str | None,
        Field(description="CIBERSORTx account email, required (with credentials_token) to run the CBSX method (included in methods by default) -- without both, multideconv skips CBSX with a warning. Register for free at https://cibersortx.stanford.edu if the user doesn't already have an account."),
    ] = None,
    credentials_token: Annotated[
        str | None,
        Field(description="CIBERSORTx API token, required (with credentials_mail) to run the CBSX method (included in methods by default). Get it from the account's settings at https://cibersortx.stanford.edu. Stored in this session's manifest.json for the R subprocess to read, but masked out of the session's exported reproducibility log."),
    ] = None,
    signatures_select: Annotated[
        list[str] | None,
        Field(description="Signature matrix names to run (e.g. tissue- or platform-specific references); all bundled signatures are used if omitted."),
    ] = None,
    normalized: Annotated[bool, Field(description="True (default): raw_counts_csv holds raw counts and is normalized to TPM internally. Set False only if raw_counts_csv already holds LINEAR TPM (it is then used as is). False does not mean 'any normalized values': log-transformed values are not accepted.")] = True,
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
    methods_sig: Annotated[
        list[Literal["DWLS", "CIBERSORTx", "MOMF", "BSeqsc"]],
        Field(
            description=(
                "Which methods to build a custom signature with, if create_signature=True "
                "(default: all four; multideconv also accepts 'CBSX' for CIBERSORTx). CIBERSORTx additionally requires credentials_mail/"
                "credentials_token to actually run; BSeq-SC additionally requires cell_markers "
                "and is silently skipped without it. Ignored unless create_signature=True."
            )
        ),
    ] = ["DWLS", "CIBERSORTx", "MOMF", "BSeqsc"],
    name_sc_signature: Annotated[
        str | None,
        Field(
            description=(
                "Name to give the custom signature being created, used in its output file names "
                "(e.g. Results/custom_signatures/DWLS-<name_sc_signature>-scRNAseq.txt under this "
                "call's run_dir). Give it a real, memorable name if you intend to reference this "
                "signature later via signatures_select in another call -- if left unset, the "
                "files get a generic name instead, and a later create_signature=True call reusing "
                "the same output_dir will overwrite them rather than creating a "
                "distinctly-named second signature. Any '_' is replaced by '-' ('_' separates "
                "method, signature and cell type in the output column names). Ignored unless "
                "create_signature=True."
            )
        ),
    ] = None,
    cell_markers: Annotated[
        dict[str, list[str]] | None,
        Field(
            description=(
                "Cell type name -> list of marker gene symbols, used only by the BSeq-SC method "
                "when building a custom signature (create_signature=True and 'BSeqsc' in "
                "methods_sig). If omitted, BSeq-SC is skipped during signature creation "
                "regardless of methods_sig. Ignored otherwise."
            )
        ),
    ] = None,
    custom_signature_files: Annotated[
        list[str] | None,
        Field(
            description=(
                "Paths to pre-computed, method-specific signature .txt files (the same format "
                "multideconv itself writes, e.g. from an earlier create_signature=True run, or "
                "hand-built following that format) to make available ADDITIONALLY to the bundled "
                "package signatures, without recomputing them. compute.deconvolution only reads "
                "custom signatures from a fixed relative folder (Results/custom_signatures/ under "
                "this call's own run directory), so each file is copied there before the R call "
                "runs -- you don't need to place them there yourself, and each file's own name is "
                "preserved (matters for signatures_select, which matches against file name minus "
                "extension). Not needed if create_signature=True: that path already creates and "
                "reads back from that same folder within one call. If output_dir/session_id are "
                "both omitted, a run directory is still resolved up front (same as the auto-named "
                "one this call would otherwise get) so the copy has somewhere to land."
            )
        ),
    ] = None,
) -> RunResult:
    """Estimate cell-type abundance from bulk RNAseq (multideconv::compute.deconvolution).

    WHAT: runs one or more deconvolution methods (and optionally signatures) against a bulk
    expression matrix, producing one abundance/proportion column per method-signature-cell-type
    combination -- e.g. `Epidish_TIL10_CD8.cells` is EpiDISH's estimate of CD8 T-cell fraction
    using the TIL10 signature. Nothing is merged or reconciled at this stage; that happens next.
    Cell type proportions sum to 1 (100%) only within each method_signature combination, not per
    row, because multideconv concatenates the cell-type columns from every method_signature
    combination together.

    WHEN: this is the first step of every pipeline on this server. Its output feeds
    analyze_deconvolution to reconcile disagreement across methods, and from there into
    pipeML-server, CellTFusion-server, or RaCInG-server.

    Before calling this tool, ask the user whether they already have pre-computed custom
    signature file(s) (not built via create_signature) they'd like included alongside the
    bundled ones -- don't assume none exist just because none were mentioned. If they do, ask
    where those files are and pass their paths via custom_signature_files; do not place or copy
    them anywhere yourself outside of that parameter -- see its own description for why a fixed
    folder matters here.

    CIBERSORTx (CBSX) credentials: CBSX is part of the default `methods` set, but it is the one
    method here that needs a CIBERSORTx Docker account -- credentials_mail and credentials_token
    -- which nothing in this pipeline can supply on its own. Before calling this tool with CBSX
    still in `methods` (i.e. whenever the caller hasn't deliberately removed it), ask the user
    for their CIBERSORTx email and API token.
      - If they already have credentials, pass them via credentials_mail/credentials_token and
        keep CBSX in methods.
      - If they don't, tell them to register a free account and generate a token at
        https://cibersortx.stanford.edu, then offer two options and let the user pick:
        (1) run now with CBSX excluded from methods (the other first-generation methods are
        unaffected -- CBSX only adds one more method's worth of agreement signal), or
        (2) wait until they have the credentials and call this tool then, with CBSX left in
        methods and credentials_mail/credentials_token filled in.
      - Don't silently drop CBSX and don't silently run without asking -- CBSX being in the
        default set is intentional and should stay the default whenever credentials are or
        become available. Note that multideconv itself no longer fails when CBSX is requested
        without credentials: it emits an R warning and skips CBSX (the call fails only if no
        method at all produced results), so a run without credentials silently lacks CBSX
        columns -- all the more reason to ask first.

    Returns a samples x features matrix of cell-type proportions.
    """
    if custom_signature_files:
        resolved_output_dir = resolve_output_dir(
            runs_root=server_home("multideconv") / "runs", tool_name="run_deconvolution",
            output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        )
        dest = Path(resolved_output_dir) / "Results" / "custom_signatures"
        dest.mkdir(parents=True, exist_ok=True)
        for src in custom_signature_files:
            src_path = Path(src).expanduser().resolve()
            shutil.copy2(src_path, dest / src_path.name)
        output_dir = resolved_output_dir

    args = args_from(
        **{
            "raw.counts": csv_arg(raw_counts_csv, row_names=1, as_matrix=True),
            "methods": lit(methods),
            "signatures_select": lit(signatures_select) if signatures_select else None,
            "normalized": lit(normalized),
            "doParallel": lit(do_parallel),
            "workers": lit(workers) if workers is not None else None,
            "return": lit(True),
            "credentials.mail": lit(credentials_mail) if credentials_mail else None,
            "credentials.token": lit(credentials_token) if credentials_token else None,
            "create_signature": lit(create_signature),
            "cells_extra": lit(cells_extra) if cells_extra else None,
            "file_name": lit(file_name) if file_name else None,
            "sc_deconv": lit(sc_deconv),
            "sc_matrix": csv_arg(sc_matrix_csv, row_names=1, as_matrix=True) if sc_matrix_csv else None,
            "sc_metadata": csv_arg(sc_metadata_csv, row_names=1) if sc_metadata_csv else None,
            "cell_label": csv_column_arg(sc_metadata_csv, cell_label_column) if (sc_metadata_csv and cell_label_column) else None,
            "sample_label": csv_column_arg(sc_metadata_csv, sample_label_column) if (sc_metadata_csv and sample_label_column) else None,
            "methods_sc": lit(methods_sc) if sc_deconv else None,
            "methods_sig": lit(methods_sig) if create_signature else None,
            "name_sc_signature": lit(name_sc_signature) if name_sc_signature else None,
            "cell_markers": named_args_arg(cell_markers) if cell_markers else None,
        }
    )
    return await _run(
        package="multideconv", function="compute.deconvolution", args=args, tool_name="run_deconvolution",
        ctx=ctx, timeout=28800, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def aggregate_cell_groups(
    deconvolution_csv: Annotated[str, Field(description="RAW deconvolution matrix CSV (samples as rows, method_signature_celltype columns) from run_deconvolution's result.csv -- not analyze_deconvolution's harmonized subgroups.")],
    cell_groups: Annotated[
        dict[str, list[str]],
        Field(
            description=(
                "Group name -> the cell types to sum into it, written as in multideconv's "
                "nomenclature, e.g. {\"Myeloid.cells\": [\"Macrophages.cells\", \"Macrophages.M0\", "
                "\"Macrophages.M1\", \"Macrophages.M2\", \"Monocytes\", \"Dendritic.cells\"], "
                "\"Lymphocytes\": [\"B.cells\", \"CD4.cells\", \"CD8.cells\", \"NK.cells\"]}. A group "
                "name can be a cell type of the nomenclature (e.g. Myeloid.cells): method-signature "
                "combinations that already estimate it are left as they are. Any other group name "
                "must not contain '_' nor the name of a nomenclature cell type (e.g. "
                "'Lymphocytes' is fine, 'All_T' or 'T.cells.helper.group' are not). The cell types "
                "of one group shouldn't overlap -- don't list a cell type together with its own "
                "subtypes (e.g. Macrophages.cells AND Macrophages.M1) if a signature reports both "
                "the total and the subtypes, or the total gets counted twice."
            )
        ),
    ],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    min_types: Annotated[int, Field(ge=1, description="Minimum number of the group's cell types a method-signature combination must estimate to get the group feature. Combinations with fewer are skipped (with 1, the group would just copy a single cell type).")] = 2,
) -> RunResult:
    """Sum cell types into coarser, user-defined groups within each method-signature combination
    (multideconv::aggregate_cell_groups).

    WHAT: for every method-signature combination, adds a new feature <method>_<signature>_<group>
    holding the sum of that group's cell types (e.g. Myeloid cells = macrophages + monocytes +
    dendritic cells). Cell types are only summed within the same method-signature combination
    (where fractions are on a common scale), and all original features are kept. The R log
    (log_tail) lists which cell types were summed, kept or skipped per combination.

    WHEN: an optional step right after run_deconvolution, before analyze_deconvolution, when the
    question is about a broader compartment (total myeloid, total lymphoid, total T cells ...)
    than the individual cell types the signatures report -- e.g. "how much lymphocyte infiltration"
    rather than "how many CD8 T cells". The result is a deconvolution matrix like any other: pass
    its result.csv to analyze_deconvolution, benchmark_deconvolution, prepare_deconvolution_folds
    or pipeML-server.

    Two things are needed for the groups to survive downstream (see warnings):
      - cells_extra: group names that aren't in multideconv's nomenclature (e.g. Lymphocytes) must
        be passed as cells_extra to analyze_deconvolution, benchmark_deconvolution and
        prepare_deconvolution_folds (or in train_ml_model's fold_construction_args_fixed),
        otherwise they are discarded as unknown cell types. Listing every group name is safe.
      - New cohorts: before replicate_subgroups, run this tool with the same cell_groups on the
        new cohort's deconvolution matrix, otherwise the group features are missing there and
        become NA.

    Returns the deconvolution matrix with the group features added as new columns (result.csv).
    """
    group_names = list(cell_groups)

    def reminders(summary: dict) -> list[str]:
        return [
            f"To keep the aggregated groups downstream, pass cells_extra={group_names} to "
            "analyze_deconvolution, benchmark_deconvolution and prepare_deconvolution_folds (names "
            "already in multideconv's nomenclature are simply ignored there), and aggregate the same "
            "cell_groups on any new cohort before replicate_subgroups. Check log_tail for the "
            "combinations that were skipped (fewer than min_types of a group's cell types)."
        ]

    args = args_from(
        **{
            "deconvolution": csv_arg(deconvolution_csv, row_names=1),
            "cell_groups": named_args_arg(cell_groups),
            "min_types": lit(min_types),
            "verbose": lit(True),
        }
    )
    return await _run(
        package="multideconv", function="aggregate_cell_groups", args=args, tool_name="aggregate_cell_groups",
        ctx=ctx, timeout=300, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=reminders,
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
                "(presumably) the same cell type into one merged subgroup. Grouping is "
                "complete-linkage: features end up in the same subgroup only if EVERY pair of "
                "them correlates at least this much (non-significant correlations count as 0), "
                "so the result doesn't depend on column order. Higher (e.g. 0.9) is "
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
    cv_thr: Annotated[float, Field(ge=0, description="Minimum coefficient of variation (standard deviation / mean) across samples; features below it are removed as near-constant (uninformative). Relative to each feature's own mean, so rare cell types aren't removed just for having small values.")] = 0.1,
    batch_csv: Annotated[
        str | None,
        Field(
            description=(
                "Optional CSV with a batch/cohort covariate column, matched to "
                "deconvolution_csv's samples by row name (first column). When given (with batch_column), correlations are computed as "
                "*partial* correlations controlling for batch (one indicator per batch, so 3 or "
                "more cohorts are handled correctly; the column is always treated as categorical, "
                "even if coded as numbers) -- use this for multi-cohort data "
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
    method/signature estimates of presumably-the-same cell type by pairwise correlation
    (complete-linkage hierarchical clustering: a subgroup forms only when every pair of its
    members correlates >= corr) and replaces each subgroup by the median of its members, named
    <CellType>_Subgroup.<i>; features without a partner are kept as they are. The result is a
    smaller set of cell-type features you can trust because multiple estimates converged on them,
    rather than one method's idiosyncratic estimate, and it doesn't depend on column order.

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
            "cv_thr": lit(cv_thr),
            "batch": csv_column_arg(batch_csv, batch_column, align_to=deconvolution_csv, as_character=True) if (batch_csv and batch_column) else None,
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

    Samples are matched between the two CSVs by row name: samples missing from groundtruth_csv
    are dropped with an R warning (see log_tail), and the call fails if no sample names are shared.

    Returns a correlation matrix (cell types as rows, method-signature combinations as columns)
    with an "average" row (mean over cell types) that orders the combinations; the heatmap labels
    show how many cell types each average is based on, and scatter plots report one correlation
    per cell type. If deconvolution_csv holds harmonized subgroups (analyze_deconvolution's
    output), columns are Subgroup.1, Subgroup.2, ... (the i-th subgroup of each cell type) and no
    average is computed, since such a column mixes unrelated features. Low correlations can mean
    either genuinely poor deconvolution performance or a silent cell-type name mismatch between
    the two inputs (see warnings).
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
    height: Annotated[float | None, Field(gt=0, description="Heatmap height in inches; chosen from the number of subgroups if unset.")] = None,
    width: Annotated[float | None, Field(gt=0, description="Heatmap width in inches; chosen from the number of pathways if unset.")] = None,
    pval: Annotated[float, Field(ge=0, le=1, description="P-value above which a correlation is left unstarred on the heatmap.")] = 0.05,
    corr_type: Annotated[Literal["pearson", "spearman"], Field(description="Correlation between subgroup abundance and pathway activity. Pearson (default) or Spearman (rank-based, more robust to skewed proportions and outliers).")] = "pearson",
) -> RunResult:
    """Correlate cell-type subgroups with pathway activity, per cell type
    (multideconv::compute.subgroup.pathways).

    WHAT: for each harmonized cell-type subgroup from analyze_deconvolution, correlates its
    abundance against each pathway's activity score across samples -- e.g. does CD8 T-cell
    abundance track with interferon-gamma pathway activity in this cohort. Saves one heatmap PDF
    per cell type. It helps to know what each subgroup of a specific cell type corresponds to
    (e.g. CD8 has 3 subgroups: CD8.cells_Subgroup.1, CD8.cells_Subgroup.2, ...); pathway
    association tells which subgroup corresponds to which biological behavior. Only cell types
    with more than one feature are compared.

    WHEN: after analyze_deconvolution, once you have (or compute separately, e.g. via
    CellTFusion-server) a pathway activity matrix you want to relate cell composition to.

    pathways_csv's row names must be the sample names used in the deconvolution: the call fails
    when fewer than 5 samples are shared.

    Returns a list with one element per cell type, each holding `correlations` and `pvalues`
    (subgroups as rows, pathways as columns) -- written as result_<cell type>_correlations.csv /
    _pvalues.csv (see csv_outputs) -- plus one heatmap PDF per cell type under run_dir/Results
    (see output_files).
    """
    args = args_from(
        **{
            "subgroups": rds_arg(subgroups_rds),
            "pathways": csv_arg(pathways_csv, row_names=1),
            "file_name": lit(file_name),
            "height": lit(height) if height is not None else None,
            "width": lit(width) if width is not None else None,
            "pval": lit(pval),
            "corr_type": lit(corr_type),
        }
    )
    return await _run(
        package="multideconv", function="compute.subgroup.pathways", args=args, tool_name="relate_subgroups_to_pathways",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def replicate_subgroups(
    deconv_res_rds: Annotated[str, Field(description="result.rds from a prior analyze_deconvolution call, defining the subgroup structure to replicate.")],
    deconvolution_test_csv: Annotated[str, Field(description="Deconvolution matrix CSV from a new/held-out cohort to project onto the trained subgroup definitions. If cell groups were aggregated (aggregate_cell_groups) before the training cohort's analyze_deconvolution, aggregate the same groups on this cohort first -- otherwise those features are missing here and come out as NA.")],
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
    analysis, or validating a model trained on the discovery cohort's harmonized features when 
    using pipeML-server.

    Returns a data.frame with the same features, in the same order, as the training "Deconvolution
    matrix": each subgroup is the median of its member features in the new cohort. Subgroups with
    no member in deconvolution_test_csv, and training features missing from it, are NA (with an R
    warning in log_tail); subgroups computed from only part of their members are also reported.
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


@mcp.tool(annotations=READ_ANNOTATIONS)
async def prepare_deconvolution_folds(
    deconvolution_csv: Annotated[
        str,
        Field(
            description=(
                "Raw (unharmonized) deconvolution matrix CSV (samples as rows, features as "
                "columns) from run_deconvolution's result.csv -- the same input shape "
                "analyze_deconvolution takes. For classification, this CSV must already include "
                "the outcome as extra columns: a literal 'target' column with each sample's class "
                "label (classification), or 'time' and 'event' columns (survival; event 1 = "
                "event occurred, 0 = censored) -- add them before calling if they aren't there "
                "yet. These outcome columns are kept out of the harmonization and added back to "
                "the returned train/test data, exactly as pipeML hands them to this function."
            )
        ),
    ],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    compute_full_dataset: Annotated[
        bool,
        Field(
            description=(
                "If True (default), harmonizes deconvolution_csv once across its whole cohort, "
                "attaching target/time/event -- equivalent to what pipeML's fold_construction_fun "
                "contract does on its own final full-data refit after cross-validation. If False, "
                "instead constructs per-fold training/test subsets using folds (required, see that "
                "parameter) -- the code path pipeML-server's "
                "train_ml_model(fold_construction_fun='prepare_multideconv_folds') drives per fold: "
                "each fold's training rows are harmonized and its held-out rows projected into "
                "that fold's own subgroups (via replicate_subgroups' logic)."
            )
        ),
    ] = True,
    folds: Annotated[
        dict[str, list[int]] | None,
        Field(
            description=(
                "Required if compute_full_dataset=False: fold name -> 1-indexed row numbers of "
                "deconvolution_csv to use as that fold's TRAINING set (the test set is every "
                "other row, implicitly). This mirrors what train_ml_model constructs internally "
                "via caret when driving this function as its fold_construction_fun -- building it "
                "by hand here is mainly for inspecting or debugging one specific fold's harmonized "
                "output, e.g. {\"Fold1\": [1, 2, 3, 4, 5], \"Fold2\": [6, 7, 8, 9, 10]}."
            )
        ),
    ] = None,
    ncores: Annotated[int | None, Field(ge=1, description="CPU cores for parallel fold processing when compute_full_dataset=False; default detectCores()-2 (at least 1).")] = None,
    cells_extra: CellTypeList = None,
    corr: Annotated[float, Field(ge=0, le=1, description="Same as analyze_deconvolution's corr -- minimum correlation for merging method/signature estimates into a subgroup.")] = 0.7,
    corr_type: Annotated[Literal["spearman", "pearson"], Field(description="Same as analyze_deconvolution's corr_type.")] = "spearman",
    zero_thr: Annotated[float, Field(ge=0, le=1, description="Same as analyze_deconvolution's zero_thr.")] = 0.9,
    cv_thr: Annotated[float, Field(ge=0, description="Same as analyze_deconvolution's cv_thr.")] = 0.1,
    batch_csv: Annotated[str | None, Field(description="Optional CSV with a batch/cohort covariate column, matched to deconvolution_csv's samples by row name -- same partial-correlation behavior as analyze_deconvolution's batch_csv.")] = None,
    batch_column: Annotated[str | None, Field(description="Column in batch_csv identifying each sample's batch/cohort.")] = None,
) -> RunResult:
    """Inspect what pipeML's custom fold-construction integration would train on
    (multideconv::prepare_multideconv_folds).

    WHAT: this is the same harmonization multideconv::compute.deconvolution.analysis() performs
    (see analyze_deconvolution), but packaged to match pipeML's fold_construction_fun contract --
    it's what pipeML-server's train_ml_model calls internally, per fold, when given
    fold_construction_fun="prepare_multideconv_folds". Calling it here standalone lets you inspect
    the harmonized subgroup features before committing to a full, potentially slow train_ml_model
    run, or reproduce/debug one specific fold's projected test features.

    WHEN: before train_ml_model with fold_construction_fun="prepare_multideconv_folds", to sanity
    check what the deconvolution harmonization would actually produce for this cohort -- or
    afterwards, to inspect a specific fold's output directly.

    Returns, with compute_full_dataset=True, an unnamed list of (1) the harmonized feature matrix
    with target (or time/event) attached, (2) the full analyze_deconvolution-style result, and
    (3) the bestune marker. With compute_full_dataset=False, a list named by fold, each with
    train_data (harmonized training rows + the outcome columns), test_data (held-out rows
    projected into that fold's subgroups, plus time/event for survival), obs_test (class labels,
    or a time/event data frame for survival), rowIndex, and fold_name -- each fold is also saved to
    run_dir/Results/fold_<name>.rds (stale fold_*.rds from earlier runs in the same run_dir are
    removed first).
    """
    if not compute_full_dataset and not folds:
        raise ToolError("folds is required when compute_full_dataset=False (fold name -> 1-indexed training rows).")
    args = args_from(
        **{
            "data": csv_arg(deconvolution_csv, row_names=1),
            "folds": named_args_arg(folds) if folds else None,
            "bestune": lit(True) if compute_full_dataset else None,
            "ncores": lit(ncores) if ncores is not None else None,
            "cells_extra": lit(cells_extra) if cells_extra else None,
            "corr": lit(corr),
            "corr_type": lit(corr_type),
            "zero_thr": lit(zero_thr),
            "cv_thr": lit(cv_thr),
            "batch": csv_column_arg(batch_csv, batch_column, align_to=deconvolution_csv, as_character=True) if (batch_csv and batch_column) else None,
        }
    )
    return await _run(
        package="multideconv", function="prepare_multideconv_folds", args=args, tool_name="prepare_deconvolution_folds",
        ctx=ctx, timeout=1800, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
