"""RaCInG-server: MCP tools for cell-cell interaction network reconstruction

Wraps four entry points from the RaCInG R package: building the ligand-receptor input matrices from raw counts (prepare_racing_inputs), the fast deterministic kernel-based communication workflow (run_racing_kernel), the slower Monte Carlo simulation-based workflow (run_racing_montecarlo) for patient-specific cell-cell interaction networks, and the per-ligand-receptor-pair decomposition of one communication feature (explain_lr_contributions).

Each tool shells out to system R via the shared common/r_bridge.py dispatcher
(common/r/run_tool.R). RaCInG must already be installed in the system R
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
from mcp_tme_servers.common.r_bridge import (
    args_from,
    csv_arg,
    lit,
    r_function_arg,
    rds_arg,
    rds_element_arg,
    rds_element_rownames_arg,
)
from mcp_tme_servers.common.session import SessionConfig
from mcp_tme_servers.common.tools import (
    EXPRESSION_SCALE_INSTRUCTIONS,
    NETWORK_ANNOTATIONS,
    OutputDir,
    RunResult,
    SessionIdParam,
    make_run,
    register_session_tools,
)

logger = logging.getLogger(__name__)

mcp = MCPServer(
    "RaCInG-server",
    instructions=(
        'Tools for reconstructing patient-specific cell-cell communication networks from bulk RNAseq and extracting network-motif features from them. Pipeline order: prepare_racing_inputs once, then reuse its result via input_data_rds in run_racing_kernel and/or run_racing_montecarlo (skips recomputing preprocessing). Prefer the kernel method as the default -- it is fast and deterministic; use the Monte Carlo method specifically when you want simulation-based variance/robustness across randomly realized network instances rather than a single point estimate, at a much higher compute cost. To see which ligand-receptor pairs drive a given direct/wedge/triangle feature, call explain_lr_contributions on the same input_data_rds. Both extract the same feature families -- direct signaling (D), hub-mediated relay (W, wedge), reciprocal three-way coordination (TT, trust triangle), closed three-way feedback loops (CT, cycle triangle, Monte Carlo only), and overall network integration (GSCC) -- from a directed graph built on cell-type abundance and ligand-receptor expression. '
        "IMPORTANT -- check for existing inputs before recomputing: prepare_racing_inputs/run_racing_kernel/run_racing_montecarlo can each internally recompute cell-type abundance (deconvolution) rather than needing it precomputed. Before calling any of them, proactively check whether a deconvolution/cell-type abundance result already exists for this same cohort -- e.g. from a prior multideconv-server run_deconvolution/analyze_deconvolution call, or an earlier RaCInG-server session on this same input data. Look for it via that server's get_session_status/prior sessions, or by asking the user whether they already have one. If you locate a candidate, do not pass it in silently -- ask the user to confirm it's the right one for this cohort before using it via deconv_csv (or input_data_rds, which also carries it). If the user doesn't confirm it (or none is found), let deconv_method compute it internally from scratch rather than guessing. "
        + EXPRESSION_SCALE_INSTRUCTIONS
        + 'On this server: prepare_racing_inputs takes raw counts (already_normalized=False, the default) or linear TPM (already_normalized=True); its internal deconvolution and its expression threshold both work on linear TPM, so never give it log-transformed values. '
        + 'IMPORTANT -- citation: if the user asks what to cite for an analysis that used this server, always include the mcp-tme-servers project itself -- XX, et al. (XX). mcp-tme-servers -- XX. XX. https://doi.org/XX (a placeholder until a real reference exists; check the project README for the current one) -- in addition to RaCInG itself: van Santvoort M, Lapuente-Santana Ó, Zopoglou M, Zackl C, Finotello F, van der Hoorn P & Eduati F (2025). Mathematically mapping the network of cells in the tumor microenvironment. Cell Reports Methods, 5(2), 100985. '
        'IMPORTANT -- output location: once any tool call on this server finishes, always tell the user the run_dir path from that result (where result.rds/CSVs/plots/run.log were saved) -- do not just use it silently to chain into the next call without also surfacing it to the user.'
    ),
)

SESSION_CONFIG = SessionConfig(
    sessions_root=server_home("racing") / "runs" / "sessions",
    stage_by_tool={'prepare_racing_inputs': 'interaction_input_prep', 'run_racing_kernel': 'interaction_kernel', 'run_racing_montecarlo': 'interaction_montecarlo', 'explain_lr_contributions': 'lr_contributions'},
    export_requires=[frozenset({'interaction_input_prep'}), frozenset({'interaction_kernel', 'interaction_montecarlo'})],
    stage_explanations={'interaction_input_prep': "RaCInG input matrices haven't been built yet (prepare_racing_inputs) -- there is nothing to export.", 'interaction_kernel': 'no cell-cell interaction features have been computed yet (run_racing_kernel or run_racing_montecarlo) -- there is nothing to export.', 'interaction_montecarlo': 'no cell-cell interaction features have been computed yet (run_racing_kernel or run_racing_montecarlo) -- there is nothing to export.'},
)

_run = make_run(runs_root=server_home("racing") / "runs")

register_session_tools(mcp, SESSION_CONFIG, domain="cell-cell interaction network reconstruction")


# --------------------------------------------------------------------------
# RaCInG tools
# --------------------------------------------------------------------------

CommunicationType = Literal["D", "W", "TT", "GSCC"]
CommunicationTypeMC = Literal["D", "W", "TT", "CT", "GSCC"]

_COMM_TYPE_DOC = (
    "Which network-motif feature families to extract, computed on a directed graph where an "
    "edge/signal runs from a sending to a receiving cell type, weighted by ligand-receptor "
    "expression and cell-type abundance:\n"
    "- 'D' (direct): raw signaling strength from one specific cell type straight to another. "
    "High D between a pair = strong direct paracrine/autocrine signaling between them.\n"
    "- 'W' (wedge): a two-hop relay -- cell type A signals to hub B, and B also signals to/from "
    "C. High W around a cell type = it's acting as a communication relay between other "
    "populations, not just a direct sender/receiver.\n"
    "- 'TT' (trust triangle): i signals to both j and k, and j and k also signal to each other -- "
    "a closed triangle of reinforcing/redundant paths. High TT = strong reciprocal coordination "
    "among a three-way group, not just pairwise signaling.\n"
    "- 'CT' (cycle triangle, Monte Carlo only): i->j->k->i, a full directed feedback loop among "
    "three cell types. High CT = evidence of a closed signaling loop, not just one-way influence "
    "(the kernel method bundles this together with TT instead of separating it).\n"
    "- 'GSCC' (giant strongly connected component): the largest set of cell types where every "
    "type can reach every other via directed paths -- a global integration measure. Large GSCC = "
    "a highly interconnected 'everyone talks to everyone' TME; small/fragmented GSCC = "
    "compartmentalized signaling.\n"
    "Requesting several types in one call reuses the same computed kernel/simulated graphs "
    "instead of redoing the expensive step per type, so ask for everything you need in one call."
)

_KERNEL_GLOSSARY = {
    "kernel": (
        "The raw sender x receiver x patient communication kernel -- abundance-weighted "
        "signaling strength between every cell-type pair, per patient."
    ),
    "kernel_norm": (
        "Only present if norm=True: a null-model kernel built by giving every structurally "
        "possible ligand-receptor pair equal uniform strength instead of its real measured "
        "expression. Dividing kernel by kernel_norm cancels out cell-type abundance and topology, "
        "leaving a composition-independent specificity/enrichment score -- 'is this pair "
        "communicating more than expected from network structure and cell abundance alone', not "
        "just raw absolute strength (which is confounded by how common the cell types are)."
    ),
    "features": (
        "The derived per-patient feature matrix (or, with multiple communication_type entries, a "
        "named list of feature matrices) -- one column per cell-type combination and motif type. "
        "This is what you'd feed into pipeML-server for downstream modeling."
    ),
}

_MC_GLOSSARY = {
    "input": "The RaCInG input matrices actually used for the simulation (same structure as prepare_racing_inputs' output).",
    "output": (
        "Mean/SD of motif counts across the Ngraphs simulated realizations per patient (or, with "
        "multiple communication_type entries, a named list of such matrices) -- the SD across "
        "realizations is what the Monte Carlo method gives you that the deterministic kernel "
        "method doesn't: a sense of how variable/robust a given feature is under random graph "
        "sampling, not just its point estimate."
    ),
}

_DECONV_CSV_DOC = (
    "Optional precomputed patient x cell-type abundance CSV. Computed internally if omitted -- "
    "via multideconv's deconvolution plus subgroup analysis, with each cell type's expression "
    "profile fitted per method/signature block and then merged, which is the most rigorous "
    "path. A supplied matrix is instead treated as ONE composition (fractions summing to ~1 per "
    "patient) when fitting the per-cell-type expression profiles (non-negative least squares), "
    "so pass a single method's output (e.g. one method/signature block of multideconv-server's "
    "run_deconvolution), not the concatenated multi-method matrix or analyze_deconvolution's "
    "harmonized subgroups, which aren't valid compositions."
)

_INPUT_GLOSSARY = {
    "Lmatrix": "Cell type x ligand matrix: estimated ligand expression per cell type.",
    "Rmatrix": "Cell type x receptor matrix: estimated receptor expression per cell type.",
    "Cmatrix": "Patient x cell type matrix: normalized cell-type abundances (from deconvolution) used to weight the simulated/kernel graphs.",
    "LRmatrix": "Ligand x receptor x patient 3-D tensor: per-patient interaction-strength weight for every ligand-receptor pair.",
    "celltypes": "The cell type vocabulary used to build the graphs.",
    "ligands": "The ligand vocabulary used to build the graphs.",
    "receptors": "The receptor vocabulary used to build the graphs.",
    "Sign_matrix": "Optional activating/inhibitory sign per ligand-receptor pair, if signed=True and a sign matrix was found.",
    "CC_table": "The filtered cell-cell ligand-receptor-celltype triples that passed expr_threshold -- the effective interaction universe the kernel/simulation is built from.",
}


@mcp.tool(annotations=NETWORK_ANNOTATIONS)
async def prepare_racing_inputs(
    counts_csv: Annotated[str, Field(description="CSV of RNAseq counts: genes as rows (first column = gene symbol), samples as columns.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    deconv_csv: Annotated[str | None, Field(description=_DECONV_CSV_DOC)] = None,
    cc_network_csv: Annotated[str | None, Field(description="Optional ligand-receptor prior network CSV (source/target gene symbol columns); fetched from OmniPath via liana (and cached indefinitely) the first time this is called if omitted.")] = None,
    fun_lr: Annotated[Literal["min", "prod"], Field(description="How to combine a ligand's and a receptor's expression into one interaction strength. 'min' (default): the weaker-expressed partner limits the interaction -- a biologically conservative choice, since signaling needs both sides present. 'prod': the product of both -- rewards pairs where both are highly expressed, more sensitive to either partner's magnitude.")] = "min",
    source_column: Annotated[str, Field(description="Ligand identifier column name in cc_network_csv.")] = "source_genesymbol",
    target_column: Annotated[str, Field(description="Receptor identifier column name in cc_network_csv.")] = "target_genesymbol",
    deconv_method: Annotated[list[str], Field(description="Deconvolution method(s) used internally when deconv_csv is not supplied.")] = ["Quantiseq"],
    file_name: Annotated[str | None, Field(description="File stem for the generated L/R/C/LR CSV files.")] = None,
    signed: Annotated[bool, Field(description="Also try to load an activating/inhibitory sign matrix from the output folder, if available.")] = False,
    already_normalized: Annotated[
        bool,
        Field(
            description=(
                "Set True only if counts_csv already holds linear-scale TPM (not raw counts) -- "
                "e.g. a public cohort distributed pre-normalized. When True, the internal "
                "gene-length/library-size normalization is skipped. Do not pass log-transformed "
                "values with this set True; convert back to linear scale first (e.g. 2^x - 1) -- "
                "expr_threshold is compared against this scale, so a log-scale value compared "
                "against it would silently apply the wrong cutoff."
            )
        ),
    ] = False,
    expr_threshold: Annotated[float, Field(ge=0, description="Minimum linear-scale TPM a ligand or receptor must reach in its sending/receiving cell type to be kept as a real interaction -- filters out pairs that are only nominally expressed.")] = 10,
) -> RunResult:
    """Build RaCInG's ligand-receptor input matrices from raw counts
    (RaCInG::prepare_input_files).

    WHAT: converts bulk counts (plus cell-type abundance, from deconvolution) into the
    cell-type x ligand, cell-type x receptor, patient x cell-type, and ligand x receptor x
    patient matrices that the kernel and Monte Carlo methods both need -- essentially "who has how
    much of which signaling machinery, and how much of each cell type does each patient have."

    WHEN: run this once per cohort and reuse its run_dir/result.rds as `input_data_rds` in
    run_racing_kernel / run_racing_montecarlo, instead of passing counts_csv to those directly --
    this avoids recomputing preprocessing (including deconvolution) on every call.

    Returns a named list with the processed input matrices (see field_glossary). If
    cc_network_csv is omitted, the first call fetches a curated ligand-receptor network from
    OmniPath over the network and caches it indefinitely; later calls reuse the cache without a
    network round-trip.
    """
    args = args_from(
        **{
            "counts": csv_arg(counts_csv, row_names=1, as_matrix=True),
            "deconv": csv_arg(deconv_csv, row_names=1) if deconv_csv else None,
            "cc_network": csv_arg(cc_network_csv, row_names=None) if cc_network_csv else None,
            "fun_LR": r_function_arg(fun_lr),
            "source": lit(source_column),
            "target": lit(target_column),
            "deconv_method": lit(deconv_method),
            "file_name": lit(file_name) if file_name else None,
            "signed": lit(signed),
            "already_normalized": lit(already_normalized),
            "expr_threshold": lit(expr_threshold),
        }
    )
    return await _run(
        package="RaCInG", function="prepare_input_files", args=args, tool_name="prepare_racing_inputs",
        ctx=ctx, timeout=900, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary=_INPUT_GLOSSARY,
    )


@mcp.tool(annotations=NETWORK_ANNOTATIONS)
async def run_racing_kernel(
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    counts_csv: Annotated[str | None, Field(description="CSV of RNAseq counts; required unless input_data_rds is given.")] = None,
    input_data_rds: Annotated[str | None, Field(description="result.rds from a prior prepare_racing_inputs call -- prefer this over counts_csv to skip recomputing preprocessing.")] = None,
    deconv_csv: Annotated[str | None, Field(description=_DECONV_CSV_DOC + " Ignored if input_data_rds is given.")] = None,
    cc_network_csv: Annotated[str | None, Field(description="Optional ligand-receptor prior network CSV; ignored if input_data_rds is given.")] = None,
    fun_lr: Annotated[Literal["min", "prod"], Field(description="How to combine ligand/receptor expression into interaction strength; only used when computing inputs from counts_csv.")] = "min",
    source_column: str = "source_genesymbol",
    target_column: str = "target_genesymbol",
    deconv_method: list[str] = ["Quantiseq"],
    file_name: Annotated[str | None, Field(description="File stem for intermediate input files.")] = None,
    n_patients: Annotated[str | int, Field(description="Number of patients to process, or 'all'.")] = "all",
    communication_type: Annotated[list[CommunicationType], Field(description=_COMM_TYPE_DOC)] = ["W"],
    norm: Annotated[bool, Field(description="Also compute a normalized baseline kernel and express features as an enrichment ratio over it, isolating communication specificity from raw abundance/topology (see field_glossary's kernel_norm). Default False returns the raw, abundance-weighted magnitude.")] = False,
    pt_idx: Annotated[int | None, Field(description="Process only this single patient index instead of all patients.")] = None,
    remove_direction: Annotated[bool, Field(description="Merge directionally equivalent features (A->B and B->A treated as one) instead of keeping sender/receiver direction separate.")] = True,
    zero_threshold: Annotated[float, Field(ge=0, le=1, description="Drop a feature once this fraction of patients have it at exactly zero. Default 1 only drops features that are zero for every patient; lower it (e.g. 0.9) to also drop merely zero-inflated, mostly-uninformative features.")] = 1,
    expr_threshold: Annotated[float, Field(ge=0, description="Minimum linear-scale TPM for a ligand/receptor to count; only used when computing inputs from counts_csv.")] = 10,
    signed: bool = False,
) -> RunResult:
    """Run the deterministic kernel-based cell-cell communication workflow
    (RaCInG::compute_racing_kernel).

    WHAT: computes a sender x receiver x patient communication kernel via a closed-form weighted
    matrix product (ligand weights x LR-interaction matrix x receptor weights), then derives
    network-motif features from it -- fast, deterministic, one point estimate per feature per
    patient. This is the recommended default method.

    WHEN: use this as your first choice for cell-cell interaction features -- it's much faster
    than run_racing_montecarlo and deterministic (no simulation noise to average over). Reach for
    run_racing_montecarlo instead specifically when you want simulation-based variance/robustness
    across randomly realized network instances, not just a point value.

    Prefer input_data_rds (from prepare_racing_inputs) over counts_csv, especially when running
    multiple communication_type extractions -- the kernel itself is computed once regardless of
    how many types are requested, so ask for all the types you need in one call. Unlike
    prepare_racing_inputs, this has no already_normalized flag: if counts_csv holds
    already-normalized TPM rather than raw counts, call prepare_racing_inputs first (with
    already_normalized=True) and pass its result here via input_data_rds instead of counts_csv.

    Returns a list with the kernel array(s) and derived feature matrix/matrices (see
    field_glossary).
    """
    args = args_from(
        **{
            "counts": csv_arg(counts_csv, row_names=1, as_matrix=True) if (counts_csv and not input_data_rds) else None,
            "input_data": rds_arg(input_data_rds) if input_data_rds else None,
            "deconv": csv_arg(deconv_csv, row_names=1) if (deconv_csv and not input_data_rds) else None,
            "cc_network": csv_arg(cc_network_csv, row_names=None) if (cc_network_csv and not input_data_rds) else None,
            # Upstream defaults output_folder to a hardcoded ~/Documents/racing/vignettes/: keep the
            # intermediate input files inside this call's run_dir instead
            "output_folder": lit("Results/"),
            "fun_LR": r_function_arg(fun_lr),
            "source": lit(source_column),
            "target": lit(target_column),
            "signed": lit(signed),
            "deconv_method": lit(deconv_method),
            "file_name": lit(file_name) if file_name else None,
            "nPatients": lit(n_patients),
            "communication_type": lit(communication_type),
            "norm": lit(norm),
            "pt_idx": lit(pt_idx) if pt_idx is not None else None,
            "remove_direction": lit(remove_direction),
            "zero_threshold": lit(zero_threshold),
            "expr_threshold": lit(expr_threshold),
        }
    )
    return await _run(
        package="RaCInG", function="compute_racing_kernel", args=args, tool_name="run_racing_kernel",
        ctx=ctx, timeout=28800, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary=_KERNEL_GLOSSARY,
    )


@mcp.tool(annotations=NETWORK_ANNOTATIONS)
async def run_racing_montecarlo(
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    counts_csv: Annotated[str | None, Field(description="CSV of RNAseq counts; required unless input_data_rds is given.")] = None,
    input_data_rds: Annotated[str | None, Field(description="result.rds from a prior prepare_racing_inputs call -- prefer this over counts_csv to skip recomputing preprocessing.")] = None,
    deconv_csv: Annotated[str | None, Field(description=_DECONV_CSV_DOC + " Ignored if input_data_rds is given.")] = None,
    cc_network_csv: Annotated[str | None, Field(description="Optional ligand-receptor prior network CSV; ignored if input_data_rds is given.")] = None,
    fun_lr: Literal["min", "prod"] = "min",
    source_column: str = "source_genesymbol",
    target_column: str = "target_genesymbol",
    deconv_method: list[str] = ["Quantiseq"],
    pt_idx: Annotated[int | None, Field(description="Simulate only this single patient index.")] = None,
    file_name: Annotated[str | None, Field(description="File stem for intermediate files.")] = None,
    n_patients: Annotated[str | int, Field(description="Number of patients to process, or 'all'.")] = "all",
    communication_type: Annotated[list[CommunicationTypeMC], Field(description=_COMM_TYPE_DOC)] = ["W"],
    n_cells: Annotated[int, Field(ge=100, description="Cells per simulated graph -- larger gives a more granular/stable per-realization estimate at higher compute cost.")] = 10000,
    n_graphs: Annotated[int, Field(ge=1, description="Monte Carlo iterations (independent random graph realizations) per patient -- the main runtime driver. Start with 10-20 for a quick exploratory pass; raise toward the 100 default once you're committing to a result, since more realizations mean a less noisy mean/SD estimate.")] = 100,
    n_degree: Annotated[int, Field(ge=1, description="Target average node degree in each simulated graph -- controls how densely connected the simulated cell-cell network is.")] = 20,
    remove_direction: bool = True,
    norm: Annotated[bool, Field(description="Also run a second, uniformized baseline simulation and express features as an enrichment ratio over it (isolates specificity from raw abundance/topology, at the cost of a second, noisier simulation pass -- budget a larger n_graphs if enabling this). Default False returns the raw, abundance-weighted magnitude: simpler, cheaper, and less noisy at a given n_graphs.")] = False,
    ncores: Annotated[int, Field(ge=1, description="Cores to parallelize across patients (patients are simulated independently, so this scales close to linearly).")] = 1,
    signed: bool = False,
) -> RunResult:
    """Run the stochastic Monte Carlo cell-cell communication workflow
    (RaCInG::compute_racing_montecarlo).

    WHAT: for each patient, simulates n_graphs independent random directed cell-cell graphs
    (n_cells cells, target average degree n_degree) that respect the patient's cell-type
    abundances and the cohort's ligand-receptor interaction probabilities, then counts network
    motifs across those realizations. Unlike the kernel method's single closed-form estimate,
    this gives you a mean *and* an SD per feature across simulated realizations -- a direct
    sense of how variable/robust a given communication estimate is under random sampling, at the
    cost of much more compute.

    WHEN: reach for this specifically when you want that simulation-based variance/robustness
    information, not just a point value -- for most exploratory work, run_racing_kernel is faster
    and sufficient. This is the slowest tool on this server: runtime scales with
    n_patients * n_graphs * n_cells. Start with a small n_graphs (e.g. 10-20) for a quick pass,
    raise ncores to parallelize across patients, and prefer input_data_rds from
    prepare_racing_inputs to avoid recomputing preprocessing. Requesting multiple
    communication_type entries in one call reuses the same simulated graphs instead of
    resimulating per type. Unlike prepare_racing_inputs, this has no already_normalized flag: if
    counts_csv holds already-normalized TPM rather than raw counts, call prepare_racing_inputs
    first (with already_normalized=True) and pass its result here via input_data_rds instead of
    counts_csv.

    Returns a list with the inputs used and the mean/SD feature matrix/matrices (see
    field_glossary).
    """
    args = args_from(
        **{
            "counts": csv_arg(counts_csv, row_names=1, as_matrix=True) if (counts_csv and not input_data_rds) else None,
            "input_data": rds_arg(input_data_rds) if input_data_rds else None,
            "deconv": csv_arg(deconv_csv, row_names=1) if (deconv_csv and not input_data_rds) else None,
            "cc_network": csv_arg(cc_network_csv, row_names=None) if (cc_network_csv and not input_data_rds) else None,
            "output_folder": lit("Results/"),  # see run_racing_kernel
            "fun_LR": r_function_arg(fun_lr),
            "source": lit(source_column),
            "target": lit(target_column),
            "signed": lit(signed),
            "deconv_method": lit(deconv_method),
            "pt_idx": lit(pt_idx) if pt_idx is not None else None,
            "file_name": lit(file_name) if file_name else None,
            "nPatients": lit(n_patients),
            "communication_type": lit(communication_type),
            "Ncells": lit(n_cells),
            "Ngraphs": lit(n_graphs),
            "Ndegree": lit(n_degree),
            "remove_direction": lit(remove_direction),
            "norm": lit(norm),
            "ncores": lit(ncores),
        }
    )
    return await _run(
        package="RaCInG", function="compute_racing_montecarlo", args=args, tool_name="run_racing_montecarlo",
        ctx=ctx, timeout=28800, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary=_MC_GLOSSARY,
    )


_LR_FUNCTIONS = {
    "direct": "computeLRContributions",
    "wedge": "computeWedgeLRContributions",
    "triangle": "computeTriangleLRContributions",
}


def _lr_diagnostics(summary: dict) -> list[str]:
    if summary.get("class") != "NULL":
        return []
    return [
        "No ligand-receptor pair connects these cell types in any patient -- check the names "
        "against the input's celltypes, or pick a feature run_racing_kernel actually returned."
    ]


@mcp.tool(annotations=NETWORK_ANNOTATIONS)
async def explain_lr_contributions(
    input_data_rds: Annotated[str, Field(description="result.rds from a prior prepare_racing_inputs call (the same input the kernel/Monte Carlo tools use).")],
    motif: Annotated[
        Literal["direct", "wedge", "triangle"],
        Field(
            description=(
                "Which communication feature to decompose: 'direct' (D, sender -> receiver), 'wedge' "
                "(W, two leaves relayed through a hub) or 'triangle' (TT, three mutually communicating "
                "cell types)."
            )
        ),
    ],
    cell_types: Annotated[
        list[str],
        Field(
            min_length=2,
            max_length=3,
            description=(
                "The cell types of the feature, as named in the input's celltypes: [sender, receiver] "
                "for 'direct'; [leaf_i, hub_j, leaf_k] for 'wedge' (the hub in the middle); [i, j, k] "
                "for 'triangle'. The column names of run_racing_kernel's features name them."
            )
        ),
    ],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    bundle: Annotated[bool, Field(description="'direct' only: pool both directions (sender -> receiver and back) into one edge, matching run_racing_kernel's default remove_direction=True features. False keeps only sender -> receiver.")] = True,
    top_n: Annotated[int | None, Field(ge=1, description="Keep only the top N ligand-receptor pairs per patient (by contribution); all nonzero pairs if unset.")] = None,
) -> RunResult:
    """Break a cell-cell communication feature down into the ligand-receptor pairs driving it
    (RaCInG::computeLRContributions / computeWedgeLRContributions / computeTriangleLRContributions).

    WHAT: the kernel sums every ligand-receptor pair into one number per cell-type pair, so a
    feature like "CD8 T cells -> macrophages" loses which signals make it up. Because that sum is
    linear, each pair's term can be computed exactly: for 'direct', the contributions add back up
    to the kernel's raw score. Wedges and triangles multiply edges rather than sum them, so there
    each edge is decomposed separately and PctOfTotal is relative to that edge.

    WHEN: after run_racing_kernel (or pipeML-server/SHAP) singles out an interesting feature, to
    ask which ligand-receptor pairs explain it -- e.g. whether a strong CD8 -> tumor edge is
    driven by checkpoint ligands or by cytokines -- and whether that differs between patients.

    Returns a data frame with one row per patient, ligand-receptor pair and edge: Patient, Edge
    (rows sharing it share one PctOfTotal denominator), Sender, Receiver, Ligand, Receptor,
    Contribution (in the units of the raw kernel score) and PctOfTotal (that pair's share of the
    edge's score for that patient); wedges add Path. Pairs contributing exactly zero are omitted,
    and a NULL result means no pair connects those cell types in any patient.
    """
    expected = 2 if motif == "direct" else 3
    if len(cell_types) != expected:
        raise ToolError(f"motif '{motif}' needs exactly {expected} cell_types, got {len(cell_types)}.")

    if motif == "direct":
        cells = {"sender": lit(cell_types[0]), "receiver": lit(cell_types[1]), "bundle": lit(bundle)}
    else:
        cells = {"i": lit(cell_types[0]), "j": lit(cell_types[1]), "k": lit(cell_types[2])}

    args = args_from(
        **{
            "Lmatrix": rds_element_arg(input_data_rds, "Lmatrix"),
            "Rmatrix": rds_element_arg(input_data_rds, "Rmatrix"),
            "Cmatrix": rds_element_arg(input_data_rds, "Cmatrix"),
            "LRmatrix": rds_element_arg(input_data_rds, "LRmatrix"),
            "celltypes": rds_element_arg(input_data_rds, "celltypes"),
            **cells,
            "patient_names": rds_element_rownames_arg(input_data_rds, "Cmatrix"),
            "top_n": lit(top_n) if top_n is not None else None,
        }
    )
    return await _run(
        package="RaCInG", function=_LR_FUNCTIONS[motif], args=args, tool_name="explain_lr_contributions",
        ctx=ctx, timeout=1800, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_lr_diagnostics,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
