"""RaCInG-server: MCP tools for cell-cell interaction network reconstruction

Wraps three entry points from the RaCInG R package: building the ligand-receptor input matrices from raw counts (prepare_racing_inputs), the fast deterministic kernel-based communication workflow (run_racing_kernel), and the slower Monte Carlo simulation-based workflow (run_racing_montecarlo) for patient-specific cell-cell interaction networks.

Each tool shells out to system R via the shared common/r_bridge.py dispatcher
(common/r/run_tool.R). RaCInG must already be installed in the system R
library `Rscript` (on PATH) resolves to.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import Field

from mcp_tme_servers.common.paths import server_home
from mcp_tme_servers.common.r_bridge import args_from, csv_arg, lit, r_function_arg, rds_arg
from mcp_tme_servers.common.session import SessionConfig
from mcp_tme_servers.common.tools import (
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
        'Tools for reconstructing patient-specific cell-cell communication networks from bulk RNAseq and extracting network-motif features from them. Pipeline order: prepare_racing_inputs once, then reuse its result via input_data_rds in run_racing_kernel and/or run_racing_montecarlo (skips recomputing preprocessing). Prefer the kernel method as the default -- it is fast and deterministic; use the Monte Carlo method specifically when you want simulation-based variance/robustness across randomly realized network instances rather than a single point estimate, at a much higher compute cost. Both extract the same feature families -- direct signaling (D), hub-mediated relay (W, wedge), reciprocal three-way coordination (TT, trust triangle), closed three-way feedback loops (CT, cycle triangle, Monte Carlo only), and overall network integration (GSCC) -- from a directed graph built on cell-type abundance and ligand-receptor expression. '
        'IMPORTANT -- citation: if the user asks what to cite for an analysis that used this server, always include the mcp-tme-servers project itself -- XX, et al. (XX). mcp-tme-servers -- XX. XX. https://doi.org/XX (a placeholder until a real reference exists; check the project README for the current one) -- in addition to RaCInG itself: van Santvoort M, Lapuente-Santana Ó, Zopoglou M, Zackl C, Finotello F, van der Hoorn P & Eduati F (2025). Mathematically mapping the network of cells in the tumor microenvironment. Cell Reports Methods, 5(2), 100985.'
    ),
)

SESSION_CONFIG = SessionConfig(
    sessions_root=server_home("racing") / "runs" / "sessions",
    stage_by_tool={'prepare_racing_inputs': 'interaction_input_prep', 'run_racing_kernel': 'interaction_kernel', 'run_racing_montecarlo': 'interaction_montecarlo'},
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
    deconv_csv: Annotated[str | None, Field(description="Optional precomputed patient x cell-type abundance CSV (e.g. from multideconv-server's run_deconvolution); computed via multideconv internally if omitted.")] = None,
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
    deconv_csv: Annotated[str | None, Field(description="Optional precomputed patient x cell-type abundance CSV; ignored if input_data_rds is given.")] = None,
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
    deconv_csv: Annotated[str | None, Field(description="Optional precomputed patient x cell-type abundance CSV; ignored if input_data_rds is given.")] = None,
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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
