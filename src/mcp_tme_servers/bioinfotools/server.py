"""Bioinfo-tools-server: MCP tools for general-purpose bulk RNAseq analysis

Wraps every exported function in Bioinfo-tools' functions.r: TF-activity inference
(from gene expression (logTPM) or from a DEG signature with permutation p-values),
decoupleR-scored pathway activity, edgeR/limma-voom differential expression with its
volcano plot, ORA/GSEA enrichment (GO/KEGG/REACTOME/PROGENy/Hallmark), Kaplan-Meier
survival analysis, TPM/z-score normalization, ComBat-seq batch correction, PCA, and
trait-association testing against both a continuous variable (Pearson, Spearman,
Fisher's exact) and a categorical one (t-test, Wilcoxon, ANOVA, Kruskal-Wallis).

IMPORTANT -- "footprint" methods: four of this server's activity-inference resources
are footprint-based (they infer a regulator's activity from its targets'/responsive
genes' expression pattern via a decoupleR statistic -- default `statistic='consensus'`,
an ensemble of MLM/ULM/normalized WSUM; VIPER is just one of several selectable
statistics, not the default -- rather than from the regulator's own expression):
CollecTRI, Dorothea, and ARACNE for transcription-factor activity (compute_tf_activity,
compute_tf_activity_deg -- via `tf_collection`), and PROGENy for pathway activity
(compute_pathway_activity's default `pathway_source`). CollecTRI, Dorothea, and PROGENy
draw on curated, literature-derived prior-knowledge networks (PKN); ARACNE
(`tf_collection="ARACNE"`) differs only in where its network comes from -- a
data-driven gene regulatory network (GRN) inferred per-cohort from the input
expression itself -- not in whether it counts as footprint-based (it does). The
genuine non-footprint alternatives are KEGG/REACTOME/Hallmark (`pathway_source`),
unsigned MSigDB gene set collections scored the same decoupleR way but without
PROGENy's curated, signed pathway-responsive weights, and compute_enrichment's
GO/KEGG/REACTOME/PROGENy/Hallmark databases, which are plain ORA/GSEA gene-set
testing, not footprint inference at all.

Unlike the other three servers in this package, Bioinfo-tools is not an installed R
package (no DESCRIPTION/NAMESPACE) -- it's a single script, functions.r, meant to be
source()'d. Each tool here still shells out to system R via the shared
common/r_bridge.py dispatcher (common/r/run_tool.R), but in "source" mode rather than
"package" mode: the script is source()'d fresh into its own environment for every call
rather than looked up in an installed package's namespace. See
common/paths.py:bioinfotools_script() for how its location is resolved (default
~/Documents/Bioinfo-tools/functions.r, override with BIOINFOTOOLS_SCRIPT). All of
functions.r's own R package dependencies (decoupleR, GSVA, edgeR, limma, clusterProfiler,
ReactomePA, org.Hs.eg.db, EnhancedVolcano, survival, survminer, WGCNA, ADImpute, sva,
ggpubr, ggstatsplot, ...) must already be installed in the system R library `Rscript`
(on PATH) resolves to.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import Field

from mcp_tme_servers.common.paths import bioinfotools_script, server_home
from mcp_tme_servers.common.r_bridge import (
    args_from,
    csv_arg,
    csv_column_arg,
    csv_rownames_arg,
    lit,
    path_arg,
    rds_arg,
)
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
        "file: transcription factor activity from expression (decoupleR-scored via "
        "CollecTRI/Dorothea/ARACNE, compute_tf_activity, default statistic 'consensus' -- not "
        "VIPER, which is just one of several decoupleR statistics available) or from a DEG "
        "signature with permutation p-values "
        "(compute_tf_activity_deg); decoupleR-scored pathway activity (PROGENy/KEGG/REACTOME/ "
        "Hallmark, compute_pathway_activity); edgeR/limma-voom differential expression "
        "(run_deg_analysis) with its volcano plot (compute_volcano_plot); ORA/GSEA enrichment "
        "against GO/KEGG/REACTOME/PROGENy/Hallmark (compute_enrichment); Kaplan-Meier survival "
        "analysis (compute_survival_analysis); TPM/z-score normalization "
        "(compute_normalization); ComBat-seq batch correction (compute_batch_correction); PCA "
        "(compute_pca_analysis); and trait-association testing between a score matrix and "
        "either a continuous trait (Pearson, Spearman, Fisher's exact -- scores_stat_analysis or "
        "the individual scores_*_test tools) or a categorical one (t-test, Wilcoxon, ANOVA, "
        "Kruskal-Wallis -- scores_stat_analysis_categorical or the individual scores_*_test "
        "tools). "
        "IMPORTANT: CollecTRI, Dorothea, and ARACNE (compute_tf_activity/compute_tf_activity_deg's "
        "tf_collection) and PROGENy (compute_pathway_activity's default pathway_source) are all "
        "footprint-based -- they score a regulator's activity from its targets'/responsive genes' "
        "expression, not the regulator's own expression, via a decoupleR statistic (default "
        "'consensus', an ensemble of MLM/ULM/normalized WSUM -- VIPER is only one of several "
        "selectable statistics, not the default). CollecTRI and Dorothea are curated "
        "prior-knowledge networks (PKN) fetched from OmnipathR; ARACNE differs only in where its "
        "network comes from -- a data-driven gene regulatory network (GRN) inferred per-cohort "
        "from the input expression itself, rather than a curated PKN -- it is scored the same "
        "footprint way once that network exists. The KEGG/REACTOME/Hallmark gene set collections "
        "are the genuine non-footprint alternative for pathway activity (unsigned MSigDB gene "
        "sets, scored the same decoupleR way but without PROGENy's curated, signed weights). "
        "These are standalone utilities rather than one fixed pipeline -- reach for whichever "
        "function fits the current step. Common chains: run compute_normalization on raw counts "
        "before compute_tf_activity/compute_pathway_activity (both expect normalized "
        "expression); run_deg_analysis on raw counts for a two-group comparison, whose full "
        "signature (t or logFC) can feed compute_tf_activity_deg or compute_enrichment "
        "(method='GSEA'), and whose significant genes can feed compute_enrichment "
        "(method='ORA'); then relate any resulting score matrix to a clinical trait with "
        "scores_stat_analysis/scores_stat_analysis_categorical (or one of the individual "
        "scores_*_test tools) or to survival with compute_survival_analysis. Score matrices from "
        "this server, or from multideconv-server/CellTFusion-server/RaCInG-server in this same "
        "package, are typical inputs to the trait-association and survival tools here. "
        "IMPORTANT -- citation: if the user asks what to cite for an analysis that used this "
        "server, always include the mcp-tme-servers project itself -- XX, et al. (XX). "
        "mcp-tme-servers -- XX. XX. https://doi.org/XX (a placeholder until a real reference "
        "exists; check the project README for the current one) -- in addition to whichever "
        "specific decoupleR statistics/resources (e.g. CollecTRI, PROGENy, ARACNE) were "
        "actually used."
    ),
)

SESSION_CONFIG = SessionConfig(
    sessions_root=server_home("bioinfotools") / "runs" / "sessions",
    stage_by_tool={
        "compute_tf_activity": "tf_activity",
        "compute_tf_activity_deg": "tf_activity_deg",
        "compute_pathway_activity": "pathway_activity",
        "run_deg_analysis": "differential_expression",
        "compute_volcano_plot": "differential_expression",
        "compute_enrichment": "enrichment",
        "compute_survival_analysis": "survival_analysis",
        "compute_normalization": "normalization",
        "compute_batch_correction": "batch_correction",
        "compute_pca_analysis": "pca",
        "compute_correlation": "correlation",
        "scores_pearson_test": "trait_association",
        "scores_spearman_test": "trait_association",
        "scores_fisher_test_continuous": "trait_association",
        "scores_stat_analysis": "trait_association",
        "scores_ttest_test": "trait_association",
        "scores_wilcox_test": "trait_association",
        "scores_anova_test": "trait_association",
        "scores_kruskal_test": "trait_association",
        "scores_stat_analysis_categorical": "trait_association",
    },
    export_requires=[
        frozenset(
            {
                "tf_activity",
                "tf_activity_deg",
                "pathway_activity",
                "differential_expression",
                "enrichment",
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
        "tf_activity_deg": "no analysis tool on this server has been run yet for this session -- there is nothing to export.",
        "pathway_activity": "no analysis tool on this server has been run yet for this session -- there is nothing to export.",
        "differential_expression": "no analysis tool on this server has been run yet for this session -- there is nothing to export.",
        "enrichment": "no analysis tool on this server has been run yet for this session -- there is nothing to export.",
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
            "A previously saved .rds samples x features score matrix/data.frame to test (e.g. "
            "chained directly from another tool's run_dir/result.rds -- compute_tf_activity's "
            "and compute_pathway_activity's result.rds are already this shape). Mutually "
            "exclusive with scores_csv; exactly one is required."
        )
    ),
]
TraitParam = Annotated[str, Field(description="Name of the (numeric) column in coldata_csv holding the continuous clinical/experimental variable to test against, e.g. age, tumor purity, or a continuous score.")]
CategoricalTraitParam = Annotated[str, Field(description="Name of the column in coldata_csv holding the categorical clinical/experimental variable to test against, e.g. treatment arm, response group, or risk class.")]
PvalParam = Annotated[float, Field(ge=0, le=1, description="P-value threshold below which a feature is kept as significant.")]

DecoupleRStatistic = Literal["consensus", "aucell", "udt", "mdt", "wmean", "ulm", "mlm", "wsum", "viper", "gsva", "ora", "fgsea"]

TfCollectionParam = Annotated[
    Literal["CollecTRI", "Dorothea", "ARACNE"],
    Field(
        description=(
            "Source of the TF-target regulon network -- all three are scored the same "
            "FOOTPRINT way (a decoupleR statistic over each TF's targets, default 'consensus'; "
            "VIPER is just one selectable statistic among several, not the default or the only "
            "one). 'CollecTRI' (default) and 'Dorothea' are prebuilt, literature-curated "
            "prior-knowledge networks (PKN) fetched from OmnipathR (and cached to "
            "Results/TF_target_collection.csv under run_dir for reuse within the same "
            "output_dir) -- generic, not tissue/cancer-specific; use 'Dorothea' with confidence "
            "to restrict to its highest-confidence regulons. 'ARACNE' instead correlates each TF "
            "against its targets in a user-supplied, data-driven gene regulatory network (GRN) "
            "rather than a curated PKN -- pick this for a cancer-type/cohort-specific regulatory "
            "network; requires aracne_network to point at a 3-column network.txt file "
            "(Regulator, Target, mutual information, e.g. from ARACNe-AP)."
        )
    ),
]
ConfidenceParam = Annotated[
    list[Literal["A", "B", "C", "D", "E"]],
    Field(description="Dorothea confidence levels to keep, from 'A' (highest) to 'E' (lowest). Ignored unless tf_collection='Dorothea'."),
]
AracneNetworkParam = Annotated[
    str | None,
    Field(
        description=(
            "Path to the ARACNE network.txt file (3 columns: Regulator, Target, mutual "
            "information), required when tf_collection='ARACNE' (ignored otherwise). Read "
            "directly from disk by the R function itself, not through this server's CSV loader."
        )
    ),
]


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
    rna_counts_csv: Annotated[str, Field(description="Normalized, log-scale gene expression matrix CSV -- log2(TPM + pseudocount), log2CPM, etc., e.g. from compute_normalization(..., log=True, zscore=False) -- genes as rows, samples as columns. Not raw or linear-scale TPM/counts.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    tf_collection: TfCollectionParam = "CollecTRI",
    min_targets_size: Annotated[int, Field(ge=1, description="Minimum number of a TF's target genes that must be present in rna_counts_csv for that TF to be scored. Lower it if too few TFs get activity scores (see warnings); raise it to require more evidence per TF.")] = 5,
    universe_csv: Annotated[str | None, Field(description="Optional user-supplied TF-target network CSV (source/target columns, i.e. regulator/target gene symbols) to use instead of fetching tf_collection.")] = None,
    confidence: ConfidenceParam = ["A", "B"],
    aracne_network: AracneNetworkParam = None,
    statistic: Annotated[
        DecoupleRStatistic,
        Field(
            description=(
                "Which decoupleR::decouple() statistic to extract. Default 'consensus' (an "
                "ensemble score across the top-performing methods: MLM, ULM, and normalized "
                "WSUM). Any single underlying method can be requested instead: 'aucell', 'udt', "
                "'mdt', 'wmean', 'ulm', 'mlm', 'wsum', 'viper', 'gsva', 'ora', or 'fgsea'."
            )
        ),
    ] = "consensus",
    cores: Annotated[int, Field(ge=1, description="Reserved for parallelizing VIPER inference; not currently used by the underlying R function.")] = 3,
    file_name: Annotated[str | None, Field(description="File name suffix for the TF activity matrix CSV written to run_dir/Results.")] = None,
) -> RunResult:
    """Infer transcription factor activity from expression via decoupleR
    (Bioinfo-tools::compute.TFs.activity).

    WHAT: infers each TF's regulatory activity (not its own expression) from how coordinately
    up/down its known target genes move in each sample, using a decoupleR statistic (default
    'consensus', an ensemble of MLM/ULM/normalized WSUM -- VIPER is just one of several
    selectable statistics via the statistic parameter, not the default) against a TF-target
    regulon network. Higher activity means the TF's targets show more coordinated up-regulation
    in that sample -- a functional readout of regulatory state that raw TF gene expression alone
    doesn't capture. All three tf_collection options are FOOTPRINT-based (scored this same way);
    'CollecTRI'/'Dorothea' draw on curated, literature-derived prior-knowledge networks (PKN),
    while 'ARACNE' instead uses a data-driven gene regulatory network (GRN) inferred per-cohort
    from the input expression -- see this server's own instructions for the full footprint-method
    rundown.

    WHEN: run on already-normalized, log-scale expression (e.g. compute_normalization's output)
    -- not raw counts. The resulting samples x TFs matrix is a typical input to
    compute_survival_analysis (via its features parameter), scores_stat_analysis(_categorical),
    or compute_correlation.

    Returns a samples x TFs data frame of TF activity (not z-scored -- unlike
    compute_pathway_activity, this function does not scale its output).
    """
    args = args_from(
        **{
            "RNA.counts": csv_arg(rna_counts_csv, row_names=1, as_matrix=True),
            "TF.collection": lit(tf_collection),
            "min_targets_size": lit(min_targets_size),
            "universe": csv_arg(universe_csv, row_names=None) if universe_csv else None,
            "confidence": lit(confidence),
            "aracne.network": path_arg(aracne_network) if aracne_network else None,
            "statistic": lit(statistic),
            "cores": lit(cores),
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
async def compute_tf_activity_deg(
    signature_csv: Annotated[
        str,
        Field(
            description=(
                "CSV with a per-gene differential expression ranking statistic, gene symbols as "
                "row names -- e.g. run_deg_analysis(..., pval=1, plot=False)'s full, unfiltered "
                "output. Paired with signature_column to select which column holds the "
                "statistic. We recommend the 't' statistic (combines effect size and precision), "
                "but the choice (e.g. logFC) is up to the caller."
            )
        ),
    ],
    signature_column: Annotated[str, Field(description="Column in signature_csv holding the per-gene ranking statistic to score (e.g. 't' or 'logFC').")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    tf_collection: TfCollectionParam = "CollecTRI",
    min_targets_size: Annotated[int, Field(ge=1, description="Minimum number of a TF's target genes that must be present in signature_csv for that TF to be scored (used for the observed score and every permutation).")] = 5,
    universe_csv: Annotated[str | None, Field(description="Optional user-supplied TF-target network CSV (source/target columns), reused for the observed score and every permutation, instead of fetching/caching tf_collection.")] = None,
    confidence: ConfidenceParam = ["A", "B"],
    aracne_network: AracneNetworkParam = None,
    rna_counts_csv: Annotated[
        str | None,
        Field(
            description=(
                "The full genes x samples normalized expression matrix, required only when "
                "tf_collection='ARACNE': unlike CollecTRI/Dorothea, ARACNE's mode of regulation "
                "isn't pre-signed, but derived from cross-sample Spearman correlation between "
                "each TF and its targets -- which needs real multi-sample expression, not the "
                "single-column signature being scored here. Ignored for other tf_collection "
                "values."
            )
        ),
    ] = None,
    statistic: Annotated[
        DecoupleRStatistic,
        Field(
            description=(
                "Which decoupleR::decouple() statistic to score the signature with. Default "
                "'consensus'. Same options as compute_tf_activity. Note 'consensus', 'udt', "
                "'mdt', 'ora', 'fgsea', and 'gsva' are considerably slower per permutation than "
                "'viper'/'ulm'/'mlm'/'wsum'/'wmean'/'aucell'."
            )
        ),
    ] = "consensus",
    n_perm: Annotated[int, Field(ge=1, description="Number of gene-label permutations used to build the null distribution for the empirical p-value. Scored in memory-bounded batches, far faster than one decouple() call per permutation.")] = 1000,
    seed: Annotated[int, Field(description="Random seed for the permutations.")] = 42,
    top_n: Annotated[int, Field(ge=1, description="Number of top up- and down-activity TFs shown in the bar plot.")] = 10,
    file_name: Annotated[str | None, Field(description="File name suffix used when saving the activity table (CSV) and plot (PDF) to run_dir/Results.")] = None,
) -> RunResult:
    """TF activity from a DEG signature, with permutation-based significance
    (Bioinfo-tools::compute.TFs.activity.DEG).

    WHAT: computes TF activity directly from a differential expression signature -- a single,
    per-gene ranking statistic rather than per-sample expression -- by treating it as a
    single-column "sample" scored the same way as compute_tf_activity. In addition to that score,
    an empirical p-value is computed by repeatedly shuffling which gene gets which signature
    value (breaking the gene-TF regulon relationship while preserving the value distribution),
    rescoring every TF against each shuffled signature, and comparing the observed score to that
    null distribution.

    WHEN: use when you have a DEG ranking statistic (e.g. from run_deg_analysis(..., pval=1)) and
    want TF activity for that single contrast with a significance estimate, rather than
    per-sample TF activity from compute_tf_activity.

    Returns a list with `activity` (one row per TF: TF, score, and the empirical p_value, sorted
    by score descending) and `plot` (a ggplot2 bar plot of the top/bottom top_n TFs, also saved as
    a PDF; the activity table is also saved as a CSV) -- see field_glossary.
    """
    args = args_from(
        **{
            "signature": csv_column_arg(signature_csv, signature_column, row_names=1),
            "TF.collection": lit(tf_collection),
            "min_targets_size": lit(min_targets_size),
            "universe": csv_arg(universe_csv, row_names=None) if universe_csv else None,
            "confidence": lit(confidence),
            "aracne.network": path_arg(aracne_network) if aracne_network else None,
            "RNA.counts": csv_arg(rna_counts_csv, row_names=1, as_matrix=True) if rna_counts_csv else None,
            "statistic": lit(statistic),
            "n_perm": lit(n_perm),
            "seed": lit(seed),
            "top_n": lit(top_n),
            "return": lit(True),
            "file.name": lit(file_name) if file_name else None,
        }
    )
    return await _run(
        source=_SOURCE, function="compute.TFs.activity.DEG", args=args, tool_name="compute_tf_activity_deg",
        ctx=ctx, timeout=1800, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        glossary={
            "activity": "Per-TF data frame: TF name, statistic score from the observed signature, and empirical (permutation-based) two-sided p-value, sorted by score descending.",
            "plot": "ggplot2 bar plot of the top/bottom top_n TFs by score, labeled with each TF's empirical p-value (also saved as a PDF under run_dir/Results).",
        },
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def compute_pathway_activity(
    rna_tpm_csv: Annotated[str, Field(description="Normalized, log-scale gene expression matrix CSV -- log2(TPM + pseudocount), not raw or linear-scale TPM, e.g. from compute_normalization(..., log=True, zscore=False) -- genes as rows, samples as columns.")],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    pathway_source: Annotated[
        Literal["PROGENy", "KEGG", "REACTOME", "Hallmark"],
        Field(
            description=(
                "Pathway resource scored (and returned). 'PROGENy' (default) is a curated, "
                "signed, FOOTPRINT-based collection of ~14 core signaling pathways' "
                "perturbation-responsive genes -- see this server's own instructions for the "
                "full footprint-method rundown. 'KEGG', 'REACTOME', and 'Hallmark' are unsigned "
                "MSigDB gene set collections instead (not footprint-based), scored the same "
                "decoupleR way."
            )
        ),
    ] = "PROGENy",
    gene_sets_rds: Annotated[
        str | None,
        Field(
            description=(
                "Optional .rds holding a named list of gene sets (character vectors of gene "
                "symbols), e.g. custom or MSigDB Hallmark signatures. If given, these sets are "
                "also scored with statistic (independently of pathway_source) and saved to "
                "run_dir/Results as their own CSV -- but they are NOT part of this tool's return "
                "value/summary, only pathway_source's matrix is."
            )
        ),
    ] = None,
    paths_csv: Annotated[str | None, Field(description="Optional PROGENy pathway-gene weight table CSV (source/target/weight columns) to use instead of fetching the bundled human PROGENy resource (top 500 genes, cached to Results/Pathways_collection_PROGENy.csv under run_dir). Ignored unless pathway_source='PROGENy'.")] = None,
    statistic: Annotated[
        DecoupleRStatistic,
        Field(
            description=(
                "Which decoupleR::decouple() statistic to score pathway_source (and gene_sets_rds, "
                "if given) with. Default 'consensus' (an ensemble score across the top-performing "
                "methods: MLM, ULM, and normalized WSUM). Any single underlying method can be "
                "requested instead: 'aucell', 'udt', 'mdt', 'wmean', 'ulm', 'mlm', 'wsum', 'viper', "
                "'gsva', 'ora', or 'fgsea'. 'mlm' (and therefore the default 'consensus') fits all "
                "sources jointly and can fail with a collinearity error on a large, highly "
                "redundant collection (e.g. pathway_source='REACTOME') -- use a per-source method "
                "(e.g. 'ulm', 'viper', 'wsum') in that case."
            )
        ),
    ] = "consensus",
    min_targets_size: Annotated[int, Field(ge=1, description="Minimum number of a source's (pathway's or gene set's) target genes that must be present in rna_tpm_csv for it to be scored; sources with fewer are dropped.")] = 5,
    file_name: Annotated[str | None, Field(description="File name suffix for the pathway activity matrix CSV(s) written to run_dir/Results.")] = None,
) -> RunResult:
    """Compute sample-level pathway activity scores (Bioinfo-tools::compute.pathway.activity).

    WHAT: scores each sample's pathway_source activity with the requested decoupleR statistic
    (default 'consensus'), z-scored across samples. PROGENy (default pathway_source) is a
    curated, signed, footprint-based collection of ~14 core signaling pathways; KEGG/REACTOME/
    Hallmark are unsigned MSigDB gene set collections instead. gene_sets_rds, if given, is scored
    the same way alongside pathway_source and saved to disk, but not returned.

    WHEN: run on already-normalized, log-scale expression (e.g. compute_normalization's output).
    Useful on its own, or as the pathways input to multideconv-server's
    relate_subgroups_to_pathways.

    Returns a samples x pathways (or samples x gene sets, for KEGG/REACTOME/Hallmark) data frame
    for pathway_source only, z-scored across samples.
    """
    args = args_from(
        **{
            "RNA.tpm": csv_arg(rna_tpm_csv, row_names=1, as_matrix=True),
            "gene_sets": rds_arg(gene_sets_rds) if gene_sets_rds else None,
            "paths": csv_arg(paths_csv, row_names=None) if paths_csv else None,
            "pathway_source": lit(pathway_source),
            "statistic": lit(statistic),
            "min_targets_size": lit(min_targets_size),
            "return": lit(True),
            "file.name": lit(file_name) if file_name else None,
        }
    )
    return await _run(
        source=_SOURCE, function="compute.pathway.activity", args=args, tool_name="compute_pathway_activity",
        ctx=ctx, timeout=900, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
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
    pval: Annotated[float, Field(ge=0, le=1, description="Adjusted p-value cutoff used both to filter the returned table and to mark significance on the volcano plot.")] = 0.05,
    logfc: Annotated[float, Field(ge=0, description="Absolute log2 fold-change cutoff marked on the volcano plot -- does not filter the returned table.")] = 1.0,
    plot: Annotated[bool, Field(description="If True (default), save a volcano plot (compute.volcano.plot()) to run_dir/Results, built from the full, unfiltered set of tested genes.")] = True,
    file_name: Annotated[str | None, Field(description="File name suffix used when saving the volcano plot to run_dir/Results (plot=True only).")] = None,
) -> RunResult:
    """Differential expression analysis with edgeR/limma-voom (Bioinfo-tools::run_deg_analysis).

    WHAT: filters low-expression genes (edgeR::filterByExpr), TMM-normalizes, applies the voom
    mean-variance transformation, and fits a linear model contrasting group_col's two levels --
    the standard edgeR/limma-voom workflow for RNAseq count data.

    WHEN: use on raw counts (not normalized expression) for a simple two-group comparison. For
    more than two groups or a covariate-adjusted design, this function's fixed one-factor
    ~ group_col model isn't sufficient -- build a custom edgeR/limma script instead. The signature
    (t statistic recommended) from a pval=1 call feeds compute_tf_activity_deg or
    compute_enrichment(method='GSEA'); this tool's significant genes feed
    compute_enrichment(method='ORA').

    Returns a data frame of genes with adj.P.Val <= pval (logFC, AveExpr, t, P.Value, adj.P.Val,
    B), ranked by significance. An empty result (0 rows) means no gene passed the threshold (see
    warnings), not a tool failure -- pass pval=1 to get every tested gene instead (e.g. to build a
    ranking vector for GSEA/compute_tf_activity_deg).
    """
    args = args_from(
        **{
            "counts": csv_arg(counts_csv, row_names=1, as_matrix=True),
            "coldata": csv_arg(coldata_csv, row_names=1),
            "group_col": lit(group_col),
            "ref_level": lit(ref_level) if ref_level else None,
            "pval": lit(pval),
            "logFC": lit(logfc),
            "plot": lit(plot),
            "file.name": lit(file_name) if file_name else None,
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
            "No gene reached the adj.P.Val threshold -- this usually means too few samples per "
            "group, a genuinely weak effect, a pval cutoff that's too strict, or that "
            "group_col/ref_level don't reflect the intended contrast (double-check the factor "
            "levels)."
        ]
    return []


@mcp.tool(annotations=READ_ANNOTATIONS)
async def compute_volcano_plot(
    deg_csv: Annotated[
        str,
        Field(
            description=(
                "Differential expression result CSV with a logFC column and an adj.P.Val "
                "column, gene symbols as row names -- e.g. run_deg_analysis(..., pval=1, "
                "plot=False)'s full, unfiltered output written to CSV. For a meaningful volcano "
                "plot this should be the full set of tested genes, not only the significant ones."
            )
        ),
    ],
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    pval: Annotated[float, Field(ge=0, le=1, description="Adjusted p-value cutoff marked on the plot.")] = 0.05,
    logfc: Annotated[float, Field(ge=0, description="Absolute log2 fold-change cutoff marked on the plot.")] = 1.0,
    title: Annotated[str | None, Field(description="Plot title. Defaults to 'Volcano plot' if omitted.")] = None,
    xlim: Annotated[
        list[float] | None,
        Field(
            description=(
                "[min, max] x-axis (log2 fold-change) display range. If omitted, EnhancedVolcano's "
                "own default is used (data range padded by a fixed +/-1.5), which can dwarf small "
                "effect sizes and compress every point near zero -- set this explicitly (e.g. "
                "[-0.2, 0.2]) to zoom into the actual spread of logFC in deg_csv."
            ),
            min_length=2,
            max_length=2,
        ),
    ] = None,
    ylim: Annotated[
        list[float] | None,
        Field(
            description=(
                "[min, max] y-axis (-log10 adjusted p-value) display range. If omitted, "
                "EnhancedVolcano's own default is used (0 to the tallest point + 5)."
            ),
            min_length=2,
            max_length=2,
        ),
    ] = None,
    file_name: Annotated[str | None, Field(description="File name suffix used when saving the plot as a PDF to run_dir/Results.")] = None,
) -> RunResult:
    """Volcano plot for a differential expression result (Bioinfo-tools::compute.volcano.plot).

    WHAT: plots log2 fold-change against adjusted p-value via EnhancedVolcano, highlighting genes
    passing both the pval and logfc cutoffs. Called automatically by run_deg_analysis(plot=True),
    but usable standalone on any differential expression result (e.g. from a custom edgeR/limma
    script, not only run_deg_analysis's).

    WHEN: use directly when you already have a DEG table from elsewhere, want a different
    pval/logfc marking than run_deg_analysis's defaults without rerunning the model, or want the
    plot saved separately from the DEG call itself. Pass xlim/ylim to zoom the axes -- useful for
    small targeted panels (e.g. a NanoString gene set) where logFC values never approach
    EnhancedVolcano's default +/-1.5 padding, leaving every point squeezed near the origin.

    Returns the ggplot object (also saved as a PDF under run_dir/Results).
    """
    args = args_from(
        **{
            "deg": csv_arg(deg_csv, row_names=1),
            "pval": lit(pval),
            "logFC": lit(logfc),
            "title": lit(title) if title else None,
            "xlim": lit(xlim) if xlim else None,
            "ylim": lit(ylim) if ylim else None,
            "return": lit(True),
            "file.name": lit(file_name) if file_name else None,
        }
    )
    return await _run(
        source=_SOURCE, function="compute.volcano.plot", args=args, tool_name="compute_volcano_plot",
        ctx=ctx, timeout=300, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def compute_enrichment(
    ctx: Context,
    database: Annotated[
        Literal["GO_BP", "GO_CC", "GO_MF", "KEGG", "REACTOME", "PROGENy", "Hallmark"],
        Field(
            description=(
                "Resource to test against. Plain gene-set testing (ORA/GSEA), not footprint "
                "inference -- even for 'PROGENy'/'Hallmark', which double as footprint/gene-set "
                "resources elsewhere on this server (compute_tf_activity, "
                "compute_pathway_activity)."
            )
        ),
    ] = "GO_BP",
    method: Annotated[
        Literal["ORA", "GSEA"],
        Field(
            description=(
                "'ORA' (over-representation, default): tests a fixed gene list -- requires "
                "genes_csv, ignores ranking_csv/ranking_column. 'GSEA' (gene set enrichment): "
                "tests a full ranked gene list -- requires ranking_csv/ranking_column, ignores "
                "genes_csv/universe_csv."
            )
        ),
    ] = "ORA",
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    genes_csv: Annotated[
        str | None,
        Field(
            description=(
                "method='ORA' only: CSV whose row names are the gene SYMBOLs to test, e.g. "
                "run_deg_analysis's output (its significant genes) written to CSV. Required for "
                "'ORA'; ignored for 'GSEA'."
            )
        ),
    ] = None,
    universe_csv: Annotated[
        str | None,
        Field(
            description=(
                "method='ORA' only: optional CSV whose row names are the background gene "
                "SYMBOLs (e.g. every gene tested in the DEG analysis, from the raw counts "
                "matrix). If omitted, each underlying tool's own default universe is used."
            )
        ),
    ] = None,
    ranking_csv: Annotated[
        str | None,
        Field(
            description=(
                "method='GSEA' only: CSV with a per-gene ranking statistic, gene SYMBOLs as row "
                "names, for every tested gene -- not pre-filtered to significant ones (e.g. "
                "run_deg_analysis(..., pval=1)'s output). Paired with ranking_column. Required "
                "for 'GSEA'; ignored for 'ORA'."
            )
        ),
    ] = None,
    ranking_column: Annotated[str | None, Field(description="method='GSEA' only: column in ranking_csv holding the ranking statistic (e.g. 't' or 'logFC'). Required together with ranking_csv.")] = None,
    pval: Annotated[float, Field(ge=0, le=1, description="Adjusted p-value cutoff for both ORA and GSEA.")] = 0.05,
    top_n: Annotated[int, Field(ge=1, description="Number of top terms shown in the dot plot.")] = 20,
    file_name: Annotated[str | None, Field(description="File name suffix used when saving the dot plot to run_dir/Results.")] = None,
) -> RunResult:
    """Gene set enrichment analysis (ORA or GSEA) across multiple databases
    (Bioinfo-tools::compute.enrichment).

    WHAT: tests a gene list for over-representation (ORA, via clusterProfiler/ReactomePA) or a
    ranked gene list for gene set enrichment (GSEA) against Gene Ontology (BP/CC/MF), KEGG,
    REACTOME, PROGENy, or MSigDB Hallmark. This is plain ORA/GSEA gene-set testing, not the
    footprint-based activity inference compute_tf_activity/compute_pathway_activity do.

    WHEN: use ORA (genes_csv) on a fixed significant-gene list, e.g. run_deg_analysis's default
    output; use GSEA (ranking_csv/ranking_column) on a full ranking statistic across every tested
    gene, e.g. run_deg_analysis(..., pval=1)'s output's 't' column, when you want enrichment
    sensitive to direction/magnitude rather than a hard significance cutoff.

    Returns a clusterProfiler/ReactomePA enrichment result object (enrichResult for ORA,
    gseaResult for GSEA), or NULL (see warnings) if no term passed pval. A dot plot of the top
    top_n terms is saved as a PDF under run_dir/Results when a result is found.
    """
    args = args_from(
        **{
            "genes": (
                csv_column_arg(ranking_csv, ranking_column, row_names=1)
                if (method == "GSEA" and ranking_csv and ranking_column)
                else (csv_rownames_arg(genes_csv) if (method == "ORA" and genes_csv) else None)
            ),
            "database": lit(database),
            "method": lit(method),
            "universe": csv_rownames_arg(universe_csv) if (universe_csv and method == "ORA") else None,
            "pval": lit(pval),
            "top_n": lit(top_n),
            "return": lit(True),
            "file.name": lit(file_name) if file_name else None,
        }
    )
    return await _run(
        source=_SOURCE, function="compute.enrichment", args=args, tool_name="compute_enrichment",
        ctx=ctx, timeout=900, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_enrichment_diagnostics,
    )


def _enrichment_diagnostics(summary: dict) -> list[str]:
    if summary.get("class") == "NULL":
        return [
            "No term reached the pval threshold for this database/method -- the call succeeded "
            "but nothing was significant. Try a higher pval, a different database, or (for ORA) "
            "confirm genes_csv/universe_csv use the same gene symbol convention as the "
            "database's annotation."
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
    zscore: Annotated[bool, Field(description="Z-score scale each gene (row) across samples after TPM/log normalization. Default False, since compute_tf_activity and compute_pathway_activity expect log-TPM expression, not per-gene standardized values.")] = False,
    tr_length_csv: Annotated[str | None, Field(description="Optional CSV with a transcript-length column, row names matching raw_counts_csv's genes, passed through to ADImpute::NormalizeTPM(). Requires tr_length_column.")] = None,
    tr_length_column: Annotated[str | None, Field(description="Column in tr_length_csv giving each gene's transcript length.")] = None,
) -> RunResult:
    """Normalize raw counts to log-TPM and optionally z-score scale
    (Bioinfo-tools::compute.normalization).

    WHAT: converts raw counts to TPM via ADImpute::NormalizeTPM (optionally log-transformed),
    then optionally centers/scales each gene to mean 0, unit variance across samples.

    WHEN: run this first on raw counts before compute_tf_activity, compute_pathway_activity, or
    any tool here documented as expecting normalized expression -- they do not normalize
    internally. Leave zscore=False for that use (the default); z-score scaling is for other
    downstream uses (e.g. PCA/clustering on gene expression directly), not for feeding
    compute_tf_activity/compute_pathway_activity.

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
    normalize: Annotated[bool, Field(description="If True (default), the batch-corrected counts are also run through compute_normalization (log-TPM, no z-score by default) before being returned, matching the ComBat_seq -> NormalizeTPM pattern used elsewhere in this toolkit's Shiny apps/scripts. If False, returns raw corrected counts.")] = True,
    log: Annotated[bool, Field(description="Passed to compute_normalization when normalize=True.")] = True,
    zscore: Annotated[bool, Field(description="Passed to compute_normalization when normalize=True. Default False, since compute_tf_activity and compute_pathway_activity expect log-TPM expression, not per-gene standardized values.")] = False,
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
    batch-corrected log-TPM (optionally z-scored, off by default) expression if normalize=True
    (default).
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
    prefer scores_spearman_test if the relationship is monotonic but not linear, or the
    trait/scores are skewed with outliers.

    WHEN: a targeted single-method test. For all three continuous methods
    (pearson/spearman/fisher) behind one interface, use scores_stat_analysis instead.

    Returns the significant-feature subset of the score matrix (NULL if none significant, see
    warnings). Each significant feature's scatter plot (with fitted regression line) is saved as a
    PDF under run_dir/Results.
    """
    args = args_from(
        **{
            "scores": csv_arg(scores_csv, row_names=1) if scores_csv else rds_arg(scores_rds),
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
            "scores": csv_arg(scores_csv, row_names=1) if scores_csv else rds_arg(scores_rds),
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

    WHEN: when neither a linear (scores_pearson_test) nor a monotonic (scores_spearman_test)
    relationship is a safe assumption, or you specifically want a High/Low-group framing (e.g. to
    match how compute_survival_analysis splits features).

    Returns the significant-feature subset of the score matrix (NULL if none significant, see
    warnings). Each significant feature's High/Low bar chart is saved as a PDF under
    run_dir/Results.
    """
    args = args_from(
        **{
            "scores": csv_arg(scores_csv, row_names=1) if scores_csv else rds_arg(scores_rds),
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
    scores_csv: ScoresCsv = None,
    scores_rds: ScoresRds = None,
    method: Annotated[
        Literal["pearson", "spearman", "fisher"],
        Field(
            description=(
                "Which single test to run: 'pearson' (linear correlation), 'spearman' (monotonic "
                "rank correlation), or 'fisher' (both variables median-split, Fisher's exact test "
                "on the 2x2 table). See scores_pearson_test/scores_spearman_test/"
                "scores_fisher_test_continuous's own docstrings for when to prefer each. For a "
                "categorical trait instead, use scores_stat_analysis_categorical."
            )
        ),
    ] = "pearson",
    pval: PvalParam = 0.05,
) -> RunResult:
    """Unified dispatcher for the scores_*_test family against a continuous trait
    (Bioinfo-tools::scores.stat.analysis.continuous).

    WHAT: the same three tests as scores_pearson_test/scores_spearman_test/
    scores_fisher_test_continuous, selected via method behind one interface -- convenient when
    method is chosen dynamically or you'd rather not call three separate tools.

    WHEN: prefer this over calling the individual scores_*_test tools when you already know which
    single method you want (set method accordingly).

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


@mcp.tool(annotations=READ_ANNOTATIONS)
async def scores_ttest_test(
    coldata_csv: Annotated[str, Field(description="Sample annotation CSV including the trait, row names matching the score matrix.")],
    trait: CategoricalTraitParam,
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    scores_csv: ScoresCsv = None,
    scores_rds: ScoresRds = None,
    pval: PvalParam = 0.05,
) -> RunResult:
    """Welch's t-test between each score column and a 2-group categorical trait
    (Bioinfo-tools::scores.ttest.test).

    WHAT: for every column of the score matrix, tests the difference in means between trait's two
    groups via stats::t.test() (Welch's, unequal variances). trait must have exactly 2 groups --
    use scores_anova_test for 3+.

    WHEN: a targeted single-method test for a 2-group categorical trait (e.g. Responder/
    Non-responder) when a roughly normal score distribution is a reasonable assumption; prefer
    scores_wilcox_test if it isn't. For all four categorical methods behind one interface, use
    scores_stat_analysis_categorical.

    Returns the significant-feature subset of the score matrix (NULL if none significant, see
    warnings). Each significant feature's boxplot is saved as a PDF under run_dir/Results.
    """
    args = args_from(
        **{
            "scores": csv_arg(scores_csv, row_names=1) if scores_csv else rds_arg(scores_rds),
            "coldata": csv_arg(coldata_csv, row_names=1),
            "trait": lit(trait),
            "pval": lit(pval),
        }
    )
    return await _run(
        source=_SOURCE, function="scores.ttest.test", args=args, tool_name="scores_ttest_test",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_no_significant_features_diagnostics("t-test"),
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def scores_wilcox_test(
    coldata_csv: Annotated[str, Field(description="Sample annotation CSV including the trait, row names matching the score matrix.")],
    trait: CategoricalTraitParam,
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    scores_csv: ScoresCsv = None,
    scores_rds: ScoresRds = None,
    pval: PvalParam = 0.05,
) -> RunResult:
    """Wilcoxon rank-sum test between each score column and a 2-group categorical trait
    (Bioinfo-tools::scores.wilcox.test).

    WHAT: for every column of the score matrix, tests the difference between trait's two groups
    via stats::wilcox.test() (Mann-Whitney U) -- a non-parametric alternative to
    scores_ttest_test that does not assume normally distributed scores. trait must have exactly 2
    groups -- use scores_kruskal_test for 3+.

    WHEN: default choice over scores_ttest_test when a normal score distribution isn't a safe
    assumption, or scores are skewed with outliers.

    Returns the significant-feature subset of the score matrix (NULL if none significant, see
    warnings). Each significant feature's boxplot is saved as a PDF under run_dir/Results.
    """
    args = args_from(
        **{
            "scores": csv_arg(scores_csv, row_names=1) if scores_csv else rds_arg(scores_rds),
            "coldata": csv_arg(coldata_csv, row_names=1),
            "trait": lit(trait),
            "pval": lit(pval),
        }
    )
    return await _run(
        source=_SOURCE, function="scores.wilcox.test", args=args, tool_name="scores_wilcox_test",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_no_significant_features_diagnostics("Wilcoxon"),
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def scores_anova_test(
    coldata_csv: Annotated[str, Field(description="Sample annotation CSV including the trait, row names matching the score matrix.")],
    trait: CategoricalTraitParam,
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    scores_csv: ScoresCsv = None,
    scores_rds: ScoresRds = None,
    pval: PvalParam = 0.05,
) -> RunResult:
    """One-way ANOVA F-test between each score column and a categorical trait (2+ groups)
    (Bioinfo-tools::scores.anova.test).

    WHAT: for every column of the score matrix, tests whether trait's groups (two or more) differ
    in mean score via stats::aov(). Prefer scores_kruskal_test if a normal score distribution
    isn't a safe assumption.

    WHEN: a targeted single-method test for a categorical trait with more than two groups (e.g. a
    risk class or treatment arm); for exactly two groups, scores_ttest_test/scores_wilcox_test
    give the same kind of result more directly.

    Returns the significant-feature subset of the score matrix (NULL if none significant, see
    warnings). Each significant feature's boxplot is saved as a PDF under run_dir/Results.
    """
    args = args_from(
        **{
            "scores": csv_arg(scores_csv, row_names=1) if scores_csv else rds_arg(scores_rds),
            "coldata": csv_arg(coldata_csv, row_names=1),
            "trait": lit(trait),
            "pval": lit(pval),
        }
    )
    return await _run(
        source=_SOURCE, function="scores.anova.test", args=args, tool_name="scores_anova_test",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_no_significant_features_diagnostics("ANOVA"),
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def scores_kruskal_test(
    coldata_csv: Annotated[str, Field(description="Sample annotation CSV including the trait, row names matching the score matrix.")],
    trait: CategoricalTraitParam,
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    scores_csv: ScoresCsv = None,
    scores_rds: ScoresRds = None,
    pval: PvalParam = 0.05,
) -> RunResult:
    """Kruskal-Wallis test between each score column and a categorical trait (2+ groups)
    (Bioinfo-tools::scores.kruskal.test).

    WHAT: for every column of the score matrix, tests whether trait's groups (two or more) differ
    via stats::kruskal.test() -- a non-parametric alternative to scores_anova_test that does not
    assume normally distributed scores.

    WHEN: default choice over scores_anova_test when a normal score distribution isn't a safe
    assumption, or scores are skewed with outliers.

    Returns the significant-feature subset of the score matrix (NULL if none significant, see
    warnings). Each significant feature's boxplot is saved as a PDF under run_dir/Results.
    """
    args = args_from(
        **{
            "scores": csv_arg(scores_csv, row_names=1) if scores_csv else rds_arg(scores_rds),
            "coldata": csv_arg(coldata_csv, row_names=1),
            "trait": lit(trait),
            "pval": lit(pval),
        }
    )
    return await _run(
        source=_SOURCE, function="scores.kruskal.test", args=args, tool_name="scores_kruskal_test",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_no_significant_features_diagnostics("Kruskal-Wallis"),
    )


@mcp.tool(annotations=READ_ANNOTATIONS)
async def scores_stat_analysis_categorical(
    coldata_csv: Annotated[str, Field(description="Sample annotation CSV including the trait, row names matching the score matrix.")],
    trait: CategoricalTraitParam,
    ctx: Context,
    output_dir: OutputDir = None,
    session_id: SessionIdParam = None,
    scores_csv: ScoresCsv = None,
    scores_rds: ScoresRds = None,
    method: Annotated[
        Literal["ttest", "wilcox", "anova", "kruskal"],
        Field(
            description=(
                "Which single test to run: 'ttest' (Welch's, parametric, exactly 2 groups), "
                "'wilcox' (Wilcoxon rank-sum, non-parametric, exactly 2 groups), 'anova' "
                "(one-way ANOVA F-test, parametric, 2+ groups), or 'kruskal' (Kruskal-Wallis, "
                "non-parametric, 2+ groups). See scores_ttest_test/scores_wilcox_test/"
                "scores_anova_test/scores_kruskal_test's own docstrings for when to prefer each. "
                "For a continuous trait instead, use scores_stat_analysis."
            )
        ),
    ] = "ttest",
    pval: PvalParam = 0.05,
) -> RunResult:
    """Unified dispatcher for the scores_*_test family against a categorical trait
    (Bioinfo-tools::scores.stat.analysis.categorical).

    WHAT: the same four tests as scores_ttest_test/scores_wilcox_test/scores_anova_test/
    scores_kruskal_test, selected via method behind one interface -- convenient when method is
    chosen dynamically or you'd rather not call four separate tools. Mirrors scores_stat_analysis
    for the categorical case.

    WHEN: prefer this over calling the individual scores_*_test tools when you already know which
    single method you want (set method accordingly).

    Returns the significant-feature subset of the score matrix (NULL if none significant, see
    warnings), with per-feature boxplots saved as PDFs under run_dir/Results.
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
        source=_SOURCE, function="scores.stat.analysis.categorical", args=args, tool_name="scores_stat_analysis_categorical",
        ctx=ctx, timeout=600, output_dir=output_dir, session_id=session_id, session_config=SESSION_CONFIG,
        diagnostics=_no_significant_features_diagnostics("selected"),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
