# mcp-tme-servers

Five [Model Context Protocol](https://modelcontextprotocol.io/) servers for
tumor-microenvironment (TME) analysis of bulk RNAseq, wrapping four R
packages plus one general-purpose R toolkit so an LLM agent can call them
directly:

| Server | Analysis role | Upstream R code |
|---|---|---|
| multideconv | Bulk RNAseq cell-type deconvolution, harmonized across methods | [multideconv](https://github.com/VeraPancaldiLab/multideconv) (package) |
| pipeML | Leakage-free classification/survival model training | [pipeML](https://github.com/VeraPancaldiLab/pipeML) (package) |
| CellTFusion | TF-activity-based tumor-microenvironment state characterization | [CellTFusion](https://github.com/VeraPancaldiLab/CellTFusion) (package) |
| RaCInG | Cell-cell interaction network reconstruction | [RaCInG](https://github.com/VeraPancaldiLab/RaCInG_package) (package) |
| Bioinfo-tools | General-purpose bulk RNAseq analysis utilities (TF/pathway activity, DEG, survival, normalization, PCA, trait-association testing) | [Bioinfo-tools](https://github.com/mhurtado13/Bioinfo-tools) (script, `functions.r`) |

All five servers use MCP over stdio and are distributed together as
`mcp-tme-servers`.

## Requirements

- Python >= 3.12.
- MCP Python SDK 2.x, installed automatically with this package.
- R with `multideconv`, `pipeML`, `CellTFusion`, and `RaCInG` installed in the
  system R library that `Rscript` (must be on `PATH`) resolves to -- these
  are not pip/CRAN packages and aren't installed automatically; see each
  package's own repository for install instructions.
- A checkout of [Bioinfo-tools](https://github.com/mhurtado13/Bioinfo-tools)
  (its `functions.r`, not a package, so `library()` won't find it) with its
  own R dependencies (decoupleR, GSVA, edgeR, limma, survival, survminer,
  gridExtra, WGCNA, ADImpute, sva, ggpubr, ggstatsplot, ...) installed in the
  same R library. Defaults to `~/Documents/Bioinfo-tools/functions.r`;
  override with `BIOINFOTOOLS_SCRIPT` if checked out elsewhere.

## How it works

Each server's tools are thin Python wrappers (`src/mcp_tme_servers/<server>/server.py`)
around one shared subprocess bridge (`src/mcp_tme_servers/common/r_bridge.py`)
that shells out to a single generic R dispatcher
(`src/mcp_tme_servers/common/r/run_tool.R`) -- one dispatcher, one marshaling
contract, reused by all five servers rather than duplicated per server. See
`common/r_bridge.py` docstrings for the manifest format. Four servers run it
in "package" mode (`library()` + `asNamespace()` lookup); Bioinfo-tools runs
it in "source" mode instead (`source()`s `functions.r` fresh into its own
environment per call), since it isn't an installed package.

Runtime output (result files, plots, session logs) defaults to
`~/.mcp-tme-servers/<server>/runs/`, one directory per server so they never
collide; override with the `MCP_TME_SERVERS_HOME` environment variable, or
per-call with a tool's `output_dir` parameter.

## Installation

### Run from source with uv (current setup)

```bash
uv sync
uv run pytest        # regression tests against each package's own example data
```

```bash
uv run mcp-multideconv-server
uv run mcp-pipeml-server
uv run mcp-celltfusion-server
uv run mcp-racing-server
uv run mcp-bioinfotools-server
```

### Register with Claude Code

```bash
claude mcp add multideconv-server --scope user -- uv --directory /home/marcelo.hurtado/Documents/mcp-tme-servers run mcp-multideconv-server
claude mcp add pipeML-server --scope user -- uv --directory /home/marcelo.hurtado/Documents/mcp-tme-servers run mcp-pipeml-server
claude mcp add CellTFusion-server --scope user -- uv --directory /home/marcelo.hurtado/Documents/mcp-tme-servers run mcp-celltfusion-server
claude mcp add RaCInG-server --scope user -- uv --directory /home/marcelo.hurtado/Documents/mcp-tme-servers run mcp-racing-server
claude mcp add Bioinfo-tools-server --scope user -- uv --directory /home/marcelo.hurtado/Documents/mcp-tme-servers run mcp-bioinfotools-server
```

## Configure an MCP client generically

```jsonc
{
  "servers": {
    "multideconv":   { "type": "stdio", "command": "uv", "args": ["--directory", "/path/to/mcp-tme-servers", "run", "mcp-multideconv-server"] },
    "pipeml":        { "type": "stdio", "command": "uv", "args": ["--directory", "/path/to/mcp-tme-servers", "run", "mcp-pipeml-server"] },
    "celltfusion":   { "type": "stdio", "command": "uv", "args": ["--directory", "/path/to/mcp-tme-servers", "run", "mcp-celltfusion-server"] },
    "racing":        { "type": "stdio", "command": "uv", "args": ["--directory", "/path/to/mcp-tme-servers", "run", "mcp-racing-server"] },
    "bioinfotools":  { "type": "stdio", "command": "uv", "args": ["--directory", "/path/to/mcp-tme-servers", "run", "mcp-bioinfotools-server"] }
  }
}
```

## Sessions, results, and errors

Each server has three extra tools -- `start_workflow_session`,
`get_session_status`, `export_session_results` -- for tracking a multi-step
analysis on one cohort/dataset:

- `start_workflow_session(cohort_id, description?)` returns a `session_id`.
  Pass it to any analysis tool on that same server: it namespaces that
  call's output under the session's own cohort directory, appends the call
  (tool, parameters, timestamp, outcome) to `session_log.jsonl` for
  reproducibility, marks the corresponding pipeline stage complete, and
  **refuses** to chain in a result that belongs to a *different* session --
  the guard against accidentally mixing two cohorts' intermediate results in
  one conversation.
- `get_session_status(session_id)` -- stage completion, full step log,
  export-readiness.
- `export_session_results(session_id, output_dir)` -- bundles every result
  file plus the reproducibility log into one directory. Refuses until each
  server's own minimum pipeline stages have completed (see each server's own
  README section below for what's required).

Every analysis tool's docstring explains what it does, when to use it, and
the biological meaning of its parameters -- not just types -- and returns a
`field_glossary` mapping each part of its result to a plain-English
explanation, plus a `warnings` field flagging methodologically notable
outcomes (e.g. very few cell-type subgroups surviving filtering, near-chance
ML performance) with a suggested next step, rather than a bare pass/fail.

Tool call failures raise a descriptive error including the R error message
and the tail of R's console log -- not a bare stack trace.

## Servers

### multideconv (8 tools: 5 analysis + 3 session)

| Tool | R function |
|---|---|
| `run_deconvolution` | `multideconv::compute.deconvolution` |
| `analyze_deconvolution` | `multideconv::compute.deconvolution.analysis` |
| `benchmark_deconvolution` | `multideconv::compute.benchmark` |
| `relate_subgroups_to_pathways` | `multideconv::compute.subgroup.pathways` |
| `replicate_subgroups` | `multideconv::replicate_deconvolution_subgroups` |

Pipeline: `run_deconvolution` -> `analyze_deconvolution` -> optionally
`benchmark_deconvolution` / `relate_subgroups_to_pathways` /
`replicate_subgroups`. Export requires `deconvolution` + `harmonization`.

### pipeML (5 tools: 2 analysis + 3 session)

| Tool | R function |
|---|---|
| `train_ml_model` | `pipeML::compute_features.training.ML` |
| `train_and_evaluate_ml_model` | `pipeML::compute_features.ML` |

Typically consumes feature tables produced by the multideconv or CellTFusion
servers. Export requires at least one model trained.

**Known upstream issue:** any call that doesn't explicitly pass
`metric="AUROC"` or `"AUPRC"` currently crashes inside pipeML itself
(`R/machine_learning.R`) -- `hyperparams` is only assigned inside the
`if (metric == "AUROC" || metric == "AUPRC")` branch but read unconditionally
afterward. Both tools' docstrings carry this workaround, and
`tests/test_pipeml.py::test_default_metric_surfaces_known_upstream_bug`
documents the failure directly so it's caught if pipeML's pinned version
changes without the bug being fixed. Also: `compute_features.ML`'s `coldata`
must have at least two columns (a single-column `coldata` collapses to a
vector on subsetting and raises `incorrect number of dimensions`).

### CellTFusion (4 tools: 1 analysis + 3 session)

| Tool | R function |
|---|---|
| `run_celltfusion` | `CellTFusion::CellTFusion` |

The heaviest tool across all four servers (WGCNA + GSVA + NMF). Can take a
precomputed deconvolution matrix (e.g. from the multideconv server) instead
of recomputing it internally. Export requires `run_celltfusion` to have
completed once. `TME_states` in the result only populates when `cancer_type`
is given and a matching bundled TCGA meta-program reference exists (`blca`,
`luad`, `skcm` at time of writing).

### RaCInG (6 tools: 3 analysis + 3 session)

| Tool | R function |
|---|---|
| `prepare_racing_inputs` | `RaCInG::prepare_input_files` |
| `run_racing_kernel` | `RaCInG::compute_racing_kernel` |
| `run_racing_montecarlo` | `RaCInG::compute_racing_montecarlo` |

Pipeline: `prepare_racing_inputs` once, then reuse its result via
`input_data_rds` in `run_racing_kernel` and/or `run_racing_montecarlo`.
Prefer the kernel method by default (fast, deterministic); the Monte Carlo
method trades a lot of runtime for simulation-based variance/robustness.
Export requires `interaction_input_prep` plus at least one of
`interaction_kernel`/`interaction_montecarlo`.

**Known upstream issue:** `compute_racing_kernel`/`compute_racing_montecarlo`
don't expose the `already_normalized` flag that `prepare_input_files` has,
even though they call it internally when given raw `counts`. Workaround
(documented in both tools): call `prepare_racing_inputs` first and pass its
result via `input_data_rds`.

### Bioinfo-tools (14 tools: 13 analysis + session tools*)

| Tool | R function |
|---|---|
| `compute_tf_activity` | `compute.TFs.activity` |
| `compute_pathway_activity` | `compute.pathway.activity` |
| `run_deg_analysis` | `run_deg_analysis` |
| `compute_survival_analysis` | `compute.survival.analysis` |
| `compute_normalization` | `compute.normalization` |
| `compute_batch_correction` | `compute.batch.correction` |
| `compute_pca_analysis` | `compute.pca.analysis` |
| `compute_correlation` | `compute.correlation` |
| `scores_pearson_test` | `scores.pearson.test` |
| `scores_spearman_test` | `scores.spearman.test` |
| `scores_kendall_test` | `scores.kendall.test` |
| `scores_lm_test` | `scores.lm.test` |
| `scores_fisher_test_continuous` | `scores.fisher.test.continuous` |
| `scores_stat_analysis` | `scores.stat.analysis.continuous` |

\* 11 analysis tools plus the 3 session tools = 14; `scores_stat_analysis`
above is itself a dispatcher over the five `scores_*_test` tools, so 13
distinct analyses are exposed as 14 tool entry points.

Standalone utilities, not one fixed pipeline -- reach for whichever function
fits the current step. `compute_normalization` typically runs first on raw
counts, since `compute_tf_activity`/`compute_pathway_activity` expect
already-normalized expression; results from any of these (or from
multideconv-server/CellTFusion-server/RaCInG-server) are typical inputs to
`compute_survival_analysis` or the `scores_*`/`scores_stat_analysis`
trait-association tools. Export requires at least one analysis tool to have
completed.

Unlike the other four servers, this one runs `run_tool.R` in "source" mode
(see [How it works](#how-it-works)): it needs a local checkout of
[Bioinfo-tools](https://github.com/mhurtado13/Bioinfo-tools) rather than an
installed R package -- see [Requirements](#requirements) for the
`BIOINFOTOOLS_SCRIPT` override.

## Shared upstream issue (all four package servers)

`CellTFusion()` and RaCInG's `prepare_input_files()`/internal helpers call
`multideconv::compute.deconvolution()` (and related functions) without an
`importFrom`/`pkg::fn()` qualification, so they only resolve when
`multideconv` is separately attached in the R session, not merely installed
as a declared dependency. `common/r/run_tool.R` works around this by
attaching all four upstream packages regardless of which server's tool is
actually running -- worth fixing upstream via each package's
`NAMESPACE`/roxygen `@importFrom`.

## Example prompts

- "Start a session for cohort LUAD_batch1, deconvolve and harmonize
  `~/data/luad_counts.csv`."
- "Run CellTFusion on that same cohort using the harmonized deconvolution as
  input, cancer_type luad."
- "Train a classifier predicting Response from the harmonized features,
  metric AUROC, leave-one-dataset-out CV by cohort."
- "Build the RaCInG inputs from the same counts and get me direct and wedge
  communication features."
- "Normalize `~/data/luad_counts.csv` to log-TPM, then compute TF activity
  and pathway activity on the result."
- "Test whether any TF activity column correlates with tumor purity in
  coldata.csv -- try Spearman first."
- "Screen the harmonized deconvolution features for anything prognostic in
  the LUAD_batch1 cohort's survival data."
- "Export the LUAD_batch1 session's results so I can attach the
  reproducibility log to the methods section."
