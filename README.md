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
- R, with `Rscript` on `PATH`. `multideconv`, `pipeML`, `CellTFusion`, and
  `RaCInG` are auto-installed the first time a tool needs one that isn't
  already present in that R's library -- see [Automatic R package
  installation](#automatic-r-package-installation) below for exactly what
  that does and does not cover.
- A checkout of [Bioinfo-tools](https://github.com/mhurtado13/Bioinfo-tools)
  (its `functions.r`, not a package, so `library()` won't find it) with its
  own R dependencies (decoupleR, GSVA, edgeR, limma, survival, survminer,
  gridExtra, WGCNA, ADImpute, sva, ggpubr, ggstatsplot, ...) installed in the
  same R library. Defaults to `~/Documents/Bioinfo-tools/functions.r`;
  override with `BIOINFOTOOLS_SCRIPT` if checked out elsewhere. Not covered
  by auto-install (see below) -- it's a script, not a package, and will move
  to its own installable package later.

**Recommended: a dedicated R environment.** Since this project just shells
out to whichever `Rscript` is first on `PATH` (no hardcoded R install path
anywhere in this codebase), running it inside a dedicated conda/mamba
environment with its own R and package library -- rather than your system R,
or an R shared with other projects -- avoids version conflicts between these
four packages' dependencies and anything else you have installed. No code
changes are needed for this: just make sure that environment is activated
(so its `Rscript` resolves first) in whichever shell you use to register/run
these servers, e.g. via `claude mcp add`.

## Automatic R package installation

`multideconv`, `pipeML`, `CellTFusion`, and `RaCInG` are installed
automatically, on first use, if missing:

| Package | Install command |
|---|---|
| multideconv | `pak::pkg_install("VeraPancaldiLab/multideconv")` |
| pipeML | `pak::pkg_install("VeraPancaldiLab/pipeML")` |
| CellTFusion | `pak::pkg_install("VeraPancaldiLab/CellTFusion")` |
| RaCInG | `remotes::install_github("VeraPancaldiLab/RaCInG_package")` |

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

### multideconv (9 tools: 6 analysis + 3 session)

| Tool | R function |
|---|---|
| `run_deconvolution` | `multideconv::compute.deconvolution` |
| `analyze_deconvolution` | `multideconv::compute.deconvolution.analysis` |
| `benchmark_deconvolution` | `multideconv::compute.benchmark` |
| `relate_subgroups_to_pathways` | `multideconv::compute.subgroup.pathways` |
| `replicate_subgroups` | `multideconv::replicate_deconvolution_subgroups` |
| `prepare_deconvolution_folds` | `multideconv::prepare_multideconv_folds` |

Pipeline: `run_deconvolution` -> `analyze_deconvolution` -> optionally
`benchmark_deconvolution` / `relate_subgroups_to_pathways` /
`replicate_subgroups`. Export requires `deconvolution` + `harmonization`.
`prepare_deconvolution_folds` is a separate, optional inspection tool (not
required for export) -- it packages the same harmonization as
`analyze_deconvolution` to match pipeML's `fold_construction_fun` contract
(see the pipeML section below); it's what `train_ml_model` calls internally
per fold when driven with `fold_construction_fun="prepare_multideconv_folds"`.

`run_deconvolution`'s `create_signature=True` builds a custom signature
matrix from a matched single-cell reference (`sc_matrix_csv`/`sc_metadata_csv`)
and uses it in that same call -- `methods_sig` picks which signature-building
methods run, `name_sc_signature` names the resulting files (give it a real
name if you'll want to select this exact signature again later via
`signatures_select`, otherwise repeat runs silently overwrite each other's
generically-named files), and `cell_markers` is required for the BSeq-SC
method specifically. If you already have signature files computed elsewhere
(not built via `create_signature`) that you want included alongside the
bundled ones, pass their paths via `custom_signature_files` instead -- the
underlying R function only ever reads custom signatures from a fixed folder
relative to the run directory (`Results/custom_signatures/`), so this tool
copies each file there for you before calling R.

**Known upstream issue (TEMP -- a fix is pending, remove this bullet once
confirmed):** `prepare_multideconv_folds()`'s per-fold construction path
(`compute_full_dataset=False` on `prepare_deconvolution_folds`, or when
driven internally by `train_ml_model`'s `fold_construction_fun`) computes and
saves each fold's data to `Results/fold_*.rds` but returns `NULL` instead of
the fold list, due to a missing `return()` statement in
`multideconv::prepare_multideconv_folds`. The fold files are still written
and listed under `output_files`, but the returned summary is empty, and this
also breaks `train_ml_model(fold_construction_fun="prepare_multideconv_folds")`
once cross-validation reaches this path. `compute_full_dataset=True` (the
default) uses a different, unaffected code path.

### pipeML (6 tools: 3 analysis + 3 session)

| Tool | R function |
|---|---|
| `train_ml_model` | `pipeML::compute_features.training.ML` |
| `predict_ml_model` | `pipeML::compute_prediction` |
| `compute_shap_values` | `pipeML::compute_shap_values` |

Pipeline: `train_ml_model` (train + cross-validate) -> `predict_ml_model`
(score a held-out test set) -> optionally `compute_shap_values` (per-feature
importance). Typically consumes feature tables produced by the multideconv
or CellTFusion servers. Export requires at least one model trained.

`train_ml_model` supports standard repeated/LODO k-fold CV by default, or a
custom `fold_construction_fun` for **sample-dependent features** -- ones
computed relative to the whole training cohort (deconvolution/harmonization
subgroups, WGCNA modules, GSVA scores, PCA components, clustering) rather
than independently per sample. Computing such features once on the full
dataset before cross-validating leaks test-fold information into training;
`fold_construction_fun` recomputes the feature space fresh per fold instead.
Only `"prepare_multideconv_folds"` (multideconv's bundled implementation,
also exposed standalone as multideconv-server's `prepare_deconvolution_folds`
for inspection) is available, since an arbitrary R function object can't
cross this project's JSON-over-subprocess bridge -- see that parameter's own
description for how to write a custom one following pipeML's own vignette
pattern (only usable by calling pipeML directly in R, not through this tool).
When `fold_construction_fun` was used for training, `predict_ml_model`'s test
data must first be projected into that same learned feature space (e.g. via
multideconv-server's `replicate_subgroups`) rather than fed in raw -- see
`train_ml_model`'s `Custom_output` field and `predict_ml_model`'s own
docstring.

`predict_ml_model` and `compute_shap_values` both read `train_ml_model`'s
result.rds's `Model` element internally -- pass that call's `run_dir/result.rds`
directly as `model_rds`, not a manually extracted sub-object.

**Known upstream issue:** any `train_ml_model` call that doesn't explicitly
pass `metric="AUROC"` or `"AUPRC"` for classification currently crashes
inside pipeML itself (`R/machine_learning.R`) -- `hyperparams` is only
assigned inside the `if (metric == "AUROC" || metric == "AUPRC")` branch but
read unconditionally afterward. `train_ml_model`'s docstring carries this
workaround, and
`tests/test_pipeml.py::test_default_metric_surfaces_known_upstream_bug`
documents the failure directly so it's caught if pipeML's pinned version
changes without the bug being fixed.

This server previously also exposed `train_and_evaluate_ml_model`
(`pipeML::compute_features.ML`), a combined train+evaluate call. It has been
removed: besides the `hyperparams` bug above, its classification branch
separately references an undeclared `fold_models_dir` variable (present on
`compute_features.training.ML`'s signature but not on `compute_features.ML`'s
own), raising `object 'fold_models_dir' not found` for any classification
call -- doubly broken, and no test exercised it. Use `train_ml_model` then
`predict_ml_model` instead, which compose the same two steps working
correctly and independently.

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

### Bioinfo-tools (21 tools: 18 analysis + session tools*)

| Tool | R function |
|---|---|
| `compute_tf_activity` | `compute.TFs.activity` |
| `compute_tf_activity_deg` | `compute.TFs.activity.DEG` |
| `compute_pathway_activity` | `compute.pathway.activity` |
| `run_deg_analysis` | `run_deg_analysis` |
| `compute_volcano_plot` | `compute.volcano.plot` |
| `compute_enrichment` | `compute.enrichment` |
| `compute_survival_analysis` | `compute.survival.analysis` |
| `compute_normalization` | `compute.normalization` |
| `compute_batch_correction` | `compute.batch.correction` |
| `compute_pca_analysis` | `compute.pca.analysis` |
| `compute_correlation` | `compute.correlation` |
| `scores_pearson_test` | `scores.pearson.test` |
| `scores_spearman_test` | `scores.spearman.test` |
| `scores_fisher_test_continuous` | `scores.fisher.test.continuous` |
| `scores_stat_analysis` | `scores.stat.analysis.continuous` |
| `scores_ttest_test` | `scores.ttest.test` |
| `scores_wilcox_test` | `scores.wilcox.test` |
| `scores_anova_test` | `scores.anova.test` |
| `scores_kruskal_test` | `scores.kruskal.test` |
| `scores_stat_analysis_categorical` | `scores.stat.analysis.categorical` |

\* 18 analysis tools plus the 3 session tools = 21; `scores_stat_analysis`
and `scores_stat_analysis_categorical` are themselves dispatchers over the
three continuous and four categorical `scores_*_test` tools, respectively,
so 16 distinct analyses are exposed as 18 tool entry points.

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
