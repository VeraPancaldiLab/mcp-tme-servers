# mcp-tme-servers

Four [Model Context Protocol](https://modelcontextprotocol.io/) servers for
tumor-microenvironment (TME) analysis of bulk RNAseq, wrapping four R
packages so an LLM agent can call them directly:

| Server | Analysis role | Upstream R package |
|---|---|---|
| multideconv | Bulk RNAseq cell-type deconvolution, harmonized across methods | [multideconv](https://github.com/VeraPancaldiLab/multideconv) |
| pipeML | Leakage-free classification/survival model training | [pipeML](https://github.com/VeraPancaldiLab/pipeML) |
| CellTFusion | TF-activity-based tumor-microenvironment state characterization | [CellTFusion](https://github.com/VeraPancaldiLab/CellTFusion) |
| RaCInG | Cell-cell interaction network reconstruction | [RaCInG](https://github.com/VeraPancaldiLab/RaCInG_package) |

All four servers use MCP over stdio and are distributed together as
`mcp-tme-servers`, following the same one-repo/one-package/multiple-servers
pattern as [mcp-biomodelling-servers](https://github.com/marcorusc/mcp-biomodelling-servers).

## Requirements

- Python >= 3.12.
- MCP Python SDK 2.x, installed automatically with this package.
- R with the four upstream packages (`multideconv`, `pipeML`, `CellTFusion`,
  `RaCInG`) installed in the system R library that `Rscript` (must be on
  `PATH`) resolves to -- these are not pip/CRAN packages and aren't installed
  automatically; see each package's own repository for install instructions.

## How it works

Each server's tools are thin Python wrappers (`src/mcp_tme_servers/<server>/server.py`)
around one shared subprocess bridge (`src/mcp_tme_servers/common/r_bridge.py`)
that shells out to a single generic R dispatcher
(`src/mcp_tme_servers/common/r/run_tool.R`) -- one dispatcher, one marshaling
contract, reused by all four servers rather than duplicated per server. See
`common/r_bridge.py` docstrings for the manifest format.

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
```

### Register with Claude Code

```bash
claude mcp add multideconv-server --scope user -- uv --directory /home/marcelo.hurtado/Documents/mcp-tme-servers run mcp-multideconv-server
claude mcp add pipeML-server --scope user -- uv --directory /home/marcelo.hurtado/Documents/mcp-tme-servers run mcp-pipeml-server
claude mcp add CellTFusion-server --scope user -- uv --directory /home/marcelo.hurtado/Documents/mcp-tme-servers run mcp-celltfusion-server
claude mcp add RaCInG-server --scope user -- uv --directory /home/marcelo.hurtado/Documents/mcp-tme-servers run mcp-racing-server
```

## Configure an MCP client generically

```jsonc
{
  "servers": {
    "multideconv": { "type": "stdio", "command": "uv", "args": ["--directory", "/path/to/mcp-tme-servers", "run", "mcp-multideconv-server"] },
    "pipeml":      { "type": "stdio", "command": "uv", "args": ["--directory", "/path/to/mcp-tme-servers", "run", "mcp-pipeml-server"] },
    "celltfusion": { "type": "stdio", "command": "uv", "args": ["--directory", "/path/to/mcp-tme-servers", "run", "mcp-celltfusion-server"] },
    "racing":      { "type": "stdio", "command": "uv", "args": ["--directory", "/path/to/mcp-tme-servers", "run", "mcp-racing-server"] }
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

## Shared upstream issue (all four servers)

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
- "Export the LUAD_batch1 session's results so I can attach the
  reproducibility log to the methods section."
