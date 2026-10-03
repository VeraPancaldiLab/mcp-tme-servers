# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                          # install/update Python deps
uv run pytest                                    # full suite -- runs real R subprocesses, not mocked
uv run pytest tests/test_multideconv.py          # one server's tests
uv run pytest tests/test_multideconv.py::test_analyze_deconvolution -v   # one test

uv run mcp-multideconv-server                    # run a server directly (stdio)
uv run mcp-pipeml-server
uv run mcp-celltfusion-server
uv run mcp-racing-server
uv run mcp-bioinfotools-server

uv run --with pyflakes python -m pyflakes src/mcp_tme_servers   # no configured linter -- this is the practical check
Rscript -e 'invisible(parse("src/mcp_tme_servers/common/r/run_tool.R"))'  # syntax-check the shared R dispatcher after editing it
```

Tests require R on `PATH` with the five underlying packages installed (see
"R package auto-install" below) -- they exercise the real `mcp.Client` against
each server's in-process `mcp` object and shell out to actual R, so they're
slower than typical unit tests and will install a missing package on first
run.

## Architecture

Five independent MCP servers (`multideconv`, `pipeml`, `celltfusion`,
`racing`, `bioinfotools`, each `src/mcp_tme_servers/<server>/server.py`) wrap
five R packages/toolkits, each exposing its analysis functions as MCP tools.
All five share one Python package and one dispatch mechanism in
`src/mcp_tme_servers/common/` -- that shared layer is the part that matters
most for understanding how any single tool actually runs.

### The manifest contract (`common/r_bridge.py` <-> `common/r/run_tool.R`)

Every tool call becomes a JSON "manifest" (package name, R function name,
a working directory, and an `args` dict) written to disk, then run via
`Rscript run_tool.R <manifest.json>` as a subprocess. Python never marshals
R values directly -- `run_tool.R` resolves each arg into a real R object,
`do.call()`s the target function, and writes back a structured JSON summary
(class/dim/names/preview, not the full object) plus any tabular result
elements as CSVs, between two stdout markers `r_bridge.py` parses out.

Each `args` entry is built by one of `r_bridge.py`'s typed constructors
(`csv_arg`, `csv_column_arg`, `csv_list_arg`, `csv_rownames_arg`, `rds_arg`,
`rds_element_arg`, `rds_element_rownames_arg`, `path_arg`, `r_function_arg`, `named_r_function_arg`,
`lit`, `named_args_arg`), matched by a same-named `"__type__"` case in
`run_tool.R`'s `resolve_arg()`. Adding a new kind of argument means adding a
constructor on the Python side and a matching resolver case in R -- both
files must change together. Sample alignment lives in the bridge, not in the
servers: `csv_column_arg`/`csv_arg` take `align_to` (+ `align_to_columns` for a
genes x samples CSV) to follow another CSV's sample IDs, `csv_arg` can also
`transpose` and append `add_columns`, and `named_args_arg` values may be nested
specs (e.g. a `csv_arg` for a data-valued fold-function argument). `_run`'s
cross-session check walks nested specs too.

### Server file template

Every `server.py` follows the same shape: module docstring -> imports ->
`mcp = MCPServer(name, instructions=...)` -> `SESSION_CONFIG =
SessionConfig(...)` -> `_run = make_run(runs_root=server_home(...))` ->
`register_session_tools(mcp, SESSION_CONFIG, domain=...)` -> one
`@mcp.tool(...)` async function per analysis capability -> `main()`. Each
tool function builds `args = args_from(**{...})` (dropping `None` entries)
and ends with `return await _run(package=..., function=..., args=args,
tool_name=..., ctx=ctx, timeout=..., output_dir=output_dir,
session_id=session_id, session_config=SESSION_CONFIG, glossary=...,
diagnostics=...)`.

`_run` (built by `common/tools.py:make_run`) wraps `run_r_function` with:
session bookkeeping (namespacing output under a session's cohort dir,
appending to `session_log.jsonl`, refusing to chain in a `*_rds`/`*_csv`
path that belongs to a *different* session), secret redaction before
logging (any arg whose name contains "token"/"password"/"secret"/
"credential"), and assembly of the generic `RunResult` model every tool
returns (`run_dir`, `summary`, `field_glossary`, `csv_outputs`,
`output_files`, `warnings`, `log_tail`, `session_id`) -- there is no
bespoke per-tool return schema. `glossary` (a dict) and `diagnostics` (a
`summary -> list[str]` callback) are optional per-tool additions that turn
the generic R-structure summary into biologically meaningful field
explanations and methodological warnings (e.g. "only 2 cell-type subgroups
survived filtering").

### Sessions (`common/session.py`)

A `Session` groups a sequence of tool calls against one cohort/dataset,
persisted on disk under `<server>/runs/sessions/<session_id>/` (not server
memory). `SessionConfig.export_requires` is a list of stage-name groups
(each a frozenset), checked as a CNF of ORs -- a singleton group means that
stage is mandatory before `export_session_results` will succeed; a
multi-member group means any one stage in it is enough.

### R package installation (`run_tool.R`)

`multideconv`, `pipeML`, `CellTFusion`, `RaCInG` are always attached
together regardless of which one a call targets, because they call into
each other's exported functions without `pkg::fn()` qualification (a real
upstream issue, worked around here rather than fixed). `BioinfoTools` is
independent and only attached when it's the actual target. All five are
installed automatically the first time they're needed if missing
(`requireNamespace()` check, cheap and local when already installed) --
this **never** reinstalls or updates an already-installed package, so a
push to one of the five upstream repos requires manually reinstalling it.

### Known upstream bugs are documented, not patched

When a wrapped R package has a real bug, this repo documents the bug in the
tool's own docstring (and README if user-facing) rather than patching the
upstream package -- this repo only wraps upstream code, it doesn't fork it.
Where the bug is cheap to detect from the arguments, the tool also rejects
the call up front with a `ToolError` instead of letting a long R run crash
at the end. Search for "Known upstream issue" and "TEMP" across
`server.py`/tests when touching a tool that has one; a `TEMP` marker means a
fix is expected soon and should be removed once confirmed fixed upstream,
not treated as permanent.

No upstream bug is currently open. (CellTFusion's incompatibility with
multideconv 0.2.0 -- the removed `seed` argument and `.Iteration.<n>`
subgroup names -- was fixed in CellTFusion `24f8c69`, which now requires
`multideconv (>= 0.2.0)`.)

### Upstream API as of the last sync (2026-10-03)

Synced against pipeML `e727861` **plus its uncommitted working-tree edits** of
`R/machine_learning.R`, `R/data.R` and `DESCRIPTION` (input validation,
`preprocess`, `n_groups`, guarded parallelism), multideconv `628ef5b` (0.2.0 +
`aggregate_cell_groups`), CellTFusion `24f8c69`, RaCInG `fef8ff0` **plus its uncommitted working-tree
edit** of `R/RaCInG_input_generation.R` (subgroup-aware expression profiles), and
BioinfoTools `2b0a0e2` (README-only since the installed `5be0c79`, so not
reinstalled). The others were reinstalled with `R CMD INSTALL
/home/marcelo.hurtado/Documents/<repo>` (repos: `pipeML`, `multideconv`,
`CellTFusion`, `RaCInG_package`, `Bioinfo-tools`). Things the servers
depend on:

- `pipeML::compute_shap_values(model_trained, task_type, seed)` explains
  the **final** model on all training samples (one fixed feature definition,
  also for `fold_construction_fun` features) and stores the mean prediction
  in `attr(, "baseline")` -- run_tool.R's `summarize()` now surfaces small
  atomic attributes as `summary.attributes`. `plot_shap_stability()`,
  per-fold models and `fold_models_dir` are gone everywhere.
- `compute_features.training.ML()`: classification `metric` is only
  `"AUROC"` (default when NULL) or `"AUPRC"`; the server never sends
  `metric` for survival (pipeML warns). `ncores` also parallelizes standard
  survival CV; every parallel path is guarded by `ncores > 1`, so the
  server passes `ncores` through untouched (with `fold_construction_fun`,
  classification trains sequentially and survival only uses it for tunable
  parameter combinations). `preprocess = TRUE` (default) removes
  near-constant and |r| > 0.9 features -- once before CV, or inside each
  fold with a fold function; `FALSE` trains on the features as given.
  `compute_prediction()` gained `n_groups` (Kaplan-Meier risk groups,
  survival). pipeML now stops up front on a target with NAs or a single
  class, events not coded 0/1, LODO without `batch_var` or with a cohort
  smaller than `k_folds`, and when no model can be selected (it used to
  return NULL). In the custom-fold classification path one algorithm's fit
  error still aborts the run (e.g. treebag on ~10 training samples -- why
  `test_custom_folds_train_then_shap` pins `seed`). Fold functions get
  the outcome inside `data` (`target` coded no/yes, or `time`/`event`), so
  survival + `prepare_multideconv_folds` works.
- `compute_prediction()` returns `Curve_bands` and errors when the target
  length doesn't match the test rows -- hence `csv_column_arg(...,
  align_to=...)`, which matches a coldata column to a feature CSV by row name.
- multideconv 0.2.0: `compute.deconvolution.analysis(deconvolution, corr,
  corr_type, zero_thr, cv_thr, batch, cells_extra, file_name, return,
  verbose)` -- subgroups come from complete-linkage clustering (every pair
  >= `corr`), are named `<CellType>_Subgroup.<i>` (no `.Iteration.`), and the
  output has 6 elements (no "Discarded groups with equal method"/"High
  correlated ..."). `prune_thr`, `seed` and `var_quantile` are gone (also
  from `prepare_multideconv_folds`); the low-variance filter is `cv_thr`
  (sd/mean, default 0.1). A character/factor `batch` is categorical (one
  indicator per batch), a numeric one a linear covariate -- the server sends
  it with `csv_column_arg(..., as_character=True)`. `compute.subgroup.pathways()`
  gained `corr_type`, `height`/`width` default to NULL, and it returns
  per-cell-type `correlations`/`pvalues` (run_tool.R writes tables nested one
  level in a plain list as `result_<element>_<table>.csv`). `computeCBSX`/
  `computeDWLS`/`computeMOMF` are internal. Deconvolution caches are keyed
  by the input data.
- `multideconv::aggregate_cell_groups(deconvolution, cell_groups, min_types,
  verbose)` (wrapped as `aggregate_cell_groups`) adds
  `<method>_<signature>_<group>` sum features to a raw deconvolution matrix;
  new group names must then go in `cells_extra` downstream, and the same
  groups must be aggregated on a new cohort before replication.
- `multideconv::prepare_multideconv_folds()` reads the outcome from
  `data`'s `target` or `time`+`event` columns; `time_var`/`event_var`/
  `trait.positive` were removed (and from `prepare_deconvolution_folds`).
  CBSX requested without credentials warns and is skipped in multideconv.
- `CellTFusion()`: default `deconv_methods` no longer includes CBSX (and it
  still *stops* without credentials -- the tool rejects that up front);
  `cancer_type` is one of blca/luad/skcm or NULL (mapping skipped);
  `batch = TRUE` needs coldata + batch_id; ARACNE networks are read from
  `input/ARACNE/<cancer_type>/network/network.txt` relative to the run dir,
  so `run_celltfusion` copies `aracne_network_file` there via
  `common/tools.py:resolve_output_dir` (also used by `run_deconvolution`).
- `CellTFusion::prepare_celltfusion_folds(data, folds, bestune, deconv,
  raw.counts, coldata, batch, batch_id, ncores, return, verbose, ...)` is the
  second pipeML fold function (`train_ml_model`'s `_FOLD_FUNCTION_PACKAGES`
  maps each registered name to its package). It needs `deconv` (and
  `raw.counts`/`coldata`) as data, which `train_ml_model` injects into
  `fold_construction_args_fixed` from `fold_deconv_csv`/`features_train_csv`/
  `fold_coldata_csv`; `features_as_rows=True` takes the usual genes x samples
  counts CSV. No tunable args, no `dt`/`tfs`/`pathways`, no ARACNE (all
  rejected up front). CellTFusion-server exposes it standalone
  (`prepare_celltfusion_folds`, outcome appended via `add_columns`) and
  `project_test_factors` as `project_celltfusion_factors` (from a
  train_ml_model result's `Custom_output`, or a run_celltfusion result) --
  the projection a test cohort needs before `predict_ml_model`. Each fold is a
  full CellTFusion run, hence the 8 h timeout for custom-fold training.
- RaCInG exports `computeLRContributions`/`computeWedgeLRContributions`/
  `computeTriangleLRContributions` (wrapped as `explain_lr_contributions`,
  fed from `prepare_racing_inputs`' result.rds). `compute_racing_kernel`/
  `compute_racing_montecarlo` default `output_folder` to a hardcoded
  `~/Documents/racing/vignettes/`, so the server always passes `"Results/"`.
  A user-supplied `deconv` is fit as ONE composition when estimating
  expression profiles (only the internally computed one is fit per
  method/signature block).

If a package is reinstalled to a version predating these changes, these
tools would break again (missing/unused-argument errors) -- see "R package
installation" above for why that can happen silently.
