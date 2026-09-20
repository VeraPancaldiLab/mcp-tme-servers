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
`rds_element_arg`, `path_arg`, `r_function_arg`, `named_r_function_arg`,
`lit`, `named_args_arg`), matched by a same-named `"__type__"` case in
`run_tool.R`'s `resolve_arg()`. Adding a new kind of argument means adding a
constructor on the Python side and a matching resolver case in R -- both
files must change together.

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

When a wrapped R package has a real bug (e.g. pipeML's
`compute_features.training.ML`/`compute_k_fold_CV` reading `hyperparams`
outside the branch that sets it when `metric` isn't `"AUROC"`/`"AUPRC"`; the
now-deleted `compute_features.ML`'s undeclared `fold_models_dir` reference;
`multideconv::prepare_multideconv_folds()`'s per-fold branch never
`return()`-ing its computed folds), this repo documents the bug in the
tool's own docstring and the README rather than patching the upstream
package -- this repo only wraps upstream code, it doesn't fork it. Search
for "Known upstream issue" and "TEMP" across `server.py`/README when
touching a tool that has one; a `TEMP` marker specifically means a fix is
expected soon (the multideconv `prepare_multideconv_folds` one) and should
be removed once confirmed fixed upstream, not treated as permanent.
