"""Subprocess bridge from Python MCP tools to R package functions. Shared by
all five servers in this repo -- one dispatcher, one marshaling contract.

Each call shells out to `common/r/run_tool.R`, a generic dispatcher that
loads one package, calls one exported function with JSON-described
arguments, and writes the (possibly large) result to disk rather than trying
to marshal it back through JSON. See `r/run_tool.R` for the manifest format.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

R_RUNNER = Path(__file__).resolve().parent / "r" / "run_tool.R"

_START_MARKER = "===TME_RESULT_JSON_START==="
_END_MARKER = "===TME_RESULT_JSON_END==="

_PROGRESS_INTERVAL_S = 5.0


class RToolError(Exception):
    def __init__(self, message: str, log_tail: str = ""):
        super().__init__(message)
        self.log_tail = log_tail


def _abs(path: str) -> str:
    """Resolve a user-given path relative to the server process's cwd.

    Required because run_tool.R calls setwd(workdir) before reading any file
    arguments, so a relative path in the manifest would otherwise resolve
    against the run directory instead of wherever the caller meant it.
    """
    return str(Path(path).expanduser().resolve())


def csv_arg(
    path: str,
    row_names: int | None = 1,
    as_matrix: bool = False,
    transpose: bool = False,
    align_to: str | None = None,
    align_to_columns: bool = False,
    add_columns: dict[str, dict] | None = None,
) -> dict:
    """A CSV file, read as a data.frame (optionally coerced to a matrix).

    Applied in this order: `transpose` (e.g. a genes x samples counts CSV read as samples x
    genes); `align_to`, which reorders the rows to follow another CSV's sample IDs (its row
    names, or its column names with align_to_columns=True) with the same fallback rules as
    csv_column_arg; `add_columns`, a {name: csv_column_arg(...)} dict of columns appended to
    the data frame (build them with align_to=path, align_to_columns=transpose so they follow
    this CSV's samples); then `as_matrix`.
    """
    spec: dict[str, Any] = {"__type__": "csv", "path": _abs(path), "row_names": row_names, "as_matrix": as_matrix}
    if transpose:
        spec["transpose"] = True
    if align_to is not None:
        spec["align_to"] = _abs(align_to)
        spec["align_to_columns"] = align_to_columns
    if add_columns:
        spec["add_columns"] = add_columns
    return spec


def csv_column_arg(
    path: str,
    column: str,
    row_names: int | None = 1,
    align_to: str | None = None,
    align_to_columns: bool = False,
    as_character: bool = False,
) -> dict:
    """A single named column of a CSV file, returned as an R vector.

    as_character=True returns it as a character vector whatever its content -- for a
    grouping column (e.g. a batch coded 1/2/3) that R must treat as categorical, not numeric.

    With align_to (another CSV, e.g. the feature matrix this column annotates), the column
    is reordered to follow align_to's sample IDs -- its row names, or its column names with
    align_to_columns=True (e.g. a genes x samples counts CSV) -- so a coldata/target CSV
    doesn't have to share the feature CSV's exact samples and order. If some of those IDs
    are missing from this CSV, it falls back to the old positional contract when both have
    the same number of samples, and errors (naming the missing IDs) otherwise.
    """
    spec: dict[str, Any] = {"__type__": "csv_column", "path": _abs(path), "column": column, "row_names": row_names}
    if as_character:
        spec["as_character"] = True
    if align_to is not None:
        spec["align_to"] = _abs(align_to)
        spec["align_to_columns"] = align_to_columns
    return spec


def csv_list_arg(path: str, row_names: int | None = 1, as_matrix: bool = False) -> dict:
    """A CSV file read as above, then wrapped in a one-element R list: list(df).

    Some Bioinfo-tools functions (the scores.*.test() family) index their `scores`
    argument as scores[[1]] rather than accepting a bare matrix/data.frame directly
    (unlike scores.stat.analysis.continuous(), which self-wraps). This produces the
    shape those functions expect from a plain samples x features CSV.
    """
    return {"__type__": "csv_list", "path": _abs(path), "row_names": row_names, "as_matrix": as_matrix}


def csv_rownames_arg(path: str) -> dict:
    """A CSV's row names, read back as a plain character vector -- e.g. a DEG result's
    gene symbols for compute.enrichment()'s `genes` (ORA) or `universe` argument.
    """
    return {"__type__": "csv_rownames", "path": _abs(path)}


def rds_arg(path: str) -> dict:
    """A .rds file, read back with readRDS() (preserves full R object structure)."""
    return {"__type__": "rds", "path": _abs(path)}


def rds_element_arg(path: str, element: str | list[str]) -> dict:
    """One named element of a .rds file's top-level list, e.g. readRDS(path)[["Model"]] --
    for a downstream R function (like pipeML's compute_prediction/compute_shap_values) that
    expects the inner trained-model object itself, not the wrapping result list a tool's own
    result.rds holds (Model, ML_Models, AUROC_median, ...). A list of names walks down nested
    lists, e.g. ["Custom_output", "Cell_groups"] is readRDS(path)[["Custom_output"]][["Cell_groups"]].
    """
    return {"__type__": "rds_element", "path": _abs(path), "element": element}


def rds_element_rownames_arg(path: str, element: str) -> dict:
    """The row names of one named element of a .rds file's top-level list, e.g.
    rownames(readRDS(path)[["Cmatrix"]]) -- for an argument that wants labels (e.g. RaCInG's
    patient_names) that only exist as a prior tool's matrix row names.
    """
    return {"__type__": "rds_element_rownames", "path": _abs(path), "element": element}


def path_arg(path: str) -> dict:
    """A raw file path, absolutized against the caller's cwd, passed through as a
    literal string -- for an argument the R function reads from disk itself (e.g. a
    3-column ARACNE network.txt read via read.table()), unlike csv_arg/rds_arg, where
    this bridge does the reading.
    """
    return {"__type__": "literal", "value": _abs(path)}


def r_function_arg(name: str) -> dict:
    """A base-R function looked up by name, e.g. "min" or "prod"."""
    return {"__type__": "r_function", "value": name}


def named_r_function_arg(name: str, package: str) -> dict:
    """A function exported by an installed package (not base R), looked up by name --
    e.g. multideconv::prepare_multideconv_folds, for a tool parameter (like pipeML's
    fold_construction_fun) that expects an actual R function object rather than data.

    Since an arbitrary R closure can't cross this JSON-manifest bridge, callers should
    restrict the Python-side parameter to a closed set of known, pre-registered function
    names (e.g. a Literal[...] type) rather than accepting an arbitrary string -- this
    builder only carries the name+package through; it doesn't validate that the name is
    safe to expose.
    """
    return {"__type__": "r_named_function", "value": name, "package": package}


def lit(value: Any) -> dict:
    """A literal scalar, string, bool, number, or flat list of those."""
    return {"__type__": "literal", "value": value}


def named_args_arg(value: dict) -> dict:
    """A named list of heterogeneously-typed arguments (e.g. {"corr": 0.7, "cells_extra":
    ["TypeA"]}) meant to be spliced into another R function call via do.call() -- e.g.
    pipeML's fold_construction_args_fixed/fold_construction_args_tunable, which are handed
    to fold_construction_fun as `do.call(fold_construction_fun, c(list(...), these_args))`.

    Unlike lit(), which -- for a flat, all-scalar dict -- flattens it into a plain R named
    vector (fine when spliced with c(), but wrong for a dict with a vector-valued entry
    like cells_extra, which would stay a nested list instead of becoming a character
    vector), this always resolves to a proper R list() at the top level, normalizing each
    entry independently. An entry may also be another arg spec (e.g. csv_arg(...)), which is
    resolved into its R object first -- for data-valued arguments such as
    CellTFusion::prepare_celltfusion_folds' `deconv`.
    """
    return {"__type__": "named_args", "value": value}


def args_from(**kwargs: Any) -> dict[str, dict]:
    """Build a manifest args dict from {r_param_name: arg_spec}, dropping None specs."""
    return {k: v for k, v in kwargs.items() if v is not None}


async def run_r_function(
    *,
    package: str,
    function: str,
    args: dict[str, dict],
    tool_name: str,
    runs_root: Path,
    ctx: Any = None,
    timeout: float = 1800.0,
    output_dir: str | None = None,
) -> dict:
    """Run one R package function via run_tool.R and return its parsed result dict.

    `package` is an installed R package, with `function` looked up in it via
    asNamespace(). All five servers in this repo run this way (multideconv,
    pipeML, CellTFusion, RaCInG, and BioinfoTools).

    Raises RToolError on a non-zero exit, a timeout, or an R-side error caught
    by run_tool.R's tryCatch (the R error message becomes the exception message).

    `runs_root` is where a fresh auto-named run directory is created when
    output_dir isn't given -- each server passes its own default (see that
    server's server.py), so results from different servers never collide.

    If output_dir is given, results are written there (created if needed) instead
    of a fresh auto-named directory under runs_root. Reusing the same output_dir
    across calls is fine for the R function's own file_name-based outputs (they
    won't collide as long as file_name differs), but this tool's own fixed-name
    files -- manifest.json, result.rds, result.csv / result_<name>.csv -- will be
    overwritten by each subsequent call, so a downstream tool chaining off an
    earlier result.rds must run before the next call reuses that directory.
    """
    if output_dir is not None:
        workdir = Path(output_dir).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        runs_root.mkdir(parents=True, exist_ok=True)
        run_id = f"{tool_name}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
        workdir = runs_root / run_id
        workdir.mkdir(parents=True, exist_ok=True)

    manifest = {"package": package, "function": function, "workdir": str(workdir), "args": args}
    manifest_path = workdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    proc = await asyncio.create_subprocess_exec(
        "Rscript",
        str(R_RUNNER),
        str(manifest_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(workdir),
        limit=1 << 24,  # 16 MiB; R can emit very long single lines (e.g. unwrapped
        # vector dumps, the final JSON blob), well past asyncio's 64 KiB default
    )

    lines: list[str] = []
    start = time.monotonic()
    last_report = 0.0
    timed_out = False

    async def pump() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            lines.append(raw.decode(errors="replace").rstrip("\n"))

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            try:
                await asyncio.wait_for(asyncio.shield(pump_task), timeout=_PROGRESS_INTERVAL_S)
                break
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - start
                if ctx is not None and elapsed - last_report >= _PROGRESS_INTERVAL_S:
                    last_report = elapsed
                    msg = next((l for l in reversed(lines) if l.strip()), "running...")
                    await ctx.report_progress(progress=elapsed, message=msg[:200])
                if elapsed > timeout:
                    timed_out = True
                    proc.kill()
                    await proc.wait()
                    break
        await proc.wait()
    finally:
        if not pump_task.done():
            pump_task.cancel()

    full_log = "\n".join(lines)
    log_tail = "\n".join(lines[-60:])

    if timed_out:
        raise RToolError(f"{tool_name} timed out after {timeout:.0f}s", log_tail)

    if proc.returncode != 0:
        raise RToolError(f"{tool_name}: R process exited with code {proc.returncode}", log_tail)

    if _START_MARKER not in full_log or _END_MARKER not in full_log:
        raise RToolError(f"{tool_name}: R script produced no result JSON", log_tail)

    json_str = full_log.split(_START_MARKER, 1)[1].split(_END_MARKER, 1)[0].strip()
    result = json.loads(json_str)
    result["run_dir"] = str(workdir)
    result["log_tail"] = log_tail

    if not result.get("ok", False):
        raise RToolError(f"{tool_name}: {result.get('error') or 'unknown R error'}", log_tail)

    return result
