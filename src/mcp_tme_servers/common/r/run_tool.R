#!/usr/bin/env Rscript
#
# Generic dispatcher used by every tool on this server.
#
# Usage: Rscript run_tool.R <manifest.json>
#
# manifest.json:
#   {
#     "package": "multideconv",
#     "function": "compute.deconvolution",
#     "workdir": "/abs/path/to/run/dir",
#     "args": {
#       "raw.counts": {"__type__": "csv", "path": "...", "row_names": 1, "as_matrix": true},
#       "methods":    {"__type__": "literal", "value": ["Quantiseq", "Epidish"]},
#       "fun_LR":     {"__type__": "r_function", "value": "min"}
#     }
#   }
#
# Other arg "__type__"s: "csv_column" (one named column, as a named vector),
# "csv_list" (a CSV wrapped as list(df)), "csv_rownames" (a CSV's row names as
# a plain character vector), "rds" (readRDS()'d as-is). See common/r_bridge.py
# for the Python-side constructors.
#
# Exactly one of "package" (an installed R package, function looked up via
# asNamespace()) or "source" (a standalone .r script, source()'d into its own
# environment -- for Bioinfo-tools, which isn't an installed package) must be
# given.
#
# Writes workdir/result.rds (full return value), workdir/result_summary.json,
# and any CSVs for tabular elements of the result. The function itself is run
# with workdir as the working directory, so its own relative "Results/..."
# writes land under workdir/Results as usual.
#
# On stdout, between markers, a JSON blob describing what happened -- this is
# what the Python side parses.

suppressMessages(library(jsonlite))

argv <- commandArgs(trailingOnly = TRUE)
if (length(argv) < 1) stop("usage: run_tool.R <manifest.json>")
manifest <- fromJSON(argv[[1]], simplifyVector = FALSE)

workdir <- manifest$workdir
dir.create(workdir, recursive = TRUE, showWarnings = FALSE)
setwd(workdir)

resolve_arg <- function(spec) {
  if (is.null(spec)) return(NULL)
  type <- spec[["__type__"]]
  if (is.null(type)) type <- "literal"

  if (type == "literal") {
    val <- spec$value
    if (is.list(val)) {
      scalar <- all(vapply(val, function(v) is.character(v) || is.numeric(v) || is.logical(v) || is.null(v), logical(1)))
      if (scalar) val <- unlist(val)
    }
    return(val)
  }

  if (type == "csv") {
    row_names <- spec$row_names
    if (is.null(row_names)) row_names <- NULL else row_names <- as.integer(row_names)
    df <- utils::read.csv(spec$path, row.names = row_names, check.names = FALSE)
    if (isTRUE(spec$as_matrix)) df <- as.matrix(df)
    return(df)
  }

  if (type == "csv_column") {
    row_names <- spec$row_names
    if (is.null(row_names)) row_names <- 1L else row_names <- as.integer(row_names)
    df <- utils::read.csv(spec$path, row.names = row_names, check.names = FALSE)
    col <- df[[spec$column]]
    names(col) <- rownames(df)
    return(col)
  }

  if (type == "csv_list") {
    row_names <- spec$row_names
    if (is.null(row_names)) row_names <- NULL else row_names <- as.integer(row_names)
    df <- utils::read.csv(spec$path, row.names = row_names, check.names = FALSE)
    if (isTRUE(spec$as_matrix)) df <- as.matrix(df)
    return(list(df))
  }

  if (type == "csv_rownames") {
    df <- utils::read.csv(spec$path, row.names = 1, check.names = FALSE)
    return(rownames(df))
  }

  if (type == "rds") {
    return(readRDS(spec$path))
  }

  if (type == "r_function") {
    return(get(spec$value, mode = "function", envir = baseenv()))
  }

  stop("Unknown arg type: ", type)
}

raw_args <- manifest$args
call_args <- list()
for (nm in names(raw_args)) {
  call_args[[nm]] <- resolve_arg(raw_args[[nm]])
}

if (!is.null(manifest$source)) {
  # Bioinfo-tools isn't an installed package -- source() its script into its
  # own environment and look the function up there. Its functions use %>%
  # without an explicit library(magrittr)/library(dplyr) of their own (every
  # other call is already `pkg::fun()`-qualified), so attach magrittr for that
  # operator. Its functions also expect a "Results/" dir under the working
  # directory to already exist (same convention documented in its own
  # README/usage example), which the other four packages' functions create
  # for themselves internally.
  suppressMessages(library(magrittr))
  dir.create(file.path(workdir, "Results"), recursive = TRUE, showWarnings = FALSE)
  fn_env <- new.env(parent = globalenv())
  sys.source(manifest$source, envir = fn_env)
  fun <- get(manifest$`function`, mode = "function", envir = fn_env)
} else {
  # These four packages call into each other's exported functions without an
  # explicit `pkg::fun()` prefix (e.g. CellTFusion() calls
  # multideconv::compute.deconvolution() as a bare compute.deconvolution()), which
  # only resolves if that sister package is attached, not merely installed. Attach
  # all four regardless of which one is this server's target package.
  for (sister_pkg in c("multideconv", "pipeML", "CellTFusion", "RaCInG")) {
    suppressMessages(library(sister_pkg, character.only = TRUE))
  }
  fun <- get(manifest$`function`, envir = asNamespace(manifest$package))
}

is_error <- FALSE
error_msg <- NULL
result <- tryCatch(
  do.call(fun, call_args),
  error = function(e) {
    is_error <<- TRUE
    error_msg <<- conditionMessage(e)
    NULL
  }
)

summarize <- function(x, depth = 0) {
  if (is.null(x)) return(list(class = "NULL"))
  cls <- class(x)[1]
  if (is.data.frame(x) || is.matrix(x)) {
    info <- list(
      class = cls,
      dim = as.list(dim(x)),
      colnames = utils::head(colnames(x), 25),
      rownames = utils::head(rownames(x), 10)
    )
    numeric_vals <- suppressWarnings(as.numeric(as.matrix(x)))
    if (any(!is.na(numeric_vals))) {
      info$numeric_summary <- list(
        min = min(numeric_vals, na.rm = TRUE),
        mean = mean(numeric_vals, na.rm = TRUE),
        max = max(numeric_vals, na.rm = TRUE),
        n_na = sum(is.na(numeric_vals))
      )
    }
    return(info)
  }
  if (is.list(x) && depth < 1) {
    return(list(
      class = cls,
      names = names(x),
      elements = lapply(x, summarize, depth = depth + 1)
    ))
  }
  if (is.character(x) || is.numeric(x) || is.logical(x)) {
    return(list(class = cls, length = length(x), preview = utils::head(x, 20)))
  }
  list(class = cls)
}

write_if_tabular <- function(x, path) {
  if (is.data.frame(x) || is.matrix(x)) {
    utils::write.csv(as.data.frame(x), path, row.names = TRUE)
    return(TRUE)
  }
  FALSE
}

csv_outputs <- list()
if (!is_error) {
  saveRDS(result, file.path(workdir, "result.rds"))
  if (is.data.frame(result) || is.matrix(result)) {
    write_if_tabular(result, file.path(workdir, "result.csv"))
    csv_outputs[["result"]] <- "result.csv"
  } else if (is.list(result)) {
    for (nm in names(result)) {
      el <- result[[nm]]
      if (is.data.frame(el) || is.matrix(el)) {
        fname <- paste0("result_", make.names(nm), ".csv")
        write_if_tabular(el, file.path(workdir, fname))
        csv_outputs[[nm]] <- fname
      }
    }
  }
}

output_files <- as.list(list.files(workdir, recursive = TRUE))

out <- list(
  ok = !is_error,
  error = error_msg,
  summary = if (is_error) list() else summarize(result),
  csv_outputs = csv_outputs,
  output_files = output_files
)

out_json <- jsonlite::toJSON(out, auto_unbox = TRUE, null = "null", force = TRUE)
writeLines(out_json, file.path(workdir, "result_summary.json"))

cat("\n===TME_RESULT_JSON_START===\n")
cat(out_json)
cat("\n===TME_RESULT_JSON_END===\n")
