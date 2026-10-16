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
# a plain character vector), "rds" (readRDS()'d as-is), "rds_element" (one
# named element of a .rds file's top-level list, e.g. a prior tool's
# result.rds$Model), "rds_element_rownames" (that element's row names, e.g.
# a prior RaCInG input's patient labels), "r_named_function" (a function exported by an installed
# sister package, looked up by name -- e.g. multideconv::prepare_multideconv_folds,
# for a parameter that expects an actual R function object), "named_args" (a
# named list of heterogeneously-typed values, e.g. for pipeML's
# fold_construction_args_fixed, meant to be spliced into another call via
# do.call() rather than passed as one argument; an entry may itself be a
# typed spec, resolved first). A "csv" spec can also be transposed, reordered
# to another CSV's samples ("align_to") and extended with "add_columns";
# "csv_column" can be aligned the same way. See common/r_bridge.py for the
# Python-side constructors.
#
# "package" is an installed R package; "function" is looked up in it via
# asNamespace(). All five servers in this repo (multideconv, pipeML,
# CellTFusion, RaCInG, BioinfoTools) run this way.
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

# multideconv's .onLoad no longer creates this in non-interactive sessions
# (Rscript), and BioinfoTools/CellTFusion functions write to "Results/..."
# directly -- create it up front, unconditionally, so every package's
# functions can rely on it existing regardless of which one this call targets.
dir.create(file.path(workdir, "Results"), recursive = TRUE, showWarnings = FALSE)

is_scalarish <- function(v) is.character(v) || is.numeric(v) || is.logical(v) || is.null(v)

# Sample IDs of a CSV: its row names, or its column names (e.g. a genes x samples counts CSV)
csv_sample_ids <- function(path, columns = FALSE) {
  if (isTRUE(columns)) {
    return(colnames(utils::read.csv(path, row.names = 1, check.names = FALSE, nrows = 1)))
  }
  rownames(utils::read.csv(path, row.names = 1, check.names = FALSE))
}

# Positions of `ids` within `have` (the samples of `from`), for following another CSV's sample
# order. Falls back to the given order when some IDs are missing but both have the same number
# of samples (the old positional contract); errors when neither is possible.
match_samples <- function(have, ids, from, to, what) {
  missing_ids <- setdiff(ids, have)
  if (length(missing_ids) == 0) return(match(ids, have))
  if (length(have) == length(ids)) return(seq_along(have))
  stop(length(missing_ids), " sample(s) of ", basename(to), " are missing from ", basename(from),
       " (e.g. ", paste(utils::head(missing_ids, 5), collapse = ", "),
       "), and the two files have different numbers of samples, so ", what,
       " cannot be matched to them")
}

resolve_arg <- function(spec) {
  if (is.null(spec)) return(NULL)
  type <- spec[["__type__"]]
  if (is.null(type)) type <- "literal"

  if (type == "literal") {
    val <- spec$value
    if (is.list(val)) {
      scalar <- all(vapply(val, is_scalarish, logical(1)))
      if (scalar) val <- unlist(val)
    }
    return(val)
  }

  if (type == "csv") {
    row_names <- spec$row_names
    if (is.null(row_names)) row_names <- NULL else row_names <- as.integer(row_names)
    df <- utils::read.csv(spec$path, row.names = row_names, check.names = FALSE)
    if (isTRUE(spec$transpose)) df <- as.data.frame(t(as.matrix(df)), optional = TRUE)
    if (!is.null(spec$align_to)) {
      ids <- csv_sample_ids(spec$align_to, spec$align_to_columns)
      df <- df[match_samples(rownames(df), ids, spec$path, spec$align_to, "its rows"), , drop = FALSE]
    }
    for (nm in names(spec$add_columns)) {
      values <- resolve_arg(spec$add_columns[[nm]])
      if (length(values) != nrow(df)) {
        stop("Column '", nm, "' has ", length(values), " values for ", nrow(df), " samples of ", basename(spec$path))
      }
      df[[nm]] <- unname(values)
    }
    if (isTRUE(spec$as_matrix)) df <- as.matrix(df)
    return(df)
  }

  if (type == "csv_column") {
    row_names <- spec$row_names
    if (is.null(row_names)) row_names <- 1L else row_names <- as.integer(row_names)
    df <- utils::read.csv(spec$path, row.names = row_names, check.names = FALSE)
    if (!spec$column %in% colnames(df)) {
      stop("Column '", spec$column, "' not found in ", spec$path)
    }
    col <- df[[spec$column]]
    if (isTRUE(spec$as_character)) col <- as.character(col)
    names(col) <- rownames(df)
    # Optionally follow another CSV's sample order instead of this one's
    if (!is.null(spec$align_to)) {
      ids <- csv_sample_ids(spec$align_to, spec$align_to_columns)
      col <- col[match_samples(names(col), ids, spec$path, spec$align_to, paste0("'", spec$column, "'"))]
    }
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

  if (type == "rds_element") {
    # One name, or a path of names into nested lists
    x <- readRDS(spec$path)
    for (el in unlist(spec$element)) {
      if (is.null(x[[el]])) stop("Element '", el, "' is missing or NULL in ", basename(spec$path))
      x <- x[[el]]
    }
    return(x)
  }

  if (type == "rds_element_rownames") {
    return(rownames(readRDS(spec$path)[[spec$element]]))
  }

  if (type == "r_function") {
    return(get(spec$value, mode = "function", envir = baseenv()))
  }

  if (type == "r_named_function") {
    return(get(spec$value, mode = "function", envir = asNamespace(spec$package)))
  }

  if (type == "named_args") {
    val <- spec$value
    if (is.null(val)) return(list())
    val <- lapply(val, function(v) {
      # A nested arg spec (e.g. a csv_arg for a data-valued argument) is resolved into its R object
      if (is.list(v) && !is.null(v[["__type__"]])) return(resolve_arg(v))
      if (is.list(v)) {
        scalar <- all(vapply(v, is_scalarish, logical(1)))
        if (scalar) v <- unlist(v)
      }
      v
    })
    return(val)
  }

  stop("Unknown arg type: ", type)
}

raw_args <- manifest$args
call_args <- list()
for (nm in names(raw_args)) {
  call_args[[nm]] <- resolve_arg(raw_args[[nm]])
}

# Install-if-missing only -- this never reinstalls/updates an already-installed
# package, so a newer push to one of these repos requires manually reinstalling
# it yourself; see this project's README.
ensure_installed <- function(pkg, installer) {
  if (requireNamespace(pkg, quietly = TRUE)) return(invisible(NULL))
  installer_pkg <- if (identical(pkg, "RaCInG")) "remotes" else "pak"
  if (!requireNamespace(installer_pkg, quietly = TRUE)) {
    message("'", installer_pkg, "' not found -- installing it from CRAN first...")
    utils::install.packages(installer_pkg, repos = "https://cloud.r-project.org")
  }
  message("Package '", pkg, "' not found -- installing from GitHub (this only happens once)...")
  installer()
}

# These four packages call into each other's exported functions without an
# explicit `pkg::fun()` prefix (e.g. CellTFusion() calls
# multideconv::compute.deconvolution() as a bare compute.deconvolution()), which
# only resolves if that sister package is attached, not merely installed. Attach
# all four regardless of which one is this server's target package.
sister_install_cmds <- list(
  multideconv = function() pak::pkg_install("VeraPancaldiLab/multideconv"),
  pipeML = function() pak::pkg_install("VeraPancaldiLab/pipeML"),
  CellTFusion = function() pak::pkg_install("VeraPancaldiLab/CellTFusion"),
  RaCInG = function() remotes::install_github("VeraPancaldiLab/RaCInG_package")
)
for (sister_pkg in names(sister_install_cmds)) {
  ensure_installed(sister_pkg, sister_install_cmds[[sister_pkg]])
  suppressMessages(library(sister_pkg, character.only = TRUE))
}

# BioinfoTools is a standalone package -- none of the above call into it, nor
# it into them -- so it's only installed/attached when it's actually this
# call's target, rather than unconditionally like the four above.
if (identical(manifest$package, "BioinfoTools")) {
  ensure_installed("BioinfoTools", function() pak::pkg_install("mhurtado13/Bioinfo-tools"))
  suppressMessages(library(BioinfoTools))
}

fun <- get(manifest$`function`, envir = asNamespace(manifest$package))

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
    # Small atomic attributes a function attaches on purpose (e.g. pipeML SHAP's "baseline")
    extra <- setdiff(names(attributes(x)), c("names", "row.names", "class", "dim", "dimnames"))
    extra <- Filter(function(a) is.atomic(attr(x, a)) && length(attr(x, a)) <= 20, extra)
    if (length(extra) > 0) info$attributes <- lapply(stats::setNames(extra, extra), function(a) attr(x, a))
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
      } else if (identical(class(el), "list") && length(el) > 0 && !is.null(names(el)) &&
                 all(vapply(el, function(z) is.data.frame(z) || is.matrix(z), logical(1)))) {
        # A plain named list holding only tables (e.g. multideconv's per-cell-type
        # correlations/pvalues): one CSV per table, keyed "<element>/<table>"
        for (sub in names(el)) {
          fname <- paste0("result_", make.names(nm), "_", make.names(sub), ".csv")
          write_if_tabular(el[[sub]], file.path(workdir, fname))
          csv_outputs[[paste0(nm, "/", sub)]] <- fname
        }
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
