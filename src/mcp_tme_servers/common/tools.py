"""Shared tool-building blocks used by all five servers in this repo: the
generic `_run()` dispatcher wrapper, its result model, and the three
session-management tools (registered once per server via
`register_session_tools`).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from mcp_tme_servers.common.r_bridge import RToolError, run_r_function
from mcp_tme_servers.common.session import Session, SessionConfig, SessionError

READ_ANNOTATIONS = ToolAnnotations(read_only_hint=False, open_world_hint=False, destructive_hint=False)
NETWORK_ANNOTATIONS = ToolAnnotations(read_only_hint=False, open_world_hint=True, destructive_hint=False)
_SESSION_ANNOTATIONS = ToolAnnotations(read_only_hint=False, open_world_hint=False, destructive_hint=False)

_SECRET_KEY_MARKERS = ("token", "password", "secret", "credential")

# One rule for every server that takes an expression matrix, so the model is told the same thing
# everywhere: verify the scale first, deconvolution on linear TPM, everything else on log2(TPM + 1).
EXPRESSION_SCALE_INSTRUCTIONS = (
    "IMPORTANT -- expression scale: verify what an expression matrix holds BEFORE running any tool on it, "
    "every time, even when the file name seems to say it. Look at the values: raw counts (non-negative "
    "integers, sample totals of millions that differ between samples), linear TPM (non-negative, not "
    "integers, large values, each sample summing to about 1e6 when all genes are present), log-transformed "
    "values such as log2(TPM + 1) (small range, usually no value above ~20), or another normalization "
    "(CPM, FPKM/RPKM, VST, z-scores -- negative values mean centred or z-scored data). If you cannot tell "
    "with confidence, ask the user to confirm the scale instead of guessing. Then follow these rules: "
    "(1) cell-type deconvolution is ALWAYS run on linear TPM (or on raw counts that the tool itself "
    "normalizes to TPM) and never on log-transformed values -- a log2(TPM + 1) matrix must be converted "
    "back first (2^x - 1); "
    "(2) every other expression-based feature -- transcription-factor activity, pathway activity, "
    "GSEA/enrichment scores, PCA of expression -- uses log2(TPM + 1); "
    "(3) if linear TPM cannot be obtained (only z-scores, VST or another normalization that cannot be "
    "converted back), do not run deconvolution silently: tell the user that it will be run on that other "
    "type of normalization instead of TPM. Tell the user which scale each step received. "
)

# Rule shared by every server, appended to each server's instructions.
SCOPE_INSTRUCTIONS = (
    "IMPORTANT -- stay within what was asked: do exactly the analysis the user requested and nothing more. "
    "Do not add extra analyses, tests or outputs that the user did not ask for (for example association "
    "with response or other clinical variables, marker-gene validation, survival or statistical tests, "
    "additional plots or extra classifications). If you think an additional analysis would help, propose it "
    "in one sentence and wait for the user's answer before running it. "
)

DEFAULTS_INSTRUCTIONS = (
    "IMPORTANT -- default values: run every tool with its default parameter values unless the user "
    "explicitly asks for specific ones. Do not change a parameter on your own initiative (for example a "
    "threshold, the number of folds or repetitions, a seed, the list of methods or signatures, a "
    "normalization or a filter). If you think another value would be better for this data, do not apply "
    "it: tell the user which parameter, which value and why, and ask before deciding. Parameters that the "
    "data itself requires (for example telling a tool whether the expression is already normalized TPM, "
    "file paths or output folders) are not a choice and must be set correctly. "
)


def _redact_secrets(args: dict[str, dict]) -> dict[str, dict]:
    """Copy of `args` with any credential-looking literal value masked out.

    The session log (session_log.jsonl) is a durable, exportable artifact --
    unlike the ephemeral run manifest.json a tool call needs to actually
    execute, so any r_param whose name suggests a secret (token, password,
    credential, ...) gets its literal value replaced before it's recorded.
    """
    redacted = {}
    for key, spec in args.items():
        if (
            isinstance(spec, dict)
            and spec.get("__type__") == "literal"
            and any(marker in key.lower() for marker in _SECRET_KEY_MARKERS)
        ):
            redacted[key] = {"__type__": "literal", "value": "***REDACTED***"}
        else:
            redacted[key] = spec
    return redacted


class RunResult(BaseModel):
    """Generic result of one R analysis run.

    `summary` mirrors the R return value's own structure (its top-level names,
    each element's class/dimensions/preview) rather than a fixed schema, since
    different tools on this server return different shapes. `field_glossary`
    is what makes that self-describing: where available, it maps each name in
    `summary` to a plain-English explanation of what that piece of the result
    biologically means -- read it before interpreting `summary`'s values,
    rather than guessing from the name alone.
    """

    run_dir: str = Field(description="Directory with result.rds, any CSV/PDF outputs, and run.log")
    summary: dict[str, Any] = Field(description="Structural summary of the R return value (class, dim, names) -- see field_glossary for what each name means")
    field_glossary: dict[str, str] = Field(
        default_factory=dict,
        description="For each name in summary that this server documents: a plain-English explanation of its biological/methodological meaning, not just its R type.",
    )
    csv_outputs: dict[str, str] = Field(description="R-element name -> CSV filename written under run_dir, for tabular elements")
    output_files: list[str] = Field(description="All files present under run_dir after the run, relative paths")
    warnings: list[str] = Field(
        default_factory=list,
        description="Methodological flags noticed about this specific result (e.g. weak cross-method agreement, very few samples, near-empty output) with a suggested next step. Not fatal -- the call still succeeded -- but worth reading before trusting the result at face value.",
    )
    log_tail: str = Field(description="Last lines of R console output, for diagnosing warnings or slow steps")
    session_id: str | None = Field(default=None, description="Workflow session this call was recorded under, if any (see start_workflow_session)")


OutputDir = Annotated[
    str | None,
    Field(
        description=(
            "Directory to write results into (created if needed). Defaults to a fresh "
            "timestamped directory, or -- if session_id is given -- a directory "
            "namespaced under that session's own cohort. Reusing the same directory "
            "across calls is fine for the underlying R function's own file_name-based "
            "outputs, but this tool's own result.rds/result.csv will be overwritten by "
            "the next call that reuses the same directory -- chain off a run's "
            "result.rds before reusing its output_dir for another call."
        )
    ),
]

SessionIdParam = Annotated[
    str | None,
    Field(
        description=(
            "Session id from start_workflow_session, to track this call as part of one "
            "cohort's multi-step workflow. Effects: (1) this call's output is namespaced "
            "under the session's own cohort directory unless output_dir is also given "
            "explicitly, (2) the call (tool, parameters, timestamp, success/failure) is "
            "appended to the session's reproducibility log, (3) this tool's pipeline "
            "stage is marked complete on success, and (4) chaining in a *_rds/*_csv "
            "argument that belongs to a *different* session is refused, since that is "
            "almost always an accidental cross-cohort mix rather than a deliberate "
            "choice. When running the same kind of analysis across several cohorts in "
            "one conversation, start one session per cohort and keep each cohort's "
            "calls on its own session_id -- this is what keeps their results from "
            "blending together."
        )
    ),
]


def _spec_paths(spec: Any) -> list[str]:
    """Every file path an arg spec reads, including specs nested in named_args values or
    csv_arg's add_columns -- so the cross-session check also covers those."""
    if isinstance(spec, list):
        return [p for item in spec for p in _spec_paths(item)]
    if not isinstance(spec, dict):
        return []
    paths = [spec[key] for key in ("path", "align_to") if isinstance(spec.get(key), str) and "__type__" in spec]
    for value in spec.values():
        if isinstance(value, (dict, list)):
            paths.extend(_spec_paths(value))
    return paths


def resolve_output_dir(
    *,
    runs_root: Path,
    tool_name: str,
    output_dir: str | None,
    session_id: str | None,
    session_config: SessionConfig,
) -> str:
    """The run directory a call will use, resolved before it runs -- for tools that must stage
    files the R function reads from a fixed relative path (e.g. multideconv's
    Results/custom_signatures/, CellTFusion's input/ARACNE/). Pass the result as that call's
    output_dir so `_run` uses the same directory.
    """
    if output_dir is not None:
        resolved = output_dir
    elif session_id is not None:
        try:
            session = Session.load(session_config, session_id)
        except SessionError as e:
            raise ToolError(str(e)) from e
        resolved = str(session.run_dir_for(tool_name))
    else:
        resolved = str(runs_root / f"{tool_name}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}")
    return str(Path(resolved).expanduser().resolve())


def make_run(*, runs_root: Path) -> Callable:
    """Build this server's `_run()` dispatcher, bound to its own runs_root."""

    async def _run(
        *,
        package: str,
        function: str,
        args: dict[str, dict],
        tool_name: str,
        ctx: Context,
        session_config: SessionConfig,
        timeout: float = 1800.0,
        output_dir: str | None = None,
        session_id: str | None = None,
        glossary: dict[str, str] | None = None,
        diagnostics: Callable[[dict], list[str]] | None = None,
    ) -> RunResult:
        session: Session | None = None
        if session_id is not None:
            try:
                session = Session.load(session_config, session_id)
                for arg_name, spec in args.items():
                    for path in _spec_paths(spec):
                        session.check_not_foreign(arg_name, path)
            except SessionError as e:
                raise ToolError(str(e)) from e
            if output_dir is None:
                output_dir = str(session.run_dir_for(tool_name))

        try:
            result = await run_r_function(
                package=package,
                function=function,
                args=args,
                tool_name=tool_name,
                runs_root=runs_root,
                ctx=ctx,
                timeout=timeout,
                output_dir=output_dir,
            )
        except RToolError as e:
            if session is not None:
                session.record_step(tool=tool_name, args=_redact_secrets(args), status="failed", run_dir=output_dir, error=str(e))
            raise ToolError(f"{e}\n\n--- R log tail ---\n{e.log_tail}") from e

        if session is not None:
            session.record_step(tool=tool_name, args=_redact_secrets(args), status="success", run_dir=result["run_dir"])

        # jsonlite serializes an empty R named list as `[]`, not `{}` -- normalize
        # back to a dict here rather than trying to fight that on the R side.
        summary = result.get("summary", {})
        summary = summary if isinstance(summary, dict) else {}
        csv_outputs = result.get("csv_outputs", {})
        csv_outputs = csv_outputs if isinstance(csv_outputs, dict) else {}

        field_glossary = {}
        if glossary:
            field_glossary = {name: glossary[name] for name in (summary.get("names") or []) if name in glossary}

        return RunResult(
            run_dir=result["run_dir"],
            summary=summary,
            field_glossary=field_glossary,
            csv_outputs=csv_outputs,
            output_files=result.get("output_files", []),
            warnings=diagnostics(summary) if diagnostics else [],
            log_tail=result.get("log_tail", ""),
            session_id=session_id,
        )

    return _run


class StepRecord(BaseModel):
    timestamp: str
    tool: str
    stage: str = Field(description="Pipeline stage this tool represents")
    status: Literal["success", "failed"]
    run_dir: str | None
    error: str | None = None


class SessionInfo(BaseModel):
    session_id: str
    cohort_id: str = Field(description="The dataset/cohort this session is scoped to")
    description: str | None
    created_at: str
    stages_completed: list[str] = Field(description="Pipeline stages that have completed successfully at least once")
    export_ready: bool = Field(description="Whether export_session_results would currently succeed")
    missing_for_export: list[str] = Field(description="Stages still needed before export_session_results will succeed")
    steps: list[StepRecord] = Field(description="Every successful call recorded under this session, in order")


class ExportResult(BaseModel):
    export_dir: str
    files_copied: int
    session_log: str = Field(description="Path to the exported session_log.jsonl -- the full reproducibility transcript (every call, its parameters, and its outcome)")


def register_session_tools(mcp: MCPServer, session_config: SessionConfig, *, domain: str) -> None:
    """Register start_workflow_session / get_session_status / export_session_results
    on `mcp`, bound to `session_config`. `domain` is a short phrase (e.g. "cell-type
    deconvolution") used only in tool docstrings."""

    # Registered imperatively (mcp.add_tool) rather than via @mcp.tool(...) so
    # start_workflow_session's docstring can interpolate `domain` -- an f-string
    # isn't captured as __doc__ by Python, and setting .__doc__ after a
    # decorator has already run is too late, so the function is fully built
    # (docstring included) before it's handed to the server.

    async def start_workflow_session(
        cohort_id: Annotated[
            str,
            Field(
                description=(
                    "A short, stable identifier for the dataset/cohort this session tracks, "
                    "e.g. 'LUAD_batch1' or 'TCGA-SKCM-discovery'. Every call you route through "
                    "this session (via its session_id) is assumed to belong to this one cohort."
                )
            ),
        ],
        description: Annotated[str | None, Field(description="Optional free-text note on what this session/analysis is for")] = None,
    ) -> SessionInfo:
        session = Session.create(session_config, cohort_id=cohort_id, description=description)
        return SessionInfo(**session.status())

    start_workflow_session.__doc__ = (
        f"Start tracking a multi-step {domain} workflow for one cohort.\n\n"
        "Call this once per cohort/dataset before running the analysis tools, then pass "
        "the returned session_id to each tool call. Doing so gives you three things a "
        "bare tool call doesn't: (1) that call's output is namespaced under this "
        "session's own cohort-labeled directory instead of an anonymous timestamped "
        "one, (2) the call is appended to this session's reproducibility log (tool, "
        "parameters, timestamp, success/failure) -- exportable later as a full audit "
        "trail of the analysis, and (3) chaining a result from a *different* session "
        "into this one is refused, which is what prevents accidentally mixing two "
        "cohorts' intermediate results together in the same workflow.\n\n"
        "If you're analyzing several cohorts in one conversation, start one session per "
        "cohort and keep each cohort's tool calls on its own session_id -- that is the "
        "mechanism that keeps them from blending together."
    )
    mcp.add_tool(start_workflow_session, name="start_workflow_session", annotations=_SESSION_ANNOTATIONS)

    async def get_session_status(
        session_id: Annotated[str, Field(description="Session id returned by start_workflow_session")],
    ) -> SessionInfo:
        """Check a workflow session's progress: which pipeline stages have completed,
        the full step-by-step call log, and whether it's ready to export.

        Use this before export_session_results to see what's missing, or any time you've
        lost track of what's already been run for a cohort in a long conversation.
        """
        try:
            session = Session.load(session_config, session_id)
        except SessionError as e:
            raise ToolError(str(e)) from e
        return SessionInfo(**session.status())

    mcp.add_tool(get_session_status, name="get_session_status", annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))

    async def export_session_results(
        session_id: Annotated[str, Field(description="Session id returned by start_workflow_session")],
        output_dir: Annotated[str, Field(description="Directory to export into (created if needed); the session's contents are copied into a subdirectory named after session_id")],
    ) -> ExportResult:
        """Bundle a session's results and reproducibility log into one directory.

        Copies every tool call's output (result.rds/CSVs/plots) plus session_log.jsonl
        (the full transcript: every call's tool, parameters, timestamp, and outcome) and
        session.json (the stage/status summary) into output_dir/<session_id>/.

        Refuses if the session hasn't completed the minimum stages needed for the
        export to mean anything -- check get_session_status first if you're not sure
        why this refused.
        """
        try:
            session = Session.load(session_config, session_id)
        except SessionError as e:
            raise ToolError(str(e)) from e
        state = session.status()
        if not state["export_ready"]:
            raise ToolError(
                f"Session '{session_id}' isn't ready to export -- missing stage(s): "
                f"{', '.join(state['missing_for_export'])}. "
                f"{session_config.explain_missing_stages(state['missing_for_export'])}"
            )
        dest, n_files = session.export(output_dir)
        return ExportResult(export_dir=str(dest), files_copied=n_files, session_log=str(dest / "session_log.jsonl"))

    mcp.add_tool(export_session_results, name="export_session_results", annotations=_SESSION_ANNOTATIONS)
