"""Workflow session tracking for multi-step, multi-cohort analyses. Shared by
all four servers in this repo -- each server supplies its own SessionConfig
(its own sessions directory and its own tool/stage/export-readiness mapping).

A session groups a sequence of tool calls against one cohort/dataset so that:
(a) intermediate results are namespaced by cohort, so a conversation working
    across several cohorts can't accidentally mix their outputs together,
(b) every call routed through the session is logged (tool, parameters,
    timestamp, run_dir, success/failure) for reproducibility, and
(c) a workflow can't be exported until the minimum pipeline stages that make
    its output meaningful have actually completed.

State lives on disk under <server>/sessions/<session_id>/, not in server
memory, so it survives server restarts within a conversation and the
directory itself doubles as the exportable audit trail.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SessionConfig:
    sessions_root: Path
    # Maps each tool name to the pipeline stage it represents, in workflow
    # terms rather than raw tool/function names.
    stage_by_tool: dict[str, str]
    # Export readiness is checked as a CNF of ORs: a list of stage-name
    # groups, each of which needs at least one completed member -- every
    # group must be satisfied. A singleton group means that stage is
    # mandatory; a multi-member group means any one of those stages is enough.
    export_requires: list[frozenset[str]]
    stage_explanations: dict[str, str]

    def explain_missing_stages(self, missing: list[str]) -> str:
        """`missing` entries may be a single stage name or "a or b" (an unmet
        OR-group, joined for display) -- try each candidate name against
        stage_explanations."""
        parts = []
        for entry in missing:
            explanation = None
            for candidate in entry.split(" or "):
                candidate = candidate.strip()
                if candidate in self.stage_explanations:
                    explanation = self.stage_explanations[candidate]
                    break
            parts.append(explanation or f"stage '{entry}' hasn't completed yet.")
        return " ".join(parts)


class SessionError(Exception):
    """A session-level problem (not found, cross-session mixing) -- always becomes a ToolError."""


class Session:
    def __init__(self, config: SessionConfig, session_id: str, cohort_id: str, description: str | None):
        self.config = config
        self.session_id = session_id
        self.cohort_id = cohort_id
        self.description = description
        self.dir = config.sessions_root / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "session.json"
        self.log_path = self.dir / "session_log.jsonl"

    @classmethod
    def create(cls, config: SessionConfig, cohort_id: str, description: str | None = None) -> "Session":
        safe_cohort = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in cohort_id.strip()) or "cohort"
        session_id = f"{safe_cohort}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
        session = cls(config, session_id, cohort_id, description)
        session._write_state(
            {
                "session_id": session_id,
                "cohort_id": cohort_id,
                "description": description,
                "created_at": _now(),
                "stages_completed": [],
                "steps": [],
            }
        )
        return session

    @classmethod
    def load(cls, config: SessionConfig, session_id: str) -> "Session":
        state_path = config.sessions_root / session_id / "session.json"
        if not state_path.exists():
            raise SessionError(
                f"No session '{session_id}' found under {config.sessions_root}. "
                "Start one with start_workflow_session, or check get_session_status "
                "wasn't given a typo'd id."
            )
        state = json.loads(state_path.read_text())
        return cls(config, session_id, state["cohort_id"], state.get("description"))

    def _read_state(self) -> dict:
        return json.loads(self.state_path.read_text())

    def _write_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, default=str))

    def run_dir_for(self, tool_name: str) -> Path:
        """A fresh, cohort-namespaced directory for one tool call's output."""
        d = self.dir / self.cohort_id / f"{tool_name}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def owns_path(self, path: str) -> bool:
        try:
            Path(path).expanduser().resolve().relative_to(self.dir.resolve())
            return True
        except ValueError:
            return False

    def check_not_foreign(self, label: str, path: str | None) -> None:
        """Raise if `path` sits inside a *different* session's directory tree.

        This is the cohort-mixing guard: chaining a .rds/CSV that lives under
        another session's sessions/<other_id>/ into this session's call is
        almost always an accidental cross-cohort mix, not a deliberate
        choice, so it's refused rather than silently allowed. Paths outside
        any session tree (e.g. a plain file you pointed at directly) are
        unaffected.
        """
        if not path:
            return
        resolved = Path(path).expanduser().resolve()
        try:
            rel = resolved.relative_to(self.config.sessions_root.resolve())
        except ValueError:
            return
        if not self.owns_path(str(resolved)):
            other = rel.parts[0] if rel.parts else "another session"
            raise SessionError(
                f"{label} ('{path}') belongs to session '{other}', not the current "
                f"session '{self.session_id}' (cohort '{self.cohort_id}'). Chaining a "
                f"result across sessions/cohorts like this is almost always an accidental "
                f"mix-up. If it's deliberate, call the tool without session_id so this "
                f"guard doesn't apply."
            )

    def record_step(
        self,
        *,
        tool: str,
        args: dict,
        status: str,
        run_dir: str | None,
        error: str | None = None,
    ) -> None:
        entry = {
            "timestamp": _now(),
            "tool": tool,
            "stage": self.config.stage_by_tool.get(tool, tool),
            "cohort_id": self.cohort_id,
            "args": args,
            "status": status,
            "run_dir": run_dir,
            "error": error,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        if status == "success":
            state = self._read_state()
            state["steps"].append(entry)
            if entry["stage"] not in state["stages_completed"]:
                state["stages_completed"].append(entry["stage"])
            self._write_state(state)

    def status(self) -> dict:
        state = self._read_state()
        completed = set(state["stages_completed"])
        missing_groups = [sorted(group) for group in self.config.export_requires if not (group & completed)]
        state["export_ready"] = not missing_groups
        # Flatten for display: one representative stage name per unmet group.
        state["missing_for_export"] = [" or ".join(g) for g in missing_groups]
        return state

    def export(self, output_dir: str) -> tuple[Path, int]:
        dest = Path(output_dir).expanduser().resolve() / self.session_id
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.dir, dest, dirs_exist_ok=True)
        n_files = sum(1 for _ in dest.rglob("*") if _.is_file())
        return dest, n_files


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
