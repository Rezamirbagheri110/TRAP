"""
Ablation run manifest writer and output path resolver.

Every run must call write_manifest() before finishing so that any file in
results/ablations/ can be traced back to a specific experiment configuration
without relying on directory naming conventions alone.

Directory layout produced by get_run_dirs():
    results/ablations/
        <study_id>/
            <experiment_id>/           # A, B, C, D, E
                <seed>/
                    raw/               # result.json, per-behavior JSONL
                    reports/           # Markdown, CSV, XLSX, PNG
            aggregate/                 # cross-experiment comparison tables
"""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from utils.schemas import SCHEMA_VERSION, RunManifest

_OURS_ROOT = Path(__file__).parent.parent
_RESULTS_ROOT = _OURS_ROOT / "results" / "ablations"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

class AblationPaths:
    """Resolved paths for a single experiment run."""

    def __init__(self, study_id: str, experiment_id: str, seed: int):
        self.study_root = _RESULTS_ROOT / study_id
        self.run_root = self.study_root / experiment_id / str(seed)
        self.raw = self.run_root / "raw"
        self.reports = self.run_root / "reports"
        self.aggregate = self.study_root / "aggregate"
        self.kb = self.run_root / "kb"
        self.state = self.run_root / "state"

    def makedirs(self) -> "AblationPaths":
        for d in (self.raw, self.reports, self.aggregate, self.kb, self.state):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def manifest_path(self) -> Path:
        return self.raw / "manifest.json"

    def result_json_path(self) -> Path:
        return self.raw / "result.json"

    def behavior_jsonl_path(self) -> Path:
        return self.raw / "behaviors.jsonl"

    def behavior_csv_path(self) -> Path:
        return self.reports / "behaviors.csv"

    def rl_curve_csv_path(self) -> Path:
        return self.reports / "rl_learning_curve.csv"

    def summary_md_path(self) -> Path:
        return self.reports / "summary.md"

    def excel_path(self) -> Path:
        return self.reports / "results.xlsx"

    def as_dict(self, relative_to: Optional[Path] = None) -> Dict[str, str]:
        base = relative_to or self.run_root
        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(base))
            except ValueError:
                return str(p)

        return {
            "manifest":     _rel(self.manifest_path()),
            "result_json":  _rel(self.result_json_path()),
            "behaviors_jsonl": _rel(self.behavior_jsonl_path()),
            "behaviors_csv": _rel(self.behavior_csv_path()),
            "rl_curve_csv": _rel(self.rl_curve_csv_path()),
            "summary_md":   _rel(self.summary_md_path()),
            "excel":        _rel(self.excel_path()),
            "kb_dir":       _rel(self.kb),
            "state_dir":    _rel(self.state),
        }


def get_run_dirs(study_id: str, experiment_id: str, seed: int) -> AblationPaths:
    """Return an AblationPaths object and create the directory tree."""
    return AblationPaths(study_id, experiment_id, seed).makedirs()


def prepare_kb_snapshot(
    paths: AblationPaths,
    source_path: Optional[Path],
) -> Dict[str, Any]:
    """Freeze the starting KB into a run-local, immutable snapshot.

    The first time a run is prepared we copy the live source KB into
    ``<exp>/<seed>/kb/initial_snapshot.pkl`` and never touch it again. Every
    later rerun reuses that frozen file, so the snapshot's sha256 stays stable
    even when the shared global KB keeps drifting (e.g. an online main run that
    writes back learned examples before the ablation starts). This is what lets
    the ablation cache hit on rerun instead of re-executing every experiment.
    """
    target = paths.kb / "initial_snapshot.pkl"

    # Already frozen by a previous run — reuse it verbatim. Do NOT re-copy from
    # the live source, otherwise the snapshot hash would change whenever the
    # global KB changed and silently invalidate the ablation cache.
    if target.exists():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return {
            "source": str(Path(source_path).resolve()) if source_path else "",
            "path": str(target),
            "sha256": digest,
            "exists": True,
        }

    if source_path is None or not Path(source_path).exists():
        return {"source": "", "path": "", "sha256": "", "exists": False}
    source = Path(source_path).resolve()
    shutil.copy2(source, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {
        "source": str(source),
        "path": str(target),
        "sha256": digest,
        "exists": True,
    }


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_info() -> Dict[str, Any]:
    def _run(*args: str) -> str:
        try:
            result = subprocess.run(
                list(args),
                capture_output=True,
                text=True,
                cwd=str(_OURS_ROOT),
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    commit = _run("git", "rev-parse", "HEAD")
    branch = _run("git", "rev-parse", "--abbrev-ref", "HEAD")
    dirty_output = _run("git", "status", "--porcelain")
    dirty = bool(dirty_output)

    return {"git_commit": commit, "git_branch": branch, "git_dirty": dirty}


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

def write_manifest(
    paths: AblationPaths,
    *,
    study_id: str,
    experiment_id: str,
    experiment_name: str,
    seed: int,
    target_model: str,
    evaluator_model: str,
    rag_enabled: bool,
    tree_variant: str,
    tree_leaves: Optional[int],
    tree_max_depth: Optional[int],
    success_threshold: float,
    max_iterations_per_episode: int,
    episode_budget: int,
    # Summary metrics (can be partial — filled after run)
    total_behaviors: int = 0,
    success_count: int = 0,
    asr_percent: float = 0.0,
    avg_best_score: float = 0.0,
    avg_iterations_per_behavior: float = 0.0,
    # Experiment D fallback tracking (only populated for agent_mode == "rag_only")
    rag_only_selection_stats: Optional[Dict[str, Any]] = None,
    rag_only_selection_shares: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write manifest.json to paths.raw/.

    Call this twice:
      1. Before the run starts (omit summary metrics) to mark in-progress.
      2. After the run completes with full summary metrics.
    """
    git = _git_info()

    manifest: RunManifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_model": target_model,
        "evaluator_model": evaluator_model,
        "rag_enabled": rag_enabled,
        "tree_variant": tree_variant,
        "tree_leaves": tree_leaves,
        "tree_max_depth": tree_max_depth,
        "success_threshold": success_threshold,
        "max_iterations_per_episode": max_iterations_per_episode,
        "episode_budget": episode_budget,
        "git_commit": git["git_commit"],
        "git_branch": git["git_branch"],
        "git_dirty": git["git_dirty"],
        "total_behaviors": total_behaviors,
        "success_count": success_count,
        "asr_percent": asr_percent,
        "avg_best_score": avg_best_score,
        "avg_iterations_per_behavior": avg_iterations_per_behavior,
        "paths": paths.as_dict(relative_to=paths.study_root),
    }

    if rag_only_selection_stats is not None:
        manifest["rag_only_selection_stats"] = rag_only_selection_stats  # type: ignore[index]
    if rag_only_selection_shares is not None:
        manifest["rag_only_selection_shares"] = rag_only_selection_shares  # type: ignore[index]

    if extra:
        manifest.update(extra)  # type: ignore[arg-type]

    manifest_path = paths.manifest_path()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return manifest_path


def load_manifest(run_raw_dir: Path) -> Dict[str, Any]:
    """Load a manifest.json from a raw/ directory."""
    p = run_raw_dir / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_raw_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def collect_study_manifests(study_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Return all manifests in a study keyed by experiment_id.

    Looks for results/ablations/<study_id>/*/*/raw/manifest.json.
    When multiple seeds exist, returns the one with the most recent timestamp.
    """
    study_root = _RESULTS_ROOT / study_id
    manifests: Dict[str, Dict[str, Any]] = {}

    for mf in study_root.rglob("raw/manifest.json"):
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        exp_id = data.get("experiment_id", "")
        if not exp_id:
            continue
        existing = manifests.get(exp_id)
        if existing is None or data.get("timestamp", "") > existing.get("timestamp", ""):
            manifests[exp_id] = data

    return manifests
