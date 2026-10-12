"""
Behavior-level aggregate exporter.

Produces a flat CSV/JSONL where each row is one (behavior, ablation_variant)
pair containing all metrics needed for cross-ablation comparison:

    behavior_idx, goal, ablation_variant, experiment_name,
    success, max_score, final_score,
    iterations_to_success, first_success_iteration,
    iterations_count, rag_hit_count, chosen_leaf,
    avg_score, score_std, timed_out

This is the source-of-truth table for:
  - ASR computation per variant
  - Behaviour-level win/loss between pairs
  - "Which variant works better on which behaviours" analysis
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from utils.schemas import normalise_behavior, BehaviorRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _behavior_to_row(
    beh: BehaviorRecord,
    ablation_variant: str,
    experiment_name: str,
    seed: Optional[int] = None,
    run_id: str = "",
) -> Dict[str, Any]:
    iters = beh.get("iterations", [])
    scores = [float(it.get("score", 0.0)) for it in iters]

    return {
        "behavior_idx":           beh.get("behavior_idx", -1),
        "goal":                   beh.get("goal", ""),
        "ablation_variant":       ablation_variant,
        "experiment_name":        experiment_name,
        "seed":                   seed,
        "run_id":                 run_id,
        "success":                int(bool(beh.get("success", False))),
        "max_score":              beh.get("max_score", 0.0),
        "final_score":            beh.get("final_score", scores[-1] if scores else 0.0),
        "iterations_to_success":  beh.get("iterations_to_success"),   # None = never
        "first_success_iteration": beh.get("first_success_iteration"),
        "iterations_count":       beh.get("iterations_count", len(iters)),
        "rag_hit_count":          beh.get("rag_hit_count", 0),
        "chosen_leaf":            beh.get("chosen_leaf", ""),
        "avg_score":              round(sum(scores) / len(scores), 4) if scores else 0.0,
        "score_std":              round(_std(scores), 4),
        "timed_out":              int(bool(beh.get("timed_out", False))),
    }


CSV_FIELDNAMES = [
    "behavior_idx", "goal", "ablation_variant", "experiment_name", "seed", "run_id",
    "success", "max_score", "final_score",
    "iterations_to_success", "first_success_iteration",
    "iterations_count", "rag_hit_count", "chosen_leaf",
    "avg_score", "score_std", "timed_out",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_behaviors_from_result(
    result: Any,
) -> List[BehaviorRecord]:
    """
    Extract and normalise a list of BehaviorRecords from a result.json value.

    Supports all formats encountered in the project:
      - dict with metrics.results list (comparison run format)
      - list of behavior dicts at top level
      - flat dict with iterations/attempts at top level (single-behavior run)
    """
    # Format 2: list at top level
    if isinstance(result, list):
        return [normalise_behavior(b) for b in result if isinstance(b, dict)]

    if not isinstance(result, dict):
        return []

    # Format 1: {method, metrics: {results: [...]}, ...}
    metrics = result.get("metrics", {})
    raw_list = metrics.get("results") if isinstance(metrics, dict) else None

    if isinstance(raw_list, list):
        return [normalise_behavior(b) for b in raw_list if isinstance(b, dict)]

    # Format 1.5: {behaviors: [...], ...} (ablation run format)
    raw_behaviors = result.get("behaviors")
    if isinstance(raw_behaviors, list):
        return [normalise_behavior(b) for b in raw_behaviors if isinstance(b, dict)]

    # Format 3: flat dict with iterations at top level (single-behavior run)
    if "iterations" in result or "attempts" in result:
        return [normalise_behavior(result)]

    return []


def build_behavior_rows(
    behaviors: List[BehaviorRecord],
    ablation_variant: str,
    experiment_name: str,
    seed: Optional[int] = None,
    run_id: str = "",
) -> List[Dict[str, Any]]:
    """Convert normalised BehaviorRecords to flat dicts for CSV/JSONL."""
    return [
        _behavior_to_row(
            b,
            ablation_variant,
            experiment_name,
            seed=seed,
            run_id=run_id,
        )
        for b in behaviors
    ]


def write_behavior_csv(rows: List[Dict[str, Any]], out_path: Path) -> Path:
    """Write rows to a UTF-8 CSV file, creating parent dirs as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_behavior_jsonl(rows: List[Dict[str, Any]], out_path: Path) -> Path:
    """Write rows to a JSONL file (one JSON object per line)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return out_path


def aggregate_study_behaviors(
    study_dir: Path,
    experiment_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Collect behavior rows from all experiments in a study directory.

    Args:
        study_dir: Path to results/ablations/<study_id>/
        experiment_map: {experiment_id -> experiment_name}, e.g. {"A": "Full+RAG"}

    Returns:
        Combined list of row dicts across all experiments.
    """
    all_rows: List[Dict[str, Any]] = []

    for exp_id, exp_name in experiment_map.items():
        # Find result.json files under <study_dir>/<exp_id>/*/raw/result.json
        for result_path in study_dir.glob(f"{exp_id}/*/raw/result.json"):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[WARN] Could not read {result_path}: {exc}")
                continue

            behaviors = extract_behaviors_from_result(result)
            try:
                seed = int(result_path.parent.parent.name)
            except ValueError:
                seed = None
            run_id = result_path.parent.parent.name
            rows = build_behavior_rows(
                behaviors,
                exp_id,
                exp_name,
                seed=seed,
                run_id=run_id,
            )
            all_rows.extend(rows)
            print(f"  [{exp_id}] {len(rows)} behaviors from {result_path}")

    return all_rows


def write_aggregate_behavior_table(
    study_dir: Path,
    experiment_map: Dict[str, str],
) -> Dict[str, Path]:
    """
    Build and write the combined behavior-level table for a study.

    Produces:
        <study_dir>/aggregate/behaviors_all.csv
        <study_dir>/aggregate/behaviors_all.jsonl
    """
    agg_dir = study_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    rows = aggregate_study_behaviors(study_dir, experiment_map)

    csv_path = write_behavior_csv(rows, agg_dir / "behaviors_all.csv")
    jsonl_path = write_behavior_jsonl(rows, agg_dir / "behaviors_all.jsonl")

    print(f"[OK] Wrote {len(rows)} behavior rows to {csv_path}")
    print(f"[OK] Wrote {len(rows)} behavior rows to {jsonl_path}")

    return {"csv": csv_path, "jsonl": jsonl_path}
