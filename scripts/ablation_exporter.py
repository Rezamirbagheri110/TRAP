"""
Ablation study exporter.

Reads experiment_registry.yaml, loads all experiment results from the unified
ablation directory structure, and produces:

Run-level artifacts (per experiment):
  - raw/result.json          (full raw data)
  - raw/behaviors.jsonl      (one JSON per behavior)
  - reports/behaviors.csv    (flat behavior-level table)
  - reports/rl_learning_curve.csv
  - reports/summary.md
  - reports/results.xlsx     (multi-sheet)

Study-level artifacts (in aggregate/):
  - behaviors_all.csv / behaviors_all.jsonl   (combined table)
  - comparison_<pair>.csv                     (metric+diff table per pair)
  - comparison_<pair>.md                      (human-readable with interpretation)
  - ablation_summary.csv                      (one row per experiment)
  - ablation_summary.xlsx                     (multi-sheet workbook)
  - ablation_summary.parquet                  (if pandas + pyarrow available)

Usage:
    python scripts/ablation_exporter.py --study <study_id>
    python scripts/ablation_exporter.py --study <study_id> --registry config/experiment_registry.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_OURS_DIR = Path(__file__).parent.parent
if str(_OURS_DIR) not in sys.path:
    sys.path.insert(0, str(_OURS_DIR))

import yaml

from utils.ablation_manifest import (
    AblationPaths,
    _RESULTS_ROOT,
    collect_study_manifests,
)
from utils.behavior_aggregate import (
    CSV_FIELDNAMES,
    aggregate_study_behaviors,
    extract_behaviors_from_result,
    write_behavior_csv,
    write_behavior_jsonl,
)
from utils.schemas import PairComparisonRecord, normalise_behavior

_REGISTRY_PATH = _OURS_DIR / "config" / "experiment_registry.yaml"


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------

def load_registry(path: Path = _REGISTRY_PATH) -> Dict[str, Any]:
    from utils.experiment_config import load_experiment_registry
    return load_experiment_registry(path)


# ---------------------------------------------------------------------------
# Metric computation from behavior rows
# ---------------------------------------------------------------------------

def _compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {"asr": 0.0, "avg_score": 0.0, "avg_iters": 0.0, "n": 0}

    n = len(rows)
    success_count = sum(r["success"] for r in rows)
    avg_score = sum(r["max_score"] for r in rows) / n
    iters_vals = [r["iterations_count"] for r in rows if r.get("iterations_count") is not None]
    avg_iters = sum(iters_vals) / len(iters_vals) if iters_vals else 0.0

    return {
        "asr": 100.0 * success_count / n,
        "avg_score": round(avg_score, 4),
        "avg_iters": round(avg_iters, 4),
        "n": n,
        "success_count": success_count,
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def _bootstrap_ci(values: List[float], samples: int = 2000) -> Tuple[float, float]:
    """Deterministic percentile bootstrap CI for a mean."""
    if not values:
        return 0.0, 0.0
    rng = random.Random(42)
    estimates = []
    for _ in range(samples):
        estimates.append(_mean([values[rng.randrange(len(values))] for _ in values]))
    estimates.sort()
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def _paired_significance(
    idx_to_a: Dict[Tuple[Any, Any], Dict[str, Any]],
    idx_to_b: Dict[Tuple[Any, Any], Dict[str, Any]],
) -> Dict[str, float]:
    common = sorted(set(idx_to_a) & set(idx_to_b), key=str)
    wins_a = sum(bool(idx_to_a[k]["success"]) and not bool(idx_to_b[k]["success"]) for k in common)
    wins_b = sum(bool(idx_to_b[k]["success"]) and not bool(idx_to_a[k]["success"]) for k in common)
    score_a = [float(idx_to_a[k]["max_score"]) for k in common]
    score_b = [float(idx_to_b[k]["max_score"]) for k in common]
    try:
        from scipy.stats import binomtest, wilcoxon
        discordant = wins_a + wins_b
        mcnemar_p = float(binomtest(min(wins_a, wins_b), discordant, 0.5).pvalue) if discordant else 1.0
        wilcoxon_p = float(wilcoxon(score_a, score_b).pvalue) if score_a and score_a != score_b else 1.0
    except Exception:
        mcnemar_p = 1.0
        wilcoxon_p = 1.0
    return {"mcnemar_exact_p": mcnemar_p, "wilcoxon_score_p": wilcoxon_p}


def _compute_pair_comparison(
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    exp_a: str,
    exp_b: str,
    pair_key: str,
    interpretation_note: str,
) -> PairComparisonRecord:
    m_a = _compute_metrics(rows_a)
    m_b = _compute_metrics(rows_b)

    # per-behavior win/loss — match on behavior_idx
    def _pair_key(row: Dict[str, Any]) -> Tuple[Any, Any]:
        return (row.get("seed"), row.get("behavior_idx"))

    idx_to_a = {_pair_key(r): r for r in rows_a}
    idx_to_b = {_pair_key(r): r for r in rows_b}
    common = set(idx_to_a) & set(idx_to_b)

    wins_a = wins_b = ties = 0
    for row_key in common:
        sa = idx_to_a[row_key]["success"]
        sb = idx_to_b[row_key]["success"]
        if sa and not sb:
            wins_a += 1
        elif sb and not sa:
            wins_b += 1
        else:
            ties += 1

    asr_a, asr_b = m_a["asr"], m_b["asr"]
    asr_abs = round(asr_a - asr_b, 4)
    denom = max(asr_b, 1e-9)
    asr_rel = round(100.0 * (asr_a - asr_b) / denom, 4)

    avg_a, avg_b = m_a["avg_score"], m_b["avg_score"]
    iters_a, iters_b = m_a["avg_iters"], m_b["avg_iters"]
    asr_diff_samples = [
        100.0 * (float(bool(idx_to_a[k]["success"])) - float(bool(idx_to_b[k]["success"])))
        for k in common
    ]
    ci_low, ci_high = _bootstrap_ci(asr_diff_samples)
    significance = _paired_significance(idx_to_a, idx_to_b)

    return {  # type: ignore[return-value]
        "pair_key": pair_key,
        "experiment_a": exp_a,
        "experiment_b": exp_b,
        "interpretation_note": interpretation_note,
        "asr_a": asr_a,
        "asr_b": asr_b,
        "asr_abs_diff": asr_abs,
        "asr_rel_diff_pct": asr_rel,
        "avg_score_a": avg_a,
        "avg_score_b": avg_b,
        "avg_score_abs_diff": round(avg_a - avg_b, 4),
        "avg_iters_a": iters_a,
        "avg_iters_b": iters_b,
        "avg_iters_abs_diff": round(iters_a - iters_b, 4),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "total_behaviors": len(common),
        "asr_diff_ci95_low": round(ci_low, 4),
        "asr_diff_ci95_high": round(ci_high, 4),
        **significance,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_pair_csv(record: PairComparisonRecord, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(record.keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(record)
    return out_path


def _write_pair_markdown(record: PairComparisonRecord, out_path: Path,
                          exp_names: Dict[str, str]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a, b = record["experiment_a"], record["experiment_b"]
    name_a = exp_names.get(a, a)
    name_b = exp_names.get(b, b)

    lines = [
        f"# Ablation Pair: {record['pair_key']}",
        "",
        f"**Comparison:** {a} ({name_a}) vs {b} ({name_b})",
        "",
        "## Interpretation",
        "",
        f"> {record['interpretation_note']}",
        "",
        "## Metric Table",
        "",
        f"| Metric | {a}: {name_a} | {b}: {name_b} | Abs diff (A-B) | Rel diff % |",
        f"|--------|{'-----------|' * 4}",
        f"| ASR (%) | {record['asr_a']:.2f} | {record['asr_b']:.2f} | "
        f"{record['asr_abs_diff']:+.2f} | {record['asr_rel_diff_pct']:+.2f}% |",
        f"| Avg max_score | {record['avg_score_a']:.4f} | {record['avg_score_b']:.4f} | "
        f"{record['avg_score_abs_diff']:+.4f} | — |",
        f"| Avg iterations | {record['avg_iters_a']:.2f} | {record['avg_iters_b']:.2f} | "
        f"{record['avg_iters_abs_diff']:+.2f} | — |",
        "",
        "## Per-Behavior Win/Loss",
        "",
        f"| | Count |",
        f"|---|---|",
        f"| {a} wins (success, {b} fails) | {record['wins_a']} |",
        f"| {b} wins (success, {a} fails) | {record['wins_b']} |",
        f"| Ties (both same outcome) | {record['ties']} |",
        f"| Total matched behaviors | {record['total_behaviors']} |",
        "",
        "## Statistical Analysis",
        "",
        f"- Paired bootstrap 95% CI for ASR difference: "
        f"[{record.get('asr_diff_ci95_low', 0):+.2f}, {record.get('asr_diff_ci95_high', 0):+.2f}] percentage points",
        f"- Exact McNemar p-value: {record.get('mcnemar_exact_p', 1.0):.6f}",
        f"- Paired Wilcoxon p-value for max score: {record.get('wilcoxon_score_p', 1.0):.6f}",
        "",
        "---",
        f"*Generated {datetime.now().isoformat(timespec='seconds')}*",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_summary_csv(rows: List[Dict[str, Any]], out_path: Path) -> Path:
    if not rows:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _write_summary_xlsx(
    summary_rows: List[Dict[str, Any]],
    pair_records: List[PairComparisonRecord],
    behavior_rows: List[Dict[str, Any]],
    out_path: Path,
) -> Optional[Path]:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.formatting.rule import ColorScaleRule
    except ImportError:
        print("[WARN] openpyxl not installed — XLSX skipped")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)

    def _add_sheet(ws, rows_: List[Dict[str, Any]], fields: List[str]) -> None:
        for ci, col in enumerate(fields, 1):
            cell = ws.cell(row=1, column=ci, value=col)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
        for ri, row in enumerate(rows_, 2):
            for ci, col in enumerate(fields, 1):
                ws.cell(row=ri, column=ci, value=row.get(col))

    # Sheet 1: experiment summary
    ws1 = wb.active
    ws1.title = "Summary"
    if summary_rows:
        _add_sheet(ws1, summary_rows, list(summary_rows[0].keys()))

    # Sheet 2: pair comparisons
    ws2 = wb.create_sheet("Pair Comparisons")
    if pair_records:
        _add_sheet(ws2, [dict(r) for r in pair_records], list(pair_records[0].keys()))

    # Sheet 3: behavior-level data (trimmed — no full prompts)
    ws3 = wb.create_sheet("Behaviors")
    trimmed_fields = [f for f in CSV_FIELDNAMES if f != "goal"]
    trimmed = [{k: r.get(k) for k in trimmed_fields} for r in behavior_rows]
    if trimmed:
        _add_sheet(ws3, trimmed, trimmed_fields)

    wb.save(out_path)
    return out_path


def _write_parquet(rows: List[Dict[str, Any]], out_path: Path) -> Optional[Path]:
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("[WARN] pandas/pyarrow not installed — Parquet skipped")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df), str(out_path))
    return out_path


def _write_run_xlsx(rows: List[Dict[str, Any]], out_path: Path) -> Optional[Path]:
    """Write a single-run Excel workbook with one Behaviors sheet.

    Mirrors the manifest's ``paths.excel`` entry so every run carries the
    behaviour table in xlsx form. Silently skipped when openpyxl is absent.
    """
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Behaviors"
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for ci, col in enumerate(CSV_FIELDNAMES, 1):
        cell = ws.cell(row=1, column=ci, value=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
    for ri, row in enumerate(rows, 2):
        for ci, col in enumerate(CSV_FIELDNAMES, 1):
            ws.cell(row=ri, column=ci, value=row.get(col))
    wb.save(out_path)
    return out_path


def _write_run_summary_md(
    *,
    rows: List[Dict[str, Any]],
    result: Dict[str, Any],
    manifest_path: Path,
    out_path: Path,
) -> Path:
    """Write a compact per-experiment/seed summary next to behavior exports."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = _compute_metrics(rows)
    manifest_rel = "../raw/manifest.json"
    result_rel = "../raw/result.json"
    behavior_rel = "behaviors.csv"
    curve_rel = "rl_learning_curve.csv"
    lines = [
        f"# Ablation Run Summary: {result.get('experiment_id', 'unknown')}",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Behaviors | {metrics.get('n', 0)} |",
        f"| Success count | {metrics.get('success_count', 0)} |",
        f"| ASR (%) | {metrics.get('asr', 0.0):.2f} |",
        f"| Avg max score | {metrics.get('avg_score', 0.0):.4f} |",
        f"| Avg iterations | {metrics.get('avg_iters', 0.0):.4f} |",
        f"| Elapsed seconds | {result.get('elapsed_seconds', '')} |",
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        f"| [{manifest_path.name}]({manifest_rel}) | Run manifest |",
        f"| [result.json]({result_rel}) | Full raw result with iterations |",
        f"| [behaviors.csv]({behavior_rel}) | One row per behavior |",
        f"| [results.xlsx](results.xlsx) | Behaviour table (Excel) |",
        f"| [rl_learning_curve.csv]({curve_rel}) | Iteration-level learning curve |",
        "",
    ]
    if result.get("rag_only_selection_shares"):
        lines.extend([
            "## Retrieval-Driven Selection Shares",
            "",
            "```json",
            json.dumps(result.get("rag_only_selection_shares"), indent=2, ensure_ascii=False),
            "```",
            "",
        ])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _generate_ablation_visualizations(
    summary_rows: List[Dict[str, Any]],
    pair_records: List[PairComparisonRecord],
    behavior_rows: List[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, str]:
    """Generate compact ablation-specific plots for quick inspection."""
    paths: Dict[str, str] = {}
    if not summary_rows:
        return paths
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable - ablation plots skipped: {exc}")
        return paths

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        labels = [str(r["experiment_id"]) for r in summary_rows]
        asr = [float(r.get("asr_pct", 0.0) or 0.0) for r in summary_rows]
        ci_low = [float(r.get("asr_seed_ci95_low", v) or v) for r, v in zip(summary_rows, asr)]
        ci_high = [float(r.get("asr_seed_ci95_high", v) or v) for r, v in zip(summary_rows, asr)]
        yerr = [
            [max(0.0, v - lo) for v, lo in zip(asr, ci_low)],
            [max(0.0, hi - v) for v, hi in zip(asr, ci_high)],
        ]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
        ax.bar(labels, asr, color="#4C78A8", yerr=yerr, capsize=4)
        ax.set_ylabel("ASR (%)")
        ax.set_xlabel("Ablation")
        ax.set_ylim(0, 105)
        ax.set_title("Ablation ASR with Seed CI")
        for i, value in enumerate(asr):
            ax.text(i, min(103, value + 2), f"{value:.1f}", ha="center", fontsize=9)
        fig.tight_layout()
        path = out_dir / "ablation_asr_ci.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths["ablation_asr_ci"] = str(path)
    except Exception as exc:
        print(f"[WARN] Could not create ASR CI plot: {exc}")

    try:
        labels = [str(r["experiment_id"]) for r in summary_rows]
        scores = [float(r.get("avg_max_score", 0.0) or 0.0) for r in summary_rows]
        iters = [float(r.get("avg_iterations", 0.0) or 0.0) for r in summary_rows]
        fig, ax1 = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
        x = list(range(len(labels)))
        ax1.bar([i - 0.18 for i in x], scores, width=0.36, color="#59A14F", label="Avg max score")
        ax2 = ax1.twinx()
        ax2.bar([i + 0.18 for i in x], iters, width=0.36, color="#F28E2B", label="Avg iterations")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel("Avg max score")
        ax2.set_ylabel("Avg iterations")
        ax1.set_title("Score vs Iteration Cost")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        path = out_dir / "ablation_score_vs_iterations.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths["ablation_score_vs_iterations"] = str(path)
    except Exception as exc:
        print(f"[WARN] Could not create score/iteration plot: {exc}")

    try:
        if pair_records:
            labels = [str(r["pair_key"]) for r in pair_records]
            diffs = [float(r.get("asr_abs_diff", 0.0) or 0.0) for r in pair_records]
            fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.5), 5))
            colors = ["#59A14F" if d >= 0 else "#E15759" for d in diffs]
            ax.bar(labels, diffs, color=colors)
            ax.axhline(0, color="#333333", linewidth=1)
            ax.set_ylabel("ASR diff, A-B (percentage points)")
            ax.set_title("Pairwise Ablation Effects")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            path = out_dir / "ablation_pairwise_effects.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths["ablation_pairwise_effects"] = str(path)
    except Exception as exc:
        print(f"[WARN] Could not create pairwise effects plot: {exc}")

    try:
        if behavior_rows:
            experiments = sorted({str(r.get("ablation_variant")) for r in behavior_rows})
            behavior_ids = sorted({str(r.get("behavior_idx")) for r in behavior_rows}, key=lambda x: int(x) if x.isdigit() else x)
            matrix = []
            for behavior_id in behavior_ids:
                row = []
                for exp in experiments:
                    matches = [
                        r for r in behavior_rows
                        if str(r.get("behavior_idx")) == behavior_id and str(r.get("ablation_variant")) == exp
                    ]
                    value = sum(float(bool(m.get("success"))) for m in matches) / len(matches) if matches else 0.0
                    row.append(value)
                matrix.append(row)
            fig, ax = plt.subplots(figsize=(max(8, len(experiments) * 0.9), max(4, len(behavior_ids) * 0.35)))
            im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
            ax.set_xticks(range(len(experiments)))
            ax.set_xticklabels(experiments)
            ax.set_yticks(range(len(behavior_ids)))
            ax.set_yticklabels(behavior_ids)
            ax.set_xlabel("Ablation")
            ax.set_ylabel("Behavior")
            ax.set_title("Per-Behavior Success Heatmap")
            fig.colorbar(im, ax=ax, label="Success share")
            fig.tight_layout()
            path = out_dir / "ablation_behavior_success_heatmap.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths["ablation_behavior_success_heatmap"] = str(path)
    except Exception as exc:
        print(f"[WARN] Could not create behavior heatmap: {exc}")

    return paths


# ---------------------------------------------------------------------------
# RL learning curve CSV
# ---------------------------------------------------------------------------

def write_rl_curve_csv(result: Dict[str, Any], out_path: Path) -> Optional[Path]:
    """
    Extract reward and log_prob from every iteration across all behaviors
    and write a CSV for plotting learning curves.
    """
    from utils.behavior_aggregate import extract_behaviors_from_result

    behaviors = extract_behaviors_from_result(result)
    if not behaviors:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["behavior_idx", "iteration", "score", "reward", "log_prob",
              "rl_metrics_available", "template_name", "rag_enhanced"]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for beh in behaviors:
            bidx = beh.get("behavior_idx", -1)
            for it in beh.get("iterations", []):
                rl = it.get("rl_metrics") or {}
                writer.writerow({
                    "behavior_idx":         bidx,
                    "iteration":            it.get("iteration", 0),
                    "score":                it.get("score", 0.0),
                    "reward":               it.get("reward", ""),
                    "log_prob":             it.get("log_prob", ""),
                    "rl_metrics_available": int(bool(rl)),
                    "template_name":        it.get("template_name", ""),
                    "rag_enhanced":         int(bool(it.get("rag_enhanced", False))),
                })
    return out_path


# ---------------------------------------------------------------------------
# Main export orchestrator
# ---------------------------------------------------------------------------

def export_study(study_id: str, registry_path: Path = _REGISTRY_PATH) -> None:
    registry = load_registry(registry_path)
    experiments: Dict[str, Any] = registry.get("experiments", {})
    comparison_matrix: Dict[str, List[str]] = registry.get("comparison_matrix", {})

    study_dir = _RESULTS_ROOT / study_id
    if not study_dir.exists():
        print(f"[ERROR] Study directory not found: {study_dir}")
        sys.exit(1)

    agg_dir = study_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    exp_names = {k: v.get("name", k) for k, v in experiments.items()}

    # ---- 1. Load all result.json files and build behavior rows per experiment ----
    exp_rows: Dict[str, List[Dict[str, Any]]] = {}

    for exp_id in experiments:
        rows: List[Dict[str, Any]] = []
        for result_path in study_dir.glob(f"{exp_id}/*/raw/result.json"):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[WARN] Cannot read {result_path}: {exc}")
                continue

            behaviors = extract_behaviors_from_result(result)
            exp_name = exp_names.get(exp_id, exp_id)
            try:
                seed = int(result_path.parent.parent.name)
            except ValueError:
                seed = None
            run_id = result_path.parent.parent.name
            from utils.behavior_aggregate import build_behavior_rows
            run_rows = build_behavior_rows(
                behaviors,
                exp_id,
                exp_name,
                seed=seed,
                run_id=run_id,
            )
            rows.extend(run_rows)

            reports_dir = result_path.parent.parent / "reports"
            write_behavior_csv(run_rows, reports_dir / "behaviors.csv")
            _write_run_summary_md(
                rows=run_rows,
                result=result,
                manifest_path=result_path.parent / "manifest.json",
                out_path=reports_dir / "summary.md",
            )
            _write_run_xlsx(run_rows, reports_dir / "results.xlsx")

            # RL curve
            curve_path = reports_dir / "rl_learning_curve.csv"
            write_rl_curve_csv(result, curve_path)
            print(f"  [{exp_id}/{run_id}] wrote per-run reports")

        exp_rows[exp_id] = rows
        print(f"  [{exp_id}] {len(rows)} behavior rows loaded")

    # ---- 2. Combined behavior table ----
    all_rows = [r for rows in exp_rows.values() for r in rows]
    write_behavior_csv(all_rows, agg_dir / "behaviors_all.csv")
    write_behavior_jsonl(all_rows, agg_dir / "behaviors_all.jsonl")

    # ---- 3. Per-experiment summary rows ----
    summary_rows: List[Dict[str, Any]] = []
    for exp_id, rows in exp_rows.items():
        m = _compute_metrics(rows)
        exp_cfg = experiments.get(exp_id, {})
        per_seed_asr: List[float] = []
        per_seed_score: List[float] = []
        per_seed_iters: List[float] = []
        for seed in sorted({r.get("seed") for r in rows if r.get("seed") is not None}):
            seed_rows = [r for r in rows if r.get("seed") == seed]
            sm = _compute_metrics(seed_rows)
            per_seed_asr.append(sm["asr"])
            per_seed_score.append(sm["avg_score"])
            per_seed_iters.append(sm["avg_iters"])
        asr_ci_low, asr_ci_high = _bootstrap_ci(per_seed_asr)
        score_ci_low, score_ci_high = _bootstrap_ci(per_seed_score)
        summary_rows.append({
            "experiment_id":        exp_id,
            "experiment_name":      exp_names.get(exp_id, exp_id),
            "n_behaviors":          m["n"],
            "success_count":        m.get("success_count", 0),
            "asr_pct":              round(m["asr"], 4),
            "avg_max_score":        m["avg_score"],
            "avg_iterations":       m["avg_iters"],
            "rag_enabled":          exp_cfg.get("rag_enabled", ""),
            "tree_variant":         exp_cfg.get("tree_variant", ""),
            "tree_leaves":          exp_cfg.get("tree_leaves", ""),
            "rag_mode":             exp_cfg.get("rag_mode", ""),
            "n_seeds":              len(per_seed_asr),
            "asr_seed_std":         round(_sample_std(per_seed_asr), 4),
            "asr_seed_ci95_low":    round(asr_ci_low, 4),
            "asr_seed_ci95_high":   round(asr_ci_high, 4),
            "score_seed_mean":      round(_mean(per_seed_score), 4),
            "score_seed_std":       round(_sample_std(per_seed_score), 4),
            "score_seed_ci95_low":  round(score_ci_low, 4),
            "score_seed_ci95_high": round(score_ci_high, 4),
            "iters_seed_mean":      round(_mean(per_seed_iters), 4),
            "iters_seed_std":       round(_sample_std(per_seed_iters), 4),
        })

    _write_summary_csv(summary_rows, agg_dir / "ablation_summary.csv")

    # ---- 4. Per-pair comparison tables ----
    pair_records: List[PairComparisonRecord] = []
    for pair_key, (exp_a, exp_b) in comparison_matrix.items():
        exp_cfg_a = experiments.get(exp_a, {})
        exp_cfg_b = experiments.get(exp_b, {})
        note_a = exp_cfg_a.get("interpretation_note", "")
        note_b = exp_cfg_b.get("interpretation_note", "")
        note = note_a or note_b or f"Comparison of {exp_a} vs {exp_b}."

        rows_a = exp_rows.get(exp_a, [])
        rows_b = exp_rows.get(exp_b, [])

        rec = _compute_pair_comparison(rows_a, rows_b, exp_a, exp_b, pair_key, note)
        pair_records.append(rec)

        _write_pair_csv(rec, agg_dir / f"comparison_{pair_key}.csv")
        _write_pair_markdown(rec, agg_dir / f"comparison_{pair_key}.md", exp_names)
        print(f"  [pair:{pair_key}] ASR {exp_a}={rec['asr_a']:.1f}%  {exp_b}={rec['asr_b']:.1f}%  "
              f"diff={rec['asr_abs_diff']:+.1f}%")

    # ---- 5. Multi-format aggregate outputs ----
    xlsx_path = _write_summary_xlsx(
        summary_rows, pair_records, all_rows,
        agg_dir / "ablation_summary.xlsx",
    )
    if xlsx_path:
        print(f"  [OK] XLSX: {xlsx_path}")

    pq_path = _write_parquet(all_rows, agg_dir / "behaviors_all.parquet")
    if pq_path:
        print(f"  [OK] Parquet: {pq_path}")

    viz_paths = _generate_ablation_visualizations(summary_rows, pair_records, all_rows, agg_dir)
    for name, path in viz_paths.items():
        print(f"  [OK] Plot {name}: {path}")

    # ---- 6. Master index ----
    _write_aggregate_index(agg_dir, summary_rows, pair_records, exp_names, viz_paths)
    print(f"\n[DONE] Study export complete -> {agg_dir}")


def _write_aggregate_index(
    agg_dir: Path,
    summary_rows: List[Dict[str, Any]],
    pair_records: List[PairComparisonRecord],
    exp_names: Dict[str, str],
    viz_paths: Optional[Dict[str, str]] = None,
) -> Path:
    index_path = agg_dir / "INDEX.md"
    lines = [
        "# Ablation Study — Aggregate Results",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Experiment Summary",
        "",
        "| ID | Name | N | Seeds | ASR % | ASR std | ASR 95% CI | Score mean±std | Avg Iters | RAG | Tree |",
        "|----|------|---|-------|-------|---------|-----------|----------------|-----------|-----|------|",
    ]
    for r in summary_rows:
        asr_ci = f"[{r.get('asr_seed_ci95_low', 0):+.2f}, {r.get('asr_seed_ci95_high', 0):+.2f}]"
        score_ms = f"{r.get('score_seed_mean', r['avg_max_score']):.3f}±{r.get('score_seed_std', 0):.3f}"
        lines.append(
            f"| {r['experiment_id']} | {r['experiment_name']} | {r['n_behaviors']} | "
            f"{r.get('n_seeds', '?')} | {r['asr_pct']:.2f} | {r.get('asr_seed_std', 0):.2f} | "
            f"{asr_ci} | {score_ms} | {r['avg_iterations']:.2f} | {r['rag_enabled']} | {r['tree_variant']} |"
        )

    lines += [
        "",
        "## Pairwise Comparisons",
        "",
        "| Pair | Exp A | Exp B | ASR A | ASR B | Abs Diff | Rel Diff | Wins A | Wins B |",
        "|------|-------|-------|-------|-------|----------|----------|--------|--------|",
    ]
    for rec in pair_records:
        pk = rec['pair_key']
        lines.append(
            f"| [{pk}](comparison_{pk}.md) | "
            f"{rec['experiment_a']} | {rec['experiment_b']} | "
            f"{rec['asr_a']:.2f}% | {rec['asr_b']:.2f}% | "
            f"{rec['asr_abs_diff']:+.2f}% | {rec['asr_rel_diff_pct']:+.2f}% | "
            f"{rec['wins_a']} | {rec['wins_b']} |"
        )

    lines += [
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| [ablation_summary.csv](ablation_summary.csv) | One row per experiment |",
        "| [ablation_summary.xlsx](ablation_summary.xlsx) | Multi-sheet workbook |",
        "| [behaviors_all.csv](behaviors_all.csv) | All behaviors × all variants |",
        "| [behaviors_all.jsonl](behaviors_all.jsonl) | Same in JSONL |",
        "| [behaviors_all.parquet](behaviors_all.parquet) | Same in Parquet |",
    ]
    for rec in pair_records:
        pk = rec["pair_key"]
        lines.append(f"| [comparison_{pk}.md](comparison_{pk}.md) | Pair report with interpretation |")
        lines.append(f"| [comparison_{pk}.csv](comparison_{pk}.csv) | Pair metrics CSV |")

    if viz_paths:
        lines += [
            "",
            "## Plots",
            "",
            "| File | Description |",
            "|------|-------------|",
        ]
        descriptions = {
            "ablation_asr_ci": "ASR per ablation with seed confidence intervals",
            "ablation_score_vs_iterations": "Average score compared with iteration cost",
            "ablation_pairwise_effects": "Pairwise ASR differences from the comparison matrix",
            "ablation_behavior_success_heatmap": "Per-behavior success share by ablation",
        }
        for name, path in sorted(viz_paths.items()):
            filename = Path(path).name
            lines.append(f"| [{filename}]({filename}) | {descriptions.get(name, name)} |")

    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export ablation study results")
    parser.add_argument("--study", required=True, help="Study ID (subdirectory of results/ablations/)")
    parser.add_argument("--registry", default=str(_REGISTRY_PATH),
                        help="Path to experiment_registry.yaml")
    args = parser.parse_args()
    export_study(args.study, Path(args.registry))


if __name__ == "__main__":
    main()
