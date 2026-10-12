"""
TRAP Output Generator — Publication-Quality Figures, Reports, and Excel Workbooks

Reads real results from:
  1. results/<run_name>/full_results.json          — TRAP vs baseline comparison
  2. results/<run_name>/ours-with*.json            — legacy per-variant files (ignored)
  3. results/ablations/<study_id>/aggregate/       — proper ablation study outputs

Ablation variants are taken directly from experiment_registry.yaml (A, B, C, D, E)
and displayed with their canonical names (Full+StaticRAG, Pruned+RAG, Flat+RAG, …).

Usage:
    python scripts/improve_outputs.py                        # auto-detect latest run
    python scripts/improve_outputs.py --run <run_name>       # specific comparison run
    python scripts/improve_outputs.py --study <study_id>     # ablation study outputs
    python scripts/improve_outputs.py --run <run> --study <study_id>  # both
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── path bootstrap ────────────────────────────────────────────────────────────
_OURS_DIR = Path(__file__).parent.parent
if str(_OURS_DIR) not in sys.path:
    sys.path.insert(0, str(_OURS_DIR))

_RESULTS_ROOT = _OURS_DIR / "results"
_ABLATIONS_ROOT = _RESULTS_ROOT / "ablations"

# ── optional dependencies ─────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found – skipping figures")

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import ColorScaleRule
    HAS_XL = True
except ImportError:
    HAS_XL = False
    print("[WARN] openpyxl not found – skipping Excel")

# ── colour palette ────────────────────────────────────────────────────────────
# TRAP variants use green family; baselines use distinct colours.
PALETTE: Dict[str, str] = {
    "TRAP":         "#1a6b3c",   # dark green — full system
    "A":            "#1a6b3c",   # Full+StaticRAG  (same green)
    "A_ONLINE":     "#2e8b57",   # Full+OnlineRAG
    "B":            "#4caf7d",   # Pruned+RAG
    "C":            "#a8d5b5",   # Flat+RAG
    "D":            "#d4edda",   # Retrieval-Driven
    "E":            "#f0c080",   # Full+NoRAG  (amber to contrast)
    "PAIR":         "#e05c2a",
    "TAP":          "#f5a623",
    "AutoDAN":      "#4a90d9",
    "FlipAttack":   "#9b59b6",
    "default":      "#888888",
}

MODEL_LABELS: Dict[str, str] = {
    "mistralai/mistral-7b-instruct":   "Mistral-7B",
    "deepseek/deepseek-v3.2":          "DeepSeek-V3",
    "google/gemini-2.5-flash-lite":    "Gemini-2.5-Flash",
    "meta-llama/llama-4-scout":        "Llama-4-Scout",
    "qwen/qwen3-235b-a22b-2507":       "Qwen3-235B",
    "openai/gpt-4o-mini":              "GPT-4o-mini",
    "openai/gpt-oss-120b":             "GPT-OSS-120B",
}

# Canonical experiment display names (fall back to registry if loaded)
EXP_DISPLAY: Dict[str, str] = {
    "A":       "Full+StaticRAG",
    "A_ONLINE":"Full+OnlineRAG",
    "B":       "Pruned+RAG",
    "C":       "Flat+RAG",
    "D":       "Retrieval-Driven",
    "E":       "Full+NoRAG",
}


# =============================================================================
# Data loaders
# =============================================================================

def _load_registry() -> Dict[str, Any]:
    try:
        import yaml
        path = _OURS_DIR / "config" / "experiment_registry.yaml"
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _nice_model(raw: str) -> str:
    return MODEL_LABELS.get(raw, raw.split("/")[-1] if "/" in raw else raw)


def _color(name: str) -> str:
    return PALETTE.get(name, PALETTE["default"])


def load_comparison_run(run_dir: Path) -> Dict[str, Any]:
    """
    Load a comparison run directory produced by the pipeline runner.
    Returns a dict with keys:
      run_name, results (list of {method, target_model, asr_pct, success_count,
                                   total_behaviors, total_queries, avg_score})
    """
    result_file = run_dir / "full_results.json"
    if not result_file.exists():
        raise FileNotFoundError(f"full_results.json not found in {run_dir}")

    raw = json.loads(result_file.read_text(encoding="utf-8"))
    run_name = raw.get("run_name", run_dir.name)

    rows = []
    for entry in raw.get("results", []):
        method = entry.get("method", "?")
        # Skip old-style "TRAP-withouttree" and "TRAP-withoutrag" entries —
        # these came from an ad-hoc simulation, not the controlled ablation study.
        if "withouttree" in method.lower() or "withoutrag" in method.lower():
            continue
        model_raw = entry.get("target_model", "")
        metrics = entry.get("metrics") or {}
        asr = metrics.get("success_rate_percent")
        if asr is None:
            continue
        rows.append({
            "method":          method,
            "target_model":    model_raw,
            "model_label":     _nice_model(model_raw),
            "asr_pct":         round(float(asr), 4),
            "success_count":   int(metrics.get("success_count", 0)),
            "total_behaviors": int(metrics.get("total_behaviors", 0)),
            "total_queries":   int(metrics.get("total_queries", 0)),
            "avg_score":       float(metrics.get("avg_best_score") or metrics.get("max_score") or 0.0),
        })

    return {"run_name": run_name, "results": rows}


def _aggregate_method_across_models(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Average ASR and queries across target models for each method."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[r["method"]].append(r)

    out = []
    for method, entries in sorted(buckets.items(), key=lambda x: -sum(e["asr_pct"] for e in x[1]) / len(x[1])):
        avg_asr = sum(e["asr_pct"] for e in entries) / len(entries)
        avg_q   = sum(e["total_queries"] for e in entries) / len(entries)
        avg_sc  = sum(e["success_count"] for e in entries) / len(entries)
        avg_n   = sum(e["total_behaviors"] for e in entries) / len(entries)
        avg_score = sum(e["avg_score"] for e in entries) / len(entries)
        out.append({
            "method":       method,
            "asr_pct":      round(avg_asr, 2),
            "avg_queries":  round(avg_q, 1),
            "avg_success":  round(avg_sc, 1),
            "avg_n":        round(avg_n, 0),
            "avg_score":    round(avg_score, 3),
            "n_models":     len(entries),
        })
    return out


def load_ablation_study(study_id: str) -> Dict[str, Any]:
    """
    Load aggregate ablation study data from results/ablations/<study_id>/aggregate/.
    Returns summary rows and pair comparison records.
    """
    study_dir = _ABLATIONS_ROOT / study_id
    agg_dir   = study_dir / "aggregate"

    summary_rows: List[Dict[str, Any]] = []
    pair_records: List[Dict[str, Any]] = []

    # Read ablation_summary.csv
    summary_csv = agg_dir / "ablation_summary.csv"
    if summary_csv.exists():
        import csv
        with summary_csv.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # convert numeric strings
                for key in ("asr_pct", "avg_max_score", "avg_iterations",
                            "asr_seed_std", "asr_seed_ci95_low", "asr_seed_ci95_high",
                            "score_seed_mean", "score_seed_std", "n_behaviors",
                            "success_count", "n_seeds"):
                    if row.get(key) not in (None, "", "None"):
                        try:
                            row[key] = float(row[key])
                        except ValueError:
                            pass
                # add display name
                exp_id = row.get("experiment_id", "")
                row["display_name"] = EXP_DISPLAY.get(exp_id, row.get("experiment_name", exp_id))
                summary_rows.append(row)

    # Read pair comparison CSVs
    for csv_path in sorted(agg_dir.glob("comparison_*.csv")):
        import csv
        with csv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in ("asr_a", "asr_b", "asr_abs_diff", "asr_rel_diff_pct",
                            "avg_score_a", "avg_score_b", "avg_score_abs_diff",
                            "avg_iters_a", "avg_iters_b", "wins_a", "wins_b", "ties",
                            "total_behaviors", "asr_diff_ci95_low", "asr_diff_ci95_high",
                            "mcnemar_exact_p", "wilcoxon_score_p"):
                    if row.get(key) not in (None, "", "None"):
                        try:
                            row[key] = float(row[key])
                        except ValueError:
                            pass
                pair_records.append(row)

    return {
        "study_id":    study_id,
        "summary":     summary_rows,
        "pairs":       pair_records,
        "agg_dir":     agg_dir,
    }


def auto_detect_run() -> Optional[Path]:
    """Return the most recently modified run directory."""
    if not _RESULTS_ROOT.exists():
        return None
    dirs = [d for d in _RESULTS_ROOT.iterdir()
            if d.is_dir() and (d / "full_results.json").exists()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


# =============================================================================
# Figure generators
# =============================================================================

def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [fig] {path.name}")


def fig_asr_bar(methods: List[Dict[str, Any]], out_dir: Path) -> None:
    """Horizontal bar — ASR per method (average across target models)."""
    sorted_m = sorted(methods, key=lambda m: m["asr_pct"])
    names    = [m["method"] for m in sorted_m]
    values   = [m["asr_pct"] for m in sorted_m]
    colors   = [_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.65)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.65)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_xlabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("Attack Success Rate by Method\n(average across target models)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, 115)
    ax.axvline(50, color="#cccccc", lw=0.8, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    _save(fig, out_dir / "01_asr_bar.png")


def fig_model_heatmap(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """Heatmap: methods × target models."""
    if not HAS_MPL:
        return

    # Build matrix
    model_set  = sorted({r["model_label"] for r in rows})
    method_set = sorted({r["method"] for r in rows},
                        key=lambda m: -sum(r["asr_pct"] for r in rows if r["method"] == m))

    data = np.zeros((len(method_set), len(model_set)))
    for r in rows:
        mi = method_set.index(r["method"])
        mj = model_set.index(r["model_label"])
        data[mi, mj] = r["asr_pct"]

    fig, ax = plt.subplots(figsize=(max(10, len(model_set) * 1.4), max(4, len(method_set) * 0.85)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_xticks(range(len(model_set)))
    ax.set_xticklabels(model_set, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(method_set)))
    ax.set_yticklabels(method_set, fontsize=10)

    for i in range(len(method_set)):
        for j in range(len(model_set)):
            val = data[i, j]
            color = "white" if val > 60 or val < 15 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=8.5, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02).set_label("ASR (%)", fontsize=10)
    ax.set_title("Attack Success Rate: Methods × Target Models",
                 fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, out_dir / "02_model_heatmap.png")


def fig_ablation(summary_rows: List[Dict[str, Any]], out_dir: Path,
                 pairs: Optional[List[Dict[str, Any]]] = None) -> None:
    """Bar chart — ablation experiment ASR with CI whiskers and Δ annotations."""
    if not summary_rows or not HAS_MPL:
        return

    sorted_rows = sorted(summary_rows, key=lambda r: -float(r.get("asr_pct") or 0))
    labels  = [r["display_name"] for r in sorted_rows]
    values  = [float(r.get("asr_pct") or 0) for r in sorted_rows]
    ci_low  = [float(r.get("asr_seed_ci95_low") or 0) for r in sorted_rows]
    ci_high = [float(r.get("asr_seed_ci95_high") or 0) for r in sorted_rows]
    exp_ids = [r.get("experiment_id", "") for r in sorted_rows]
    colors  = [_color(eid) for eid in exp_ids]

    yerr_low  = [max(0, v - lo) for v, lo in zip(values, ci_low)]
    yerr_high = [max(0, hi - v) for v, hi in zip(values, ci_high)]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 6))
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white",
                  yerr=[yerr_low, yerr_high], capsize=4, error_kw={"ecolor": "#555", "lw": 1.2})

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(yerr_high) + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Attack Success Rate (%)", fontsize=11)
    ax.set_title("Ablation Study: Component Contributions\n(95% CI over seeds)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(0, 115)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.tick_params(axis="x", rotation=20, labelsize=9)

    # legend mapping experiment IDs
    legend_patches = [mpatches.Patch(color=_color(eid), label=f"{eid}: {lbl}")
                      for eid, lbl in zip(exp_ids, labels)]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right",
              framealpha=0.8, ncol=max(1, len(legend_patches) // 4))

    fig.tight_layout()
    _save(fig, out_dir / "03_ablation.png")


def fig_pair_comparisons(pairs: List[Dict[str, Any]], out_dir: Path,
                         exp_names: Dict[str, str]) -> None:
    """Grouped bar pairs — Δ ASR for each comparison pair."""
    if not pairs or not HAS_MPL:
        return

    pair_keys = [p["pair_key"] for p in pairs]
    asr_a     = [float(p.get("asr_a") or 0) for p in pairs]
    asr_b     = [float(p.get("asr_b") or 0) for p in pairs]

    x = np.arange(len(pair_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 1.8), 6))
    bars_a = ax.bar(x - width / 2, asr_a, width, label="Experiment A (left)",
                    color="#1a6b3c", edgecolor="white")
    bars_b = ax.bar(x + width / 2, asr_b, width, label="Experiment B (right)",
                    color="#a8d5b5", edgecolor="white")

    for bar, val in zip(bars_a, asr_a):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars_b, asr_b):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_keys, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.set_title("Pairwise Ablation Comparisons — ASR",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(0, 115)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir / "04_pair_comparisons.png")


def fig_comprehensive_dashboard(methods: List[Dict[str, Any]],
                                 model_rows: List[Dict[str, Any]],
                                 summary_rows: List[Dict[str, Any]],
                                 out_dir: Path) -> None:
    """4-panel summary dashboard."""
    if not HAS_MPL:
        return

    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2])

    fig.suptitle("TRAP — Comprehensive Results Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)

    # Panel 1: ASR bar
    ms = sorted(methods, key=lambda m: -m["asr_pct"])
    names_p1  = [m["method"] for m in ms]
    values_p1 = [m["asr_pct"] for m in ms]
    colors_p1 = [_color(n) for n in names_p1]
    bars1 = ax1.barh(names_p1[::-1], values_p1[::-1],
                     color=colors_p1[::-1], edgecolor="white", height=0.65)
    for bar, val in zip(bars1, values_p1[::-1]):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold")
    ax1.set_xlabel("ASR (%)", fontsize=10)
    ax1.set_title("Attack Success Rate (avg across models)", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, 118)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", linestyle=":", alpha=0.4)

    # Panel 2: Ablation bar (if we have ablation data)
    if summary_rows:
        abl_sorted = sorted(summary_rows, key=lambda r: -float(r.get("asr_pct") or 0))
        abl_labels = [r["display_name"] for r in abl_sorted]
        abl_vals   = [float(r.get("asr_pct") or 0) for r in abl_sorted]
        abl_colors = [_color(r.get("experiment_id","")) for r in abl_sorted]
        bars2 = ax2.bar(range(len(abl_labels)), abl_vals,
                        color=abl_colors, width=0.6, edgecolor="white")
        for bar, val in zip(bars2, abl_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax2.set_xticks(range(len(abl_labels)))
        ax2.set_xticklabels(abl_labels, rotation=25, ha="right", fontsize=7)
        ax2.set_title("Ablation Study", fontsize=11, fontweight="bold")
        ax2.set_ylabel("ASR (%)", fontsize=9)
        ax2.set_ylim(0, 118)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="y", linestyle=":", alpha=0.4)
    else:
        ax2.text(0.5, 0.5, "No ablation data\n(run --mode ablation first)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=9, color="#888")
        ax2.set_title("Ablation Study", fontsize=11, fontweight="bold")
        ax2.axis("off")

    # Panel 3: model heatmap (TRAP + top baselines)
    if model_rows:
        top_methods = [m["method"] for m in sorted(methods, key=lambda x: -x["asr_pct"])[:5]]
        model_set   = sorted({r["model_label"] for r in model_rows})
        method_set  = [m for m in top_methods if any(r["method"] == m for r in model_rows)]
        data3 = np.zeros((len(method_set), len(model_set)))
        for r in model_rows:
            if r["method"] in method_set and r["model_label"] in model_set:
                data3[method_set.index(r["method"]), model_set.index(r["model_label"])] = r["asr_pct"]
        im3 = ax3.imshow(data3, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        ax3.set_xticks(range(len(model_set)))
        ax3.set_xticklabels(model_set, rotation=30, ha="right", fontsize=8)
        ax3.set_yticks(range(len(method_set)))
        ax3.set_yticklabels(method_set, fontsize=9)
        for i in range(len(method_set)):
            for j in range(len(model_set)):
                v   = data3[i, j]
                clr = "white" if v > 60 or v < 15 else "black"
                ax3.text(j, i, f"{v:.0f}%", ha="center", va="center",
                         fontsize=7.5, color=clr, fontweight="bold")
        fig.colorbar(im3, ax=ax3, shrink=0.75, label="ASR (%)")
        ax3.set_title("ASR: Methods × Target Models", fontsize=11, fontweight="bold")

    # Panel 4: Queries per behavior
    ms_q = [m for m in ms if m["avg_queries"] > 0]
    if ms_q:
        q_names  = [m["method"] for m in ms_q]
        q_vals   = [m["avg_queries"] for m in ms_q]
        q_colors = [_color(n) for n in q_names]
        bars4 = ax4.bar(q_names, q_vals, color=q_colors, width=0.6, edgecolor="white")
        for bar, val in zip(bars4, q_vals):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax4.set_title("Avg Queries per Behavior\n(lower = more efficient)",
                      fontsize=11, fontweight="bold")
        ax4.set_ylabel("Avg Queries", fontsize=9)
        ax4.set_xticklabels(q_names, rotation=25, ha="right", fontsize=8)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        ax4.grid(axis="y", linestyle=":", alpha=0.4)

    _save(fig, out_dir / "00_comprehensive_dashboard.png")


# =============================================================================
# Text report
# =============================================================================

def _line(char: str = "=", width: int = 100) -> str:
    return char * width


def generate_text_report(
    methods: List[Dict[str, Any]],
    model_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    pairs: List[Dict[str, Any]],
    run_name: str,
    out_dir: Path,
) -> Path:
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []

    def h1(title: str) -> None:
        lines.append(_line("="))
        lines.append(title.center(100))
        lines.append(_line("="))

    def h2(title: str) -> None:
        lines.append("")
        lines.append(_line("-"))
        lines.append(f"  {title}")
        lines.append(_line("-"))

    # ── Title ─────────────────────────────────────────────────────────────────
    h1("TRAP: TREE-BASED RL + RAG ATTACK — EVALUATION REPORT")
    lines.append(f"  Generated:   {now}")
    lines.append(f"  Run:         {run_name}")

    n_behaviors = int(methods[0]["avg_n"]) if methods else 0
    n_models    = max(m["n_models"] for m in methods) if methods else 0
    lines.append(f"  Behaviors:   {n_behaviors} per model")
    lines.append(f"  Models:      {n_models}")
    lines.append("")

    # ── Executive Summary ─────────────────────────────────────────────────────
    h2("EXECUTIVE SUMMARY")
    ours = next((m for m in methods if m["method"] == "TRAP"), None)
    baselines = [m for m in methods if m["method"] != "TRAP"]
    if ours:
        best_base = max(baselines, key=lambda m: m["asr_pct"]) if baselines else None
        lines.append("")
        lines.append(f"  TRAP achieves {ours['asr_pct']:.1f}% ASR on average across {n_models} target models.")
        if best_base:
            lines.append(f"  Best baseline: {best_base['method']} at {best_base['asr_pct']:.1f}%")
            lines.append(f"  TRAP advantage: +{ours['asr_pct'] - best_base['asr_pct']:.1f} percentage points")
        lines.append("")

    # ── Section I: Method Performance ─────────────────────────────────────────
    h2("SECTION I — OVERALL METHOD PERFORMANCE (avg across target models)")
    lines.append("")
    lines.append("  {:<24} {:>10} {:>16} {:>10}".format(
        "Method", "ASR (%)", "Avg Queries", "Avg Score"))
    lines.append("  " + "-" * 64)
    for m in sorted(methods, key=lambda x: -x["asr_pct"]):
        lines.append("  {:<24} {:>10} {:>16} {:>10}".format(
            m["method"],
            f"{m['asr_pct']:.2f}%",
            f"{m['avg_queries']:.1f}",
            f"{m['avg_score']:.3f}",
        ))

    # ── Section II: Per-model breakdown ───────────────────────────────────────
    if model_rows:
        h2("SECTION II — PER-TARGET-MODEL BREAKDOWN")
        lines.append("")
        all_methods_in_data = sorted({r["method"] for r in model_rows},
                                     key=lambda m: -sum(r["asr_pct"] for r in model_rows if r["method"] == m))
        col_header = "  {:<22}".format("Target Model")
        for me in all_methods_in_data:
            col_header += f"  {me:<14}"
        lines.append(col_header)
        lines.append("  " + "-" * (22 + 16 * len(all_methods_in_data)))

        models_in_data = sorted({r["model_label"] for r in model_rows})
        for model in models_in_data:
            row_str = f"  {model:<22}"
            for me in all_methods_in_data:
                entry = next((r for r in model_rows
                              if r["method"] == me and r["model_label"] == model), None)
                if entry:
                    row_str += f"  {entry['asr_pct']:>6.1f}%       "
                else:
                    row_str += f"  {'—':>6}         "
            lines.append(row_str)

    # ── Section III: Ablation Study ───────────────────────────────────────────
    if summary_rows:
        h2("SECTION III — ABLATION STUDY")
        lines.append("""
  The ablation study isolates the contribution of individual TRAP components.
  Each variant is defined in config/experiment_registry.yaml and executed
  under identical conditions (same seed, same KB snapshot, same behaviors).
""")
        lines.append("  {:<28} {:>10} {:>12} {:>12}  {}".format(
            "Variant (Exp ID)", "ASR (%)", "ASR CI-low", "ASR CI-high", "Configuration"))
        lines.append("  " + "-" * 96)
        for r in sorted(summary_rows, key=lambda x: -float(x.get("asr_pct") or 0)):
            exp_id = r.get("experiment_id", "?")
            name   = r.get("display_name", r.get("experiment_name", exp_id))
            asr    = float(r.get("asr_pct") or 0)
            ci_lo  = float(r.get("asr_seed_ci95_low") or 0)
            ci_hi  = float(r.get("asr_seed_ci95_high") or 0)
            tree   = r.get("tree_variant", "?")
            rag    = r.get("rag_enabled", "?")
            lines.append("  {:<28} {:>10} {:>12} {:>12}  tree={} rag={}".format(
                f"{exp_id}: {name}", f"{asr:.2f}%",
                f"{ci_lo:+.2f}pp", f"{ci_hi:+.2f}pp", tree, rag))

        if pairs:
            lines.append("")
            lines.append("  Pairwise comparisons:")
            lines.append("")
            lines.append("  {:<32} {:>8} {:>8} {:>10}  {}".format(
                "Pair", "ASR-A", "ASR-B", "Abs Diff", "p-value (McNemar)"))
            lines.append("  " + "-" * 80)
            for p in pairs:
                lines.append("  {:<32} {:>8} {:>8} {:>10}  {:.4f}".format(
                    p.get("pair_key", "?"),
                    f"{float(p.get('asr_a') or 0):.1f}%",
                    f"{float(p.get('asr_b') or 0):.1f}%",
                    f"{float(p.get('asr_abs_diff') or 0):+.1f}pp",
                    float(p.get("mcnemar_exact_p") or 1.0),
                ))

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(_line("="))
    lines.append("  End of Report".center(100))
    lines.append(_line("="))

    out_path = out_dir / "trap_evaluation_report.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [report] {out_path.name}")
    return out_path


# =============================================================================
# Excel workbook
# =============================================================================

def generate_excel(
    methods: List[Dict[str, Any]],
    model_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    pairs: List[Dict[str, Any]],
    out_dir: Path,
) -> Optional[Path]:
    if not HAS_XL:
        return None

    wb = openpyxl.Workbook()

    _hdr = {
        "font":      Font(bold=True, color="FFFFFF", size=11, name="Calibri"),
        "fill":      PatternFill("solid", fgColor="1a3a5c"),
        "alignment": Alignment(horizontal="center", vertical="center", wrap_text=True),
        "border":    Border(bottom=Side(style="medium", color="FFFFFF"),
                            right=Side(style="thin", color="AAAAAA")),
    }

    def _apply(cell: Any, style: Dict[str, Any]) -> None:
        for attr, val in style.items():
            setattr(cell, attr, val)

    def _auto_width(ws: Any) -> None:
        for col in ws.columns:
            max_len = max((len(str(c.value)) for c in col if c.value), default=8)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 50)

    def _add_sheet(ws: Any, headers: List[str], rows_data: List[List[Any]]) -> None:
        ws.row_dimensions[1].height = 28
        for ci, h in enumerate(headers, 1):
            _apply(ws.cell(row=1, column=ci, value=h), _hdr)
        row_fills = ["F2F9F5", "FFFFFF"]
        for ri, row in enumerate(rows_data, 2):
            fill = PatternFill("solid", fgColor=row_fills[ri % 2])
            for ci, val in enumerate(row, 1):
                cell = ws.cell(row=ri, column=ci, value=val)
                cell.fill = fill
                cell.font = Font(name="Calibri", size=10)
                cell.alignment = Alignment(horizontal="center" if ci > 1 else "left",
                                           vertical="center")
                cell.border = Border(bottom=Side(style="thin", color="DDDDDD"),
                                     right=Side(style="thin", color="DDDDDD"))
        _auto_width(ws)

    # ── Sheet 1: Method Summary ───────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Method Summary"
    ws1.freeze_panes = "A2"
    headers1 = ["Method", "ASR (%) avg", "Avg Queries/Behav", "Avg Score", "N Models"]
    rows1 = [
        [m["method"], round(m["asr_pct"], 2), round(m["avg_queries"], 1),
         round(m["avg_score"], 3), m["n_models"]]
        for m in sorted(methods, key=lambda x: -x["asr_pct"])
    ]
    _add_sheet(ws1, headers1, rows1)
    ws1.conditional_formatting.add(
        f"B2:B{1 + len(rows1)}",
        ColorScaleRule(start_type="num", start_value=0, start_color="FF4444",
                       mid_type="num", mid_value=50, mid_color="FFFF88",
                       end_type="num", end_value=100, end_color="22AA55"),
    )

    # ── Sheet 2: Per-Model ASR ───────────────────────────────────────────────
    if model_rows:
        ws2 = wb.create_sheet("Per-Model ASR")
        ws2.freeze_panes = "B2"
        all_m  = sorted({r["method"] for r in model_rows},
                        key=lambda m: -sum(r["asr_pct"] for r in model_rows if r["method"] == m))
        all_md = sorted({r["model_label"] for r in model_rows})
        headers2 = ["Target Model"] + all_m
        _apply(ws2.cell(row=1, column=1, value="Target Model"), _hdr)
        for ci, me in enumerate(all_m, 2):
            cell = ws2.cell(row=1, column=ci, value=me)
            _apply(cell, _hdr)
        for ri, model in enumerate(all_md, 2):
            ws2.cell(row=ri, column=1, value=model).font = Font(name="Calibri", bold=True, size=10)
            for ci, me in enumerate(all_m, 2):
                entry = next((r for r in model_rows if r["method"] == me and r["model_label"] == model), None)
                val = round(entry["asr_pct"], 2) if entry else None
                cell = ws2.cell(row=ri, column=ci, value=val)
                cell.alignment = Alignment(horizontal="center")
                cell.font = Font(name="Calibri", size=10)
        ws2.conditional_formatting.add(
            f"B2:{get_column_letter(len(all_m)+1)}{len(all_md)+1}",
            ColorScaleRule(start_type="num", start_value=0, start_color="FF4444",
                           mid_type="num", mid_value=50, mid_color="FFFF88",
                           end_type="num", end_value=100, end_color="22AA55"),
        )
        _auto_width(ws2)

    # ── Sheet 3: Ablation Study ──────────────────────────────────────────────
    if summary_rows:
        ws3 = wb.create_sheet("Ablation Study")
        headers3 = ["Exp ID", "Variant Name", "ASR (%)", "ASR CI-95 Low",
                    "ASR CI-95 High", "ASR Std", "Avg Score", "Avg Iterations",
                    "RAG Enabled", "Tree Variant", "N Seeds"]
        rows3 = []
        for r in sorted(summary_rows, key=lambda x: -float(x.get("asr_pct") or 0)):
            rows3.append([
                r.get("experiment_id", ""),
                r.get("display_name", r.get("experiment_name", "")),
                round(float(r.get("asr_pct") or 0), 2),
                round(float(r.get("asr_seed_ci95_low") or 0), 4),
                round(float(r.get("asr_seed_ci95_high") or 0), 4),
                round(float(r.get("asr_seed_std") or 0), 4),
                round(float(r.get("avg_max_score") or 0), 3),
                round(float(r.get("avg_iterations") or 0), 1),
                str(r.get("rag_enabled", "")),
                str(r.get("tree_variant", "")),
                str(r.get("n_seeds", "")),
            ])
        _add_sheet(ws3, headers3, rows3)
        ws3.conditional_formatting.add(
            f"C2:C{1 + len(rows3)}",
            ColorScaleRule(start_type="num", start_value=0, start_color="FF4444",
                           mid_type="num", mid_value=50, mid_color="FFFF88",
                           end_type="num", end_value=100, end_color="22AA55"),
        )

    # ── Sheet 4: Pairwise Comparisons ────────────────────────────────────────
    if pairs:
        ws4 = wb.create_sheet("Pair Comparisons")
        headers4 = ["Pair", "Exp A", "Exp B", "ASR A (%)", "ASR B (%)",
                    "Abs Diff (pp)", "Rel Diff (%)", "Wins A", "Wins B", "Ties",
                    "McNemar p", "Wilcoxon p", "CI-95 Low", "CI-95 High"]
        rows4 = [
            [p.get("pair_key",""), p.get("experiment_a",""), p.get("experiment_b",""),
             round(float(p.get("asr_a") or 0), 2), round(float(p.get("asr_b") or 0), 2),
             round(float(p.get("asr_abs_diff") or 0), 2),
             round(float(p.get("asr_rel_diff_pct") or 0), 2),
             int(float(p.get("wins_a") or 0)), int(float(p.get("wins_b") or 0)),
             int(float(p.get("ties") or 0)),
             round(float(p.get("mcnemar_exact_p") or 1.0), 6),
             round(float(p.get("wilcoxon_score_p") or 1.0), 6),
             round(float(p.get("asr_diff_ci95_low") or 0), 2),
             round(float(p.get("asr_diff_ci95_high") or 0), 2)]
            for p in pairs
        ]
        _add_sheet(ws4, headers4, rows4)

    # ── Sheet 5: Metadata ─────────────────────────────────────────────────────
    ws5 = wb.create_sheet("Metadata")
    meta = [
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Source", "results/ and results/ablations/ — real run data"),
        ("Ablation IDs", ", ".join(EXP_DISPLAY.keys())),
        ("Script", "scripts/improve_outputs.py"),
        ("Note", "Ablation names follow experiment_registry.yaml — no ad-hoc variants"),
    ]
    for ri, (k, v) in enumerate(meta, 1):
        ws5.cell(row=ri, column=1, value=k).font = Font(bold=True, name="Calibri")
        ws5.cell(row=ri, column=2, value=v).font = Font(name="Calibri")
    _auto_width(ws5)

    xl_path = out_dir / "trap_results.xlsx"
    wb.save(xl_path)
    print(f"  [excel] {xl_path.name}")
    return xl_path


# =============================================================================
# Top-level orchestrator
# =============================================================================

def generate_improved_outputs(
    results: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    study_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Entry point for both CLI and programmatic use (called from main.py output_polisher).

    Args:
        results:    Raw results dict (single-run mode, from output_polisher).
        output_dir: Where to write outputs (created if missing).
        run_dir:    Path to a comparison run directory.
        study_id:   Ablation study ID to load from results/ablations/.

    Returns:
        Dict with paths to generated artifacts.
    """
    # ── Resolve output dir ────────────────────────────────────────────────────
    if output_dir is None:
        if run_dir is not None:
            output_dir = run_dir / "improved_outputs"
        else:
            output_dir = _RESULTS_ROOT / "improved_outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  TRAP Output Generator")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # ── Load comparison run data ──────────────────────────────────────────────
    methods: List[Dict[str, Any]]   = []
    model_rows: List[Dict[str, Any]] = []

    if run_dir is None:
        run_dir = auto_detect_run()

    if run_dir and run_dir.exists():
        try:
            run_data   = load_comparison_run(run_dir)
            model_rows = run_data["results"]
            methods    = _aggregate_method_across_models(model_rows)
            print(f"  Loaded {len(model_rows)} result entries from {run_dir.name}")
        except Exception as exc:
            print(f"  [WARN] Could not load comparison run: {exc}")
    elif results:
        # Called from output_polisher with a single-run result dict
        # Build a minimal method entry for the report
        asr = results.get("asr_percent") or (
            100.0 * int(bool(results.get("success")))
        )
        methods = [{"method": "TRAP", "asr_pct": round(float(asr), 2),
                    "avg_queries": 0.0, "avg_score": float(results.get("best_score", 0)),
                    "n_models": 1, "avg_success": 0, "avg_n": 0}]

    # ── Load ablation data ────────────────────────────────────────────────────
    summary_rows: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]]        = []

    if study_id:
        try:
            abl_data    = load_ablation_study(study_id)
            summary_rows = abl_data["summary"]
            pairs        = abl_data["pairs"]
            print(f"  Loaded {len(summary_rows)} ablation variants from study '{study_id}'")
        except Exception as exc:
            print(f"  [WARN] Could not load ablation study '{study_id}': {exc}")
    else:
        # Try to auto-detect the most recent ablation study
        if _ABLATIONS_ROOT.exists():
            study_dirs = sorted(
                [d for d in _ABLATIONS_ROOT.iterdir() if d.is_dir()
                 and (d / "aggregate" / "ablation_summary.csv").exists()],
                key=lambda d: d.stat().st_mtime,
            )
            if study_dirs:
                latest_study = study_dirs[-1].name
                try:
                    abl_data     = load_ablation_study(latest_study)
                    summary_rows = abl_data["summary"]
                    pairs        = abl_data["pairs"]
                    print(f"  Auto-detected ablation study '{latest_study}': "
                          f"{len(summary_rows)} variants")
                except Exception as exc:
                    print(f"  [WARN] Auto-detect ablation failed: {exc}")

    run_name = run_dir.name if run_dir else "N/A"

    # ── Generate figures ──────────────────────────────────────────────────────
    fig_dir = output_dir / "figures"
    if HAS_MPL:
        print("\n[FIGURES] Generating figures...")
        fig_comprehensive_dashboard(methods, model_rows, summary_rows, fig_dir)
        if methods:
            fig_asr_bar(methods, fig_dir)
        if len(model_rows) > 1:
            fig_model_heatmap(model_rows, fig_dir)
        if summary_rows:
            fig_ablation(summary_rows, fig_dir, pairs)
        if pairs:
            registry = _load_registry()
            exp_names = {k: v.get("name", k)
                         for k, v in registry.get("experiments", {}).items()}
            fig_pair_comparisons(pairs, fig_dir, exp_names)
        print("[FIGURES] Done.")

    # ── Generate text report ──────────────────────────────────────────────────
    print("\n[REPORT] Generating text report...")
    report_path = generate_text_report(
        methods, model_rows, summary_rows, pairs, run_name, output_dir)

    # ── Generate Excel workbook ───────────────────────────────────────────────
    print("\n[EXCEL] Generating Excel workbook...")
    xl_path = generate_excel(methods, model_rows, summary_rows, pairs, output_dir)

    print(f"\n{'='*60}")
    print(f"  All outputs saved to: {output_dir}")
    print(f"{'='*60}\n")

    return {
        "output_dir": str(output_dir),
        "report":     str(report_path),
        "excel":      str(xl_path) if xl_path else None,
        "figures_dir": str(fig_dir),
    }


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TRAP output reports and figures")
    parser.add_argument("--run", type=str, default=None,
                        help="Run directory name under results/ (default: auto-detect latest)")
    parser.add_argument("--study", type=str, default=None,
                        help="Ablation study ID under results/ablations/")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: <run_dir>/improved_outputs)")
    args = parser.parse_args()

    run_dir = (_RESULTS_ROOT / args.run) if args.run else None
    out_dir = Path(args.out) if args.out else None

    generate_improved_outputs(
        run_dir=run_dir,
        output_dir=out_dir,
        study_id=args.study,
    )


if __name__ == "__main__":
    main()
