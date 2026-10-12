"""
Comprehensive Visualization & Tracking System for TRAP Attack Method

Tracks and visualizes the full lifecycle across a dataset run:
- Model weight/preference evolution per level across behaviors
- Per-iteration input/output details until success
- Leaf node effectiveness ranking over entire dataset
- Score progression across all behaviors
- Template effectiveness heatmaps
- RAG contribution analysis
- Reward component breakdown
- Dataset-level success/failure patterns
- Real-time progress dashboard
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import math

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.patches as mpatches
    _MATPLOTLIB_OK = True
except ImportError:
    _MATPLOTLIB_OK = False

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """One iteration within a behavior attack."""
    behavior_idx: int
    behavior_goal: str
    iteration: int
    template_name: str
    leaf_name: str
    source: str
    prompt_preview: str
    response_preview: str
    judge_score: float
    on_topic: float
    is_refusal: bool
    reward: float
    prompt_len: int
    response_len: int
    timestamp: float
    rag_used: bool
    success: bool
    weight_snapshot: Optional[Dict[str, float]] = None
    prompt_text: str = ""
    response_text: str = ""


@dataclass
class BehaviorRecord:
    """Summary for one behavior/goal across all its iterations."""
    behavior_idx: int
    goal: str
    success: bool
    best_score: float
    iterations_count: int
    iterations_to_success: Optional[int]
    winning_template: Optional[str]
    winning_leaf: Optional[str]
    score_trajectory: List[float] = field(default_factory=list)
    templates_tried: List[str] = field(default_factory=list)
    leaves_tried: List[str] = field(default_factory=list)
    reward_trajectory: List[float] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class LeafStats:
    """Aggregate statistics for one leaf node across the full dataset."""
    name: str
    total_uses: int = 0
    successes: int = 0
    total_score: float = 0.0
    max_score: float = 0.0
    scores: List[float] = field(default_factory=list)
    behaviors_used_on: int = 0
    behaviors_succeeded: int = 0
    avg_iteration_when_used: float = 0.0
    iteration_positions: List[int] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.total_uses, 1)

    @property
    def avg_score(self) -> float:
        return self.total_score / max(self.total_uses, 1)


@dataclass
class TemplateStats:
    """Aggregate statistics for one template across the full dataset."""
    name: str
    total_uses: int = 0
    successes: int = 0
    total_score: float = 0.0
    max_score: float = 0.0
    scores: List[float] = field(default_factory=list)
    behaviors_used_on: int = 0
    behaviors_succeeded: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.total_uses, 1)

    @property
    def avg_score(self) -> float:
        return self.total_score / max(self.total_uses, 1)


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class DatasetTracker:
    """
    Central tracker for the entire dataset run.
    Call record_iteration() after each iteration and
    finalize_behavior() after each behavior completes.
    At the end call generate_all_reports().
    """

    def __init__(self, output_dir: str, run_tag: str = "", enabled: bool = True):
        self.enabled = enabled
        self.run_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.data_dir = self.output_dir / "data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()

        # Storage
        self.iteration_records: List[IterationRecord] = []
        self.behavior_records: List[BehaviorRecord] = []
        self.leaf_stats: Dict[str, LeafStats] = defaultdict(lambda: LeafStats(name=""))
        self.template_stats: Dict[str, TemplateStats] = defaultdict(lambda: TemplateStats(name=""))

        # Running counters
        self.total_behaviors = 0
        self.successful_behaviors = 0
        self.total_iterations = 0

        # Weight snapshots: behavior_idx -> iteration -> {leaf_name: preference_weight}
        self.weight_history: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(dict)

        # Dataset-level score trajectory (one entry per behavior = best score achieved)
        self.dataset_score_trajectory: List[float] = []

        # Per-behavior score trajectories for heatmap
        # behavior_idx -> list of scores per iteration
        self.per_behavior_scores: Dict[int, List[float]] = defaultdict(list)

        # Reward component breakdown per behavior
        # behavior_idx -> list of {component: value}
        self.reward_components: Dict[int, List[Dict[str, float]]] = defaultdict(list)

        self._start_time = time.time()
        self._current_behavior_start = time.time()
        self.live_progress_file = self.output_dir / f"live_progress_{self.run_tag}.json"

        logger.info(f"[VIZ] DatasetTracker initialized -> {self.output_dir}")
        self._write_live_progress_snapshot(extra={"event": "initialized"})

    # ------------------------------------------------------------------
    # Public API: called from run_attack_with_rag
    # ------------------------------------------------------------------

    def start_behavior(self, behavior_idx: int, goal: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._current_behavior_start = time.time()
            self.total_behaviors = max(self.total_behaviors, behavior_idx + 1)
            logger.debug(f"[VIZ] start_behavior idx={behavior_idx} goal='{goal[:60]}'")
            self._write_live_progress_snapshot(extra={
                "event": "start_behavior",
                "behavior_idx": behavior_idx,
                "goal_preview": goal[:200],
            })

    def record_iteration(
        self,
        behavior_idx: int,
        goal: str,
        iteration: int,
        template_name: str,
        leaf_name: str,
        source: str,
        prompt: str,
        response: str,
        judge_score: float,
        on_topic: float,
        is_refusal: bool,
        reward: float,
        rag_used: bool,
        success: bool,
        weight_snapshot: Optional[Dict[str, float]] = None,
        reward_components: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            rec = IterationRecord(
                behavior_idx=behavior_idx,
                behavior_goal=goal,
                iteration=iteration,
                template_name=template_name or "unknown",
                leaf_name=leaf_name or "unknown",
                source=source or "unknown",
                prompt_preview=prompt[:300],
                response_preview=response[:300],
                judge_score=float(judge_score),
                on_topic=float(on_topic),
                is_refusal=bool(is_refusal),
                reward=float(reward),
                prompt_len=len(prompt),
                response_len=len(response),
                timestamp=time.time(),
                rag_used=bool(rag_used),
                success=bool(success),
                weight_snapshot=weight_snapshot,
                prompt_text=prompt,
                response_text=response,
            )
            self.iteration_records.append(rec)
            self.total_iterations += 1

            # Update per-behavior score list
            self.per_behavior_scores[behavior_idx].append(float(judge_score))

            # Update leaf stats
            ls = self.leaf_stats[leaf_name]
            ls.name = leaf_name
            ls.total_uses += 1
            ls.total_score += float(judge_score)
            ls.max_score = max(ls.max_score, float(judge_score))
            ls.scores.append(float(judge_score))
            ls.iteration_positions.append(iteration)
            if success:
                ls.successes += 1

            # Update template stats
            ts = self.template_stats[template_name]
            ts.name = template_name
            ts.total_uses += 1
            ts.total_score += float(judge_score)
            ts.max_score = max(ts.max_score, float(judge_score))
            ts.scores.append(float(judge_score))
            if success:
                ts.successes += 1

            # Weight snapshot
            if weight_snapshot:
                self.weight_history[behavior_idx][iteration] = weight_snapshot

            # Reward components
            if reward_components:
                self.reward_components[behavior_idx].append(reward_components)
            self._write_live_progress_snapshot(extra={
                "event": "iteration_recorded",
                "behavior_idx": behavior_idx,
                "iteration": iteration,
                "score": float(judge_score),
                "success": bool(success),
            })

    def finalize_behavior(
        self,
        behavior_idx: int,
        goal: str,
        success: bool,
        best_score: float,
        iterations_count: int,
        winning_template: Optional[str] = None,
        winning_leaf: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            iters_for_b = [r for r in self.iteration_records if r.behavior_idx == behavior_idx]
            score_traj = [r.judge_score for r in iters_for_b]
            reward_traj = [r.reward for r in iters_for_b]
            templates_tried = list(dict.fromkeys(r.template_name for r in iters_for_b))
            leaves_tried = list(dict.fromkeys(r.leaf_name for r in iters_for_b))

            iters_to_success = None
            if success:
                for r in iters_for_b:
                    if r.success:
                        iters_to_success = r.iteration
                        break

            brec = BehaviorRecord(
                behavior_idx=behavior_idx,
                goal=goal,
                success=success,
                best_score=best_score,
                iterations_count=iterations_count,
                iterations_to_success=iters_to_success,
                winning_template=winning_template,
                winning_leaf=winning_leaf,
                score_trajectory=score_traj,
                templates_tried=templates_tried,
                leaves_tried=leaves_tried,
                reward_trajectory=reward_traj,
                start_time=self._current_behavior_start,
                end_time=time.time(),
            )
            self.behavior_records.append(brec)

            # Update leaf/template "behaviors" counters
            for leaf in leaves_tried:
                self.leaf_stats[leaf].behaviors_used_on += 1
                if success:
                    self.leaf_stats[leaf].behaviors_succeeded += 1

            for tmpl in templates_tried:
                self.template_stats[tmpl].behaviors_used_on += 1
                if success:
                    self.template_stats[tmpl].behaviors_succeeded += 1

            if success:
                self.successful_behaviors += 1
            self.dataset_score_trajectory.append(float(best_score))

            logger.info(
                f"[VIZ] finalized behavior {behavior_idx} | success={success} "
                f"best_score={best_score:.1f} iters={iterations_count}"
            )
            self._write_live_progress_snapshot(extra={
                "event": "behavior_finalized",
                "behavior_idx": behavior_idx,
                "best_score": float(best_score),
                "success": bool(success),
            })

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_all_reports(self) -> Dict[str, Path]:
        """Generate all charts and data files. Returns dict of path names."""
        if not self.enabled:
            return {}
        paths: Dict[str, Path] = {}
        logger.info("[VIZ] Generating all reports...")

        # 1. Export raw data to CSV/JSON
        paths.update(self._export_raw_data())

        if not _MATPLOTLIB_OK:
            logger.warning("[VIZ] matplotlib not available, skipping charts")
            return paths

        # 2. Dataset-level progress chart
        p = self._plot_dataset_progress()
        if p:
            paths["dataset_progress"] = p

        # 3. Score heatmap across behaviors x iterations
        p = self._plot_score_heatmap()
        if p:
            paths["score_heatmap"] = p

        # 4. Leaf node ranking (bar charts)
        p = self._plot_leaf_ranking()
        if p:
            paths["leaf_ranking"] = p

        # 5. Template effectiveness
        p = self._plot_template_effectiveness()
        if p:
            paths["template_effectiveness"] = p

        # 6. Weight evolution per behavior
        p = self._plot_weight_evolution()
        if p:
            paths["weight_evolution"] = p

        # 7. Per-behavior score trajectories (grid)
        p = self._plot_per_behavior_trajectories()
        if p:
            paths["behavior_trajectories"] = p

        # 8. Reward breakdown
        p = self._plot_reward_breakdown()
        if p:
            paths["reward_breakdown"] = p

        # 9. RAG vs RL contribution
        p = self._plot_rag_vs_rl()
        if p:
            paths["rag_vs_rl"] = p

        # 10. Summary dashboard
        p = self._plot_summary_dashboard()
        if p:
            paths["summary_dashboard"] = p

        # 11. Iterations-to-success distribution
        p = self._plot_iterations_to_success()
        if p:
            paths["iterations_to_success"] = p

        # 12. Score distribution
        p = self._plot_score_distribution()
        if p:
            paths["score_distribution"] = p

        logger.info(f"[VIZ] Reports done. {len(paths)} artifacts generated.")
        for k, v in paths.items():
            logger.info(f"  [{k}] -> {v}")
        return paths

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------

    def _export_raw_data(self) -> Dict[str, Path]:
        paths = {}
        ts = self.run_tag
        iteration_rows = []
        behavior_rows = []
        leaf_ranking_rows = []
        template_rows = []

        # Iteration-level CSV
        if self.iteration_records:
            for r in self.iteration_records:
                iteration_rows.append({
                    "behavior_idx": r.behavior_idx,
                    "goal": r.behavior_goal[:80],
                    "iteration": r.iteration,
                    "template": r.template_name,
                    "leaf": r.leaf_name,
                    "source": r.source,
                    "judge_score": r.judge_score,
                    "on_topic": r.on_topic,
                    "is_refusal": int(r.is_refusal),
                    "reward": r.reward,
                    "prompt_len": r.prompt_len,
                    "response_len": r.response_len,
                    "rag_used": int(r.rag_used),
                    "success": int(r.success),
                })
            p = self.data_dir / f"iterations_{ts}.csv"
            try:
                if _PANDAS_OK:
                    pd.DataFrame(iteration_rows).to_csv(p, index=False)
                else:
                    import csv
                    with open(p, "w", newline="", encoding="utf-8") as f:
                        if iteration_rows:
                            writer = csv.DictWriter(f, fieldnames=iteration_rows[0].keys())
                            writer.writeheader()
                            writer.writerows(iteration_rows)
                paths["iterations_csv"] = p
            except Exception as e:
                logger.warning(f"[VIZ] Could not export iterations CSV: {e}")

        # Behavior-level JSON
        if self.behavior_records:
            data = []
            for b in self.behavior_records:
                d = asdict(b)
                data.append(d)
            p = self.data_dir / f"behaviors_{ts}.json"
            try:
                p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
                paths["behaviors_json"] = p
            except Exception as e:
                logger.warning(f"[VIZ] Could not export behaviors JSON: {e}")

        # Leaf stats JSON
        if self.leaf_stats:
            leaf_data = {}
            for name, ls in self.leaf_stats.items():
                leaf_data[name] = {
                    "total_uses": ls.total_uses,
                    "successes": ls.successes,
                    "success_rate": round(ls.success_rate, 4),
                    "avg_score": round(ls.avg_score, 4),
                    "max_score": ls.max_score,
                    "behaviors_used_on": ls.behaviors_used_on,
                    "behaviors_succeeded": ls.behaviors_succeeded,
                }
                quality = 0.7 * ls.success_rate + 0.3 * (ls.avg_score / 10.0)
                leaf_ranking_rows.append({
                    "leaf": name,
                    "total_uses": ls.total_uses,
                    "successes": ls.successes,
                    "success_rate": round(ls.success_rate, 6),
                    "avg_score": round(ls.avg_score, 6),
                    "max_score": round(ls.max_score, 6),
                    "quality_index": round(quality, 6),
                    "behaviors_used_on": ls.behaviors_used_on,
                    "behaviors_succeeded": ls.behaviors_succeeded,
                })
            p = self.data_dir / f"leaf_stats_{ts}.json"
            try:
                p.write_text(json.dumps(leaf_data, indent=2), encoding="utf-8")
                paths["leaf_stats_json"] = p
            except Exception as e:
                logger.warning(f"[VIZ] Could not export leaf stats JSON: {e}")

        # Template stats JSON
        if self.template_stats:
            tmpl_data = {}
            for name, ts_obj in self.template_stats.items():
                tmpl_data[name] = {
                    "total_uses": ts_obj.total_uses,
                    "successes": ts_obj.successes,
                    "success_rate": round(ts_obj.success_rate, 4),
                    "avg_score": round(ts_obj.avg_score, 4),
                    "max_score": ts_obj.max_score,
                    "behaviors_used_on": ts_obj.behaviors_used_on,
                }
                template_rows.append({
                    "template": name,
                    "total_uses": ts_obj.total_uses,
                    "successes": ts_obj.successes,
                    "success_rate": round(ts_obj.success_rate, 6),
                    "avg_score": round(ts_obj.avg_score, 6),
                    "max_score": round(ts_obj.max_score, 6),
                    "behaviors_used_on": ts_obj.behaviors_used_on,
                })
            p = self.data_dir / f"template_stats_{ts}.json"
            try:
                p.write_text(json.dumps(tmpl_data, indent=2), encoding="utf-8")
                paths["template_stats_json"] = p
            except Exception as e:
                logger.warning(f"[VIZ] Could not export template stats JSON: {e}")

        # Full iteration traces JSONL (with full prompt/response + weights)
        if self.iteration_records:
            p = self.data_dir / f"iteration_traces_{ts}.jsonl"
            try:
                with open(p, "w", encoding="utf-8") as f:
                    for r in self.iteration_records:
                        obj = asdict(r)
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                paths["iteration_traces_jsonl"] = p
            except Exception as e:
                logger.warning(f"[VIZ] Could not export iteration traces JSONL: {e}")

        # Behavior journey JSON: until first success (or full if never successful)
        if self.behavior_records and self.iteration_records:
            p = self.data_dir / f"behavior_journeys_{ts}.json"
            try:
                journeys = {}
                for b in self.behavior_records:
                    iters = [r for r in self.iteration_records if r.behavior_idx == b.behavior_idx]
                    cutoff = len(iters)
                    for i, r in enumerate(iters):
                        if r.success:
                            cutoff = i + 1
                            break
                    trimmed = iters[:cutoff]
                    journeys[str(b.behavior_idx)] = {
                        "goal": b.goal,
                        "success": b.success,
                        "best_score": b.best_score,
                        "iterations_in_journey": cutoff,
                        "steps": [
                            {
                                "iteration": r.iteration,
                                "template": r.template_name,
                                "leaf": r.leaf_name,
                                "source": r.source,
                                "judge_score": r.judge_score,
                                "on_topic": r.on_topic,
                                "success": r.success,
                                "prompt": r.prompt_text,
                                "response": r.response_text,
                                "weight_snapshot": r.weight_snapshot or {},
                            }
                            for r in trimmed
                        ]
                    }
                p.write_text(json.dumps(journeys, indent=2, ensure_ascii=False), encoding="utf-8")
                paths["behavior_journeys_json"] = p
            except Exception as e:
                logger.warning(f"[VIZ] Could not export behavior journeys JSON: {e}")

        # Leaf ranking CSV with both best and worst ordering signals
        if self.leaf_stats:
            p = self.data_dir / f"leaf_ranking_{ts}.csv"
            try:
                leaf_ranking_rows.sort(key=lambda x: (x["quality_index"], x["total_uses"]), reverse=True)
                if _PANDAS_OK:
                    pd.DataFrame(leaf_ranking_rows).to_csv(p, index=False)
                else:
                    import csv
                    with open(p, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=leaf_ranking_rows[0].keys())
                        writer.writeheader()
                        writer.writerows(leaf_ranking_rows)
                paths["leaf_ranking_csv"] = p
            except Exception as e:
                logger.warning(f"[VIZ] Could not export leaf ranking CSV: {e}")

        # Weight evolution summary JSON (delta magnitude per behavior/iteration)
        if self.weight_history:
            p = self.data_dir / f"weight_evolution_summary_{ts}.json"
            try:
                summary = {}
                for bidx, iter_map in self.weight_history.items():
                    ordered = sorted(iter_map.items(), key=lambda x: x[0])
                    deltas = []
                    prev = None
                    for it_idx, weights in ordered:
                        if prev is None:
                            deltas.append({
                                "iteration": it_idx,
                                "delta_l1": 0.0,
                                "max_shift_leaf": None,
                                "max_shift_value": 0.0,
                            })
                            prev = weights
                            continue
                        all_keys = set(prev.keys()) | set(weights.keys())
                        shift = {k: abs(float(weights.get(k, 0.0)) - float(prev.get(k, 0.0))) for k in all_keys}
                        if shift:
                            max_leaf = max(shift.items(), key=lambda x: x[1])
                            max_shift_leaf, max_shift_value = max_leaf[0], float(max_leaf[1])
                            l1 = float(sum(shift.values()))
                        else:
                            max_shift_leaf, max_shift_value, l1 = None, 0.0, 0.0
                        deltas.append({
                            "iteration": it_idx,
                            "delta_l1": l1,
                            "max_shift_leaf": max_shift_leaf,
                            "max_shift_value": max_shift_value,
                        })
                        prev = weights
                    summary[str(bidx)] = {
                        "iterations": [it for it, _ in ordered],
                        "deltas": deltas,
                    }
                p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                paths["weight_evolution_summary_json"] = p
            except Exception as e:
                logger.warning(f"[VIZ] Could not export weight evolution summary JSON: {e}")

        # Behavior summary CSV (good for quick slicing in spreadsheets)
        if self.behavior_records:
            p = self.data_dir / f"behavior_summary_{ts}.csv"
            try:
                for b in self.behavior_records:
                    behavior_rows.append({
                        "behavior_idx": b.behavior_idx,
                        "goal": b.goal,
                        "success": int(b.success),
                        "best_score": b.best_score,
                        "iterations_count": b.iterations_count,
                        "iterations_to_success": b.iterations_to_success,
                        "winning_template": b.winning_template or "",
                        "winning_leaf": b.winning_leaf or "",
                        "templates_tried_count": len(b.templates_tried),
                        "leaves_tried_count": len(b.leaves_tried),
                        "duration_seconds": round(max(0.0, b.end_time - b.start_time), 3),
                    })
                if _PANDAS_OK:
                    pd.DataFrame(behavior_rows).to_csv(p, index=False)
                else:
                    import csv
                    with open(p, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=behavior_rows[0].keys())
                        writer.writeheader()
                        writer.writerows(behavior_rows)
                paths["behavior_summary_csv"] = p
            except Exception as e:
                logger.warning(f"[VIZ] Could not export behavior summary CSV: {e}")

        # Comprehensive Excel workbook (multi-sheet)
        if _PANDAS_OK:
            xlsx_path = self.data_dir / f"comprehensive_analysis_{ts}.xlsx"
            try:
                with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                    if behavior_rows:
                        pd.DataFrame(behavior_rows).to_excel(writer, sheet_name="behavior_summary", index=False)
                    if iteration_rows:
                        pd.DataFrame(iteration_rows).to_excel(writer, sheet_name="iterations", index=False)
                    if leaf_ranking_rows:
                        pd.DataFrame(leaf_ranking_rows).to_excel(writer, sheet_name="leaf_ranking", index=False)
                    if template_rows:
                        pd.DataFrame(template_rows).to_excel(writer, sheet_name="template_stats", index=False)
                    if self.reward_components:
                        reward_rows = []
                        for bidx, comp_list in self.reward_components.items():
                            for i, comp in enumerate(comp_list, start=1):
                                row = {"behavior_idx": bidx, "sequence": i}
                                row.update(comp)
                                reward_rows.append(row)
                        if reward_rows:
                            pd.DataFrame(reward_rows).to_excel(writer, sheet_name="reward_components", index=False)
                    # Pivot-ready long-form table for Excel pivots/charts
                    if iteration_rows:
                        pivot_ready_rows = []
                        for r in iteration_rows:
                            pivot_ready_rows.append({
                                "behavior_idx": r.get("behavior_idx"),
                                "iteration": r.get("iteration"),
                                "template": r.get("template"),
                                "leaf": r.get("leaf"),
                                "source": r.get("source"),
                                "rag_used": r.get("rag_used"),
                                "success": r.get("success"),
                                "metric": "judge_score",
                                "value": r.get("judge_score", 0),
                            })
                            pivot_ready_rows.append({
                                "behavior_idx": r.get("behavior_idx"),
                                "iteration": r.get("iteration"),
                                "template": r.get("template"),
                                "leaf": r.get("leaf"),
                                "source": r.get("source"),
                                "rag_used": r.get("rag_used"),
                                "success": r.get("success"),
                                "metric": "reward",
                                "value": r.get("reward", 0),
                            })
                            pivot_ready_rows.append({
                                "behavior_idx": r.get("behavior_idx"),
                                "iteration": r.get("iteration"),
                                "template": r.get("template"),
                                "leaf": r.get("leaf"),
                                "source": r.get("source"),
                                "rag_used": r.get("rag_used"),
                                "success": r.get("success"),
                                "metric": "prompt_len",
                                "value": r.get("prompt_len", 0),
                            })
                            pivot_ready_rows.append({
                                "behavior_idx": r.get("behavior_idx"),
                                "iteration": r.get("iteration"),
                                "template": r.get("template"),
                                "leaf": r.get("leaf"),
                                "source": r.get("source"),
                                "rag_used": r.get("rag_used"),
                                "success": r.get("success"),
                                "metric": "response_len",
                                "value": r.get("response_len", 0),
                            })
                        pd.DataFrame(pivot_ready_rows).to_excel(writer, sheet_name="pivot_ready_long", index=False)
                    # Ranking sheets
                    if leaf_ranking_rows:
                        leaf_sorted = sorted(leaf_ranking_rows, key=lambda x: (x["quality_index"], x["total_uses"]), reverse=True)
                        pd.DataFrame(leaf_sorted[:50]).to_excel(writer, sheet_name="top50_leaves", index=False)
                        pd.DataFrame(leaf_sorted[-50:]).to_excel(writer, sheet_name="bottom50_leaves", index=False)
                    if template_rows:
                        tmpl_sorted = sorted(template_rows, key=lambda x: (x["success_rate"], x["avg_score"]), reverse=True)
                        pd.DataFrame(tmpl_sorted[:50]).to_excel(writer, sheet_name="top50_templates", index=False)
                paths["comprehensive_excel"] = xlsx_path
            except Exception as e:
                logger.warning(f"[VIZ] Could not export comprehensive Excel workbook: {e}")

        # Human-readable text summary
        txt_path = self.output_dir / f"RUN_SUMMARY_{ts}.txt"
        try:
            top_leaf_names = [r["leaf"] for r in leaf_ranking_rows[:10]]
            bottom_leaf_names = [r["leaf"] for r in leaf_ranking_rows[-10:]] if leaf_ranking_rows else []
            top_templates = sorted(template_rows, key=lambda x: (x["success_rate"], x["avg_score"]), reverse=True)[:10]
            total_behaviors = len(self.behavior_records)
            sr = self.successful_behaviors / max(total_behaviors, 1)
            mean_best = (
                sum(b.best_score for b in self.behavior_records) / max(total_behaviors, 1)
                if self.behavior_records else 0.0
            )
            lines = [
                "TRAP DATASET RUN SUMMARY",
                f"run_tag={self.run_tag}",
                f"total_behaviors={total_behaviors}",
                f"successful_behaviors={self.successful_behaviors}",
                f"success_rate={sr:.4f}",
                f"total_iterations={self.total_iterations}",
                f"mean_best_score={mean_best:.4f}",
                "",
                "Top leaves:",
                ", ".join(top_leaf_names) if top_leaf_names else "-",
                "",
                "Bottom leaves:",
                ", ".join(bottom_leaf_names) if bottom_leaf_names else "-",
                "",
                "Top templates:",
            ]
            for t in top_templates:
                lines.append(
                    f"- {t['template']} | sr={t['success_rate']:.3f} avg={t['avg_score']:.3f} uses={t['total_uses']}"
                )
            txt_path.write_text("\n".join(lines), encoding="utf-8")
            paths["run_summary_txt"] = txt_path
        except Exception as e:
            logger.warning(f"[VIZ] Could not export text summary: {e}")

        # Markdown executive report
        md_path = self.output_dir / f"DATASET_EXECUTIVE_REPORT_{ts}.md"
        try:
            total_behaviors = len(self.behavior_records)
            sr = self.successful_behaviors / max(total_behaviors, 1)
            mean_best = (
                sum(b.best_score for b in self.behavior_records) / max(total_behaviors, 1)
                if self.behavior_records else 0.0
            )
            top5 = leaf_ranking_rows[:5]
            bottom5 = leaf_ranking_rows[-5:] if leaf_ranking_rows else []
            top_tmpl5 = sorted(template_rows, key=lambda x: (x["success_rate"], x["avg_score"]), reverse=True)[:5]
            fail_behaviors = [b for b in self.behavior_records if not b.success][:10]
            success_behaviors = [b for b in self.behavior_records if b.success][:10]

            md = []
            md.append(f"# Dataset Executive Report ({ts})")
            md.append("")
            md.append("## Run KPIs")
            md.append(f"- Total behaviors: **{total_behaviors}**")
            md.append(f"- Successful behaviors: **{self.successful_behaviors}**")
            md.append(f"- Success rate: **{sr:.2%}**")
            md.append(f"- Total iterations: **{self.total_iterations}**")
            md.append(f"- Mean best score: **{mean_best:.3f}/10**")
            md.append("")
            md.append("## Best Leaves (Top 5)")
            if top5:
                md.append("| Leaf | Uses | SR | Avg Score | Quality |")
                md.append("|---|---:|---:|---:|---:|")
                for r in top5:
                    md.append(
                        f"| {r['leaf']} | {r['total_uses']} | {r['success_rate']:.2%} | {r['avg_score']:.3f} | {r['quality_index']:.3f} |"
                    )
            else:
                md.append("- No leaf stats available.")
            md.append("")
            md.append("## Weak Leaves (Bottom 5)")
            if bottom5:
                md.append("| Leaf | Uses | SR | Avg Score | Quality |")
                md.append("|---|---:|---:|---:|---:|")
                for r in bottom5:
                    md.append(
                        f"| {r['leaf']} | {r['total_uses']} | {r['success_rate']:.2%} | {r['avg_score']:.3f} | {r['quality_index']:.3f} |"
                    )
            else:
                md.append("- No leaf stats available.")
            md.append("")
            md.append("## Best Templates (Top 5)")
            if top_tmpl5:
                md.append("| Template | Uses | SR | Avg Score |")
                md.append("|---|---:|---:|---:|")
                for t in top_tmpl5:
                    md.append(
                        f"| {t['template']} | {t['total_uses']} | {t['success_rate']:.2%} | {t['avg_score']:.3f} |"
                    )
            else:
                md.append("- No template stats available.")
            md.append("")
            md.append("## Sample Successful Behaviors (First 10)")
            if success_behaviors:
                for b in success_behaviors:
                    md.append(
                        f"- B{b.behavior_idx}: score={b.best_score:.2f}, iters={b.iterations_count}, leaf={b.winning_leaf or '-'}, template={b.winning_template or '-'}"
                    )
            else:
                md.append("- None")
            md.append("")
            md.append("## Sample Failed Behaviors (First 10)")
            if fail_behaviors:
                for b in fail_behaviors:
                    md.append(
                        f"- B{b.behavior_idx}: best_score={b.best_score:.2f}, iters={b.iterations_count}, goal={b.goal[:120]}"
                    )
            else:
                md.append("- None")
            md.append("")
            md.append("## Output Artifacts")
            md.append("- Iteration traces JSONL")
            md.append("- Behavior journeys JSON")
            md.append("- Leaf ranking CSV")
            md.append("- Behavior summary CSV")
            md.append("- Comprehensive analysis XLSX")
            md.append("- Visualization PNG charts")
            md_path.write_text("\n".join(md), encoding="utf-8")
            paths["executive_report_md"] = md_path
        except Exception as e:
            logger.warning(f"[VIZ] Could not export markdown report: {e}")

        # Artifact index for quick navigation
        try:
            idx_path = self.output_dir / f"artifact_index_{ts}.md"
            lines = [
                f"# Artifact Index ({ts})",
                "",
                "## Data Files",
            ]
            for k, v in sorted(paths.items(), key=lambda x: x[0]):
                if "csv" in k or "json" in k or "excel" in k:
                    lines.append(f"- `{k}`: `{v}`")
            lines += ["", "## Report Files"]
            for k, v in sorted(paths.items(), key=lambda x: x[0]):
                if "report" in k or "summary" in k:
                    lines.append(f"- `{k}`: `{v}`")
            lines += ["", "## Plot Files"]
            plot_files = sorted(self.plots_dir.glob(f"*_{ts}.png"))
            if plot_files:
                for pimg in plot_files:
                    lines.append(f"- `{pimg.name}`: `{pimg}`")
            else:
                lines.append("- No plot files found.")
            idx_path.write_text("\n".join(lines), encoding="utf-8")
            paths["artifact_index_md"] = idx_path
        except Exception as e:
            logger.warning(f"[VIZ] Could not export artifact index: {e}")

        return paths

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def _plot_dataset_progress(self) -> Optional[Path]:
        """Rolling success rate + cumulative avg score across behaviors."""
        if not self.behavior_records:
            return None
        try:
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            idxs = [b.behavior_idx for b in self.behavior_records]
            scores = [b.best_score for b in self.behavior_records]
            successes = [1 if b.success else 0 for b in self.behavior_records]

            # Rolling success rate (window=10)
            window = min(10, len(successes))
            rolling_sr = []
            for i in range(len(successes)):
                start = max(0, i - window + 1)
                rolling_sr.append(sum(successes[start:i+1]) / (i - start + 1))

            # Cumulative avg score
            cumulative_avg = []
            running = 0.0
            for i, s in enumerate(scores):
                running += s
                cumulative_avg.append(running / (i + 1))

            ax0 = axes[0]
            colors = ['green' if s else 'red' for s in successes]
            ax0.bar(idxs, scores, color=colors, alpha=0.7, label="Best Score")
            ax0.axhline(y=7, color='orange', linestyle='--', alpha=0.7, label='Success threshold (7)')
            ax0.set_ylabel("Best Score (0-10)")
            ax0.set_title("Best Score per Behavior (green=success, red=fail)")
            ax0.legend(loc='upper right', fontsize=8)
            ax0.set_ylim(0, 10.5)

            ax1 = axes[1]
            ax1.plot(idxs, rolling_sr, color='blue', linewidth=2, label=f"Rolling SR (w={window})")
            ax1.fill_between(idxs, rolling_sr, alpha=0.2, color='blue')
            ax1.axhline(y=sum(successes)/max(len(successes),1), color='navy',
                        linestyle='--', alpha=0.6, label=f"Overall SR={sum(successes)/max(len(successes),1):.1%}")
            ax1.set_ylabel("Success Rate")
            ax1.set_ylim(0, 1.05)
            ax1.legend(loc='upper right', fontsize=8)
            ax1.set_title("Rolling Success Rate Across Dataset")

            ax2 = axes[2]
            ax2.plot(idxs, cumulative_avg, color='purple', linewidth=2)
            ax2.fill_between(idxs, cumulative_avg, alpha=0.15, color='purple')
            ax2.set_ylabel("Cumulative Avg Score")
            ax2.set_xlabel("Behavior Index")
            ax2.set_title("Cumulative Average Score")

            plt.suptitle(
                f"Dataset Progress | {self.successful_behaviors}/{self.total_behaviors} Success "
                f"({self.successful_behaviors/max(self.total_behaviors,1):.1%})",
                fontsize=13, fontweight='bold'
            )
            plt.tight_layout()
            p = self.plots_dir / f"dataset_progress_{self.run_tag}.png"
            plt.savefig(p, dpi=120)
            plt.close(fig)
            return p
        except Exception as e:
            logger.warning(f"[VIZ] dataset_progress chart failed: {e}")
            plt.close('all')
            return None

    def _plot_score_heatmap(self) -> Optional[Path]:
        """Heatmap: behaviors (rows) x iterations (cols) colored by score."""
        if not _NUMPY_OK or not self.per_behavior_scores:
            return None
        try:
            b_ids = sorted(self.per_behavior_scores.keys())
            max_iters = max(len(v) for v in self.per_behavior_scores.values())
            if max_iters == 0:
                return None

            # Build matrix
            mat = np.full((len(b_ids), max_iters), np.nan)
            for row, bidx in enumerate(b_ids):
                scores = self.per_behavior_scores[bidx]
                mat[row, :len(scores)] = scores

            n_b = len(b_ids)
            fig_h = max(4, min(n_b * 0.25 + 2, 20))
            fig, ax = plt.subplots(figsize=(min(max_iters * 0.5 + 3, 18), fig_h))

            cmap = plt.cm.RdYlGn
            cmap.set_bad(color='lightgray')
            im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=10, interpolation='nearest')

            # Mark successes
            for row, bidx in enumerate(b_ids):
                brec = next((b for b in self.behavior_records if b.behavior_idx == bidx), None)
                if brec and brec.success and brec.iterations_to_success is not None:
                    col = brec.iterations_to_success
                    if col < max_iters:
                        ax.plot(col, row, 'w*', markersize=8, markeredgecolor='black', markeredgewidth=0.5)

            plt.colorbar(im, ax=ax, label="Judge Score (0-10)")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Behavior Index")
            ax.set_title("Score Heatmap: All Behaviors × All Iterations\n(★ = first success, gray = no data)")

            if n_b <= 40:
                ax.set_yticks(range(len(b_ids)))
                ax.set_yticklabels([f"B{i}" for i in b_ids], fontsize=6)

            plt.tight_layout()
            p = self.plots_dir / f"score_heatmap_{self.run_tag}.png"
            plt.savefig(p, dpi=120, bbox_inches='tight')
            plt.close(fig)
            return p
        except Exception as e:
            logger.warning(f"[VIZ] score_heatmap chart failed: {e}")
            plt.close('all')
            return None

    def _plot_leaf_ranking(self) -> Optional[Path]:
        """Top/bottom leaf nodes by success rate, avg score, and usage."""
        if not self.leaf_stats:
            return None
        try:
            leaves = [(name, ls) for name, ls in self.leaf_stats.items() if ls.total_uses >= 2]
            if not leaves:
                leaves = list(self.leaf_stats.items())
            if not leaves:
                return None

            leaves.sort(key=lambda x: (x[1].success_rate, x[1].avg_score), reverse=True)
            top_n = min(20, len(leaves))
            top_leaves = leaves[:top_n]
            bottom_leaves = leaves[-min(10, len(leaves)):]

            fig, axes = plt.subplots(1, 3, figsize=(20, max(6, top_n * 0.35 + 2)))

            # --- Success rate ---
            ax = axes[0]
            names_top = [l[0][:28] for l in top_leaves]
            sr_top = [l[1].success_rate for l in top_leaves]
            colors = plt.cm.RdYlGn([v for v in sr_top])
            bars = ax.barh(range(len(names_top)), sr_top, color=colors, edgecolor='gray', linewidth=0.5)
            ax.set_yticks(range(len(names_top)))
            ax.set_yticklabels(names_top, fontsize=7)
            ax.set_xlabel("Success Rate")
            ax.set_xlim(0, 1.05)
            ax.set_title(f"Top {top_n} Leaves by Success Rate")
            ax.invert_yaxis()
            for bar, val in zip(bars, sr_top):
                ax.text(min(val + 0.01, 0.95), bar.get_y() + bar.get_height()/2,
                        f'{val:.1%}', va='center', fontsize=6)

            # --- Avg score ---
            ax = axes[1]
            names_score = [l[0][:28] for l in top_leaves]
            avg_scores = [l[1].avg_score for l in top_leaves]
            colors2 = plt.cm.Blues([v/10.0 for v in avg_scores])
            bars2 = ax.barh(range(len(names_score)), avg_scores, color=colors2, edgecolor='gray', linewidth=0.5)
            ax.set_yticks(range(len(names_score)))
            ax.set_yticklabels(names_score, fontsize=7)
            ax.set_xlabel("Average Judge Score")
            ax.set_xlim(0, 10.5)
            ax.set_title(f"Top {top_n} Leaves by Avg Score")
            ax.invert_yaxis()
            for bar, val in zip(bars2, avg_scores):
                ax.text(min(val + 0.1, 9.8), bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}', va='center', fontsize=6)

            # --- Usage count + max score ---
            ax = axes[2]
            all_sorted_usage = sorted(self.leaf_stats.items(), key=lambda x: x[1].total_uses, reverse=True)
            top_usage = all_sorted_usage[:top_n]
            names_u = [l[0][:28] for l in top_usage]
            uses = [l[1].total_uses for l in top_usage]
            max_scores = [l[1].max_score for l in top_usage]
            x_pos = range(len(names_u))
            ax2_twin = ax.twiny()
            ax.barh(x_pos, uses, color='steelblue', alpha=0.6, label='Total Uses')
            ax2_twin.plot(max_scores, x_pos, 'ro-', markersize=4, label='Max Score', alpha=0.8)
            ax.set_yticks(x_pos)
            ax.set_yticklabels(names_u, fontsize=7)
            ax.set_xlabel("Total Uses", color='steelblue')
            ax2_twin.set_xlabel("Max Score", color='red')
            ax2_twin.set_xlim(0, 10.5)
            ax.set_title(f"Top {top_n} Leaves by Usage")
            ax.invert_yaxis()
            ax2_twin.invert_yaxis()

            plt.suptitle("Leaf Node Performance Across Full Dataset", fontsize=12, fontweight='bold')
            plt.tight_layout()
            p = self.plots_dir / f"leaf_ranking_{self.run_tag}.png"
            plt.savefig(p, dpi=120, bbox_inches='tight')
            plt.close(fig)
            return p
        except Exception as e:
            logger.warning(f"[VIZ] leaf_ranking chart failed: {e}")
            plt.close('all')
            return None

    def _plot_template_effectiveness(self) -> Optional[Path]:
        """Scatter + bar: template avg_score vs success_rate vs usage."""
        if not self.template_stats:
            return None
        try:
            templates = [(n, ts) for n, ts in self.template_stats.items() if ts.total_uses >= 1]
            if not templates:
                return None

            fig, axes = plt.subplots(1, 2, figsize=(18, 8))

            # Scatter: avg_score (x) vs success_rate (y), size = usage
            ax = axes[0]
            xs = [ts.avg_score for _, ts in templates]
            ys = [ts.success_rate for _, ts in templates]
            sizes = [max(20, ts.total_uses * 8) for _, ts in templates]
            colors_s = [ts.avg_score / 10.0 for _, ts in templates]
            sc = ax.scatter(xs, ys, s=sizes, c=colors_s, cmap='RdYlGn', alpha=0.7,
                            vmin=0, vmax=1, edgecolors='gray', linewidths=0.5)
            plt.colorbar(sc, ax=ax, label="Avg Score (normalized)")
            for i, (name, ts) in enumerate(templates):
                if ts.total_uses >= 3 or ts.success_rate > 0:
                    ax.annotate(name[:18], (xs[i], ys[i]), fontsize=5,
                                xytext=(3, 3), textcoords='offset points', alpha=0.8)
            ax.set_xlabel("Average Judge Score")
            ax.set_ylabel("Success Rate")
            ax.set_title("Template Effectiveness Map\n(bubble size = usage count)")
            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(-0.05, 1.1)
            ax.axvline(x=7, color='orange', linestyle='--', alpha=0.5, label='Score=7')
            ax.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='SR=50%')
            ax.legend(fontsize=8)

            # Bar: top 20 by composite score
            ax = axes[1]
            composite = sorted(templates,
                               key=lambda x: x[1].success_rate * 0.6 + x[1].avg_score / 10.0 * 0.4,
                               reverse=True)
            top20 = composite[:20]
            names = [t[0][:25] for t in top20]
            comp_scores = [t[1].success_rate * 0.6 + t[1].avg_score / 10.0 * 0.4 for t in top20]
            bar_colors = plt.cm.RdYlGn([c for c in comp_scores])
            bars = ax.barh(range(len(names)), comp_scores, color=bar_colors, edgecolor='gray', linewidth=0.5)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel("Composite Score (SR×0.6 + AvgScore/10×0.4)")
            ax.set_title("Top 20 Templates by Composite Performance")
            ax.invert_yaxis()
            for bar, (name, ts) in zip(bars, top20):
                ax.text(bar.get_width() + 0.005,
                        bar.get_y() + bar.get_height()/2,
                        f'SR={ts.success_rate:.0%} Avg={ts.avg_score:.1f} n={ts.total_uses}',
                        va='center', fontsize=5.5)

            plt.suptitle("Template Effectiveness Analysis", fontsize=12, fontweight='bold')
            plt.tight_layout()
            p = self.plots_dir / f"template_effectiveness_{self.run_tag}.png"
            plt.savefig(p, dpi=120, bbox_inches='tight')
            plt.close(fig)
            return p
        except Exception as e:
            logger.warning(f"[VIZ] template_effectiveness chart failed: {e}")
            plt.close('all')
            return None

    def _plot_weight_evolution(self) -> Optional[Path]:
        """
        How RL preferences change per behavior across iterations.
        Shows the top-N leaves' weights over iterations for each behavior.
        """
        if not self.weight_history:
            return None
        try:
            behavior_ids = sorted(self.weight_history.keys())
            n_behaviors = len(behavior_ids)
            if n_behaviors == 0:
                return None

            cols = min(3, n_behaviors)
            rows = math.ceil(n_behaviors / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3.5))
            if n_behaviors == 1:
                axes = [[axes]]
            elif rows == 1:
                axes = [axes]

            for i, bidx in enumerate(behavior_ids):
                row_i = i // cols
                col_i = i % cols
                ax = axes[row_i][col_i] if rows > 1 else axes[0][col_i]

                snapshots = self.weight_history[bidx]
                if not snapshots:
                    ax.set_visible(False)
                    continue

                iter_list = sorted(snapshots.keys())
                all_leaves = set()
                for snap in snapshots.values():
                    all_leaves.update(snap.keys())

                # Pick top-N leaves by max weight seen
                leaf_max_weight = {
                    leaf: max(snapshots[it].get(leaf, 0) for it in iter_list)
                    for leaf in all_leaves
                }
                top_leaves = sorted(leaf_max_weight.items(), key=lambda x: x[1], reverse=True)[:8]

                colors_w = plt.cm.tab10.colors
                for j, (leaf, _) in enumerate(top_leaves):
                    ys = [snapshots[it].get(leaf, float('nan')) for it in iter_list]
                    ax.plot(iter_list, ys, marker='o', markersize=3,
                            linewidth=1.5, color=colors_w[j % 10],
                            label=leaf[:20])

                brec = next((b for b in self.behavior_records if b.behavior_idx == bidx), None)
                title_sfx = "SUCCESS" if (brec and brec.success) else "fail"
                ax.set_title(f"B{bidx} [{title_sfx}]", fontsize=8)
                ax.set_xlabel("Iter", fontsize=7)
                ax.set_ylabel("Weight", fontsize=7)
                ax.legend(fontsize=4, loc='upper right', ncol=2)
                ax.tick_params(labelsize=6)

            # Hide empty subplots
            for i in range(n_behaviors, rows * cols):
                row_i = i // cols
                col_i = i % cols
                try:
                    ax = axes[row_i][col_i] if rows > 1 else axes[0][col_i]
                    ax.set_visible(False)
                except Exception:
                    pass

            plt.suptitle("RL Preference Weights Evolution (Top Leaves per Behavior)",
                         fontsize=11, fontweight='bold')
            plt.tight_layout()
            p = self.plots_dir / f"weight_evolution_{self.run_tag}.png"
            plt.savefig(p, dpi=110, bbox_inches='tight')
            plt.close(fig)
            return p
        except Exception as e:
            logger.warning(f"[VIZ] weight_evolution chart failed: {e}")
            plt.close('all')
            return None

    def _plot_per_behavior_trajectories(self) -> Optional[Path]:
        """Grid of score trajectories per behavior."""
        if not self.behavior_records:
            return None
        try:
            n = len(self.behavior_records)
            cols = min(5, n)
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.2))
            axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

            for i, brec in enumerate(self.behavior_records):
                ax = axes_flat[i] if i < len(axes_flat) else None
                if ax is None:
                    break
                traj = brec.score_trajectory
                if not traj:
                    ax.set_visible(False)
                    continue

                x = list(range(1, len(traj) + 1))
                color = 'green' if brec.success else 'crimson'
                ax.plot(x, traj, color=color, linewidth=1.5, marker='o', markersize=3)
                ax.fill_between(x, traj, alpha=0.15, color=color)
                ax.axhline(y=7, color='orange', linestyle='--', linewidth=0.8, alpha=0.7)
                ax.set_ylim(0, 10.5)

                if brec.iterations_to_success is not None:
                    ax.axvline(x=brec.iterations_to_success + 1,
                               color='green', linestyle=':', linewidth=1.5, alpha=0.9)

                goal_short = brec.goal[:25]
                ax.set_title(f"B{brec.behavior_idx}: {goal_short}",
                             fontsize=5.5, color=color)
                ax.tick_params(labelsize=5)
                ax.set_xlabel("Iter", fontsize=5)
                ax.set_ylabel("Score", fontsize=5)

            # Hide extras
            for j in range(n, len(axes_flat)):
                axes_flat[j].set_visible(False)

            plt.suptitle("Score Trajectory per Behavior\n(green=success, red=fail, orange line=threshold, dashed=success point)",
                         fontsize=9, fontweight='bold')
            plt.tight_layout()
            p = self.plots_dir / f"behavior_trajectories_{self.run_tag}.png"
            plt.savefig(p, dpi=110, bbox_inches='tight')
            plt.close(fig)
            return p
        except Exception as e:
            logger.warning(f"[VIZ] behavior_trajectories chart failed: {e}")
            plt.close('all')
            return None

    def _plot_reward_breakdown(self) -> Optional[Path]:
        """Stacked bar of reward components per behavior (if tracked)."""
        if not self.reward_components:
            return None
        try:
            # Aggregate avg component per behavior
            all_components = set()
            for comp_list in self.reward_components.values():
                for comp_dict in comp_list:
                    all_components.update(comp_dict.keys())
            if not all_components:
                return None

            all_components = sorted(all_components)
            b_ids = sorted(self.reward_components.keys())

            avgs = []
            for bidx in b_ids:
                comp_list = self.reward_components[bidx]
                if not comp_list:
                    avgs.append({c: 0 for c in all_components})
                    continue
                totals = {c: 0.0 for c in all_components}
                for cd in comp_list:
                    for c in all_components:
                        totals[c] += cd.get(c, 0.0)
                avgs.append({c: totals[c] / len(comp_list) for c in all_components})

            fig, ax = plt.subplots(figsize=(min(len(b_ids) * 0.4 + 4, 18), 6))
            bottoms = [0.0] * len(b_ids)
            colors_r = plt.cm.Set3.colors
            for ci, comp in enumerate(all_components):
                vals = [avgs[i].get(comp, 0.0) for i in range(len(b_ids))]
                ax.bar(b_ids, vals, bottom=bottoms, label=comp[:20],
                       color=colors_r[ci % len(colors_r)], edgecolor='white', linewidth=0.3)
                bottoms = [bottoms[i] + vals[i] for i in range(len(b_ids))]

            ax.set_xlabel("Behavior Index")
            ax.set_ylabel("Avg Reward Component Value")
            ax.set_title("Reward Component Breakdown per Behavior")
            ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
            plt.tight_layout()
            p = self.plots_dir / f"reward_breakdown_{self.run_tag}.png"
            plt.savefig(p, dpi=110, bbox_inches='tight')
            plt.close(fig)
            return p
        except Exception as e:
            logger.warning(f"[VIZ] reward_breakdown chart failed: {e}")
            plt.close('all')
            return None

    def _plot_rag_vs_rl(self) -> Optional[Path]:
        """Compare RAG-enhanced vs RL-only iterations by score."""
        if not self.iteration_records:
            return None
        try:
            rag_scores = [r.judge_score for r in self.iteration_records if r.rag_used]
            rl_scores = [r.judge_score for r in self.iteration_records if not r.rag_used]

            if not rag_scores and not rl_scores:
                return None

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            # Score distribution comparison
            ax = axes[0]
            bins = list(range(0, 12))
            if rl_scores:
                ax.hist(rl_scores, bins=bins, alpha=0.6, color='steelblue', label=f'RL-only (n={len(rl_scores)})', density=True)
            if rag_scores:
                ax.hist(rag_scores, bins=bins, alpha=0.6, color='orange', label=f'RAG-enhanced (n={len(rag_scores)})', density=True)
            ax.set_xlabel("Judge Score")
            ax.set_ylabel("Density")
            ax.set_title("Score Distribution: RAG vs RL-only")
            ax.legend()

            # Success rate comparison
            ax = axes[1]
            rag_sr = sum(1 for r in self.iteration_records if r.rag_used and r.success) / max(len(rag_scores), 1)
            rl_sr = sum(1 for r in self.iteration_records if not r.rag_used and r.success) / max(len(rl_scores), 1)
            rag_avg = sum(rag_scores) / max(len(rag_scores), 1)
            rl_avg = sum(rl_scores) / max(len(rl_scores), 1)
            bars = ax.bar(['RL-only', 'RAG-enhanced'],
                         [rl_sr, rag_sr],
                         color=['steelblue', 'orange'], edgecolor='gray', width=0.4)
            ax.set_ylabel("Success Rate")
            ax.set_ylim(0, 1.1)
            ax.set_title("Success Rate: RAG vs RL-only")
            for bar, val in zip(bars, [rl_sr, rag_sr]):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                        f'{val:.1%}', ha='center', fontsize=10)

            # Avg score comparison
            ax = axes[2]
            bars2 = ax.bar(['RL-only', 'RAG-enhanced'],
                          [rl_avg, rag_avg],
                          color=['steelblue', 'orange'], edgecolor='gray', width=0.4)
            ax.set_ylabel("Average Score")
            ax.set_ylim(0, 10.5)
            ax.set_title("Average Score: RAG vs RL-only")
            for bar, val in zip(bars2, [rl_avg, rag_avg]):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                        f'{val:.2f}', ha='center', fontsize=10)

            plt.suptitle("RAG Enhancement Impact Analysis", fontsize=12, fontweight='bold')
            plt.tight_layout()
            p = self.plots_dir / f"rag_vs_rl_{self.run_tag}.png"
            plt.savefig(p, dpi=120, bbox_inches='tight')
            plt.close(fig)
            return p
        except Exception as e:
            logger.warning(f"[VIZ] rag_vs_rl chart failed: {e}")
            plt.close('all')
            return None

    def _plot_iterations_to_success(self) -> Optional[Path]:
        """Distribution of how many iterations it takes to succeed."""
        if not self.behavior_records:
            return None
        try:
            its_vals = [b.iterations_to_success for b in self.behavior_records
                        if b.success and b.iterations_to_success is not None]
            fail_count = sum(1 for b in self.behavior_records if not b.success)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            if its_vals:
                ax.hist(its_vals, bins=range(0, max(its_vals) + 2), color='green', alpha=0.7, edgecolor='black')
                ax.axvline(x=sum(its_vals)/len(its_vals), color='red', linestyle='--',
                           label=f'Mean={sum(its_vals)/len(its_vals):.1f}')
                ax.legend()
            ax.set_xlabel("Iteration at First Success")
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of Iterations-to-Success\n({len(its_vals)} successful behaviors)")

            # Pie: success vs fail
            ax = axes[1]
            s_count = len(its_vals)
            total = s_count + fail_count
            wedges, texts, autotexts = ax.pie(
                [s_count, fail_count],
                labels=[f'Success\n(n={s_count})', f'Fail\n(n={fail_count})'],
                colors=['#4CAF50', '#F44336'],
                autopct='%1.1f%%',
                startangle=90,
                explode=[0.05, 0.05]
            )
            ax.set_title(f"Overall Success Rate\n({total} total behaviors)")

            plt.suptitle("Attack Efficiency Analysis", fontsize=12, fontweight='bold')
            plt.tight_layout()
            p = self.plots_dir / f"iterations_to_success_{self.run_tag}.png"
            plt.savefig(p, dpi=120, bbox_inches='tight')
            plt.close(fig)
            return p
        except Exception as e:
            logger.warning(f"[VIZ] iterations_to_success chart failed: {e}")
            plt.close('all')
            return None

    def _plot_score_distribution(self) -> Optional[Path]:
        """Score distribution across all iterations (violin + box)."""
        if not self.iteration_records:
            return None
        try:
            all_scores = [r.judge_score for r in self.iteration_records]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            ax.hist(all_scores, bins=list(range(0, 12)), color='steelblue', edgecolor='black', alpha=0.8)
            ax.axvline(x=7, color='red', linestyle='--', label='Threshold=7')
            avg_all = sum(all_scores) / max(len(all_scores), 1)
            ax.axvline(x=avg_all, color='orange', linestyle='-', label=f'Mean={avg_all:.2f}')
            ax.set_xlabel("Judge Score")
            ax.set_ylabel("Iteration Count")
            ax.set_title(f"All Iteration Scores Distribution (n={len(all_scores)})")
            ax.legend()

            ax = axes[1]
            # Scores by source
            sources = list(set(r.source for r in self.iteration_records))
            data_by_source = [[r.judge_score for r in self.iteration_records if r.source == s] for s in sources]
            data_by_source = [(s, d) for s, d in zip(sources, data_by_source) if d]
            if data_by_source:
                labels_s = [s[0][:20] for s in data_by_source]
                scores_s = [s[1] for s in data_by_source]
                bp = ax.boxplot(scores_s, labels=labels_s, patch_artist=True,
                                medianprops=dict(color='red', linewidth=2))
                colors_b = plt.cm.Set2.colors
                for patch, color in zip(bp['boxes'], colors_b):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_ylabel("Judge Score")
                ax.set_title("Score Distribution by Source Type")
                ax.tick_params(axis='x', rotation=30)

            plt.suptitle("Score Distribution Analysis", fontsize=12, fontweight='bold')
            plt.tight_layout()
            p = self.plots_dir / f"score_distribution_{self.run_tag}.png"
            plt.savefig(p, dpi=120, bbox_inches='tight')
            plt.close(fig)
            return p
        except Exception as e:
            logger.warning(f"[VIZ] score_distribution chart failed: {e}")
            plt.close('all')
            return None

    def _plot_summary_dashboard(self) -> Optional[Path]:
        """One-page summary dashboard with key metrics."""
        if not self.behavior_records:
            return None
        try:
            fig = plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

            total = len(self.behavior_records)
            n_success = sum(1 for b in self.behavior_records if b.success)
            sr = n_success / max(total, 1)
            all_scores_all = [r.judge_score for r in self.iteration_records]
            avg_score_all = sum(all_scores_all) / max(len(all_scores_all), 1)
            its_vals = [b.iterations_to_success for b in self.behavior_records
                        if b.success and b.iterations_to_success is not None]
            avg_its = sum(its_vals) / max(len(its_vals), 1) if its_vals else 0
            elapsed = time.time() - self._start_time

            # --- KPI Text Box ---
            ax_kpi = fig.add_subplot(gs[0, :2])
            ax_kpi.axis('off')
            kpi_text = (
                f"TRAP Attack — Run Summary\n"
                f"Run: {self.run_tag}\n\n"
                f"{'Behaviors:':<26}{total}\n"
                f"{'Successful:':<26}{n_success}  ({sr:.1%})\n"
                f"{'Total Iterations:':<26}{self.total_iterations}\n"
                f"{'Avg Score (all iters):':<26}{avg_score_all:.2f}/10\n"
                f"{'Avg Iters to Success:':<26}{avg_its:.1f}\n"
                f"{'Elapsed Time:':<26}{elapsed/60:.1f} min\n"
                f"{'Unique Leaves Used:':<26}{len(self.leaf_stats)}\n"
                f"{'Unique Templates Used:':<26}{len(self.template_stats)}\n"
            )
            ax_kpi.text(0.05, 0.95, kpi_text,
                       transform=ax_kpi.transAxes,
                       fontsize=10.5, verticalalignment='top',
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4f8', edgecolor='#2196F3', linewidth=2))

            # --- Rolling success rate ---
            ax_sr = fig.add_subplot(gs[0, 2:])
            b_idxs = [b.behavior_idx for b in self.behavior_records]
            b_succs = [1 if b.success else 0 for b in self.behavior_records]
            window = min(10, len(b_succs))
            rolling = []
            for i in range(len(b_succs)):
                st = max(0, i - window + 1)
                rolling.append(sum(b_succs[st:i+1]) / (i - st + 1))
            ax_sr.plot(b_idxs, rolling, color='blue', linewidth=2)
            ax_sr.fill_between(b_idxs, rolling, alpha=0.15, color='blue')
            ax_sr.axhline(y=sr, color='navy', linestyle='--', alpha=0.7, label=f'Overall={sr:.1%}')
            ax_sr.set_ylim(0, 1.05)
            ax_sr.set_title("Rolling Success Rate")
            ax_sr.set_xlabel("Behavior")
            ax_sr.set_ylabel("SR")
            ax_sr.legend(fontsize=8)

            # --- Score progression ---
            ax_prog = fig.add_subplot(gs[1, :2])
            ax_prog.bar(b_idxs, [b.best_score for b in self.behavior_records],
                       color=['#4CAF50' if b.success else '#F44336' for b in self.behavior_records],
                       alpha=0.75)
            ax_prog.axhline(y=7, color='orange', linestyle='--', linewidth=1, alpha=0.8)
            ax_prog.set_ylim(0, 10.5)
            ax_prog.set_title("Best Score per Behavior")
            ax_prog.set_xlabel("Behavior")
            ax_prog.set_ylabel("Best Score")

            # --- Top 10 leaves ---
            ax_leaves = fig.add_subplot(gs[1, 2:])
            top_leaves = sorted(self.leaf_stats.items(),
                               key=lambda x: (x[1].success_rate, x[1].avg_score), reverse=True)[:10]
            if top_leaves:
                leaf_names = [l[0][:22] for l in top_leaves]
                leaf_sr = [l[1].success_rate for l in top_leaves]
                leaf_bars = ax_leaves.barh(range(len(leaf_names)), leaf_sr,
                                          color=plt.cm.RdYlGn([v for v in leaf_sr]))
                ax_leaves.set_yticks(range(len(leaf_names)))
                ax_leaves.set_yticklabels(leaf_names, fontsize=7)
                ax_leaves.set_xlim(0, 1.1)
                ax_leaves.set_xlabel("Success Rate")
                ax_leaves.set_title("Top 10 Leaves (by SR)")
                ax_leaves.invert_yaxis()

            # --- Template top 10 ---
            ax_tmpls = fig.add_subplot(gs[2, :2])
            top_tmpls = sorted(self.template_stats.items(),
                              key=lambda x: x[1].success_rate * 0.6 + x[1].avg_score / 10 * 0.4,
                              reverse=True)[:10]
            if top_tmpls:
                tmpl_names = [t[0][:22] for t in top_tmpls]
                tmpl_sr = [t[1].success_rate for t in top_tmpls]
                ax_tmpls.barh(range(len(tmpl_names)), tmpl_sr,
                             color=plt.cm.Blues([0.3 + 0.7*v for v in tmpl_sr]))
                ax_tmpls.set_yticks(range(len(tmpl_names)))
                ax_tmpls.set_yticklabels(tmpl_names, fontsize=7)
                ax_tmpls.set_xlim(0, 1.1)
                ax_tmpls.set_xlabel("Success Rate")
                ax_tmpls.set_title("Top 10 Templates (by SR)")
                ax_tmpls.invert_yaxis()

            # --- Iterations histogram ---
            ax_hist = fig.add_subplot(gs[2, 2:])
            if its_vals:
                ax_hist.hist(its_vals, bins=range(0, max(its_vals) + 2),
                            color='#4CAF50', alpha=0.8, edgecolor='black')
                ax_hist.axvline(x=avg_its, color='red', linestyle='--', label=f'Mean={avg_its:.1f}')
                ax_hist.legend(fontsize=8)
            ax_hist.set_xlabel("Iteration at Success")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title("Iterations-to-Success Distribution")

            plt.suptitle(f"TRAP Attack — Summary Dashboard  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        fontsize=13, fontweight='bold', y=0.98)
            p = self.plots_dir / f"summary_dashboard_{self.run_tag}.png"
            plt.savefig(p, dpi=120, bbox_inches='tight')
            plt.close(fig)
            return p
        except Exception as e:
            logger.warning(f"[VIZ] summary_dashboard chart failed: {e}")
            plt.close('all')
            return None

    # ------------------------------------------------------------------
    # Real-time progress print (no matplotlib needed)
    # ------------------------------------------------------------------

    def print_progress(self, behavior_idx: int, total_behaviors: int) -> None:
        """Print a compact progress summary to stdout after each behavior."""
        if not self.enabled:
            return
        done = len(self.behavior_records)
        n_ok = self.successful_behaviors
        sr = n_ok / max(done, 1)
        elapsed = time.time() - self._start_time
        eta = (elapsed / max(done, 1)) * (total_behaviors - done) if done > 0 else 0
        last_scores = self.dataset_score_trajectory[-5:] if self.dataset_score_trajectory else []
        top_leaf = None
        bottom_leaf = None
        if self.leaf_stats:
            ranked = sorted(
                self.leaf_stats.values(),
                key=lambda ls: (0.7 * ls.success_rate + 0.3 * (ls.avg_score / 10.0), ls.total_uses),
                reverse=True
            )
            if ranked:
                top_leaf = ranked[0].name
                bottom_leaf = ranked[-1].name
        print(
            f"\r[VIZ] B{behavior_idx+1}/{total_behaviors} | "
            f"Done={done} SR={sr:.1%} | "
            f"LastScores={[f'{s:.0f}' for s in last_scores]} | "
            f"TopLeaf={top_leaf or '-'} BottomLeaf={bottom_leaf or '-'} | "
            f"Elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m",
            flush=True
        )

    def _write_live_progress_snapshot(self, extra: Optional[Dict[str, Any]] = None) -> None:
        """Persist a lightweight live progress JSON for external monitoring."""
        try:
            total_done = len(self.behavior_records)
            success_rate = self.successful_behaviors / max(total_done, 1)
            avg_best = (
                sum(b.best_score for b in self.behavior_records) / max(total_done, 1)
                if self.behavior_records else 0.0
            )
            top_leaves = []
            bottom_leaves = []
            if self.leaf_stats:
                ranked = sorted(
                    self.leaf_stats.values(),
                    key=lambda ls: (0.7 * ls.success_rate + 0.3 * (ls.avg_score / 10.0), ls.total_uses),
                    reverse=True
                )
                top_leaves = [ls.name for ls in ranked[:5]]
                bottom_leaves = [ls.name for ls in ranked[-5:]]
            payload = {
                "run_tag": self.run_tag,
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": round(time.time() - self._start_time, 2),
                "behaviors_done": total_done,
                "successful_behaviors": self.successful_behaviors,
                "success_rate": round(success_rate, 6),
                "total_iterations": self.total_iterations,
                "avg_best_score": round(avg_best, 6),
                "top_leaves": top_leaves,
                "bottom_leaves": bottom_leaves,
            }
            if extra:
                payload["event"] = extra
            self.live_progress_file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            logger.debug(f"[VIZ] live progress snapshot failed: {e}")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_dataset_tracker(
    output_dir: Optional[str] = None,
    run_tag: str = "",
    enabled: bool = True
) -> DatasetTracker:
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "logs" / "visualization")
    return DatasetTracker(output_dir=output_dir, run_tag=run_tag, enabled=enabled)
