"""
Canonical output schemas for ablation study results.

All scripts must use these TypedDicts as the single source of truth for
field names. No more score/judge_score or attempts/iterations duality.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict, Required

SCHEMA_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Iteration-level record (one entry per attack attempt within a behavior)
# ---------------------------------------------------------------------------

class IterationRecord(TypedDict, total=False):
    # Required
    iteration: Required[int]
    score: Required[float]          # canonical name — always use this
    success: Required[bool]

    # Prompt / response
    prompt: str
    system_prompt: str
    base_prompt: str
    response: str
    response_len: int

    # Action / template
    source: str
    template_name: str              # canonical name — always use this
    template_context: str
    leaf_node: str
    leaf_path: str
    navigation_steps: int
    root_category: str
    secondary_template: str
    composed: bool

    # RAG
    rag_enhanced: bool
    rag_prob: float
    rag_allowed: bool

    # RL
    reward: float
    log_prob: float                 # captured every iteration (not only on train)
    rl_metrics: Dict[str, Any]      # populated when training batch fires

    # Evaluation detail
    judge_raw: Any
    on_topic_raw: Any
    success_threshold: float
    on_topic_threshold: float
    aspect_details: Dict[str, Any]
    guardrails: Dict[str, Any]
    evaluation_metadata: Dict[str, Any]

    # Flags
    is_refusal: bool
    safety_tone: bool
    short_response: bool
    half_step: bool
    infra_failure: bool
    failure_reasons: List[str]

    # Timing
    timestamp: str
    iteration_seconds: float        # canonical name for per-iter wall time


# ---------------------------------------------------------------------------
# Behavior-level record (one entry per behavior/goal)
# ---------------------------------------------------------------------------

class BehaviorRecord(TypedDict, total=False):
    # Required
    behavior_idx: Required[int]
    goal: Required[str]
    target: Required[str]
    success: Required[bool]
    max_score: Required[float]

    # Derived metrics
    final_score: float
    iterations_to_success: Optional[int]   # None if never succeeded
    first_success_iteration: Optional[int] # same semantics, iteration number (1-based)
    iterations_count: int
    rag_hit_count: int                     # iterations where rag_enhanced=True
    chosen_leaf: str                       # leaf_node of the best iteration

    # Raw data
    iterations: List[IterationRecord]

    # Artifacts
    successful_prompt: str
    target_response: str
    rag_retrieval: Dict[str, Any]
    aspect_details: Dict[str, Any]
    guardrails: Dict[str, Any]
    evaluation_metadata: Dict[str, Any]
    timed_out: bool


# ---------------------------------------------------------------------------
# Run-level manifest (written once per run, adjacent to raw JSON)
# ---------------------------------------------------------------------------

class RunManifest(TypedDict, total=False):
    # Identity — all required
    schema_version: Required[str]
    study_id: Required[str]
    experiment_id: Required[str]       # A / B / C / D / E
    experiment_name: Required[str]     # "Full+RAG" etc.
    seed: Required[int]
    timestamp: Required[str]           # ISO-8601

    # Config
    target_model: str
    evaluator_model: str
    rag_enabled: bool
    rag_mode: str
    kb_snapshot_sha256: str
    kb_parameters: Dict[str, Any]
    tree_variant: str                  # full_active / pruned / flat / none
    tree_leaves: Optional[int]
    tree_max_depth: Optional[int]
    success_threshold: float
    max_iterations_per_episode: int
    episode_budget: int

    # Reproducibility
    git_commit: str
    git_branch: str
    git_dirty: bool

    # Metrics summary (filled after run completes)
    total_behaviors: int
    asr_diff_ci95_low: float
    asr_diff_ci95_high: float
    mcnemar_exact_p: float
    wilcoxon_score_p: float
    success_count: int
    asr_percent: float
    avg_best_score: float
    avg_iterations_per_behavior: float

    # Experiment D — retrieval-as-policy fallback tracking
    # Only populated when experiment_id == "D" (agent_mode == "rag_only").
    # Keys: "retrieval", "fallback_named", "fallback_random" (raw counts).
    rag_only_selection_stats: Dict[str, int]
    # Same data expressed as fractions [0, 1] for quick table scanning.
    # Keys: "retrieval_share", "fallback_named_share", "fallback_random_share".
    rag_only_selection_shares: Dict[str, float]

    # Artifact paths (relative to study root)
    paths: Dict[str, str]


# ---------------------------------------------------------------------------
# Study-level aggregate record (one per comparison pair)
# ---------------------------------------------------------------------------

class PairComparisonRecord(TypedDict, total=False):
    pair_key: Required[str]            # e.g. "rag_effect"
    experiment_a: Required[str]        # e.g. "A"
    experiment_b: Required[str]        # e.g. "E"
    interpretation_note: str           # verbatim from experiment_registry.yaml

    # Metrics for each side
    asr_a: float
    asr_b: float
    asr_abs_diff: float                # asr_a - asr_b
    asr_rel_diff_pct: float            # 100 * (asr_a - asr_b) / max(asr_b, 1e-9)

    avg_score_a: float
    avg_score_b: float
    avg_score_abs_diff: float

    avg_iters_a: float
    avg_iters_b: float
    avg_iters_abs_diff: float

    # Per-behavior win/loss (A wins when A.success and not B.success, etc.)
    wins_a: int
    wins_b: int
    ties: int
    total_behaviors: int

    # Statistical inference
    asr_diff_ci95_low: float
    asr_diff_ci95_high: float
    mcnemar_exact_p: float
    wilcoxon_score_p: float


# ---------------------------------------------------------------------------
# Normalisation helpers used by all scripts
# ---------------------------------------------------------------------------

def normalise_iteration(raw: Dict[str, Any]) -> IterationRecord:
    """
    Convert any raw iteration dict (old or new format) into IterationRecord.

    Handles legacy field aliases:
      judge_score  -> score
      attempts     -> (handled at behavior level)
      iter_seconds / iteration_seconds -> iteration_seconds
      action / template_name           -> template_name
    """
    out: Dict[str, Any] = dict(raw)

    # score
    if "score" not in out or out["score"] is None:
        out["score"] = float(out.pop("judge_score", 0.0) or 0.0)
    else:
        out["score"] = float(out["score"])
        out.pop("judge_score", None)

    # template_name
    if "template_name" not in out or not out["template_name"]:
        out["template_name"] = out.pop("action", "unknown")
    else:
        out.pop("action", None)

    # iteration_seconds
    if "iteration_seconds" not in out:
        out["iteration_seconds"] = float(out.pop("iter_seconds", 0.0) or 0.0)
    else:
        out.pop("iter_seconds", None)

    # success — ensure bool
    out["success"] = bool(out.get("success", False))

    # iteration — ensure int
    out["iteration"] = int(out.get("iteration", 0))

    return out  # type: ignore[return-value]


def normalise_behavior(raw: Dict[str, Any]) -> BehaviorRecord:
    """
    Convert any raw behavior dict into BehaviorRecord, normalising all
    iteration records inside it and computing derived fields.
    """
    out: Dict[str, Any] = dict(raw)

    # Legacy / alternate alias used by newer ablation writers
    if "max_score" not in out or out["max_score"] is None:
        if "best_score" in out:
            out["max_score"] = out["best_score"]

    # iterations field — accept 'attempts' as legacy alias
    raw_iters = out.pop("iterations", None) or out.pop("attempts", [])
    if not isinstance(raw_iters, list):
        raw_iters = []
    iters: List[IterationRecord] = [normalise_iteration(it) for it in raw_iters]
    out["iterations"] = iters
    out["iterations_count"] = len(iters)

    # max_score
    if "max_score" not in out or out["max_score"] is None:
        out["max_score"] = max((it["score"] for it in iters), default=0.0)

    # success — ensure bool
    out["success"] = bool(out.get("success", False))

    # iterations_to_success / first_success_iteration
    first_success = next(
        (it["iteration"] for it in iters if it.get("success")), None
    )
    out.setdefault("iterations_to_success", first_success)
    out.setdefault("first_success_iteration", first_success)

    # rag_hit_count
    out.setdefault(
        "rag_hit_count",
        sum(1 for it in iters if it.get("rag_enhanced", False)),
    )

    # chosen_leaf — leaf_node of best-scoring iteration
    best_iter = max(iters, key=lambda it: it.get("score", 0.0), default=None)
    out.setdefault(
        "chosen_leaf",
        best_iter.get("leaf_node", "") if best_iter else "",
    )

    return out  # type: ignore[return-value]
