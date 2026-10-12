"""State summaries and lightweight metric tracking for the RAG runner."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from scripts.rag_runner_runtime import logger


class RagRunnerReportingMixin:
    """Reporting behavior kept separate from attack orchestration."""

    def _record_train_metrics(self, iteration: int, train_metrics: Dict[str, Any]) -> None:
        if not train_metrics:
            return
        # Gradient-bandit diagnostics (no actor-critic loss exists for this policy).
        # We track policy entropy (exploration) and mean batch reward, which are the
        # quantities that actually move during a bandit update.
        self.training_history.append({
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "entropy": float(train_metrics.get("entropy", 0.0)),
            "avg_reward": float(train_metrics.get("avg_reward", 0.0)),
        })

    def _store_and_train_rl(
        self,
        *,
        score: float,
        reward: float,
        response: str,
        infra_failure: bool,
        is_explicit_refusal: bool,
        decision_path: Any,
        policy_attributed: bool,
        iteration_count: int,
        iter_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Apply the RL learning-gate, store the trajectory, and run a train step.

        Behavior-preserving extraction of the inline RL block from run_attack().
        Returns the train metrics dict (or None). Mutates iter_result['rl_metrics']
        and the rl_*_count counters exactly as the inline code did.

        The learning gate (self.rl_learn_gate) is the canonical RL positive/negative
        split — DISTINCT from the KB learning_threshold. Sub-gate attempts get a
        negative reward so the bandit learns to suppress weak branches; refusals and
        infra failures are filtered out entirely (never poison the buffer).
        """
        rl_learn_gate = float(self.rl_learn_gate)

        is_rl_noise = is_explicit_refusal or infra_failure or not policy_attributed
        should_store_for_learning = not is_rl_noise

        if score >= rl_learn_gate:
            rl_reward = reward
            rl_quality = "high" if score >= 7 else "partial"
        else:
            shortfall = (rl_learn_gate - max(score, 0.0)) / max(rl_learn_gate, 1e-6)
            shortfall = max(0.0, min(1.0, shortfall))
            rl_reward = -shortfall * 5.0
            rl_quality = "negative"

        if not should_store_for_learning:
            if is_explicit_refusal:
                logger.info(
                    f"[FILTER] Skipping RL storage for explicit refusal (len={len(response)}, score={score})"
                )
            elif infra_failure:
                logger.info(f"[FILTER] Skipping RL storage for infrastructure failure (score={score})")
            elif not policy_attributed:
                logger.info(
                    "[FILTER] Skipping RL storage for non-policy selection "
                    f"(source={iter_result.get('selection_source', 'unknown')})"
                )

        train_metrics = None
        if decision_path and should_store_for_learning:
            try:
                trajectory_rewards = self.rl_agent.compute_trajectory_reward(
                    final_reward=rl_reward,
                    decision_path=decision_path,
                    depth_penalty=0.0,
                )
                self.rl_agent.store_trajectory(
                    decision_path=decision_path,
                    trajectory_rewards=trajectory_rewards,
                )
                self.rl_trajectory_store_count += 1
                logger.info(
                    f"[RL-STORE] Stored trajectory with score={score:.1f}, "
                    f"rl_reward={rl_reward:.2f}, learning_quality={rl_quality}"
                )

                if self.rl_agent.should_train():
                    train_metrics = self.rl_agent.train_step(epochs=4)
                    self.rl_train_step_count += 1
                    if train_metrics:
                        logger.info(
                            f"[TRAIN] Bandit metrics: entropy={train_metrics.get('entropy', 0):.4f}, "
                            f"avg_reward={train_metrics.get('avg_reward', 0):.4f}, "
                            f"updates={int(train_metrics.get('updates', 0))}, "
                            f"buffer_size={len(self.rl_agent.experience_buffer)}"
                        )
                        self._record_train_metrics(iteration_count, train_metrics)
                        iter_result["rl_metrics"] = train_metrics
            except Exception as train_error:
                logger.warning(f"[ITERATION {iteration_count}] Training step failed: {train_error}")

        return train_metrics

    def _summarize_coverage(self) -> Dict[str, Any]:
        usage = self.stats.get("leaf_usage", {})
        summary = {"total_leaves": len(usage), "top_leaves": []}
        for name, data in usage.items():
            depth_avg = (
                data.get("depth_total", 0) / data["count"]
                if data["count"]
                else 0.0
            )
            summary["top_leaves"].append({
                "leaf": name,
                "count": data["count"],
                "success_rate": data["successes"] / max(data["count"], 1),
                "avg_score": data.get("avg_score", 0.0),
                "max_score": data.get("max_score", 0.0),
                "avg_depth": depth_avg,
                "paths": sorted(list(data.get("paths", [])))[:3],
            })
        summary["top_leaves"] = sorted(
            summary["top_leaves"],
            key=lambda item: (
                item["success_rate"],
                item["avg_score"],
                item["count"],
            ),
            reverse=True,
        )[:10]
        return summary

    def _track_template_detailed_stats(
        self,
        template_name: Optional[str],
        score: float,
        success: bool,
        iteration: int,
        goal: str,
    ) -> None:
        if not template_name:
            return
        stats = self.template_detailed_stats.setdefault(template_name, {
            "uses": 0,
            "successes": 0,
            "scores": [],
            "avg_iterations_to_success": [],
            "behavior_ids": set(),
            "first_used_timestamp": datetime.now().isoformat(),
        })
        stats["uses"] += 1
        stats["scores"].append(score)
        stats["behavior_ids"].add(goal)
        if success:
            stats["successes"] += 1
            stats["avg_iterations_to_success"].append(iteration)

    def _summarize_rag_retrieval(self) -> Dict[str, Any]:
        rag_stats = self.stats.get("rag_retrieval", {})
        queries = rag_stats.get("queries", 0)
        hits = rag_stats.get("hits", 0)
        similarities = rag_stats.get("similarities", [])
        return {
            "queries": queries,
            "hits": hits,
            "misses": rag_stats.get("misses", 0),
            "hit_rate": hits / max(queries, 1),
            "avg_similarity": (
                sum(similarities) / len(similarities)
                if similarities
                else 0.0
            ),
            "best_similarity": max(similarities) if similarities else 0.0,
            "samples": similarities[:10],
        }

    def _reconcile_prompt_counters(self, results: Dict[str, Any]) -> None:
        iterations = results.get("iterations", [])
        rag_used = sum(1 for item in iterations if item.get("rag_enhanced"))
        rag_allowed = sum(1 for item in iterations if item.get("rag_allowed"))
        total_iters = len(iterations)
        self.stats["rag_used"] = rag_used
        self.stats["rl_only"] = max(0, total_iters - rag_used)
        rag_stats = self.stats.setdefault(
            "rag_retrieval",
            {"queries": 0, "hits": 0, "misses": 0, "similarities": []},
        )
        rag_stats["hits"] = max(rag_stats.get("hits", 0), rag_used)
        rag_stats["queries"] = max(
            rag_stats.get("queries", 0),
            rag_allowed,
            rag_used,
        )
        rag_stats["misses"] = max(0, rag_stats["queries"] - rag_stats["hits"])
        self.stats["rag_retrieval"] = rag_stats
