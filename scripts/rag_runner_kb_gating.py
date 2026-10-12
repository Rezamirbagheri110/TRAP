"""Knowledge-base bootstrapping and retrieval-as-policy (ablation D) selection.

Behavior-preserving extraction of the KB-gating methods from
``run_attack_with_rag.py``. These methods operate solely on ``self`` (the
``RAGEnhancedAttackRunner`` instance) and module-level ``logger``/stdlib only,
so moving them out changes nothing at runtime.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.rag_runner_runtime import logger


class RagRunnerKbGatingMixin:
    """KB bootstrap + retrieval-driven action selection, kept separate."""

    def _bootstrap_kb_from_cache(self) -> None:
        """
        Load seed examples from extracted cache directories if KB is empty.
        Uses simple JSON import to avoid 0-example RAG startup.
        """
        if len(self.rag_kb.examples) > 0:
            return

        base_rag_dir = Path(__file__).parent.parent / "rag"
        seed_files = [
            base_rag_dir / "extracted_knowledge" / "successful_prompts_full.json",
            base_rag_dir / "extracted_knowledge" / "top_performing_prompts.json",
            base_rag_dir / "cache_extraction_complete" / "successful_attempts.json",
            base_rag_dir / "successful_prompts.json",
        ]

        total_loaded = 0
        for seed_path in seed_files:
            if not seed_path.exists():
                continue
            try:
                count = self.rag_kb.load_from_json(str(seed_path))
                total_loaded += count
                logger.info(f"[BOOTSTRAP] Loaded {count} seed examples from {seed_path.name}")
            except Exception as load_err:
                logger.warning(f"[BOOTSTRAP] Failed loading {seed_path}: {load_err}")

        if total_loaded > 0:
            try:
                self.rag_kb.remove_duplicates()
            except Exception as dedup_err:
                logger.warning(f"[BOOTSTRAP] Dedup failed: {dedup_err}")
            # Persist immediately so subsequent runs see a non-empty KB
            # (skip entirely when RAG is disabled to keep ablation E isolated).
            if self.rag_mode == "online":
                try:
                    self.rag_kb.save()
                    logger.info(f"[BOOTSTRAP] KB seeded with {len(self.rag_kb.examples)} examples and saved")
                except Exception as save_err:
                    logger.warning(f"[BOOTSTRAP] KB seeded but save failed: {save_err}")

    def _summarize_rag_only_selection(self) -> Dict[str, Any]:
        """Summarize how ablation D actually chose its actions.

        Returns the absolute counts plus the fraction of decisions that came
        from genuine retrieval vs. the named fallback (``research_framing``) vs.
        the uniform-random fallback. ``retrieval_share`` is the headline number:
        if it is low, the A<->D comparison no longer isolates
        "retrieval-as-policy" and must be reported with that caveat.
        """
        stats = dict(self.rag_only_selection_stats)
        total = sum(stats.values())
        shares = {
            f"{key}_share": (count / total if total else 0.0)
            for key, count in stats.items()
        }
        summary = {
            "counts": stats,
            "total_decisions": total,
            **shares,
            "retrieval_share": (stats.get("retrieval", 0) / total) if total else 0.0,
            "fallback_share": (
                (stats.get("fallback_named", 0) + stats.get("fallback_random", 0)) / total
            ) if total else 0.0,
        }
        logger.info(
            "[ABLATION D] action source shares: retrieval=%.1f%% "
            "fallback_named=%.1f%% fallback_random=%.1f%% (n=%d). "
            "Report these alongside ASR; retrieval-as-policy holds only when "
            "retrieval_share dominates.",
            summary["retrieval_share"] * 100,
            (stats.get("fallback_named", 0) / total * 100) if total else 0.0,
            (stats.get("fallback_random", 0) / total * 100) if total else 0.0,
            total,
        )
        return summary

    def _select_rag_only_action(self, goal: str) -> Tuple[Any, str, str]:
        """Select an action directly from the RAG KB for ablation D.

        Retrieval-as-policy works in two complementary ways, tried in order:

        1. **Prompt replay (primary).** Retrieve the most similar successful
           prompt and replay it verbatim. This is the only mechanism the seed KB
           can actually drive, because its examples carry an empty ``metadata``
           dict (no ``leaf_name``) — so leaf-name lookup alone always degraded to
           the fixed ``research_framing`` fallback and ``retrieval_share`` was 0.
           The retrieved prompt is stashed on ``last_rag_only_retrieved_prompt``
           so the generator replays it instead of templating from a leaf. The
           returned leaf is only a structural placeholder for logging.

        2. **Leaf-name lookup (when metadata exists).** If a retrieved example
           does expose a usable leaf/template name (e.g. a KB grown by online
           learning), select that action by name as before.

        Falls back to the named action and finally a uniform-random leaf only
        when retrieval yields nothing usable.
        """
        fallback_action = "research_framing"
        candidate_names: List[str] = []
        retrieved_prompt: str = ""
        if self.rag_enabled and len(self.rag_kb.examples) > 0:
            try:
                # ABLATION A<->D fairness: these retrieval parameters MUST stay
                # identical to the standard RAG path used by A
                # (RAGEnhancedPromptSelector.get_prompt_suggestion:
                # top_k=5, min_score=0.15,
                # min_success_score=max(2.0, learning_threshold),
                # weight_by_score=True). If A's retrieval settings change, mirror
                # them here, otherwise A<->D gains a hidden confound in retrieval
                # conditions. The only intentional difference is the query: D uses
                # the goal alone (no current_action), because in D the action is
                # what retrieval is choosing — it cannot be part of the query.
                similar_examples = self.rag_kb.retrieve_similar(
                    query=goal,
                    top_k=5,
                    min_score=0.15,
                    min_success_score=max(2.0, float(getattr(self.rag_selector, "learning_threshold", 5.5))),
                    weight_by_score=True,
                )
                for example, _ in similar_examples:
                    # Capture the best (first / highest-weighted) prompt to replay.
                    if not retrieved_prompt:
                        prompt_text = getattr(example, "prompt", "") or ""
                        if isinstance(prompt_text, str) and prompt_text.strip():
                            retrieved_prompt = prompt_text.strip()
                    metadata = getattr(example, "metadata", {}) or {}
                    for key in ("leaf_name", "leaf_node_name", "action_name", "template_name"):
                        value = metadata.get(key)
                        if isinstance(value, str) and value.strip():
                            candidate_names.append(value.strip())
            except Exception as rag_err:
                logger.warning(f"[ABLATION D] RAG-only action retrieval failed: {rag_err}")

        # Primary path: a successful prompt was retrieved. Replay it verbatim and
        # record this as a genuine retrieval decision. The leaf returned here is a
        # structural placeholder only (used for logging / tree bookkeeping); the
        # generator replays last_rag_only_retrieved_prompt and ignores the leaf
        # template.
        if retrieved_prompt:
            self.last_rag_only_retrieved_prompt = retrieved_prompt
            placeholder = self.action_tree.get_action_by_name(fallback_action)
            if placeholder is None and getattr(self.action_tree, "leaf_nodes", None):
                placeholder = self.action_tree.leaf_nodes[0]
            leaf_path = (
                placeholder.get_category_path()
                if placeholder is not None and hasattr(placeholder, "get_category_path")
                else fallback_action
            )
            self.last_rag_only_action_source = "retrieval"
            self.rag_only_selection_stats["retrieval"] = self.rag_only_selection_stats.get("retrieval", 0) + 1
            return placeholder, fallback_action, leaf_path

        # Secondary path: no prompt, but a leaf/template name was retrievable.
        self.last_rag_only_retrieved_prompt = None
        candidate_names.append(fallback_action)
        for candidate in candidate_names:
            leaf_node = self.action_tree.get_action_by_name(candidate)
            if leaf_node is not None:
                leaf_path = leaf_node.get_category_path() if hasattr(leaf_node, "get_category_path") else candidate
                source = "retrieval" if candidate != fallback_action else "fallback_named"
                self.last_rag_only_action_source = source
                self.rag_only_selection_stats[source] = self.rag_only_selection_stats.get(source, 0) + 1
                return leaf_node, candidate, leaf_path

        # KB empty or all candidates invalid — choose uniformly at random to avoid
        # a systematic first-leaf bias that would confound the D ablation results.
        leaf_node = random.choice(self.action_tree.leaf_nodes)
        leaf_path = leaf_node.get_category_path() if hasattr(leaf_node, "get_category_path") else leaf_node.name
        self.last_rag_only_action_source = "fallback_random"
        self.rag_only_selection_stats["fallback_random"] = self.rag_only_selection_stats.get("fallback_random", 0) + 1
        return leaf_node, leaf_node.name, leaf_path
