"""Ablation configuration and selection-agent support for the RAG runner."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.experiment_config import load_experiment_registry

logger = logging.getLogger("run_attack_with_rag")


def resolve_repo_path(path_like: Optional[str], default: Path, repo_root: Path) -> Path:
    """Resolve runner paths consistently regardless of the process cwd."""
    base = Path(path_like) if path_like else Path(default)
    return base if base.is_absolute() else repo_root / base


def resolve_experiment_config(
    experiment_name: Optional[str],
    tree_variant: Optional[str],
    rag_enabled: Optional[bool],
) -> Dict[str, Any]:
    """Resolve the controlled action-tree and RAG configuration for one run."""
    try:
        registry = load_experiment_registry()
    except Exception as exc:
        logger.warning("[ABLATION] Could not load experiment registry: %s", exc)
        registry = {}

    experiments = registry.get("experiments", {}) if isinstance(registry, dict) else {}
    exp_name = (experiment_name or "").strip().upper() or None
    exp_cfg = experiments.get(exp_name, {}) if exp_name else {}

    resolved_tree_variant = tree_variant or exp_cfg.get("tree_variant") or "full_active"
    if rag_enabled is not None:
        resolved_rag_enabled = rag_enabled
    else:
        raw_rag = exp_cfg.get("rag_enabled", True)
        resolved_rag_enabled = raw_rag is True or raw_rag == 1

    if resolved_tree_variant == "flat":
        agent_mode = "flat_bandit"
    elif resolved_tree_variant == "none":
        agent_mode = "rag_only"
    else:
        agent_mode = "hierarchical"

    return {
        "experiment_name": exp_name,
        "experiment_label": exp_cfg.get("name") or exp_name or "custom",
        "tree_variant": resolved_tree_variant,
        "rag_enabled": resolved_rag_enabled,
        "agent_mode": agent_mode,
        "registry_entry": exp_cfg,
    }


class RagOnlySelectionAgent:
    """Minimal no-learning adapter used by the RAG-only ablation."""

    def __init__(self, action_tree: Any):
        self.action_tree = action_tree
        self.experience_buffer: List[Any] = []

    def navigate_to_leaf(
        self,
        deterministic: bool = False,
        return_path: bool = True,
        min_leaf_depth: int = 0,
    ):
        del deterministic, min_leaf_depth
        leaf = (
            self.action_tree.leaf_nodes[0]
            if getattr(self.action_tree, "leaf_nodes", [])
            else self.action_tree.root
        )
        decision_path: List[Dict[str, Any]] = []
        return (leaf, decision_path) if return_path else leaf

    def compute_trajectory_reward(
        self,
        final_reward: float,
        decision_path: List[Dict[str, Any]],
        depth_penalty: float = 0.0,
    ) -> List[float]:
        del depth_penalty
        return [float(final_reward) for _ in decision_path]

    def store_trajectory(self, *args: Any, **kwargs: Any) -> None:
        return None

    def should_train(self, batch_size: Optional[int] = None) -> bool:
        del batch_size
        return False

    def train_step(
        self,
        batch_size: Optional[int] = None,
        epochs: int = 1,
    ) -> Dict[str, float]:
        del batch_size, epochs
        return {}

    def save(self, path: str, save_optimizer: bool = True) -> None:
        del save_optimizer
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"agent_mode": "rag_only", "saved_at": datetime.now().isoformat()}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def load(self, path: str) -> None:
        del path
