"""
Hierarchical tree navigation agent using Gradient Bandit updates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
import math
import random

import numpy as np
import torch


@dataclass
class AgentConfig:
    max_tree_depth: int = 8
    max_children: int = 20
    step_size: float = 0.1
    use_baseline: bool = True
    temperature: float = 1.0
    exploration_rate: float = 0.1
    batch_size: int = 32
    buffer_capacity: int = 10000
    random_seed: Optional[int] = None
    # Shrinkage strength for per-node baselines. When a node has been visited
    # few times its running mean is noisy and gradient ~ (reward - reward) ~ 0
    # which leaves the node dormant. Shrinking the node baseline toward the
    # global baseline restores a useful gradient signal for low-visit nodes.
    # Effective baseline = node_avg * n/(n+alpha) + global_avg * alpha/(n+alpha)
    baseline_shrinkage_alpha: float = 8.0


class _ExperienceBuffer:
    """O(1) append/evict ring-buffer backed by collections.deque."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: Deque[Dict[str, Any]] = deque(maxlen=capacity)

    def append(self, item: Dict[str, Any]) -> None:
        self._buf.append(item)

    def __len__(self) -> int:
        return len(self._buf)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._buf[idx]


class HierarchicalBanditAgent:
    """Non-contextual hierarchical gradient bandit over an action tree.

    The policy is a per-node softmax over child preferences. There is no
    contextual signal in the policy itself — contextual adaptation comes
    from the RAG retrieval layer, not the bandit. This keeps the bandit
    update rule a clean Sutton & Barto §2.8 gradient bandit.
    """

    def __init__(self, action_tree, config: Optional[AgentConfig] = None,
                 device: Optional[str] = None, **_: Any):
        self.action_tree = action_tree
        self.config = config or AgentConfig()

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)

        self.experience_buffer = _ExperienceBuffer(self.config.buffer_capacity)
        self._preferences: Dict[str, np.ndarray] = {}
        # Per-node baseline: each node tracks its own running average reward
        self._node_avg_reward: Dict[str, float] = {}
        self._node_reward_updates: Dict[str, int] = {}
        self.average_reward = 0.0
        self._reward_updates = 0

    def _node_key(self, node) -> str:
        if hasattr(node, "get_path"):
            return "/".join(node.get_path())
        return getattr(node, "name", str(id(node)))

    def _get_preferences(self, node) -> np.ndarray:
        key = self._node_key(node)
        n = len(getattr(node, "children", []))
        if key not in self._preferences or len(self._preferences[key]) != n:
            self._preferences[key] = np.zeros(n, dtype=np.float64)
        return self._preferences[key]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Behavior policy used for sampling actions (includes explicit epsilon mixing)."""
        z = x / max(self.config.temperature, 1e-6)
        z = z - np.max(z)
        e = np.exp(z)
        probs = e / np.clip(np.sum(e), 1e-12, None)
        # Mix softmax policy with uniform distribution for explicit exploration.
        eps = float(np.clip(self.config.exploration_rate, 0.0, 1.0))
        if eps > 0.0 and len(probs) > 0:
            probs = (1.0 - eps) * probs + eps * (1.0 / len(probs))
        return probs / np.clip(np.sum(probs), 1e-12, None)

    def _softmax_policy(self, x: np.ndarray) -> np.ndarray:
        """Parametric softmax policy used in the gradient-bandit update rule."""
        z = x / max(self.config.temperature, 1e-6)
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.clip(np.sum(e), 1e-12, None)

    def navigate_to_leaf(self, deterministic: bool = False,
                         return_path: bool = True,
                         min_leaf_depth: int = 0) -> Any:
        """Navigate the action tree to a leaf node.

        Args:
            deterministic: If True, always pick the highest-probability child.
            return_path: If True, return (leaf_node, decision_path); otherwise return leaf_node only.
            min_leaf_depth: Minimum depth the chosen leaf must be at. When the
                policy would stop at a shallower node the agent continues to
                descend (using the softmax policy) until the depth requirement is
                met or no children remain. This is used by run_attack_with_rag to
                force deeper, more diverse tree navigation when the agent is stuck.
        """
        current = self.action_tree.root
        decision_path: List[Dict[str, Any]] = []

        while getattr(current, "children", []):
            prefs = self._get_preferences(current)
            probs = self._softmax(prefs)
            current_depth = len(decision_path)

            # If the policy wants to stop here (is_leaf) but we haven't reached
            # min_leaf_depth yet, keep descending using the softmax distribution.
            at_leaf = getattr(current, "is_leaf", False)
            below_min_depth = current_depth < min_leaf_depth

            if at_leaf and not below_min_depth:
                break

            child_idx = int(np.argmax(probs)) if deterministic else int(np.random.choice(len(probs), p=probs))
            decision_path.append({
                "node": current,
                "child_idx": child_idx,
                "depth": current_depth,
                "log_prob": float(math.log(max(probs[child_idx], 1e-12))),
                "action_probs": probs.copy(),
            })
            current = current.children[child_idx]

        return (current, decision_path) if return_path else current

    def decision_path_to_leaf(self, leaf_node) -> List[Dict[str, Any]]:
        """Describe the current-policy path to an externally selected leaf.

        This is for attribution/reporting of heuristic interventions. The
        returned path must not be treated as a policy sample for training.
        """
        ancestry = []
        current = leaf_node
        while getattr(current, "parent", None) is not None:
            ancestry.append(current)
            current = current.parent
        ancestry.reverse()

        current = self.action_tree.root
        decision_path: List[Dict[str, Any]] = []
        for depth, child in enumerate(ancestry):
            if child not in getattr(current, "children", []):
                return []
            child_idx = current.children.index(child)
            probs = self._softmax(self._get_preferences(current))
            decision_path.append({
                "node": current,
                "child_idx": child_idx,
                "depth": depth,
                "log_prob": float(math.log(max(probs[child_idx], 1e-12))),
                "action_probs": probs.copy(),
                "intervention": True,
            })
            current = child
        return decision_path

    def compute_trajectory_reward(
        self,
        final_reward: float,
        decision_path: List[Dict[str, Any]],
        depth_penalty: float = 0.0,
        leaf_discount: float = 0.85,
    ) -> List[float]:
        """Distribute the terminal reward across the decision path.

        Each node receives `final_reward * leaf_discount^(distance_from_leaf)` minus
        an optional depth penalty. The leaf gets the full reward; ancestors get a
        geometrically discounted share. This addresses the credit-assignment problem
        where a uniform reward (the previous behaviour) gave the root the same signal
        as the leaf and caused root preferences to overfit on partial credit.

        Args:
            final_reward: Terminal reward returned by the environment.
            decision_path: Ordered list of decision dicts (root → leaf).
            depth_penalty: Extra subtractive penalty per depth step (default 0).
            leaf_discount: Geometric decay from leaf toward root (1.0 → uniform).
        """
        n = len(decision_path)
        if n == 0:
            return []
        leaf_discount = float(max(0.0, min(1.0, leaf_discount)))
        rewards: List[float] = []
        for i in range(n):
            dist_from_leaf = (n - 1) - i  # 0 at leaf, n-1 at root
            rewards.append(float(final_reward * (leaf_discount ** dist_from_leaf) - depth_penalty * (i + 1)))
        return rewards

    def store_trajectory(self, *args: Any, **kwargs: Any) -> None:
        """Store a trajectory in a backward-compatible format.

        Supported call forms:
            store_trajectory(decision_path, trajectory_rewards)
            store_trajectory(query_text, decision_path, trajectory_rewards)
            store_trajectory(decision_path=..., trajectory_rewards=..., query_text="...")
        """
        query_text = kwargs.pop("query_text", "")
        if "decision_path" in kwargs or "trajectory_rewards" in kwargs:
            if args:
                raise TypeError("Use either positional or keyword arguments for the trajectory payload")
            try:
                decision_path = kwargs.pop("decision_path")
                trajectory_rewards = kwargs.pop("trajectory_rewards")
            except KeyError as exc:
                raise TypeError("decision_path and trajectory_rewards are both required") from exc
        elif len(args) == 2:
            decision_path, trajectory_rewards = args
        elif len(args) == 3 and isinstance(args[0], str):
            query_text, decision_path, trajectory_rewards = args
        else:
            raise TypeError(
                "store_trajectory expects (decision_path, trajectory_rewards) or "
                "(query_text, decision_path, trajectory_rewards)"
            )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}")

        for step, reward in zip(decision_path, trajectory_rewards):
            node = step["node"]
            probs = np.array(step.get("action_probs"), dtype=np.float64)
            self.experience_buffer.append({
                "query_text": str(query_text or ""),
                "node_key": self._node_key(node),
                "node_name": getattr(node, "name", "unknown"),
                "depth": int(step.get("depth", 0)),
                "action": int(step["child_idx"]),
                "reward": float(reward),
                "action_probs": probs,
            })

    def can_train(self, batch_size: Optional[int] = None) -> bool:
        return len(self.experience_buffer) >= (batch_size or self.config.batch_size)

    def should_train(self, batch_size: Optional[int] = None) -> bool:
        """Alias for can_train(); used by run_attack_with_rag training loop."""
        return self.can_train(batch_size)

    def train_step(self, batch_size: Optional[int] = None, epochs: int = 1) -> Dict[str, float]:
        if len(self.experience_buffer) == 0:
            return {}
        bs = min(batch_size or self.config.batch_size, len(self.experience_buffer))
        idx = np.random.choice(len(self.experience_buffer), size=bs, replace=False)
        batch = [self.experience_buffer[int(i)] for i in idx]
        ent = 0.0
        updates = 0

        # Snapshot baselines BEFORE any replay so repeated epochs don't inflate
        # the running mean with the same old rewards. We also snapshot the
        # per-node visit counts so shrinkage weights are consistent within the batch.
        node_keys_in_batch = set(exp["node_key"] for exp in batch)
        baseline_snapshot: Dict[str, float] = {
            k: self._node_avg_reward.get(k, 0.0) for k in node_keys_in_batch
        }
        count_snapshot: Dict[str, int] = {
            k: self._node_reward_updates.get(k, 0) for k in node_keys_in_batch
        }
        global_baseline = float(self.average_reward) if self.config.use_baseline else 0.0
        alpha = float(getattr(self.config, "baseline_shrinkage_alpha", 8.0))

        for _ in range(max(1, epochs)):
            for exp in batch:
                key = exp["node_key"]
                n_actions = len(np.array(exp["action_probs"], dtype=np.float64))
                action = int(exp["action"])
                reward = float(exp["reward"])
                if key not in self._preferences or len(self._preferences[key]) != n_actions:
                    self._preferences[key] = np.zeros(n_actions, dtype=np.float64)

                # Use parametric softmax policy (no epsilon mixing) in the gradient
                # update — exploration noise is sampling-only and must not appear here.
                prefs = self._preferences[key]
                current_probs = self._softmax_policy(prefs)

                # SHRINKAGE BASELINE: low-visit nodes blend their noisy node-average
                # with the global running average so the gradient never collapses to ~0
                # just because a node has only been seen 2-3 times.
                if self.config.use_baseline:
                    n_visits = count_snapshot.get(key, 0)
                    node_b = baseline_snapshot.get(key, 0.0)
                    w_node = n_visits / (n_visits + alpha)
                    baseline = w_node * node_b + (1.0 - w_node) * global_baseline
                else:
                    baseline = 0.0

                mask = np.zeros(n_actions, dtype=np.float64)
                mask[action] = 1.0
                self._preferences[key] = prefs + self.config.step_size * (reward - baseline) * (mask - current_probs)

                ent += float(-np.sum(current_probs * np.log(np.clip(current_probs, 1e-12, None))))
                updates += 1

        # Update per-node AND global running means once per (key, reward) pair after all epochs.
        # Global mean is needed by the shrinkage baseline so it must stay current.
        for exp in batch:
            key = exp["node_key"]
            reward = float(exp["reward"])
            prev_updates = self._node_reward_updates.get(key, 0)
            prev_avg = self._node_avg_reward.get(key, 0.0)
            new_updates = prev_updates + 1
            self._node_reward_updates[key] = new_updates
            self._node_avg_reward[key] = prev_avg + (reward - prev_avg) / new_updates

            # Global running mean (Welford). Used as the shrinkage target.
            self._reward_updates += 1
            self.average_reward += (reward - self.average_reward) / self._reward_updates

        # NOTE: This is a gradient bandit (Sutton & Barto §2.8), not an actor-critic.
        # It has no value head and no policy-gradient "loss" scalar, so we do NOT
        # emit fake policy_loss/value_loss/kl_divergence keys (they were previously
        # hardcoded to 0.0 and logged, which misleadingly implied a PPO-style update).
        # The meaningful diagnostics for a bandit are entropy and mean batch reward.
        return {
            "entropy": ent / max(1, updates),
            "updates": float(updates),
            "avg_reward": float(np.mean([x["reward"] for x in batch])) if batch else 0.0,
        }

    def select_action(self, action_history: Optional[List[int]] = None, deterministic: bool = False) -> Tuple[int, float]:
        leaf_node, decision_path = self.navigate_to_leaf(deterministic=deterministic, return_path=True)
        action_idx = 0
        if hasattr(self.action_tree, "leaf_nodes"):
            for i, node in enumerate(self.action_tree.leaf_nodes):
                if getattr(node, "name", None) == getattr(leaf_node, "name", None):
                    action_idx = i
                    break
        return action_idx, float(sum(d.get("log_prob", 0.0) for d in decision_path))

    def get_action_name(self, action_idx: int) -> str:
        if hasattr(self.action_tree, "leaf_nodes") and 0 <= action_idx < len(self.action_tree.leaf_nodes):
            return self.action_tree.leaf_nodes[action_idx].name
        return f"action_{action_idx}"

    def save(self, path: str, save_optimizer: bool = True) -> None:
        state = {
            "config": self.config.__dict__,
            "preferences": self._preferences,
            "node_avg_reward": self._node_avg_reward,
            "node_reward_updates": self._node_reward_updates,
            "average_reward": self.average_reward,
            "reward_updates": self._reward_updates,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(p))

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self._preferences = {k: np.array(v, dtype=np.float64) for k, v in state.get("preferences", {}).items()}
        self._node_avg_reward = {k: float(v) for k, v in state.get("node_avg_reward", {}).items()}
        self._node_reward_updates = {k: int(v) for k, v in state.get("node_reward_updates", {}).items()}
        self.average_reward = float(state.get("average_reward", 0.0))
        self._reward_updates = int(state.get("reward_updates", 0))

    def store_experience(self, *args, **kwargs):
        return None


def create_hierarchical_bandit_agent(action_tree, **kwargs) -> HierarchicalBanditAgent:
    return HierarchicalBanditAgent(action_tree=action_tree, **kwargs)
