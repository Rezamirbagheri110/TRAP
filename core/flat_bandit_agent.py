"""
Flat gradient bandit over tree leaves for the Flat+RAG ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import random

import numpy as np
import torch


@dataclass
class FlatAgentConfig:
    step_size: float = 0.1
    temperature: float = 1.0
    exploration_rate: float = 0.1
    batch_size: int = 32
    buffer_capacity: int = 10000
    random_seed: Optional[int] = None


class FlatBanditAgent:
    def __init__(
        self,
        action_space_size: int,
        action_tree=None,
        step_size: float = 0.1,
        temperature: float = 1.0,
        exploration_rate: float = 0.1,
        batch_size: int = 32,
        buffer_capacity: int = 10000,
        random_seed: Optional[int] = None,
        **_: Any,
    ):
        self.action_space_size = int(action_space_size)
        self.action_tree = action_tree
        self.config = FlatAgentConfig(
            step_size=step_size,
            temperature=temperature,
            exploration_rate=exploration_rate,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            random_seed=random_seed,
        )
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.preferences = np.zeros(self.action_space_size, dtype=np.float64)
        self.average_reward = 0.0
        self.reward_updates = 0
        self.experience_buffer: List[Dict[str, Any]] = []

    def _softmax(self) -> np.ndarray:
        z = self.preferences / max(self.config.temperature, 1e-6)
        z = z - np.max(z)
        probs = np.exp(z)
        probs = probs / np.clip(np.sum(probs), 1e-12, None)
        eps = float(np.clip(self.config.exploration_rate, 0.0, 1.0))
        if eps > 0.0:
            probs = (1.0 - eps) * probs + eps * (1.0 / len(probs))
        return probs / np.clip(np.sum(probs), 1e-12, None)

    def _softmax_policy(self) -> np.ndarray:
        z = self.preferences / max(self.config.temperature, 1e-6)
        z = z - np.max(z)
        probs = np.exp(z)
        return probs / np.clip(np.sum(probs), 1e-12, None)

    def select_action(
        self,
        action_history: Optional[List[int]] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        del action_history
        probs = self._softmax()
        action = int(np.argmax(probs)) if deterministic else int(np.random.choice(len(probs), p=probs))
        return action, float(math.log(max(probs[action], 1e-12)))

    def get_action_name(self, action_idx: int) -> str:
        if self.action_tree is not None and hasattr(self.action_tree, "leaf_nodes"):
            if 0 <= action_idx < len(self.action_tree.leaf_nodes):
                return self.action_tree.leaf_nodes[action_idx].name
        return f"action_{action_idx}"

    def navigate_to_leaf(
        self,
        deterministic: bool = False,
        return_path: bool = True,
        min_leaf_depth: int = 0,
    ):
        del min_leaf_depth
        action_idx, log_prob = self.select_action(deterministic=deterministic)
        leaf_node = self.action_tree.leaf_nodes[action_idx]
        decision_path = []
        if self.action_tree is not None and getattr(self.action_tree, "root", None) is not None:
            root = self.action_tree.root
            if getattr(root, "children", None):
                decision_path = [{
                    "node": root,
                    "child_idx": action_idx,
                    "depth": 0,
                    "log_prob": log_prob,
                    "action_probs": self._softmax().copy(),
                }]
        return (leaf_node, decision_path) if return_path else leaf_node

    def decision_path_to_leaf(self, leaf_node) -> List[Dict[str, Any]]:
        """Describe an externally selected flat action for reporting only."""
        try:
            action_idx = self.action_tree.leaf_nodes.index(leaf_node)
        except (AttributeError, ValueError):
            return []
        probs = self._softmax()
        return [{
            "node": self.action_tree.root,
            "child_idx": action_idx,
            "depth": 0,
            "log_prob": float(math.log(max(probs[action_idx], 1e-12))),
            "action_probs": probs.copy(),
            "intervention": True,
        }]

    def compute_trajectory_reward(
        self,
        final_reward: float,
        decision_path: List[Dict[str, Any]],
        depth_penalty: float = 0.0,
        leaf_discount: float = 1.0,
    ) -> List[float]:
        del leaf_discount
        if not decision_path:
            return []
        return [float(final_reward - depth_penalty)]

    def store_trajectory(self, *args: Any, **kwargs: Any) -> None:
        query_text = kwargs.pop("query_text", "")
        if "decision_path" in kwargs or "trajectory_rewards" in kwargs:
            decision_path = kwargs.pop("decision_path")
            trajectory_rewards = kwargs.pop("trajectory_rewards")
        elif len(args) == 2:
            decision_path, trajectory_rewards = args
        elif len(args) == 3 and isinstance(args[0], str):
            query_text, decision_path, trajectory_rewards = args
        else:
            raise TypeError("store_trajectory expects a trajectory payload")
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}")

        for step, reward in zip(decision_path, trajectory_rewards):
            self.store_experience(
                action=int(step["child_idx"]),
                reward=float(reward),
                query_text=str(query_text or ""),
            )

    def store_experience(self, action: int, reward: float, query_text: str = "") -> None:
        self.experience_buffer.append({
            "action": int(action),
            "reward": float(reward),
            "query_text": query_text,
        })
        if len(self.experience_buffer) > self.config.buffer_capacity:
            self.experience_buffer.pop(0)

    def can_train(self, batch_size: Optional[int] = None) -> bool:
        return len(self.experience_buffer) >= (batch_size or self.config.batch_size)

    def should_train(self, batch_size: Optional[int] = None) -> bool:
        return self.can_train(batch_size)

    def train_step(self, batch_size: Optional[int] = None, epochs: int = 1) -> Dict[str, float]:
        if not self.experience_buffer:
            return {}
        bs = min(batch_size or self.config.batch_size, len(self.experience_buffer))
        idx = np.random.choice(len(self.experience_buffer), size=bs, replace=False)
        batch = [self.experience_buffer[int(i)] for i in idx]
        updates = 0

        for _ in range(max(1, epochs)):
            for exp in batch:
                action = int(exp["action"])
                reward = float(exp["reward"])
                probs = self._softmax_policy()
                mask = np.zeros(self.action_space_size, dtype=np.float64)
                mask[action] = 1.0
                baseline = self.average_reward
                self.preferences = self.preferences + self.config.step_size * (reward - baseline) * (mask - probs)
                updates += 1

        for exp in batch:
            reward = float(exp["reward"])
            self.reward_updates += 1
            self.average_reward += (reward - self.average_reward) / self.reward_updates

        # Gradient bandit (no actor-critic loss); emit only meaningful diagnostics.
        return {
            "entropy": float(-np.sum(self._softmax_policy() * np.log(np.clip(self._softmax_policy(), 1e-12, None)))),
            "updates": float(updates),
            "avg_reward": float(np.mean([x["reward"] for x in batch])) if batch else 0.0,
        }

    def save(self, path: str, save_optimizer: bool = True):
        del save_optimizer
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "preferences": self.preferences,
                "average_reward": self.average_reward,
                "reward_updates": self.reward_updates,
                "config": self.config.__dict__,
            },
            str(p),
        )

    def load(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.preferences = np.array(state.get("preferences", self.preferences), dtype=np.float64)
        self.average_reward = float(state.get("average_reward", 0.0))
        self.reward_updates = int(state.get("reward_updates", 0))


def create_flat_bandit_agent(**kwargs) -> FlatBanditAgent:
    return FlatBanditAgent(**kwargs)
