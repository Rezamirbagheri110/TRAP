"""Gradient-bandit template selector."""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class GradientBanditTemplateSelector:
    def __init__(self, templates: List[str], step_size: float = 0.1, use_baseline: bool = True,
                 temperature: float = 1.0, cold_start_boost: float = 0.5, cold_start_min_visits: int = 3):
        """
        Args:
            cold_start_boost: Additive preference bias for under-explored templates.
                Prevents newly-added or rarely-seen templates from being starved by
                already-trained competitors (the classic cold-start problem).
            cold_start_min_visits: Number of visits below which a template still
                receives the full cold_start_boost; the boost decays linearly to 0
                as counts approach this threshold.
        """
        self.templates = list(templates)
        self.step_size = step_size
        self.use_baseline = use_baseline
        self.temperature = temperature
        self.cold_start_boost = float(cold_start_boost)
        self.cold_start_min_visits = max(1, int(cold_start_min_visits))
        self.preferences = {t: 0.0 for t in self.templates}
        self.counts = {t: 0 for t in self.templates}
        self.mean_rewards = {t: 0.0 for t in self.templates}
        self.average_reward = 0.0
        self.total_selections = 0

    def _cold_start_bonus(self, template: str) -> float:
        """Linearly decaying bonus that boosts under-explored templates."""
        c = self.counts.get(template, 0)
        if c >= self.cold_start_min_visits:
            return 0.0
        decay = 1.0 - (c / self.cold_start_min_visits)
        return self.cold_start_boost * decay

    def _probs(self, available: List[str], apply_cold_start: bool = True) -> np.ndarray:
        """Compute softmax over `available` templates.

        When `apply_cold_start=True` (default, sampling), under-explored templates
        receive an additive preference bonus so cold-start templates aren't starved.
        When `apply_cold_start=False` (gradient update), the parametric softmax is
        used unchanged — the gradient must use the *unbiased* policy distribution
        so updates stay theoretically correct.
        """
        bias = np.array(
            [self._cold_start_bonus(t) if apply_cold_start else 0.0 for t in available],
            dtype=np.float64,
        )
        vals = (np.array([self.preferences[t] for t in available], dtype=np.float64) + bias) / max(self.temperature, 1e-6)
        vals = vals - np.max(vals)
        e = np.exp(vals)
        return e / np.clip(np.sum(e), 1e-12, None)

    def select(self, exclude: Optional[List[str]] = None) -> str:
        exclude = exclude or []
        available = [t for t in self.templates if t not in exclude]
        if not available:
            available = self.templates
        probs = self._probs(available, apply_cold_start=True)
        return str(np.random.choice(available, p=probs))

    def select_with_log_prob(
        self, exclude: Optional[List[str]] = None
    ) -> tuple:
        """Like select(), but also returns log-probability of the chosen action.

        Returns:
            (template_name, log_prob)  where log_prob = ln(P(template | state))
        """
        exclude = exclude or []
        available = [t for t in self.templates if t not in exclude]
        if not available:
            available = self.templates
        probs = self._probs(available, apply_cold_start=True)
        chosen = str(np.random.choice(available, p=probs))
        chosen_prob = float(probs[available.index(chosen)])
        log_prob = float(np.log(max(chosen_prob, 1e-12)))
        return chosen, log_prob

    def update(self, template: str, reward: float, available: Optional[List[str]] = None):
        """Update preferences using the gradient bandit rule.

        ``available`` must be the same candidate list that was passed to
        ``select()`` when ``template`` was chosen.  Omitting it (or passing
        ``None``) falls back to the full template list, which is correct when
        ``select()`` was called without an ``exclude`` list.  Using a mismatched
        ``available`` list would compute gradients under the wrong distribution
        and bias learning toward excluded templates.
        """
        if template not in self.preferences:
            self.templates.append(template)
            self.preferences[template] = 0.0
            self.counts[template] = 0
            self.mean_rewards[template] = 0.0

        active = available if available is not None else self.templates
        # Guard: if active is empty or template is missing, fall back to full list.
        if not active or template not in active:
            active = self.templates

        prev_avg = self.average_reward
        prev_count = self.total_selections
        baseline = prev_avg if self.use_baseline else 0.0

        # IMPORTANT: gradient must use the parametric softmax (no cold-start bonus)
        # otherwise updates become biased toward the exploration scheme.
        probs = self._probs(active, apply_cold_start=False)
        chosen_idx = active.index(template)
        for i, t in enumerate(active):
            ind = 1.0 if i == chosen_idx else 0.0
            self.preferences[t] += self.step_size * (reward - baseline) * (ind - probs[i])

        self.total_selections = prev_count + 1
        self.average_reward = prev_avg + (reward - prev_avg) / max(1, self.total_selections)

        self.counts[template] += 1
        c = self.counts[template]
        self.mean_rewards[template] += (reward - self.mean_rewards[template]) / max(1, c)

    def get_best_templates(self, top_k: int = 5) -> List[tuple]:
        rows = [(t, self.mean_rewards.get(t, 0.0), self.counts.get(t, 0)) for t in self.templates]
        rows.sort(key=lambda x: x[1], reverse=True)
        return rows[:top_k]

    def get_statistics(self) -> Dict:
        best = max(self.mean_rewards, key=self.mean_rewards.get) if self.mean_rewards else None
        return {
            "total_selections": self.total_selections,
            "templates_tried": sum(1 for c in self.counts.values() if c > 0),
            "best_template": best,
            "best_mean_reward": self.mean_rewards.get(best, 0.0) if best else 0.0,
        }

    def reset(self):
        self.preferences = {t: 0.0 for t in self.templates}
        self.counts = {t: 0 for t in self.templates}
        self.mean_rewards = {t: 0.0 for t in self.templates}
        self.average_reward = 0.0
        self.total_selections = 0
