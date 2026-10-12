"""
TemplateSelectionPolicy — extracts the per-model / global gradient-bandit
wiring previously buried inside RAGEnhancedAttackRunner.

The two parallel bandits (global + per-model) were updated with the same
reward without coordination, which inflated variance and mixed information.
This module gives them a single owner with explicit shrinkage between scales.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .gradient_bandit_template_selector import GradientBanditTemplateSelector
from .reward_system import normalize_to_unit_range, MAX_SHAPED_REWARD, MIN_SHAPED_REWARD

logger = logging.getLogger(__name__)


class TemplateSelectionPolicy:
    """Hierarchical (global → per-model) gradient bandit for template choice.

    Update flow:
        bandit_update(model_id, template, reward) →
            per_model_bandit.update(template, reward)
            global_bandit.update(template, reward * cross_model_weight)

    The global bandit acts as a prior; per-model bandits are posteriors.
    """

    def __init__(self, templates: List[str], step_size: float = 0.1,
                 cross_model_weight: float = 0.3) -> None:
        self._templates = list(templates) or ["default"]
        self._step_size = step_size
        self._cross_model_weight = float(max(0.0, min(1.0, cross_model_weight)))
        self._global = GradientBanditTemplateSelector(
            self._templates, step_size=step_size, use_baseline=True
        )
        self._per_model: Dict[str, GradientBanditTemplateSelector] = {}

    # --- accessors -------------------------------------------------------
    def global_bandit(self) -> GradientBanditTemplateSelector:
        return self._global

    def for_model(self, model_id: str) -> GradientBanditTemplateSelector:
        if model_id not in self._per_model:
            self._per_model[model_id] = GradientBanditTemplateSelector(
                self._templates, step_size=self._step_size, use_baseline=True
            )
        return self._per_model[model_id]

    # --- core --------------------------------------------------------------
    def select(self, model_id: Optional[str], exclude: Optional[List[str]] = None,
               use_global_prior: bool = True) -> str:
        """Pick a template. When `use_global_prior=True` and the per-model bandit
        is cold, defers to the global bandit so we don't waste early budget on
        an uninformed posterior."""
        per_model = self.for_model(model_id) if model_id else self._global
        if use_global_prior and per_model.total_selections < 5:
            return self._global.select(exclude=exclude)
        return per_model.select(exclude=exclude)

    def select_with_log_prob(
        self, model_id: Optional[str], exclude: Optional[List[str]] = None,
        use_global_prior: bool = True
    ) -> tuple:
        """Like select(), but also returns log-probability of the chosen action."""
        per_model = self.for_model(model_id) if model_id else self._global
        if use_global_prior and per_model.total_selections < 5:
            return self._global.select_with_log_prob(exclude=exclude)
        return per_model.select_with_log_prob(exclude=exclude)

    def update(self, model_id: Optional[str], template: str, shaped_reward: float,
               is_refusal: bool = False, infra_failure: bool = False) -> None:
        """Push reward into per-model bandit, and a downweighted copy into the
        global bandit. Reward is mapped through the central scale contract."""
        if not template or infra_failure:
            return
        # Backward-compatible input contract:
        # - shaped reward from RewardSystem (roughly [-10, 60]) -> normalize
        # - already-bounded policy reward in [-1.5, 1.5] -> use directly
        if -1.5 <= float(shaped_reward) <= 1.5:
            unit_reward = float(max(-1.0, min(1.0, shaped_reward)))
        else:
            unit_reward = normalize_to_unit_range(shaped_reward, MIN_SHAPED_REWARD, MAX_SHAPED_REWARD)
        if is_refusal:
            unit_reward -= 0.1

        # Per-model (full credit)
        per_model = self.for_model(model_id) if model_id else self._global
        per_model.update(template, unit_reward)

        # Global (downweighted — acts as cross-model prior, not the primary policy)
        if model_id is not None:
            self._global.update(template, unit_reward * self._cross_model_weight)

    def get_best_templates(self, model_id: Optional[str], top_k: int = 5):
        bandit = self.for_model(model_id) if model_id else self._global
        return bandit.get_best_templates(top_k=top_k)
