"""
Dynamic Strategy Scoring System
Adaptively scores strategies based on performance without blacklisting

Key Features:
- Dynamic weight assignment based on historical performance
- Continuous learning and adaptation
- No hard blacklisting - all strategies remain available but weighted
- Exploration-exploitation balance
"""

import json
import logging
import math
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class DynamicStrategyScorer:
    """
    Dynamic strategy scoring system that adapts weights based on performance

    Instead of blacklisting, this system:
    - Gives higher weights to successful strategies
    - Gives lower weights to failing strategies
    - Maintains exploration by keeping all strategies available
    - Adapts in real-time during attack execution
    """

    def __init__(
        self,
        meta_knowledge_path: str,
        min_weight: float = 0.1,  # Minimum weight (never 0 - always keep exploring)
        max_weight: float = 3.0,  # Maximum weight for best strategies
        learning_rate: float = 0.1,  # How fast to adapt weights
        exploration_bonus: float = 0.2,  # Bonus for rarely-tried strategies
        success_boost: float = 1.5,  # Multiplier for successful strategies
        failure_decay: float = 0.7  # Multiplier for failing strategies
    ):
        """
        Initialize dynamic strategy scorer

        Args:
            meta_knowledge_path: Path to meta knowledge JSON
            min_weight: Minimum weight for any strategy (ensures exploration)
            max_weight: Maximum weight for best strategies
            learning_rate: Speed of weight adaptation
            exploration_bonus: Bonus weight for under-explored strategies
            success_boost: How much to boost weights on success
            failure_decay: How much to reduce weights on failure
        """
        self.meta_knowledge_path = Path(meta_knowledge_path)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.learning_rate = learning_rate
        self.exploration_bonus = exploration_bonus
        self.success_boost = success_boost
        self.failure_decay = failure_decay

        # Strategy weights - all start at 1.0 (neutral)
        self.strategy_weights: Dict[str, float] = defaultdict(lambda: 1.0)

        # Strategy statistics
        self.strategy_stats: Dict = {}

        # Runtime tracking for this session
        self.session_attempts: Dict[str, int] = defaultdict(int)
        self.session_successes: Dict[str, int] = defaultdict(int)
        self.session_scores: Dict[str, List[float]] = defaultdict(list)

        self._load_and_analyze()

    def _load_and_analyze(self):
        """Load meta knowledge and initialize weights from historical data"""
        if not self.meta_knowledge_path.exists():
            logger.info(f"Meta knowledge file not found (will be created on first run): {self.meta_knowledge_path}")
            return

        try:
            with open(self.meta_knowledge_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            performance_db = meta.get('performance_db', {})

            # Analyze each strategy and set initial weights
            for category, strategies in performance_db.items():
                for strategy_name, data in strategies.items():
                    if not isinstance(data, dict):
                        continue

                    effectiveness = data.get('effectiveness', 0.5)
                    total_attempts = data.get('total_attempts', 0)
                    success_count = data.get('success_count', 0)
                    avg_score = data.get('avg_score', 0.0)

                    # Store stats
                    self.strategy_stats[strategy_name] = {
                        'effectiveness': effectiveness,
                        'total_attempts': total_attempts,
                        'success_count': success_count,
                        'avg_score': avg_score,
                        'category': category,
                        'success_rate': success_count / max(1, total_attempts)
                    }

                    # Calculate initial weight based on historical performance
                    initial_weight = self._calculate_weight_from_stats(
                        effectiveness, total_attempts, success_count, avg_score
                    )
                    self.strategy_weights[strategy_name] = initial_weight

            logger.info(f"[DYNAMIC_SCORER] Loaded {len(self.strategy_stats)} strategies")
            logger.info(f"[DYNAMIC_SCORER] Weight range: {min(self.strategy_weights.values()):.2f} to {max(self.strategy_weights.values()):.2f}")

            # Log top weighted strategies
            top_weighted = sorted(self.strategy_weights.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info(f"[DYNAMIC_SCORER] Top weighted strategies: {[(s, f'{w:.2f}') for s, w in top_weighted]}")

        except Exception as e:
            logger.error(f"Error loading meta knowledge: {e}")

    def _calculate_weight_from_stats(
        self,
        effectiveness: float,
        total_attempts: int,
        success_count: int,
        avg_score: float
    ) -> float:
        """
        Calculate strategy weight from historical statistics

        Uses multi-factor weighting:
        - Effectiveness (primary factor)
        - Success rate
        - Average score
        - Exploration bonus for under-tried strategies
        """
        # Base weight from effectiveness (0.1 to 3.0 range)
        base_weight = self.min_weight + (effectiveness * (self.max_weight - self.min_weight))

        # Exploration bonus for under-tried strategies
        if total_attempts < 5:
            exploration_factor = 1.0 + (self.exploration_bonus * (5 - total_attempts) / 5)
            base_weight *= exploration_factor

        # Boost for high average scores
        if avg_score > 5.0:
            score_boost = 1.0 + ((avg_score - 5.0) / 10.0)  # Up to 1.5x for score=10
            base_weight *= score_boost

        # Success rate consideration
        if total_attempts > 0:
            success_rate = success_count / total_attempts
            if success_rate > 0.5:
                base_weight *= (1.0 + (success_rate - 0.5))  # Up to 1.5x boost

        # Clamp to valid range
        return max(self.min_weight, min(self.max_weight, base_weight))

    def update_strategy_performance(
        self,
        strategy_name: str,
        score: float,
        success: bool,
        iteration: int = 1
    ):
        """
        Update strategy weight based on performance in current run

        Args:
            strategy_name: Name of the strategy
            score: Score achieved (0-10)
            success: Whether it was successful (score >= 7)
            iteration: Current iteration number
        """
        # Track session stats
        self.session_attempts[strategy_name] += 1
        if success:
            self.session_successes[strategy_name] += 1
        self.session_scores[strategy_name].append(score)

        # Get current weight
        current_weight = self.strategy_weights[strategy_name]

        # Calculate weight adjustment
        if success:
            # Boost weight on success
            adjustment = self.success_boost * self.learning_rate
            new_weight = current_weight * (1.0 + adjustment)
        else:
            # Reduce weight on failure (but not too harshly)
            adjustment = (1.0 - self.failure_decay) * self.learning_rate
            new_weight = current_weight * (1.0 - adjustment)

        # Additional boost for high scores even if not "success"
        if score >= 5.0:
            score_factor = (score - 4.0) / 10.0  # 0.1 to 0.6
            new_weight *= (1.0 + score_factor * self.learning_rate)

        # Clamp to valid range
        new_weight = max(self.min_weight, min(self.max_weight, new_weight))

        # Update weight
        old_weight = self.strategy_weights[strategy_name]
        self.strategy_weights[strategy_name] = new_weight

        # Log significant weight changes
        if abs(new_weight - old_weight) > 0.1:
            logger.info(
                f"[WEIGHT_UPDATE] {strategy_name}: {old_weight:.2f} -> {new_weight:.2f} "
                f"(score={score:.1f}, success={success})"
            )

    def get_strategy_weight(self, strategy_name: str) -> float:
        """
        Get current weight for a strategy

        Returns:
            Weight multiplier (min_weight to max_weight range)
        """
        return self.strategy_weights[strategy_name]

    def get_weighted_selection_probabilities(
        self,
        strategy_names: List[str]
    ) -> Dict[str, float]:
        """
        Get selection probabilities for strategies based on weights

        Args:
            strategy_names: List of strategy names to choose from

        Returns:
            Dictionary mapping strategy names to selection probabilities
        """
        if not strategy_names:
            return {}

        # Get weights for all strategies
        weights = {name: self.strategy_weights[name] for name in strategy_names}

        # Add small exploration bonus to rarely tried strategies in this session
        for name in strategy_names:
            if self.session_attempts[name] == 0:
                weights[name] *= (1.0 + self.exploration_bonus)

        # Normalize to probabilities
        total_weight = sum(weights.values())
        if total_weight == 0:
            # Fallback to uniform distribution
            prob = 1.0 / len(strategy_names)
            return {name: prob for name in strategy_names}

        probabilities = {name: w / total_weight for name, w in weights.items()}
        return probabilities

    def get_top_strategies(self, n: int = 20) -> List[Tuple[str, float, Dict]]:
        """
        Get top N strategies by current weight

        Args:
            n: Number of top strategies to return

        Returns:
            List of (strategy_name, weight, stats) tuples
        """
        sorted_strategies = sorted(
            self.strategy_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        result = []
        for name, weight in sorted_strategies[:n]:
            stats = self.strategy_stats.get(name, {})
            result.append((name, weight, stats))

        return result

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        weights = list(self.strategy_weights.values())

        session_success_rate = 0.0
        if sum(self.session_attempts.values()) > 0:
            session_success_rate = sum(self.session_successes.values()) / sum(self.session_attempts.values())

        return {
            'total_strategies': len(self.strategy_weights),
            'weight_stats': {
                'min': min(weights) if weights else 0,
                'max': max(weights) if weights else 0,
                'mean': sum(weights) / len(weights) if weights else 0,
                'median': sorted(weights)[len(weights)//2] if weights else 0
            },
            'session_stats': {
                'total_attempts': sum(self.session_attempts.values()),
                'total_successes': sum(self.session_successes.values()),
                'success_rate': session_success_rate,
                'unique_strategies_tried': len(self.session_attempts)
            },
            'high_weight_strategies': len([w for w in weights if w > 1.5]),
            'low_weight_strategies': len([w for w in weights if w < 0.5]),
            'config': {
                'min_weight': self.min_weight,
                'max_weight': self.max_weight,
                'learning_rate': self.learning_rate,
                'exploration_bonus': self.exploration_bonus
            }
        }

    def reset_session_stats(self):
        """Reset session statistics (for new attack run)"""
        self.session_attempts.clear()
        self.session_successes.clear()
        self.session_scores.clear()
        logger.info("[DYNAMIC_SCORER] Session statistics reset")


def create_dynamic_scorer(
    meta_knowledge_path: str,
    min_weight: float = 0.1,
    max_weight: float = 3.0,
    learning_rate: float = 0.1
) -> DynamicStrategyScorer:
    """Factory function to create dynamic strategy scorer"""
    return DynamicStrategyScorer(
        meta_knowledge_path=meta_knowledge_path,
        min_weight=min_weight,
        max_weight=max_weight,
        learning_rate=learning_rate
    )
