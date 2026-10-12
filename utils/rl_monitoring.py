"""
RL Training Monitoring and Stability Checks

Monitors KL divergence, entropy, gradient norms, and other metrics
to detect training instability and suggest interventions.
"""

import logging
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class RLMonitor:
    """
    Monitors RL training metrics and detects instability.

    Key metrics tracked:
    - KL divergence (policy change per update)
    - Entropy (exploration level)
    - Gradient norms (training stability)
    - Policy/value losses
    """

    def __init__(
        self,
        kl_threshold: float = 0.05,
        entropy_min: float = 0.01,
        grad_norm_max: float = 5.0,
        window_size: int = 10
    ):
        """
        Initialize RL monitor

        Args:
            kl_threshold: Warn if KL divergence exceeds this
            entropy_min: Warn if entropy drops below this
            grad_norm_max: Warn if gradient norm exceeds this
            window_size: Number of recent updates to track
        """
        self.kl_threshold = kl_threshold
        self.entropy_min = entropy_min
        self.grad_norm_max = grad_norm_max
        self.window_size = window_size

        # Rolling windows
        self.kl_history = deque(maxlen=window_size)
        self.entropy_history = deque(maxlen=window_size)
        self.grad_norm_history = deque(maxlen=window_size)
        self.policy_loss_history = deque(maxlen=window_size)
        self.value_loss_history = deque(maxlen=window_size)

        # Counters
        self.total_updates = 0
        self.kl_violations = 0
        self.entropy_warnings = 0
        self.grad_explosions = 0

        logger.info(
            f"[RL-MONITOR] Initialized: kl_threshold={kl_threshold}, "
            f"entropy_min={entropy_min}, grad_norm_max={grad_norm_max}"
        )

    def record_update(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record training update and check for issues.

        Args:
            metrics: Dictionary with training metrics from train_step()
                Expected keys: policy_loss, value_loss, gradient_norm,
                              kl_divergence (optional), entropy (optional)

        Returns:
            Dict with warnings/recommendations
        """
        self.total_updates += 1

        # Extract metrics (with defaults)
        policy_loss = metrics.get('policy_loss', 0.0)
        value_loss = metrics.get('value_loss', 0.0)
        grad_norm = metrics.get('gradient_norm', 0.0)
        kl_div = metrics.get('kl_divergence', None)
        entropy = metrics.get('entropy', None)

        # Record
        self.policy_loss_history.append(policy_loss)
        self.value_loss_history.append(value_loss)
        self.grad_norm_history.append(grad_norm)

        if kl_div is not None:
            self.kl_history.append(kl_div)
        if entropy is not None:
            self.entropy_history.append(entropy)

        # Check for issues
        warnings = []

        # 1. KL divergence spike (policy changing too fast)
        if kl_div is not None and kl_div > self.kl_threshold:
            self.kl_violations += 1
            warnings.append({
                'type': 'kl_spike',
                'severity': 'high' if kl_div > 2 * self.kl_threshold else 'medium',
                'value': kl_div,
                'threshold': self.kl_threshold,
                'message': f"KL divergence {kl_div:.4f} exceeds threshold {self.kl_threshold}",
                'recommendation': "Consider reducing learning rate or increasing Bandit clip epsilon"
            })

        # 2. Entropy collapse (loss of exploration)
        if entropy is not None and entropy < self.entropy_min:
            self.entropy_warnings += 1
            warnings.append({
                'type': 'entropy_collapse',
                'severity': 'medium',
                'value': entropy,
                'threshold': self.entropy_min,
                'message': f"Entropy {entropy:.4f} below minimum {self.entropy_min}",
                'recommendation': "Consider increasing entropy coefficient or temperature"
            })

        # 3. Gradient explosion
        if grad_norm > self.grad_norm_max:
            self.grad_explosions += 1
            warnings.append({
                'type': 'grad_explosion',
                'severity': 'high' if grad_norm > 2 * self.grad_norm_max else 'medium',
                'value': grad_norm,
                'threshold': self.grad_norm_max,
                'message': f"Gradient norm {grad_norm:.4f} exceeds max {self.grad_norm_max}",
                'recommendation': "Gradient clipping should handle this, but check for NaN/Inf"
            })

        # 4. Loss divergence (training unstable)
        if len(self.policy_loss_history) >= 3:
            recent_losses = list(self.policy_loss_history)[-3:]
            if all(recent_losses[i] > recent_losses[i-1] * 1.5 for i in range(1, len(recent_losses))):
                warnings.append({
                    'type': 'loss_divergence',
                    'severity': 'high',
                    'value': recent_losses[-1],
                    'message': "Policy loss increasing rapidly (3 consecutive increases >50%)",
                    'recommendation': "Reduce learning rate immediately"
                })

        # Log warnings
        for warning in warnings:
            if warning['severity'] == 'high':
                logger.warning(f"[RL-MONITOR] {warning['message']} | {warning['recommendation']}")
            else:
                logger.info(f"[RL-MONITOR] {warning['message']}")

        return {
            'warnings': warnings,
            'total_updates': self.total_updates,
            'kl_violations': self.kl_violations,
            'entropy_warnings': self.entropy_warnings,
            'grad_explosions': self.grad_explosions
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        summary = {
            'total_updates': self.total_updates,
            'kl_violations': self.kl_violations,
            'entropy_warnings': self.entropy_warnings,
            'grad_explosions': self.grad_explosions
        }

        if self.kl_history:
            summary['avg_kl'] = np.mean(self.kl_history)
            summary['max_kl'] = np.max(self.kl_history)

        if self.entropy_history:
            summary['avg_entropy'] = np.mean(self.entropy_history)
            summary['min_entropy'] = np.min(self.entropy_history)

        if self.grad_norm_history:
            summary['avg_grad_norm'] = np.mean(self.grad_norm_history)
            summary['max_grad_norm'] = np.max(self.grad_norm_history)

        if self.policy_loss_history:
            summary['avg_policy_loss'] = np.mean(self.policy_loss_history)

        if self.value_loss_history:
            summary['avg_value_loss'] = np.mean(self.value_loss_history)

        return summary

    def should_reduce_lr(self) -> bool:
        """Check if learning rate should be reduced"""
        # Reduce if multiple high-severity issues
        recent_kl_violations = sum(1 for kl in list(self.kl_history)[-5:] if kl > self.kl_threshold)
        recent_grad_explosions = sum(1 for gn in list(self.grad_norm_history)[-5:] if gn > self.grad_norm_max)

        return (recent_kl_violations >= 3) or (recent_grad_explosions >= 3)

    def log_summary(self):
        """Log summary statistics"""
        summary = self.get_summary()

        logger.info("[RL-MONITOR] === Training Summary ===")
        logger.info(f"  Total updates: {summary.get('total_updates', 0)}")
        logger.info(f"  KL violations: {summary.get('kl_violations', 0)}")
        logger.info(f"  Entropy warnings: {summary.get('entropy_warnings', 0)}")
        logger.info(f"  Gradient explosions: {summary.get('grad_explosions', 0)}")

        if 'avg_kl' in summary:
            logger.info(f"  Avg KL: {summary['avg_kl']:.4f}, Max: {summary['max_kl']:.4f}")

        if 'avg_entropy' in summary:
            logger.info(f"  Avg entropy: {summary['avg_entropy']:.4f}, Min: {summary['min_entropy']:.4f}")

        if 'avg_grad_norm' in summary:
            logger.info(f"  Avg grad norm: {summary['avg_grad_norm']:.4f}, Max: {summary['max_grad_norm']:.4f}")

        if summary.get('kl_violations', 0) > 0 or summary.get('grad_explosions', 0) > 0:
            logger.warning("[RL-MONITOR] Training showed instability signs - consider reducing learning rate")

