"""
Reinforcement Learning Components
Shared components for Advanced and Hierarchical RL training.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import heapq
import time
from scipy import stats

@dataclass
class PrioritizedExperience:
    """Experience with priority for replay"""
    data: Any  # Generic data payload
    priority: float
    timestamp: float = 0.0

    def __lt__(self, other):
        return self.priority < other.priority  # Min-heap for heapq compatibility

class CurriculumManager:
    """
    Manages curriculum learning by gradually increasing difficulty
    Uses exponential moving average and statistical significance testing
    """

    def __init__(self, difficulty_levels: int = 5, success_threshold: float = 0.7,
                 min_samples: int = 30, alpha: float = 0.1, confidence_level: float = 0.95):
        if difficulty_levels <= 0:
            raise ValueError("Difficulty levels must be positive")
        if not (0 < success_threshold <= 1):
            raise ValueError("Success threshold must be in range (0, 1]")
        if min_samples <= 0:
            raise ValueError("Minimum samples must be positive")
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be in range (0, 1]")
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be in range (0, 1)")

        self.difficulty_levels = difficulty_levels
        self.current_level = 0
        self.success_threshold = success_threshold
        self.min_samples = min_samples  # Minimum samples before considering advancement
        self.alpha = alpha  # EMA smoothing factor
        self.confidence_level = confidence_level  # Confidence level for statistical testing

        # Store raw results and EMA values
        self.level_performance = [[] for _ in range(difficulty_levels)]
        self.level_ema = [0.0 for _ in range(difficulty_levels)]  # Exponential moving averages
        self.level_variance = [0.0 for _ in range(difficulty_levels)]  # Running variance estimates
        
    def get_current_difficulty(self) -> int:
        """Get current difficulty level"""
        return self.current_level
    
    def record_result(self, level: int, success: bool):
        """Record training result at given difficulty level"""
        if not (0 <= level < self.difficulty_levels):
            raise ValueError(f"Level {level} is out of range [0, {self.difficulty_levels-1}]")

        result = 1.0 if success else 0.0
        self.level_performance[level].append(result)

        # Update exponential moving average
        if len(self.level_performance[level]) == 1:
            self.level_ema[level] = result
            self.level_variance[level] = 0.0
        else:
            # Update EMA
            old_ema = self.level_ema[level]
            self.level_ema[level] = self.alpha * result + (1 - self.alpha) * old_ema

            # Update running variance estimate
            error = result - old_ema
            self.level_variance[level] = (1 - self.alpha) * (self.level_variance[level] + self.alpha * error ** 2)

        # Keep only recent results to prevent memory growth
        if len(self.level_performance[level]) > 200:
            self.level_performance[level].pop(0)
    
    def should_advance_level(self) -> bool:
        """Check if should advance to next difficulty level using statistical significance"""
        if self.current_level >= self.difficulty_levels - 1:
            return False

        results = self.level_performance[self.current_level]
        if len(results) < self.min_samples:
            return False

        # Use EMA for more stable estimate
        current_success_rate = self.level_ema[self.current_level]

        # Calculate standard error
        n = len(results)
        if n > 1:
            std_error = np.sqrt(self.level_variance[self.current_level] / n)
        else:
            std_error = 0.0

        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        margin_of_error = z_score * std_error

        # Check if upper bound of confidence interval exceeds threshold
        upper_bound = min(1.0, current_success_rate + margin_of_error)

        return upper_bound >= self.success_threshold
    
    def advance_level(self):
        """Advance to next difficulty level"""
        if self.current_level < self.difficulty_levels - 1:
            self.current_level += 1
            print(f"Curriculum advanced to level {self.current_level + 1}/{self.difficulty_levels}")
    
    def get_current_success_rate(self, level: int = None) -> float:
        """Get current success rate estimate for a level"""
        if level is None:
            level = self.current_level
        if level >= self.difficulty_levels or len(self.level_performance[level]) == 0:
            return 0.0
        return self.level_ema[level]

    def get_difficulty_multiplier(self) -> float:
        """Get difficulty multiplier for current level"""
        return (self.current_level + 1) / self.difficulty_levels

    def get_statistics(self, level: int = None) -> Dict[str, float]:
        """Get detailed statistics for a level"""
        if level is None:
            level = self.current_level

        if level >= self.difficulty_levels:
            return {}

        results = self.level_performance[level]
        if len(results) == 0:
            return {'samples': 0, 'success_rate': 0.0, 'variance': 0.0}

        return {
            'samples': len(results),
            'success_rate': self.level_ema[level],
            'variance': self.level_variance[level],
            'raw_mean': np.mean(results),
            'raw_std': np.std(results) if len(results) > 1 else 0.0
        }


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Samples experiences based on priority (e.g., TD-error)
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta_start: float = 0.4, beta_end: float = 1.0, beta_steps: int = 100000):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be in range (0, 1]")
        if not (0 <= beta_start <= beta_end <= 1):
            raise ValueError("Beta values must satisfy 0 <= beta_start <= beta_end <= 1")
        if beta_steps <= 0:
            raise ValueError("Beta steps must be positive")

        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta_start = beta_start  # Initial importance sampling exponent
        self.beta_end = beta_end  # Final importance sampling exponent
        self.beta_steps = beta_steps  # Steps to anneal beta
        self.step_count = 0  # Track steps for beta annealing
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.epsilon = 1e-6  # Small epsilon to avoid zero priorities
        
    def add(self, data: Any, priority: float = None):
        """Add experience with priority"""
        max_priority = max(self.priorities) if self.priorities else 1.0
        if priority is None:
            priority = max_priority

        # Add epsilon to avoid zero priorities
        priority = abs(priority) + self.epsilon

        experience = PrioritizedExperience(data=data, priority=priority, timestamp=time.time())

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        """
        Sample batch with importance sampling weights

        Returns:
            batch_data: List of experience data
            weights: Importance sampling weights
            indices: Indices of sampled experiences (for updating priorities)
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Anneal beta for importance sampling
        current_beta = min(self.beta_start + (self.beta_end - self.beta_start) * (self.step_count / self.beta_steps), self.beta_end)
        self.step_count += 1

        # Calculate sampling probabilities
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs_sum = probs.sum()

        if probs_sum == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum

        # Sample indices
        actual_batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer),
                                    actual_batch_size,
                                    p=probs,
                                    replace=False)

        # Calculate importance sampling weights with annealed beta
        weights = (len(self.buffer) * probs[indices]) ** (-current_beta)
        weights /= weights.max() + self.epsilon  # Avoid division by zero

        batch_data = [self.buffer[idx].data for idx in indices]

        return batch_data, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = abs(priority) + self.epsilon
    
    def __len__(self):
        return len(self.buffer)


class IntrinsicMotivation:
    """
    Curiosity-driven exploration using multiple intrinsic motivation techniques:
    - Count-based exploration (novelty bonus)
    - Prediction error (learning progress)
    - Temporal coherence (state transition bonuses)
    """

    def __init__(self, intrinsic_reward_scale: float = 0.1, prediction_error_scale: float = 0.05,
                 temporal_coherence_scale: float = 0.02, decay_factor: float = 0.999,
                 max_visit_count: int = 1000):
        if intrinsic_reward_scale < 0:
            raise ValueError("Intrinsic reward scale must be non-negative")
        if prediction_error_scale < 0:
            raise ValueError("Prediction error scale must be non-negative")
        if temporal_coherence_scale < 0:
            raise ValueError("Temporal coherence scale must be non-negative")
        if not (0 < decay_factor <= 1):
            raise ValueError("Decay factor must be in range (0, 1]")
        if max_visit_count <= 0:
            raise ValueError("Max visit count must be positive")

        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.prediction_error_scale = prediction_error_scale
        self.temporal_coherence_scale = temporal_coherence_scale
        self.decay_factor = decay_factor
        self.max_visit_count = max_visit_count

        # Count-based exploration
        self.visit_counts = {}
        self.last_visit_time = {}

        # Prediction error based exploration
        self.state_predictions = {}  # state -> predicted_next_reward
        self.state_transition_counts = {}  # (state, next_state) -> count

        # Temporal coherence tracking
        self.previous_state = None
        self.state_sequence = []
        self.max_sequence_length = 10
        
    def compute_intrinsic_reward(self, state_key: str, actual_reward: float = 0.0) -> float:
        """
        Compute comprehensive intrinsic reward using multiple techniques

        Args:
            state_key: Unique identifier for the state (or state-action pair)
            actual_reward: Actual extrinsic reward received (for prediction error)

        Returns:
            Total intrinsic reward value
        """
        total_intrinsic_reward = 0.0

        # 1. Count-based exploration bonus
        visit_count = self.visit_counts.get(state_key, 0)
        self.visit_counts[state_key] = min(visit_count + 1, self.max_visit_count)
        self.last_visit_time[state_key] = time.time()

        # Apply decay to visit counts periodically
        self._decay_visit_counts()

        # Novelty bonus: inversely proportional to visit count
        count_bonus = self.intrinsic_reward_scale / np.sqrt(visit_count + 1)
        total_intrinsic_reward += count_bonus

        # 2. Prediction error based exploration
        prediction_error_bonus = self._compute_prediction_error_bonus(state_key, actual_reward)
        total_intrinsic_reward += prediction_error_bonus

        # 3. Temporal coherence bonus
        coherence_bonus = self._compute_temporal_coherence_bonus(state_key)
        total_intrinsic_reward += coherence_bonus

        # Update state sequence for temporal coherence
        self.state_sequence.append(state_key)
        if len(self.state_sequence) > self.max_sequence_length:
            self.state_sequence.pop(0)
        self.previous_state = state_key

        return total_intrinsic_reward
    
    def _decay_visit_counts(self):
        """Apply decay to visit counts to forget old experiences"""
        current_time = time.time()
        decay_threshold = 1000  # Decay every 1000 steps

        if not hasattr(self, '_last_decay_time'):
            self._last_decay_time = current_time
            return

        if current_time - self._last_decay_time > decay_threshold:
            # Apply decay to all visit counts
            for state in list(self.visit_counts.keys()):
                self.visit_counts[state] = max(1, int(self.visit_counts[state] * self.decay_factor))
                if self.visit_counts[state] <= 1:
                    del self.visit_counts[state]
                    if state in self.last_visit_time:
                        del self.last_visit_time[state]

            self._last_decay_time = current_time

    def _compute_prediction_error_bonus(self, state_key: str, actual_reward: float) -> float:
        """Compute prediction error based intrinsic reward"""
        if state_key not in self.state_predictions:
            # Initialize prediction for new state
            self.state_predictions[state_key] = 0.0
            return self.prediction_error_scale  # Bonus for discovering new state

        predicted_reward = self.state_predictions[state_key]
        prediction_error = abs(actual_reward - predicted_reward)

        # Update prediction using simple exponential moving average
        alpha = 0.1
        self.state_predictions[state_key] = alpha * actual_reward + (1 - alpha) * predicted_reward

        return self.prediction_error_scale * prediction_error

    def _compute_temporal_coherence_bonus(self, state_key: str) -> float:
        """Compute temporal coherence bonus based on state transition patterns"""
        if self.previous_state is None:
            return 0.0

        transition_key = (self.previous_state, state_key)
        transition_count = self.state_transition_counts.get(transition_key, 0)
        self.state_transition_counts[transition_key] = transition_count + 1

        # Bonus for rare transitions (high information content)
        total_transitions_from_prev = sum(count for (prev, _), count in self.state_transition_counts.items()
                                        if prev == self.previous_state)

        if total_transitions_from_prev > 0:
            transition_probability = transition_count / total_transitions_from_prev
            # Bonus inversely proportional to transition probability
            coherence_bonus = self.temporal_coherence_scale / (transition_probability + 0.1)
            return min(coherence_bonus, self.temporal_coherence_scale * 2)  # Cap the bonus
        else:
            return self.temporal_coherence_scale

    def get_visit_count(self, state_key: str) -> int:
        return self.visit_counts.get(state_key, 0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about intrinsic motivation state"""
        return {
            'total_states_visited': len(self.visit_counts),
            'total_transitions': len(self.state_transition_counts),
            'sequence_length': len(self.state_sequence),
            'avg_visit_count': np.mean(list(self.visit_counts.values())) if self.visit_counts else 0.0,
            'max_visit_count': max(self.visit_counts.values()) if self.visit_counts else 0
        }