"""
Configuration for RL-based Attack System
Contains hyperparameters and settings for all components
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@dataclass
class ModelConfig:
    """Configuration for language models"""
    # Target model
    target_model: Optional[str] = None  # Supply the target explicitly.
    target_max_tokens: int = 150
    target_temperature: float = 0.0
    target_top_p: float = 1.0
    
    # Evaluator model
    evaluator_model: str = "openai/gpt-4o-mini"
    evaluator_max_tokens: int = 10
    evaluator_temperature: float = 0.0
    evaluator_top_p: float = 1.0
    
    # Attacker model (if using for prompts)
    attacker_model: Optional[str] = "qwen/qwen-2.5-7b-instruct"
    attacker_max_tokens: int = 500
    attacker_temperature: float = 1.0
    attacker_top_p: float = 0.9


@dataclass
class BanditConfig:
    """Configuration for the gradient bandit agent.

    This is a non-contextual bandit — there is no query embedding, no
    neural network, and no PPO/UCB machinery. Context-awareness comes
    from the RAG retrieval layer, not the policy.
    """
    # Learning parameters (Sutton & Barto §2.8 gradient bandit)
    step_size: float = 0.1   # α in preference update
    temperature: float = 1.0  # softmax temperature
    exploration_rate: float = 0.1  # uniform mixing weight for behavior policy
    use_baseline: bool = True

    # Training parameters
    batch_size: int = 8
    update_epochs: int = 4

    # Experience buffer
    buffer_size: int = 10000


@dataclass
class RewardConfig:
    """Configuration for reward calculation.

    These values mirror the defaults in ``utils.reward_system.RewardConfig``
    and are the single source of truth for reward hyper-parameters.
    Use ``utils.reward_system.RewardConfig.from_rl_config(instance)`` to
    convert this into the runtime config consumed by the reward calculator.
    """
    # Core reward values — kept in sync with utils/reward_system.py defaults
    success_reward: float = 35.0
    partial_reward_scale: float = 8.0
    efficiency_bonus: float = 8.0
    failure_penalty: float = -0.5
    off_topic_penalty: float = -3.5
    diversity_bonus: float = 4.0
    template_diversity_bonus: float = 2.0
    use_normalization: bool = True
    normalization_window: int = 100
    refusal_penalty: float = -2.0
    progress_reward_scale: float = 6.0
    on_topic_bonus: float = 3.0
    partial_floor: float = 1.0
    intermediate_milestones: tuple = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    milestone_bonus: float = 3.0
    successful_strategy_bonus: float = 3.0
    failed_strategy_penalty: float = -1.0

    # Success gate — must match AttackConfig.success_threshold
    success_threshold: float = 7.0

    # Adaptive reward parameters
    use_adaptive_rewards: bool = True
    adaptation_window: int = 100


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    # Training loop
    num_epochs: int = 100
    episodes_per_epoch: int = 10
    max_steps_per_episode: int = 10
    
    # Policy updates
    update_frequency: int = 2  # Update policy every N episodes
    
    # Evaluation
    eval_frequency: int = 10  # Evaluate every N epochs
    eval_episodes: int = 5  # Number of episodes for evaluation
    
    # Checkpointing
    save_frequency: int = 20  # Save checkpoint every N epochs
    checkpoint_dir: str = "checkpoints"
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 20  # Stop if no improvement for N epochs
    min_success_rate: float = 0.8  # Target success rate


@dataclass
class ActionTreeConfig:
    """Configuration for action tree"""
    # Action space
    action_space_size: int = 164  # Canonical full_active attack-tree leaves
    
    # Action selection
    exploration_rate: float = 0.1  # Epsilon for epsilon-greedy exploration
    temperature: float = 1.0  # Temperature for softmax action selection
    
    # Diversity encouragement
    encourage_diversity: bool = True
    diversity_weight: float = 0.3


@dataclass
class AttackConfig:
    """Configuration for attack execution"""
    # Attack parameters
    max_iterations: int = 10
    deterministic_eval: bool = True  # Use greedy policy during evaluation
    
    # Success criteria
    success_threshold: float = 7.0  # Canonical unified-judge success threshold
    
    # Logging
    verbose: bool = True
    log_frequency: int = 1  # Log every N iterations
    save_results: bool = True
    results_dir: str = "results"


@dataclass
class RLSystemConfig:
    """Complete configuration for RL system"""
    # Component configs
    model_config: ModelConfig = field(default_factory=ModelConfig)
    bandit_config: BanditConfig = field(default_factory=BanditConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    action_tree_config: ActionTreeConfig = field(default_factory=ActionTreeConfig)
    attack_config: AttackConfig = field(default_factory=AttackConfig)
    
    # System settings — auto-detected at runtime; override with "cpu" or "cuda"
    device: str = field(default_factory=lambda: "cuda" if _torch_cuda_available() else "cpu")
    random_seed: int = 42
    num_workers: int = 4  # For parallel data collection
    
    # Paths
    data_path: Optional[str] = None  # Path to training data
    pretrained_model_path: Optional[str] = None  # Path to pretrained agent
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'model_config': self.model_config.__dict__,
            'bandit_config': self.bandit_config.__dict__,
            'reward_config': self.reward_config.__dict__,
            'training_config': self.training_config.__dict__,
            'action_tree_config': self.action_tree_config.__dict__,
            'attack_config': self.attack_config.__dict__,
            'device': self.device,
            'random_seed': self.random_seed,
            'num_workers': self.num_workers,
            'data_path': self.data_path,
            'pretrained_model_path': self.pretrained_model_path
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'RLSystemConfig':
        """Create config from dictionary"""
        return cls(
            model_config=ModelConfig(**config_dict.get('model_config', {})),
            bandit_config=BanditConfig(**config_dict.get('bandit_config', {})),
            reward_config=RewardConfig(**config_dict.get('reward_config', {})),
            training_config=TrainingConfig(**config_dict.get('training_config', {})),
            action_tree_config=ActionTreeConfig(**config_dict.get('action_tree_config', {})),
            attack_config=AttackConfig(**config_dict.get('attack_config', {})),
            device=config_dict.get('device', 'cuda'),
            random_seed=config_dict.get('random_seed', 42),
            num_workers=config_dict.get('num_workers', 4),
            data_path=config_dict.get('data_path'),
            pretrained_model_path=config_dict.get('pretrained_model_path')
        )
    
    def save(self, path: str):
        """Save config to JSON file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'RLSystemConfig':
        """Load config from JSON file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations for different scenarios

def get_fast_training_config() -> RLSystemConfig:
    """Configuration for fast training (debugging/testing)"""
    config = RLSystemConfig()
    config.training_config.num_epochs = 10
    config.training_config.episodes_per_epoch = 5
    config.training_config.max_steps_per_episode = 5
    config.bandit_config.batch_size = 32
    return config


def get_production_config() -> RLSystemConfig:
    """Configuration for production training"""
    config = RLSystemConfig()
    config.training_config.num_epochs = 200
    config.training_config.episodes_per_epoch = 20
    config.training_config.max_steps_per_episode = 15
    config.bandit_config.batch_size = 128
    return config


def get_evaluation_config() -> RLSystemConfig:
    """Configuration for evaluation only (no training)"""
    config = RLSystemConfig()
    config.attack_config.max_iterations = 20
    config.attack_config.deterministic_eval = True
    config.training_config.num_epochs = 0  # No training
    return config


def get_high_exploration_config() -> RLSystemConfig:
    """Configuration with high exploration (early training)"""
    config = RLSystemConfig()
    config.action_tree_config.exploration_rate = 0.3
    config.reward_config.diversity_bonus = 1.0  # Higher diversity reward
    return config


def get_exploitation_config() -> RLSystemConfig:
    """Configuration for exploitation (late training)"""
    config = RLSystemConfig()
    config.action_tree_config.exploration_rate = 0.05
    config.reward_config.efficiency_bonus = 5.0  # Emphasize efficiency
    return config


# Default configuration
DEFAULT_CONFIG = RLSystemConfig()


if __name__ == "__main__":
    # Test configuration system
    print("Testing RL Configuration System")
    print("=" * 80)
    
    # Create default config
    config = RLSystemConfig()
    
    print("\nDefault Configuration:")
    print("-" * 80)
    print(f"Bandit Step Size: {config.bandit_config.step_size}")
    print(f"Batch Size: {config.bandit_config.batch_size}")
    print(f"Success Reward: {config.reward_config.success_reward}")
    print(f"Max Iterations: {config.attack_config.max_iterations}")
    print(f"Training Epochs: {config.training_config.num_epochs}")
    
    # Save and load config
    print("\nTesting Save/Load:")
    print("-" * 80)
    config.save("test_config.json")
    loaded_config = RLSystemConfig.load("test_config.json")
    print(f"Loaded config matches: {config.to_dict() == loaded_config.to_dict()}")
    
    # Test predefined configs
    print("\nPredefined Configurations:")
    print("-" * 80)
    
    fast_config = get_fast_training_config()
    print(f"Fast Training - Epochs: {fast_config.training_config.num_epochs}, "
          f"Episodes: {fast_config.training_config.episodes_per_epoch}")
    
    prod_config = get_production_config()
    print(f"Production - Epochs: {prod_config.training_config.num_epochs}, "
          f"Batch: {prod_config.bandit_config.batch_size}")

    eval_config = get_evaluation_config()
    print(f"Evaluation - Max Iterations: {eval_config.attack_config.max_iterations}, "
          f"Deterministic: {eval_config.attack_config.deterministic_eval}")

    explore_config = get_high_exploration_config()
    print(f"High Exploration - Exploration Rate: {explore_config.action_tree_config.exploration_rate}, "
          f"Diversity: {explore_config.reward_config.diversity_bonus}")
    
    exploit_config = get_exploitation_config()
    print(f"Exploitation - Exploration Rate: {exploit_config.action_tree_config.exploration_rate}, "
          f"Efficiency Bonus: {exploit_config.reward_config.efficiency_bonus}")
    
    print("\n" + "=" * 80)
    print("Configuration test complete!")
    
    # Clean up test file
    import os
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
