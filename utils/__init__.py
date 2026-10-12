"""
Utility Functions Package
"""

from .common import *
from .loggers import *
from .log_config import get_log_file, get_subdir, LOGS_DIR, STORE_FOLDER
from .reward_system import AdaptiveRewardCalculator, RewardTracker, create_reward_calculator
from .embeddings import SemanticEmbedder, get_embedder
from .rl_components import PrioritizedReplayBuffer, IntrinsicMotivation, CurriculumManager

try:
    from .advanced_scorer import AdvancedScorer
    _has_advanced_scorer = True
except ImportError:
    _has_advanced_scorer = False

# Build __all__ dynamically
__all__ = [
    'get_log_file',
    'get_subdir',
    'LOGS_DIR',
    'STORE_FOLDER',
    'AdaptiveRewardCalculator',
    'RewardTracker',
    'create_reward_calculator',
    'SemanticEmbedder',
    'get_embedder',
    'PrioritizedReplayBuffer',
    'IntrinsicMotivation',
    'CurriculumManager'
]

if _has_advanced_scorer:
    __all__.append('AdvancedScorer')
