"""
Core RL Components Package

This package provides the core reinforcement learning components for the TRAP
attack method:
- ActionTree: hierarchical action space structure
- FlatBanditAgent: flat leaf-level bandit for ablation C
- HierarchicalBanditAgent: gradient-bandit navigation over the tree
"""

import logging

logger = logging.getLogger(__name__)

# Import with error handling
try:
    from .action_tree import create_action_tree, ActionTree, ActionNode
except ImportError as e:
    logger.warning(f"Could not import action_tree: {e}")
    create_action_tree = None
    ActionTree = None
    ActionNode = None

try:
    from .flat_bandit_agent import create_flat_bandit_agent, FlatBanditAgent
except ImportError as e:
    logger.warning(f"Could not import flat_bandit_agent: {e}")
    create_flat_bandit_agent = None
    FlatBanditAgent = None

try:
    from .hierarchical_bandit_agent import (
        create_hierarchical_bandit_agent,
        HierarchicalBanditAgent,
        AgentConfig
    )
except ImportError as e:
    logger.warning(f"Could not import hierarchical_bandit_agent: {e}")
    create_hierarchical_bandit_agent = None
    HierarchicalBanditAgent = None
    AgentConfig = None

__all__ = [
    # Action tree
    'create_action_tree',
    'ActionTree',
    'ActionNode',
    
    # Flat Bandit Agent
    'create_flat_bandit_agent',
    'FlatBanditAgent',

    # Hierarchical Bandit Agent
    'create_hierarchical_bandit_agent',
    'HierarchicalBanditAgent',
    'AgentConfig'
]

