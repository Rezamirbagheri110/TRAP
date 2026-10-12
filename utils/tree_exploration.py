"""
Tree-Based Exploration for Jailbreak Attack
Implements exponential branching with pruning and diversity enforcement.

Key features:
- Exponential branching: 1 -> 2 -> 4 -> 8 -> ...
- Diversity: Each branch uses different template/strategy
- Pruning: Keep only top-K branches based on score
- Early stopping: Stop when success found
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class BranchState:
    """Represents a single branch in the exploration tree"""
    depth: int
    parent_id: Optional[str]
    branch_id: str

    # Attack state
    goal: str
    target_str: str

    # Last iteration result
    prompt: Optional[str]
    response: Optional[str]
    score: float
    success: bool

    # Diversity tracking
    template_used: Optional[str]
    rag_enabled: bool
    strategy_variant: str  # 'aggressive', 'obfuscation', 'roleplay', etc.

    # Metadata
    iterations_count: int
    total_queries: int


class TreeExplorer:
    """
    Manages tree-based exploration of the attack space.

    Instead of sequential iterations, explores multiple paths in parallel:
    - Depth 1: 1 path
    - Depth 2: 2 paths (branch from best)
    - Depth 3: 4 paths (branch from top 2)
    - ...
    - Max: configurable cap
    """

    def __init__(
        self,
        max_branches: int = 8,
        max_depth: int = 5,
        diversity_templates: List[str] = None,
        diversity_strategies: List[str] = None
    ):
        """
        Initialize tree explorer

        Args:
            max_branches: Maximum concurrent branches (cap on exponential growth)
            max_depth: Maximum tree depth (iterations)
            diversity_templates: List of template types for diversity
            diversity_strategies: List of strategy variants
        """
        self.max_branches = max_branches
        self.max_depth = max_depth

        # Diversity pools
        self.diversity_templates = diversity_templates or [
            'roleplay', 'technical', 'obfuscation', 'emotional',
            'authority', 'fictional', 'academic', 'professional'
        ]

        self.diversity_strategies = diversity_strategies or [
            'aggressive', 'subtle', 'technical', 'emotional',
            'authority', 'creative', 'direct', 'indirect'
        ]

        # State
        self.active_branches: List[BranchState] = []
        self.all_branches: List[BranchState] = []
        self.best_branch: Optional[BranchState] = None

        logger.info(
            f"[TREE] Initialized TreeExplorer: max_branches={max_branches}, "
            f"max_depth={max_depth}, templates={len(self.diversity_templates)}"
        )

    def initialize_root(
        self,
        goal: str,
        target_str: str,
        initial_result: Dict[str, Any]
    ) -> BranchState:
        """
        Create root branch from initial result

        Args:
            goal: Attack goal
            target_str: Target response prefix
            initial_result: Result from first iteration

        Returns:
            Root branch state
        """
        root = BranchState(
            depth=0,
            parent_id=None,
            branch_id="root",
            goal=goal,
            target_str=target_str,
            prompt=initial_result.get('prompt'),
            response=initial_result.get('response'),
            score=initial_result.get('score', 0.0),
            success=initial_result.get('success', False),
            template_used=initial_result.get('template_used'),
            rag_enabled=initial_result.get('rag_enabled', False),
            strategy_variant=initial_result.get('strategy_variant', 'initial'),
            iterations_count=1,
            total_queries=1
        )

        self.active_branches = [root]
        self.all_branches = [root]
        self.best_branch = root

        logger.info(f"[TREE] Root initialized: score={root.score:.1f}, success={root.success}")

        return root

    def should_continue(self, current_depth: int) -> bool:
        """Check if exploration should continue"""
        if current_depth >= self.max_depth:
            logger.info(f"[TREE] Max depth {self.max_depth} reached")
            return False

        if any(b.success for b in self.active_branches):
            logger.info(f"[TREE] Success found, stopping exploration")
            return False

        if len(self.active_branches) == 0:
            logger.warning(f"[TREE] No active branches, stopping")
            return False

        return True

    def branch_and_prune(
        self,
        current_depth: int
    ) -> List[Dict[str, Any]]:
        """
        Create branch configurations for next depth level.

        Returns list of configurations for generating new prompts:
        [
            {
                'parent_branch': BranchState,
                'template_type': str,
                'rag_enabled': bool,
                'strategy_variant': str,
                'branch_id': str
            },
            ...
        ]
        """
        # Calculate target branch count (exponential: 2^depth, capped)
        target_branches = min(2 ** current_depth, self.max_branches)

        # Determine how many parents to branch from
        parents_needed = max(1, target_branches // 2)

        # Select top parents by score
        sorted_branches = sorted(
            self.active_branches,
            key=lambda b: b.score,
            reverse=True
        )
        parent_branches = sorted_branches[:parents_needed]

        logger.info(
            f"[TREE] Depth {current_depth}: branching from {len(parent_branches)} parents "
            f"to create {target_branches} branches"
        )

        # Create branch configurations
        configs = []
        branch_counter = 0

        for parent in parent_branches:
            # Each parent creates 2 children (or 1 if at capacity)
            children_per_parent = min(2, target_branches - len(configs))

            for child_idx in range(children_per_parent):
                # Create diverse child
                config = self._create_diverse_branch_config(
                    parent=parent,
                    child_idx=child_idx,
                    current_depth=current_depth,
                    branch_counter=branch_counter
                )
                configs.append(config)
                branch_counter += 1

        logger.info(f"[TREE] Created {len(configs)} branch configurations")

        return configs

    def _create_diverse_branch_config(
        self,
        parent: BranchState,
        child_idx: int,
        current_depth: int,
        branch_counter: int
    ) -> Dict[str, Any]:
        """
        Create configuration for a diverse branch.

        Ensures diversity from parent by varying:
        - Template type (if child_idx == 0)
        - RAG enabled/disabled (if child_idx == 1)
        - Strategy variant (random)
        """
        # Select different template
        available_templates = [t for t in self.diversity_templates if t != parent.template_used]
        if not available_templates:
            available_templates = self.diversity_templates

        # Child 0: different template, keep RAG state
        # Child 1: different RAG state, keep template (or choose new if same as child 0)
        if child_idx == 0:
            new_template = random.choice(available_templates)
            new_rag = parent.rag_enabled
        else:
            # Try to use different template than child 0 would use
            first_child_template = available_templates[0] if available_templates else parent.template_used
            other_templates = [t for t in available_templates if t != first_child_template]
            new_template = random.choice(other_templates) if other_templates else random.choice(available_templates)
            new_rag = not parent.rag_enabled  # Flip RAG state

        # Random strategy variant
        new_strategy = random.choice(self.diversity_strategies)

        branch_id = f"d{current_depth}_b{branch_counter}"

        config = {
            'parent_branch': parent,
            'template_type': new_template,
            'rag_enabled': new_rag,
            'strategy_variant': new_strategy,
            'branch_id': branch_id,
            'depth': current_depth
        }

        logger.debug(
            f"[TREE] Branch {branch_id}: template={new_template}, "
            f"rag={new_rag}, strategy={new_strategy} (parent: {parent.branch_id})"
        )

        return config

    def update_branches(
        self,
        new_results: List[Dict[str, Any]],
        configs: List[Dict[str, Any]]
    ):
        """
        Update tree with new iteration results.

        Args:
            new_results: List of iteration results (one per branch config)
            configs: List of branch configurations that were executed
        """
        new_branches = []

        for result, config in zip(new_results, configs):
            branch = BranchState(
                depth=config['depth'],
                parent_id=config['parent_branch'].branch_id,
                branch_id=config['branch_id'],
                goal=config['parent_branch'].goal,
                target_str=config['parent_branch'].target_str,
                prompt=result.get('prompt'),
                response=result.get('response'),
                score=result.get('score', 0.0),
                success=result.get('success', False),
                template_used=config['template_type'],
                rag_enabled=config['rag_enabled'],
                strategy_variant=config['strategy_variant'],
                iterations_count=config['parent_branch'].iterations_count + 1,
                total_queries=config['parent_branch'].total_queries + result.get('queries_used', 1)
            )

            new_branches.append(branch)
            self.all_branches.append(branch)

            # Update best
            if self.best_branch is None or branch.score > self.best_branch.score:
                self.best_branch = branch
                logger.info(
                    f"[TREE] New best branch: {branch.branch_id} "
                    f"(score={branch.score:.1f}, success={branch.success})"
                )

        # Prune: keep only top branches for next iteration
        keep_count = min(self.max_branches, len(new_branches))
        sorted_branches = sorted(new_branches, key=lambda b: b.score, reverse=True)
        self.active_branches = sorted_branches[:keep_count]

        pruned = len(new_branches) - len(self.active_branches)
        if pruned > 0:
            logger.info(
                f"[TREE] Pruned {pruned} low-scoring branches, "
                f"keeping top {len(self.active_branches)}"
            )

        # Log active branches
        for i, branch in enumerate(self.active_branches):
            logger.info(
                f"[TREE] Active #{i+1}: {branch.branch_id} "
                f"(score={branch.score:.1f}, template={branch.template_used})"
            )

    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """Get result from best branch"""
        if self.best_branch is None:
            return None

        return {
            'prompt': self.best_branch.prompt,
            'response': self.best_branch.response,
            'score': self.best_branch.score,
            'success': self.best_branch.success,
            'branch_id': self.best_branch.branch_id,
            'depth': self.best_branch.depth,
            'template_used': self.best_branch.template_used,
            'total_iterations': self.best_branch.iterations_count,
            'total_queries': self.best_branch.total_queries
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get exploration statistics"""
        successful_branches = [b for b in self.all_branches if b.success]

        return {
            'total_branches_explored': len(self.all_branches),
            'max_depth_reached': max((b.depth for b in self.all_branches), default=0),
            'active_branches': len(self.active_branches),
            'successful_branches': len(successful_branches),
            'best_score': self.best_branch.score if self.best_branch else 0.0,
            'total_queries_all_branches': sum(b.total_queries for b in self.all_branches),
            'avg_score': sum(b.score for b in self.all_branches) / max(1, len(self.all_branches))
        }
