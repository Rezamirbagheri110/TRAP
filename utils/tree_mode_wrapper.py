"""
Wrapper for enabling tree exploration mode in run_attack_with_rag.py
Provides backward-compatible integration without breaking sequential mode.

Usage in run_attack():
    if os.getenv("OURS_ENABLE_TREE_EXPLORATION", "0") == "1":
        return self._run_attack_tree_mode(goal, target_str, behavior_idx, ...)
    else:
        # existing sequential loop
"""

import logging
import os
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def run_attack_tree_mode(
    runner_instance,  # RAGEnhancedAttackRunner instance
    goal: str,
    target_str: str,
    behavior_idx: int = 0
) -> Dict[str, Any]:
    """
    Run attack using tree exploration mode (branching + pruning).

    This is a simplified conservative implementation that wraps the existing
    sequential generation logic but explores multiple branches in parallel.

    Args:
        runner_instance: The RAGEnhancedAttackRunner instance (has all methods)
        goal: Attack goal
        target_str: Target response prefix
        behavior_idx: Behavior index

    Returns:
        Results dict compatible with sequential mode
    """
    from utils.tree_exploration import TreeExplorer, BranchState

    # Read tree config
    max_branches = int(os.getenv("OURS_TREE_MAX_BRANCHES", "4"))
    max_depth = int(os.getenv("OURS_TREE_MAX_DEPTH", "3"))

    logger.info(
        f"[TREE-MODE] Starting tree exploration: "
        f"max_branches={max_branches}, max_depth={max_depth}"
    )

    # Initialize tree explorer
    tree = TreeExplorer(
        max_branches=max_branches,
        max_depth=max_depth,
        diversity_templates=[
            'roleplay', 'technical', 'obfuscation', 'emotional',
            'authority', 'fictional', 'academic', 'professional'
        ]
    )

    # Results structure (compatible with sequential mode)
    results = {
        'goal': goal,
        'target_str': target_str,
        'run_tag': f'tree_b{behavior_idx}',
        'iterations': [],
        'success': False,
        'best_score': 0.0,
        'best_prompt': '',
        'best_response': '',
        'start_time': time.time(),
        'timed_out': False,
        'tree_mode': True,
        'tree_stats': {}
    }

    # Generate initial prompt (depth 0)
    try:
        initial_result = _generate_single_prompt(
            runner_instance,
            goal=goal,
            target_str=target_str,
            template_hint=None,
            rag_hint=True,
            iteration_count=1
        )

        if initial_result is None:
            logger.error("[TREE-MODE] Failed to generate initial prompt")
            return _finalize_results(runner_instance, results, tree, start_time=results['start_time'])

        # Initialize root
        root = tree.initialize_root(goal, target_str, initial_result)
        results['iterations'].append(initial_result)
        results['best_score'] = root.score
        results['best_prompt'] = root.prompt
        results['best_response'] = root.response
        results['success'] = root.success

        if root.success:
            logger.info(f"[TREE-MODE] Root succeeded immediately! score={root.score}")
            return _finalize_results(runner_instance, results, tree, start_time=results['start_time'])

    except Exception as e:
        logger.error(f"[TREE-MODE] Error in initial generation: {e}")
        return _finalize_results(runner_instance, results, tree, start_time=results['start_time'])

    # Tree exploration loop
    for depth in range(1, max_depth + 1):
        if not tree.should_continue(depth):
            break

        logger.info(f"[TREE-MODE] === Depth {depth} ===")

        # Get branch configurations
        branch_configs = tree.branch_and_prune(depth)

        if not branch_configs:
            logger.warning(f"[TREE-MODE] No branches to explore at depth {depth}")
            break

        # Generate prompts for all branches
        branch_results = []
        for config in branch_configs:
            try:
                result = _generate_single_prompt(
                    runner_instance,
                    goal=goal,
                    target_str=target_str,
                    template_hint=config['template_type'],
                    rag_hint=config['rag_enabled'],
                    iteration_count=len(results['iterations']) + 1,
                    parent_prompt=config['parent_branch'].prompt
                )

                if result:
                    result['branch_id'] = config['branch_id']
                    result['template_used'] = config['template_type']
                    result['rag_enabled'] = config['rag_enabled']
                    result['strategy_variant'] = config['strategy_variant']
                    branch_results.append(result)
                    results['iterations'].append(result)

            except Exception as e:
                logger.warning(f"[TREE-MODE] Branch {config['branch_id']} failed: {e}")

        if not branch_results:
            logger.warning(f"[TREE-MODE] All branches failed at depth {depth}")
            break

        # Update tree with results
        tree.update_branches(branch_results, branch_configs)

        # Update best result
        best = tree.get_best_result()
        if best and best['score'] > results['best_score']:
            results['best_score'] = best['score']
            results['best_prompt'] = best['prompt']
            results['best_response'] = best['response']
            results['success'] = best['success']

        # Early stop on success
        if results['success']:
            logger.info(f"[TREE-MODE] Success found at depth {depth}! Stopping.")
            break

    # Finalize
    return _finalize_results(runner_instance, results, tree, start_time=results['start_time'])


def _generate_single_prompt(
    runner_instance,
    goal: str,
    target_str: str,
    template_hint: Optional[str],
    rag_hint: bool,
    iteration_count: int,
    parent_prompt: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate a single prompt using the runner's existing logic.

    This is a simplified wrapper that calls the core generation pipeline
    without all the iteration loop overhead.
    """
    try:
        # Use existing prompt generation (simplified)
        # This calls RL agent, template selection, RAG, etc.
        prompt_info = runner_instance._generate_prompt_internal(
            goal=goal,
            iteration_count=iteration_count,
            template_hint=template_hint,
            rag_hint=rag_hint,
            target_str=target_str
        )

        if not prompt_info or not prompt_info.get('prompt'):
            return None

        prompt = prompt_info['prompt']

        # Query target model
        try:
            response_list = runner_instance.target_llm.batched_generate([prompt])
            response = response_list[0] if response_list else ""
        except Exception as e:
            logger.warning(f"[TREE-GEN] Target query failed: {e}")
            return None

        # Evaluate
        eval_result = runner_instance.evaluate_response(
            prompt, response, goal, target_str, iteration_index=iteration_count
        )

        return {
            'prompt': prompt,
            'response': response,
            'score': eval_result.get('judge_score', 0.0),
            'success': eval_result.get('success', False),
            'is_refusal': eval_result.get('is_refusal', False),
            'queries_used': 1,
            'template_used': prompt_info.get('template_name'),
            'rag_enabled': prompt_info.get('rag_used', False),
            'iteration': iteration_count
        }

    except Exception as e:
        logger.error(f"[TREE-GEN] Generation failed: {e}")
        return None


def _finalize_results(
    runner_instance,
    results: Dict[str, Any],
    tree: 'TreeExplorer',
    start_time: float
) -> Dict[str, Any]:
    """Finalize results with tree statistics"""
    results['end_time'] = time.time()
    results['duration_seconds'] = results['end_time'] - start_time
    results['iterations_count'] = len(results['iterations'])

    # Add tree stats
    tree_stats = tree.get_statistics()
    results['tree_stats'] = tree_stats

    logger.info(
        f"[TREE-MODE] Completed: success={results['success']}, "
        f"best_score={results['best_score']:.1f}, "
        f"branches={tree_stats['total_branches_explored']}, "
        f"depth={tree_stats['max_depth_reached']}"
    )

    return results
