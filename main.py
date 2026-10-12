"""
Unified Main Execution Script for "TRAP" Attack Method

This script provides a simple, unified entry point to test and run the complete
RL + RAG + Template-based jailbreak attack system.

Features:
- RAG mode: Single-objective execution with retrieval and learning
- Ablation mode: Collection-level execution of manuscript configurations

Architecture:
1. HierarchicalBanditAgent: gradient-bandit strategy selection
2. PromptTemplateSystem: Pattern-based prompt generation
3. RAGKnowledgeBase: Learn from successful attacks

Usage:
    python main.py --mode rag --experiment A_ONLINE --goal "Describe a benign software testing task"
    
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime

# Keep third-party experiment trackers inert unless explicitly enabled.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

# Allow log level override for finer-grained debugging across TRAP components
LOG_LEVEL = getattr(
    logging,
    os.getenv("OURS_LOG_LEVEL", "INFO").upper(),
    logging.INFO
)

# Add necessary paths
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from utils.runtime_config import (
        get_model_default,
        get_ours_learning_threshold,
    )
except ImportError:
    from utils.runtime_config import (  # type: ignore
        get_model_default,
        get_ours_learning_threshold,
    )

# Import dotenv to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(root_dir / 'config.env')
except ImportError:
    pass

# Configure logging using centralized configuration
try:
    from utils.log_config import get_log_file, LOGS_DIR
    log_dir = LOGS_DIR
    log_file = get_log_file('main')
except ImportError:
    # Fallback for backward compatibility
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'main_execution.log'

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure UTF-8 encoding for stdout/stderr on Windows to avoid charmap errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# -----------------------------------------------------------------------------
# Component availability checks
# -----------------------------------------------------------------------------
AVAILABLE_COMPONENTS = {
    'torch': False,
    'sentence_transformers': False,
    'prompt_templates': False,
    'rag_system': False,
    'hierarchical_bandit': False
}


def _is_placeholder_secret(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in {
        "",
        "your_api_key_here",
        "your-openrouter-api-key",
        "replace_me",
        "changeme",
        "test",
        "dummy",
        "none",
        "null",
    }


def _check_huggingface_cache_access() -> Optional[str]:
    """
    Detect a common runtime failure mode: sentence-transformers is importable,
    but its cache directory is not writable in the current environment.
    """
    cache_root = Path(
        os.getenv("HF_HOME")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or (Path.home() / ".cache" / "huggingface")
    )
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        probe = cache_root / ".ours_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return None
    except Exception as exc:
        return f"Hugging Face cache is not writable at '{cache_root}': {exc}"


def get_environment_diagnostics() -> Dict[str, Any]:
    """
    Return warnings/errors for conditions that make the runtime look ready while
    still failing during real model or embedding calls.
    """
    diagnostics: Dict[str, Any] = {"errors": [], "warnings": []}

    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        diagnostics["errors"].append("OPENROUTER_API_KEY not found in environment.")
    elif _is_placeholder_secret(api_key):
        diagnostics["errors"].append("OPENROUTER_API_KEY is set to a placeholder value.")
    elif not api_key.startswith("sk-or-"):
        diagnostics["warnings"].append(
            "OPENROUTER_API_KEY does not use the expected OpenRouter prefix; auth may fail at runtime."
        )

    if AVAILABLE_COMPONENTS['sentence_transformers']:
        cache_warning = _check_huggingface_cache_access()
        if cache_warning:
            diagnostics["warnings"].append(cache_warning)

    try:
        from utils.embeddings import SemanticEmbedder

        model_name = os.getenv("OURS_EMBEDDING_MODEL") or SemanticEmbedder.DEFAULT_MODEL
        model_cfg = SemanticEmbedder.SUPPORTED_MODELS.get(model_name)
        if model_cfg and model_cfg.get("type") == "openrouter":
            diagnostics["warnings"].append(
                f"Embedding model '{model_name}' uses OpenRouter; embedding calls will fail if API auth is invalid."
            )
    except Exception:
        pass

    return diagnostics

def _check_component_availability():
    """Check which components are available"""
    global AVAILABLE_COMPONENTS
    
    try:
        import torch
        AVAILABLE_COMPONENTS['torch'] = True
    except ImportError:
        pass
    
    try:
        from sentence_transformers import SentenceTransformer
        AVAILABLE_COMPONENTS['sentence_transformers'] = True
    except ImportError:
        pass
    
    try:
        from prompts import create_prompt_system
        AVAILABLE_COMPONENTS['prompt_templates'] = True
    except ImportError:
        pass
    
    try:
        from rag import RAGKnowledgeBase
        AVAILABLE_COMPONENTS['rag_system'] = True
    except ImportError:
        pass
    
    try:
        from core.hierarchical_bandit_agent import HierarchicalBanditAgent
        AVAILABLE_COMPONENTS['hierarchical_bandit'] = True
    except ImportError:
        pass
    
# Run availability check at import
_check_component_availability()


def check_environment() -> bool:
    """
    Check if environment is properly configured for running attacks.
    
    Returns:
        True if environment is ready, False otherwise
    """
    logger.info("Checking environment configuration...")
    
    # Check for OpenRouter API key
    diagnostics = get_environment_diagnostics()
    if diagnostics["errors"]:
        for error in diagnostics["errors"]:
            logger.error(error)
        logger.error("Please fix the environment errors above before running attacks.")
        return False
    
    logger.info("[OK] OpenRouter API key found")
    
    # Check for required dependencies
    if not AVAILABLE_COMPONENTS['torch']:
        logger.error("[X] PyTorch not installed - run: pip install torch")
        return False
    
    import torch
    logger.info(f"[OK] PyTorch {torch.__version__} installed")
    
    # Check optional components
    if AVAILABLE_COMPONENTS['sentence_transformers']:
        logger.info("[OK] sentence-transformers available")
    else:
        logger.warning("[WARNING] sentence-transformers not installed - will use fallback embeddings")
        logger.warning("  For better performance, run: pip install sentence-transformers")
    
    if AVAILABLE_COMPONENTS['prompt_templates']:
        logger.info("[OK] Prompt template system available")
    else:
        logger.warning("[WARNING] Prompt template system not available")
    
    if AVAILABLE_COMPONENTS['rag_system']:
        logger.info("[OK] RAG knowledge base available")
    else:
        logger.warning("[WARNING] RAG system not available")
    
    if AVAILABLE_COMPONENTS['hierarchical_bandit']:
        logger.info("[OK] Hierarchical Bandit agent available")
    else:
        logger.warning("[WARNING] Hierarchical Bandit agent not available")

    for warning in diagnostics["warnings"]:
        logger.warning(f"[WARNING] {warning}")
    
    return True


def get_component_status() -> Dict[str, bool]:
    """
    Get the availability status of all components.
    
    Returns:
        Dictionary mapping component names to availability status
    """
    return AVAILABLE_COMPONENTS.copy()


def run_rag_mode(args) -> Dict[str, Any]:
    """
    Run RAG-enhanced RL attack with full template integration.
    
    This mode provides the most comprehensive attack by combining:
    - Hierarchical Bandit for intelligent action selection
    - Prompt template system for high-quality prompt generation
    - RAG knowledge base for learning from successful attacks
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary containing attack results with detailed metrics
    """
    logger.info("\n" + "="*80)
    logger.info("RUNNING RAG-ENHANCED RL MODE (with templates)")
    logger.info("="*80)
    
    from scripts.run_attack_with_rag import RAGEnhancedAttackRunner
    from utils.experiment_config import (
        load_experiment_registry,
        resolve_kb_parameters,
        resolve_seeds,
        resolve_snapshot_path,
    )

    registry = load_experiment_registry()
    shared_cfg = registry.get("shared", {})
    experiments = registry.get("experiments", {})
    exp_id = getattr(args, "experiment", None) or "A"
    exp_cfg = experiments.get(exp_id, {})
    seed = resolve_seeds(registry, getattr(args, "random_seed", None))[0]
    rag_enabled = exp_cfg.get("rag_enabled", True)
    rag_mode = str(exp_cfg.get("rag_mode", "online" if rag_enabled else "disabled")).lower()
    run_kb_params = resolve_kb_parameters(registry, exp_cfg)

    rag_kb_path = args.rag_kb_path
    rag_storage_dir = None
    state_dir = None
    snapshot = {"source": "", "path": "", "sha256": "", "exists": False}

    if getattr(args, "study_id", None) and exp_id:
        from utils.ablation_manifest import get_run_dirs, prepare_kb_snapshot, write_manifest

        paths = get_run_dirs(args.study_id, exp_id, seed)
        snapshot_source = resolve_snapshot_path(registry, args.rag_kb_path)
        snapshot = prepare_kb_snapshot(paths, snapshot_source if rag_enabled else None)
        rag_kb_path = snapshot["path"] if snapshot["exists"] else None
        rag_storage_dir = str(paths.kb)
        state_dir = str(paths.state)

        write_manifest(
            paths,
            study_id=args.study_id,
            experiment_id=exp_id,
            experiment_name=exp_cfg.get("name", exp_id),
            seed=seed,
            target_model=args.target_model,
            evaluator_model=args.evaluator_model,
            rag_enabled=rag_enabled,
            tree_variant=exp_cfg.get("tree_variant", "full_active"),
            tree_leaves=exp_cfg.get("tree_leaves"),
            tree_max_depth=exp_cfg.get("tree_max_depth"),
            success_threshold=shared_cfg.get("success_threshold", 7.0),
            max_iterations_per_episode=args.max_iterations,
            episode_budget=1,
            extra={
                "rag_mode": rag_mode,
                "kb_snapshot_sha256": snapshot["sha256"],
                "kb_snapshot_source": snapshot["source"],
                "kb_parameters": run_kb_params,
            },
        )

    # Create runner with template integration
    runner = RAGEnhancedAttackRunner(
        target_model=args.target_model,
        evaluator_model=args.evaluator_model,
        rag_kb_path=rag_kb_path,
        rag_storage_dir=rag_storage_dir,
        state_dir=state_dir,
        rag_mode=rag_mode,
        rag_parameters=run_kb_params,
        use_rag_prob=float(run_kb_params.get("use_rag_prob", args.use_rag_prob)) if rag_enabled else 0.0,
        learning_threshold=float(run_kb_params.get("learning_threshold", args.learning_threshold)),
        random_seed=seed,
        use_advanced_templates=getattr(args, 'use_templates', True),
        experiment_name=exp_id,
    )
    
    # Load examples if provided
    if args.load_examples and getattr(args, "study_id", None):
        logger.warning(
            "Ignoring --load-examples for controlled study run; use the configured KB snapshot instead"
        )
    elif args.load_examples and Path(args.load_examples).exists():
        try:
            count = runner.rag_kb.load_from_json(args.load_examples)
            logger.info(f"Loaded {count} examples from {args.load_examples}")
        except Exception as e:
            logger.warning(f"Failed to load examples: {e}")
    
    # Run attack
    results = runner.run_attack(
        goal=args.goal,
        target_str=args.target_str,
        max_iterations=args.max_iterations,
        save_interval=5
    )
    
    # Add mode info and component status
    results['mode'] = 'rag'
    results['study_id'] = getattr(args, "study_id", None)
    results['experiment_id'] = exp_id
    results['seed'] = seed
    results['rag_mode'] = rag_mode
    results['kb_parameters'] = run_kb_params
    results['kb_snapshot_sha256'] = snapshot["sha256"]
    results['kb_snapshot_source'] = snapshot["source"]
    results['components_used'] = {
        'hierarchical_bandit': AVAILABLE_COMPONENTS['hierarchical_bandit'],
        'prompt_templates': AVAILABLE_COMPONENTS['prompt_templates'],
        'rag_system': AVAILABLE_COMPONENTS['rag_system']
    }
    
    # Add template statistics if available
    if hasattr(runner, 'stats') and runner.stats.get('template_usage'):
        results['template_statistics'] = {
            'usage': runner.stats['template_usage'],
            'effectiveness': runner.stats.get('pattern_effectiveness', {})
        }
    
    return results


# Bump when the retrieval-as-policy (ablation D, tree_variant=none) selection
# logic changes, so previously cached D results are invalidated and re-run.
# v2: replay the retrieved successful prompt verbatim instead of degrading to the
# fixed research_framing fallback when KB examples carry no leaf metadata.
_RAG_ONLY_POLICY_VERSION = "v2_prompt_replay_20260616"


def run_ablation_mode(args) -> Dict[str, Any]:
    """
    Run the full ablation study: iterate over every experiment defined in
    experiment_registry.yaml, execute each one via RAGEnhancedAttackRunner,
    write manifests to results/ablations/<study_id>/, and export comparison
    tables when all experiments finish.

    Args:
        args: Parsed CLI arguments. Must have --study-id and --goals-file (or
              --goal).

    Returns:
        Dict summarising per-experiment ASR and paths to aggregate outputs.
    """
    import time
    from scripts.run_attack_with_rag import RAGEnhancedAttackRunner
    from utils.ablation_manifest import (
        AblationPaths, write_manifest, get_run_dirs, prepare_kb_snapshot,
    )
    from utils.experiment_config import (
        load_experiment_registry,
        resolve_kb_parameters,
        resolve_seeds,
        resolve_snapshot_path,
    )
    from scripts.ablation_exporter import export_study

    logger.info("\n" + "="*80)
    logger.info("RUNNING ABLATION STUDY MODE")
    logger.info("="*80)

    registry_path = Path(__file__).parent / "config" / "experiment_registry.yaml"
    if not registry_path.exists():
        raise FileNotFoundError(f"experiment_registry.yaml not found: {registry_path}")

    registry = load_experiment_registry(registry_path)

    shared_cfg = registry.get("shared", {})
    experiments: Dict[str, Any] = registry.get("experiments", {})

    study_id = args.study_id
    seeds = resolve_seeds(registry, getattr(args, "random_seed", None))
    max_iterations = args.max_iterations or shared_cfg.get("max_iterations_per_episode", 10)

    # Determine goals list
    goals = _load_ablation_goals(args)
    logger.info(f"Study: {study_id}  |  Seeds: {seeds}  |  Behaviors: {len(goals)}  |  Iters/behavior: {max_iterations}")

    # Filter to specific experiments if --experiments flag given
    exp_filter = set(getattr(args, "experiments", None) or experiments.keys())
    selected = {k: v for k, v in experiments.items() if k in exp_filter}
    logger.info(f"Experiments to run: {sorted(selected.keys())}")

    study_summary: Dict[str, Any] = {"study_id": study_id, "experiments": {}}

    snapshot_source = resolve_snapshot_path(registry, args.rag_kb_path)

    for seed in seeds:
      for exp_id, exp_cfg in sorted(selected.items()):
        exp_name = exp_cfg.get("name", exp_id)
        rag_enabled = exp_cfg.get("rag_enabled", True)
        tree_variant = exp_cfg.get("tree_variant", "full_active")

        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT {exp_id}: {exp_name}")
        logger.info(f"  tree_variant={tree_variant}  rag_enabled={rag_enabled}")
        logger.info(f"{'='*60}")

        paths = get_run_dirs(study_id, exp_id, seed)
        rag_mode = str(exp_cfg.get("rag_mode", "online" if rag_enabled else "disabled")).lower()
        run_kb_params = resolve_kb_parameters(registry, exp_cfg)
        # Retrieval-as-policy (ablation D / tree_variant=none) selects the prompt
        # itself from the KB. When that selection logic changes, prior D caches are
        # no longer comparable, so stamp the policy version into the cache key to
        # auto-invalidate stale D results. Other experiments are unaffected.
        if str(tree_variant).lower() == "none":
            run_kb_params = {**run_kb_params, "rag_only_policy_version": _RAG_ONLY_POLICY_VERSION}
        snapshot = prepare_kb_snapshot(paths, snapshot_source if rag_enabled else None)
        cached_result = _load_cached_ablation_result(
            paths,
            study_id=study_id,
            experiment_id=exp_id,
            experiment_name=exp_name,
            seed=seed,
            target_model=args.target_model,
            evaluator_model=args.evaluator_model,
            rag_enabled=rag_enabled,
            tree_variant=tree_variant,
            tree_leaves=exp_cfg.get("tree_leaves"),
            tree_max_depth=exp_cfg.get("tree_max_depth"),
            success_threshold=shared_cfg.get("success_threshold", 7.0),
            max_iterations_per_episode=max_iterations,
            episode_budget=len(goals),
            rag_mode=rag_mode,
            kb_snapshot_sha256=snapshot["sha256"],
            kb_parameters=run_kb_params,
        )

        if cached_result is not None:
            all_behaviors = cached_result.get("behaviors", [])
            if not isinstance(all_behaviors, list):
                all_behaviors = []
            cached_metrics = cached_result.get("metrics", {}) if isinstance(cached_result.get("metrics"), dict) else {}
            elapsed = float(cached_result.get("elapsed_seconds", 0.0) or 0.0)
            n_ok = int(cached_metrics.get("success_count", sum(1 for b in all_behaviors if b.get("success"))))
            asr = float(cached_metrics.get("asr_percent", (100.0 * n_ok / max(len(all_behaviors), 1))))
            avg_score = float(
                cached_metrics.get(
                    "avg_best_score",
                    sum(float(b.get("best_score", 0) or 0) for b in all_behaviors) / max(len(all_behaviors), 1),
                )
            )
            avg_iters = float(
                cached_metrics.get(
                    "avg_iterations",
                    sum(float(b.get("iterations_count", 0) or 0) for b in all_behaviors) / max(len(all_behaviors), 1),
                )
            )
            _d_stats = cached_result.get("rag_only_selection_stats")
            _d_shares = cached_result.get("rag_only_selection_shares")
            # result.json still exists on disk from the previous run; expose its path
            # so _integrate_ablation_outputs can read behaviors and build the combined result.
            result_path = paths.result_json_path()
        else:

            # Write in-progress manifest
            write_manifest(
                paths,
                study_id=study_id,
                experiment_id=exp_id,
                experiment_name=exp_name,
                seed=seed,
                target_model=args.target_model,
                evaluator_model=args.evaluator_model,
                rag_enabled=rag_enabled,
                tree_variant=tree_variant,
                tree_leaves=exp_cfg.get("tree_leaves"),
                tree_max_depth=exp_cfg.get("tree_max_depth"),
                success_threshold=shared_cfg.get("success_threshold", 7.0),
                max_iterations_per_episode=max_iterations,
                episode_budget=len(goals),
                extra={
                    "rag_mode": rag_mode,
                    "kb_snapshot_sha256": snapshot["sha256"],
                    "kb_snapshot_source": snapshot["source"],
                    "kb_parameters": run_kb_params,
                },
            )

            runner = RAGEnhancedAttackRunner(
                target_model=args.target_model,
                evaluator_model=args.evaluator_model,
                rag_kb_path=snapshot["path"] if snapshot["exists"] else None,
                rag_storage_dir=str(paths.kb),
                state_dir=str(paths.state),
                rag_mode=rag_mode,
                rag_parameters=run_kb_params,
                use_rag_prob=float(run_kb_params.get("use_rag_prob", args.use_rag_prob)) if rag_enabled else 0.0,
                learning_threshold=float(run_kb_params.get("learning_threshold", args.learning_threshold)),
                random_seed=seed,
                use_advanced_templates=getattr(args, "use_templates", True),
                experiment_name=exp_id,
            )

            all_behaviors = []
            run_start = time.time()

            for idx, item in enumerate(goals):
                goal = item["goal"]
                target = item.get("target", "Sure, here is")
                logger.info(f"  [B{idx+1}/{len(goals)}] {goal[:70]}")
                try:
                    result = runner.run_attack(
                        goal=goal,
                        target_str=target,
                        max_iterations=max_iterations,
                        save_interval=5,
                        behavior_idx=idx,
                    )
                    all_behaviors.append({
                        "behavior_idx": idx,
                        "goal": goal,
                        "target": target,
                        "success": result.get("success", False),
                        "max_score": result.get("best_score", 0),
                        "best_score": result.get("best_score", 0),
                        "iterations": result.get("iterations", []),
                        "iterations_count": len(result.get("iterations", [])),
                    })
                except KeyboardInterrupt:
                    logger.warning(f"  Interrupted during experiment {exp_id} — saving partial results")
                    break
                except Exception as exc:
                    logger.error(f"  [B{idx}] failed: {exc}", exc_info=True)
                    all_behaviors.append({
                        "behavior_idx": idx, "goal": goal, "target": target,
                        "success": False, "max_score": 0, "best_score": 0, "iterations": [],
                        "iterations_count": 0, "error": str(exc),
                    })

            elapsed = time.time() - run_start
            n_ok = sum(1 for b in all_behaviors if b.get("success"))
            asr = 100.0 * n_ok / max(len(all_behaviors), 1)
            avg_score = sum(b.get("best_score", 0) for b in all_behaviors) / max(len(all_behaviors), 1)
            avg_iters = sum(b.get("iterations_count", 0) for b in all_behaviors) / max(len(all_behaviors), 1)

            # Collect Experiment D fallback stats from runner (only present for rag_only mode)
            _d_stats = getattr(runner, "rag_only_selection_stats", None)
            _d_shares = None
            if _d_stats is not None:
                _total_d = sum(_d_stats.values()) or 1
                _d_shares = {
                    "retrieval_share": round(_d_stats.get("retrieval", 0) / _total_d, 4),
                    "fallback_named_share": round(_d_stats.get("fallback_named", 0) / _total_d, 4),
                    "fallback_random_share": round(_d_stats.get("fallback_random", 0) / _total_d, 4),
                }

        # Save raw result.json
        raw_result = {
            "experiment_id": exp_id,
            "experiment_name": exp_name,
            "study_id": study_id,
            "seed": seed,
            "rag_mode": rag_mode,
            "kb_snapshot_sha256": snapshot["sha256"],
            "kb_parameters": run_kb_params,
            "behaviors": all_behaviors,
            "metrics": {
                "results": all_behaviors,
                "asr_percent": round(asr, 4),
                "avg_best_score": round(avg_score, 4),
                "avg_iterations": round(avg_iters, 4),
                "n_behaviors": len(all_behaviors),
                "success_count": n_ok,
            },
            "elapsed_seconds": round(elapsed, 1),
        }
        if _d_stats is not None:
            raw_result["rag_only_selection_stats"] = _d_stats
        if _d_shares is not None:
            raw_result["rag_only_selection_shares"] = _d_shares
        result_path = paths.result_json_path()
        result_path.write_text(
            json.dumps(raw_result, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

        # Write behaviors.jsonl
        with paths.behavior_jsonl_path().open("w", encoding="utf-8") as jf:
            for beh in all_behaviors:
                jf.write(json.dumps(beh, default=str, ensure_ascii=False) + "\n")

        # Update manifest with final metrics
        write_manifest(
            paths,
            study_id=study_id,
            experiment_id=exp_id,
            experiment_name=exp_name,
            seed=seed,
            target_model=args.target_model,
            evaluator_model=args.evaluator_model,
            rag_enabled=rag_enabled,
            tree_variant=tree_variant,
            tree_leaves=exp_cfg.get("tree_leaves"),
            tree_max_depth=exp_cfg.get("tree_max_depth"),
            success_threshold=shared_cfg.get("success_threshold", 7.0),
            max_iterations_per_episode=max_iterations,
            episode_budget=len(goals),
            total_behaviors=len(all_behaviors),
            success_count=n_ok,
            asr_percent=round(asr, 4),
            avg_best_score=round(avg_score, 4),
            avg_iterations_per_behavior=round(avg_iters, 4),
            rag_only_selection_stats=_d_stats,
            rag_only_selection_shares=_d_shares,
            extra={
                "rag_mode": rag_mode,
                "kb_snapshot_sha256": snapshot["sha256"],
                "kb_snapshot_source": snapshot["source"],
                "kb_parameters": run_kb_params,
            },
        )

        logger.info(f"  [DONE] {exp_id}: ASR={asr:.1f}%  avg_score={avg_score:.2f}  elapsed={elapsed/60:.1f}min")
        logger.info(f"  Saved -> {paths.raw}")

        exp_success = len(all_behaviors) > 0
        study_summary["experiments"][f"{exp_id}:{seed}"] = {
            "asr_percent": round(asr, 4),
            "avg_best_score": round(avg_score, 4),
            "n_behaviors": len(all_behaviors),
            "result_path": str(result_path),
            "success": exp_success,
        }

    # Auto-export comparison tables
    logger.info(f"\n[EXPORT] Generating comparison tables for study '{study_id}'...")
    export_ok = False
    try:
        export_study(study_id, registry_path)
        from utils.ablation_manifest import _RESULTS_ROOT
        study_summary["aggregate_dir"] = str(_RESULTS_ROOT / study_id / "aggregate")
        logger.info(f"[EXPORT] Done -> {study_summary['aggregate_dir']}")
        export_ok = True
    except Exception as export_err:
        logger.warning(f"[EXPORT] Failed: {export_err}", exc_info=True)

    per_experiment_ok = all(
        exp_info.get("success", False)
        for exp_info in study_summary["experiments"].values()
    ) if study_summary["experiments"] else False
    study_summary["success"] = per_experiment_ok and export_ok
    study_summary["mode"] = "ablation"
    return study_summary


def _ablation_cache_matches(
    manifest: Dict[str, Any],
    *,
    study_id: str,
    experiment_id: str,
    experiment_name: str,
    seed: int,
    target_model: str,
    evaluator_model: str,
    rag_enabled: bool,
    tree_variant: str,
    tree_leaves: Optional[int],
    tree_max_depth: Optional[int],
    success_threshold: float,
    max_iterations_per_episode: int,
    episode_budget: int,
    rag_mode: str,
    kb_snapshot_sha256: str,
    kb_parameters: Dict[str, Any],
) -> bool:
    expected = {
        "study_id": study_id,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "seed": seed,
        "target_model": target_model,
        "evaluator_model": evaluator_model,
        "rag_enabled": rag_enabled,
        "tree_variant": tree_variant,
        "tree_leaves": tree_leaves,
        "tree_max_depth": tree_max_depth,
        "success_threshold": success_threshold,
        "max_iterations_per_episode": max_iterations_per_episode,
        "episode_budget": episode_budget,
        "rag_mode": rag_mode,
        "kb_snapshot_sha256": kb_snapshot_sha256,
        "kb_parameters": kb_parameters,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False
    return True


def _load_cached_ablation_result(
    paths,
    *,
    study_id: str,
    experiment_id: str,
    experiment_name: str,
    seed: int,
    target_model: str,
    evaluator_model: str,
    rag_enabled: bool,
    tree_variant: str,
    tree_leaves: Optional[int],
    tree_max_depth: Optional[int],
    success_threshold: float,
    max_iterations_per_episode: int,
    episode_budget: int,
    rag_mode: str,
    kb_snapshot_sha256: str,
    kb_parameters: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    manifest_path = paths.manifest_path()
    result_path = paths.result_json_path()
    if not manifest_path.exists() or not result_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached_result = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"[CACHE] Could not read cached ablation files for {experiment_id}:{seed}: {exc}")
        return None

    if not _ablation_cache_matches(
        manifest,
        study_id=study_id,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        seed=seed,
        target_model=target_model,
        evaluator_model=evaluator_model,
        rag_enabled=rag_enabled,
        tree_variant=tree_variant,
        tree_leaves=tree_leaves,
        tree_max_depth=tree_max_depth,
        success_threshold=success_threshold,
        max_iterations_per_episode=max_iterations_per_episode,
        episode_budget=episode_budget,
        rag_mode=rag_mode,
        kb_snapshot_sha256=kb_snapshot_sha256,
        kb_parameters=kb_parameters,
    ):
        return None

    logger.info(f"[CACHE] Reusing cached ablation result for {experiment_id}:{seed} from {result_path}")
    return cached_result


def _load_ablation_goals(args) -> List[Dict[str, str]]:
    """Load explicit objectives; reject invalid input instead of substituting a sample."""
    goals_file = getattr(args, "goals_file", None)
    if goals_file:
        raw = json.loads(Path(goals_file).read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError("--goals-file must contain a non-empty JSON list")
        out = []
        for index, item in enumerate(raw, 1):
            if isinstance(item, dict):
                goal = item.get("goal") or item.get("behavior") or item.get("prompt")
                target = item.get("target") or item.get("target_str") or "Sure, here is"
            elif isinstance(item, str):
                goal, target = item, "Sure, here is"
            else:
                raise ValueError(f"Objective {index} must be a string or object")
            if not isinstance(goal, str) or not goal.strip() or not isinstance(target, str):
                raise ValueError(f"Objective {index} needs a non-empty goal and a string target")
            out.append({"goal": goal, "target": target})
        return out
    single_goal = getattr(args, "goal", None)
    if single_goal:
        return [{"goal": single_goal, "target": getattr(args, "target_str", "Sure, here is")}]
    raise ValueError("Ablation mode requires --goals-file or --goal")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='TRAP execution entry point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RAG-enhanced attack (single behavior, single experiment)
  python main.py --mode rag --goal "Describe a benign software testing task" \\
      --experiment A_ONLINE --load-examples /path/to/private/examples.json

  # Full ablation study (all 6 manuscript configurations, 10 behaviors from file)
  python main.py --mode ablation --study-id study_v1 \\
      --goals-file /path/to/objectives.json --max-iterations 10

  # Ablation study — only experiments A and E (RAG effect)
  python main.py --mode ablation --study-id rag_effect \\
      --experiments A E --goals-file /path/to/objectives.json
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='rag',
                       choices=['rag', 'ablation'],
                       help='Execution mode: rag (RL+RAG), '
                            'ablation (selected registry configurations)')
    
    # Attack parameters
    parser.add_argument('--goal', type=str, 
                       help='Attack goal/behavior (required for rag mode)')
    parser.add_argument('--target-str', type=str, default='Sure, here is',
                       help='Target response prefix')
    
    # Model parameters
    parser.add_argument('--target-model', type=str, default=get_model_default("TARGET_MODEL"),
                       help='Target model to attack')
    parser.add_argument('--evaluator-model', type=str, default=get_model_default("UNIFIED_JUDGE_MODEL"),
                       help='Evaluator model')
    parser.add_argument('--target-max-n-tokens', type=int, default=150,
                       help='Max tokens for target model')
    parser.add_argument('--evaluator-max-n-tokens', type=int, default=10,
                       help='Max tokens for evaluator')
    parser.add_argument('--evaluator-temperature', type=float, default=0,
                       help='Temperature for evaluator')
    
    # Attack parameters
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='Maximum attack iterations')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic action selection')
    
    # RAG parameters (only for rag mode)
    parser.add_argument('--rag-kb-path', type=str,
                       help='Path to existing RAG knowledge base')
    parser.add_argument('--use-rag-prob', type=float, default=0.85,
                       help='Probability of using RAG (0-1)')
    parser.add_argument('--learning-threshold', type=float, default=get_ours_learning_threshold(),
                       help='Minimum score to learn from before adding examples to knowledge (from config.env)')
    parser.add_argument('--experiment', type=str, default='A',
                       choices=['A', 'A_ONLINE', 'B', 'C', 'D', 'E'],
                       help='Ablation experiment from config/experiment_registry.yaml (rag mode)')
    parser.add_argument('--load-examples', type=str,
                       help='Path to JSON file with successful examples')

    # Ablation study parameters (only for ablation mode)
    parser.add_argument('--study-id', type=str, default=None,
                       help='Study ID for ablation mode (e.g. "study_v1"). '
                            'Defaults to a timestamp-based ID.')
    parser.add_argument('--experiments', type=str, nargs='+',
                       choices=['A', 'A_ONLINE', 'B', 'C', 'D', 'E'],
                       help='Subset of experiments to run in ablation mode (default: all registry experiments)')
    parser.add_argument('--goals-file', type=str,
                       help='JSON file with [{goal, target}] list for ablation mode')
    parser.add_argument('--random-seed', type=int, default=None,
                       help='Random seed for ablation runs (overrides experiment_registry.yaml)')
    parser.add_argument('--use-templates', action='store_true', default=True,
                       help='Use advanced prompt template system (default: True)')
    parser.add_argument('--no-templates', action='store_false', dest='use_templates',
                       help='Disable prompt template system')
    
    # Logging parameters
    parser.add_argument('--index', type=int, default=0,
                       help='Index for logging')
    parser.add_argument('--category', type=str, default='general',
                       help='Category for logging')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--skip-output-polish', action='store_true',
                       help='Skip automatic generation of improved text/image/excel outputs')
    
    args = parser.parse_args()
    
    # Adjust logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.mode == 'ablation':
        try:
            _load_ablation_goals(args)
        except (OSError, ValueError) as exc:
            parser.error(f"Invalid objectives: {exc}")

    # Check environment
    if not check_environment():
        logger.error("Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Validate the single-objective argument
    if args.mode == 'rag' and not args.goal:
        parser.error(f"--goal is required for mode '{args.mode}'")

    # Ablation mode: auto-generate study_id if not given
    if args.mode == 'ablation' and not args.study_id:
        args.study_id = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"No --study-id given; using auto-generated: {args.study_id}")
    
    try:
        # Print header
        logger.info("\n" + "="*80)
        logger.info("TRAP - Unified Execution")
        logger.info("="*80)
        logger.info(f"Mode: {args.mode.upper()}")
        logger.info(f"Goal: {args.goal if args.goal else 'Objectives from file'}")
        logger.info(f"Target Model: {args.target_model}")
        logger.info(f"Max Iterations: {args.max_iterations}")
        logger.info("="*80 + "\n")
        
        # Run selected mode
        if args.mode == 'rag':
            results = run_rag_mode(args)
        elif args.mode == 'ablation':
            results = run_ablation_mode(args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
        
        # Print final result
        logger.info("\n" + "="*80)
        logger.info("EXECUTION COMPLETE")
        logger.info("="*80)
        logger.info(f"Mode: {args.mode.upper()}")
        logger.info(f"Success: {'[OK] YES' if results.get('success') else '[X] NO'}")
        logger.info(f"Best Score: {results.get('best_score', 0)}/10")
        logger.info(f"Total Iterations: {results.get('total_steps', len(results.get('iterations', [])))}")
        
        # Show component usage
        if results.get('components_used'):
            logger.info("\nComponents Used:")
            for comp, used in results['components_used'].items():
                status = "[OK]" if used else "[X]"
                logger.info(f"  {status} {comp}")
        
        # Show template statistics if available
        if results.get('template_statistics'):
            logger.info("\nTemplate Usage:")
            for template, count in sorted(
                results['template_statistics'].get('usage', {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                logger.info(f"  {template}: {count} uses")
        
        logger.info("="*80 + "\n")
        
        # Save results — ablation mode handles its own saving; other modes go here
        if args.mode != 'ablation':
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # For rag mode with an explicit --experiment, route output to AblationPaths
                # so single-behavior runs are traceable alongside full study runs.
                use_ablation_path = (
                    args.mode == 'rag'
                    and getattr(args, 'experiment', None)
                    and getattr(args, 'study_id', None)
                )
                if use_ablation_path:
                    from utils.ablation_manifest import get_run_dirs, write_manifest
                    from utils.experiment_config import load_experiment_registry, resolve_kb_parameters, resolve_seeds
                    _registry = load_experiment_registry()
                    _exp_cfg = _registry.get("experiments", {}).get(args.experiment, {})
                    _seed = int(results.get("seed") or resolve_seeds(_registry, getattr(args, "random_seed", None))[0])
                    _rag_mode = str(results.get(
                        "rag_mode",
                        _exp_cfg.get("rag_mode", "online" if _exp_cfg.get("rag_enabled", True) else "disabled"),
                    )).lower()
                    _kb_params = results.get("kb_parameters") or resolve_kb_parameters(_registry, _exp_cfg)
                    abl_paths = get_run_dirs(args.study_id, args.experiment, _seed)
                    output_dir = abl_paths.reports
                    result_file = abl_paths.result_json_path()
                    # Write in-progress manifest
                    write_manifest(
                        abl_paths,
                        study_id=args.study_id,
                        experiment_id=args.experiment,
                        experiment_name=_exp_cfg.get("name", args.experiment),
                        seed=_seed,
                        target_model=args.target_model,
                        evaluator_model=args.evaluator_model,
                        rag_enabled=_exp_cfg.get("rag_enabled", True),
                        tree_variant=_exp_cfg.get("tree_variant", "full_active"),
                        tree_leaves=_exp_cfg.get("tree_leaves"),
                        tree_max_depth=_exp_cfg.get("tree_max_depth"),
                        success_threshold=_registry.get("shared", {}).get("success_threshold", 7.0),
                        max_iterations_per_episode=args.max_iterations,
                        episode_budget=1,
                        extra={
                            "rag_mode": _rag_mode,
                            "kb_snapshot_sha256": results.get("kb_snapshot_sha256", ""),
                            "kb_snapshot_source": results.get("kb_snapshot_source", ""),
                            "kb_parameters": _kb_params,
                        },
                    )
                else:
                    output_dir = log_dir
                    output_dir.mkdir(exist_ok=True)
                    result_file = output_dir / f"attack_{args.mode}_{timestamp}.json"

                # Make results JSON serializable
                serializable_results = {
                    k: v for k, v in results.items()
                    if not callable(v) and k not in ['leaf_node', 'decision_path']
                }

                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_results, f, indent=2, default=str)

                logger.info(f"Results saved to: {result_file}")

                # Update manifest with final metrics (rag+experiment path only)
                if use_ablation_path:
                    _success = bool(serializable_results.get("success"))
                    _score = float(serializable_results.get("best_score", 0))
                    write_manifest(
                        abl_paths,
                        study_id=args.study_id,
                        experiment_id=args.experiment,
                        experiment_name=_exp_cfg.get("name", args.experiment),
                        seed=_seed,
                        target_model=args.target_model,
                        evaluator_model=args.evaluator_model,
                        rag_enabled=_exp_cfg.get("rag_enabled", True),
                        tree_variant=_exp_cfg.get("tree_variant", "full_active"),
                        tree_leaves=_exp_cfg.get("tree_leaves"),
                        tree_max_depth=_exp_cfg.get("tree_max_depth"),
                        success_threshold=_registry.get("shared", {}).get("success_threshold", 7.0),
                        max_iterations_per_episode=args.max_iterations,
                        episode_budget=1,
                        total_behaviors=1,
                        success_count=int(_success),
                        asr_percent=100.0 if _success else 0.0,
                        avg_best_score=_score,
                        extra={
                            "rag_mode": _rag_mode,
                            "kb_snapshot_sha256": results.get("kb_snapshot_sha256", ""),
                            "kb_snapshot_source": results.get("kb_snapshot_source", ""),
                            "kb_parameters": _kb_params,
                        },
                    )

                    # Auto-export if study_id is given (ج)
                    try:
                        from scripts.ablation_exporter import export_study
                        _reg_path = Path(__file__).parent / "config" / "experiment_registry.yaml"
                        export_study(args.study_id, _reg_path)
                        logger.info(f"[EXPORT] Comparison tables updated for study '{args.study_id}'")
                    except Exception as _export_err:
                        logger.warning(f"[EXPORT] Auto-export failed: {_export_err}")

                if not args.skip_output_polish:
                    try:
                        from scripts.output_polisher import generate_improved_outputs
                        polished = generate_improved_outputs(serializable_results, output_dir)
                        serializable_results['improved_outputs'] = polished
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(serializable_results, f, indent=2, default=str, ensure_ascii=False)
                        logger.info(f"Improved outputs generated: {polished.get('output_dir')}")
                    except Exception as polish_error:
                        logger.warning(f"Failed to generate improved outputs: {polish_error}")
            except Exception as e:
                logger.warning(f"Failed to save results: {e}")
        
        # Exit with appropriate code
        sys.exit(0 if results.get('success') else 1)
        
    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
