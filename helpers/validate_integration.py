"""
Integration validation for the current TRAP package layout.

This script is meant to be runnable from the repository root with:
    py helpers/validate_integration.py
"""

from __future__ import annotations

import io
import os
from unittest.mock import patch
import sys
from pathlib import Path
import numpy as np


if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


OURS_DIR = Path(__file__).resolve().parent.parent
ATTACKS_DIR = OURS_DIR.parent
REPO_ROOT = ATTACKS_DIR.parent
VALIDATION_KB_DIR = OURS_DIR / ".tmp" / "validation_kb"
VALIDATION_KB_DIR.mkdir(parents=True, exist_ok=True)

for path in (REPO_ROOT, OURS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def ok(message: str) -> None:
    print(f"   [OK] {message}")


def fail(message: str) -> None:
    print(f"   [FAIL] {message}")
    raise SystemExit(1)


class _TestEmbedder:
    """Deterministic in-memory embedder for integration checks without network/auth."""

    embedding_dim = 8

    def embed(self, text: str):
        values = np.zeros(self.embedding_dim, dtype=np.float32)
        encoded = text.encode("utf-8", errors="ignore")
        for index, byte in enumerate(encoded[: self.embedding_dim]):
            values[index] = byte / 255.0
        return values


print("=" * 80)
print("VALIDATING TRAP METHOD INTEGRATION")
print("=" * 80)


print("\n1. Testing core imports...")
try:
    from config.config import ATTACK_TEMP, TARGET_TEMP
    from config.rl_config import DEFAULT_CONFIG, RLSystemConfig
except Exception as exc:
    fail(f"Config import failed: {exc}")
ok("Config modules loaded")


print("\n2. Testing core RL components...")
try:
    from core.action_tree import ActionTree, create_action_tree

    tree = create_action_tree()
except Exception as exc:
    fail(f"Core components failed: {exc}")
ok(f"Action tree created ({tree.get_action_space_size()} leaf nodes)")


print("\n3. Testing utility modules...")
try:
    from utils.embeddings import SemanticEmbedder
    from utils.reward_system import AdaptiveRewardCalculator
    from utils.rl_components import IntrinsicMotivation, PrioritizedReplayBuffer

    reward_calc = AdaptiveRewardCalculator()
except Exception as exc:
    fail(f"Utils failed: {exc}")
ok("Embedding utilities imported")
ok("Reward calculator initialized")


print("\n4. Testing model modules...")
try:
    from models.evaluators import load_evaluator
    from models.language_models_openrouter import OpenRouterLLM
except Exception as exc:
    fail(f"Model modules failed: {exc}")
ok("Model modules imported")


print("\n5. Testing RAG system...")
try:
    from rag import (
        RAGAugmentedAgent,
        RAGEnhancedPromptSelector,
        RAGKnowledgeBase,
        create_rag_enhanced_agent,
        create_rag_knowledge_base,
    )

    test_embedder = _TestEmbedder()
    kb = create_rag_knowledge_base(embedder=test_embedder, storage_dir=str(VALIDATION_KB_DIR))
    selector = RAGEnhancedPromptSelector(kb)
except Exception as exc:
    fail(f"RAG system failed: {exc}")
ok("RAG knowledge base created")
ok("RAG prompt selector created")

# Regression check: unset boolean defaults must not turn "false" into True.
from utils.runtime_config import ENV_VAR_REGISTRY, get_env, _bool
with patch.dict(os.environ):
    for name, (default, coerce, _) in ENV_VAR_REGISTRY.items():
        if coerce is _bool:
            os.environ.pop(name, None)
            assert get_env(name) is _bool(default), f"Incorrect boolean default: {name}"
    online_kb = RAGKnowledgeBase(
        embedder=test_embedder, storage_dir=str(VALIDATION_KB_DIR / "online"),
        autoload=False,
    )
    assert online_kb.read_only is False, "Default online memory must be writable"
    accepted, message = online_kb.add_example("Describe a benign software testing task", "test goal", 9.0)
    assert accepted, message
    static_kb = RAGKnowledgeBase(
        embedder=test_embedder, storage_dir=str(VALIDATION_KB_DIR / "static"),
        autoload=False, read_only=True,
    )
    accepted, _ = static_kb.add_example("Describe a benign software testing task", "test goal", 9.0)
    assert not accepted, "Static memory must reject additions"
ok("Unset boolean defaults, writable online memory, and read-only static memory verified")


print("\n6. Testing main scripts...")
expected_paths = [
    OURS_DIR / "scripts" / "run_attack_with_rag.py",
    OURS_DIR / "main.py",
]
missing = [str(path) for path in expected_paths if not path.exists()]
if missing:
    fail(f"Expected scripts missing: {missing}")
for path in expected_paths:
    ok(f"{path.name} exists")


print("\n7. Testing RL agent creation...")
try:
    from core.hierarchical_bandit_agent import AgentConfig, HierarchicalBanditAgent

    agent = HierarchicalBanditAgent(action_tree=tree, config=AgentConfig())
    leaf, path = agent.navigate_to_leaf(deterministic=True)
except Exception as exc:
    fail(f"Agent creation/navigation failed: {exc}")
ok(f"Hierarchical Bandit agent created on {agent.device}")
ok(f"Navigation works (reached: {leaf.name})")


print("\n8. Testing RAG-RL integration...")
try:
    augmented_agent = create_rag_enhanced_agent(
        base_agent=agent,
        knowledge_base_path=None,
        embedder=test_embedder,
        knowledge_base=kb,
    )
    action, rag_suggestion = augmented_agent.select_action_with_rag(
        query="test query",
        deterministic=True,
    )
except Exception as exc:
    fail(f"RAG integration failed: {exc}")
ok("RAG-augmented agent created")
ok("RAG-enhanced action selection works")


print("\n9. Testing component compatibility...")
try:
    reward = reward_calc.compute_reward(
        judge_score=8,
        on_topic_score=1,
        num_attempts=3,
        action_diversity=0.7,
        previous_best_score=5,
    )
    embedding = test_embedder.embed("test query")
    kb.add_example("test prompt", "test goal", 9.0)
    results = kb.retrieve_similar("test", top_k=1)
except Exception as exc:
    fail(f"Component compatibility failed: {exc}")
ok(f"Reward calculation works (reward: {reward:.2f})")
ok(f"Embedding works (shape: {embedding.shape})")
ok(f"KB operations work (found {len(results)} similar)")


print("\n" + "=" * 80)
print("VALIDATION COMPLETE - ALL TESTS PASSED [OK]")
print("=" * 80)
print("\nThe TRAP method is properly integrated and ready to use.")
print("\nNext steps:")
print("  1. Check options: py main.py --help")
print("  2. Configure a local retrieval snapshot with --rag-kb-path; see README.md")
print("  3. Run attack: py main.py --mode rag --experiment A_ONLINE --goal 'your goal'")
print("=" * 80)
