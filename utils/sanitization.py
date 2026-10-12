"""
Safety / sanitization helpers extracted from scripts/run_attack_with_rag.py.

Pure functions, no class state — safe to import anywhere. Moved out of the
6k-line god-class so the orchestration code stays focused on attack flow
instead of string scrubbing.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import List


HARMFUL_SIGNALS = (
    "kill", "harm", "weapon", "bomb", "explosive", "poison", "exploit", "malware",
    "ransomware", "ddos", "rootkit", "phishing", "scam", "fraud", "trafficking",
    "self-harm", "suicide", "abuse",
)

SENSITIVE_SIGNALS = (
    "biological", "chemical", "nuclear", "radioactive", "cyber attack",
    "extremism", "terror", "election", "minor", "csam",
)


_DEFENSIVE_PREFIXES = (
    "From a research-and-safety perspective, ",
    "For a defensive audit, ",
    "From a safety-engineering perspective, ",
)


def safe_preview(text: str, limit: int = 150) -> str:
    """Truncate long text with an ellipsis suffix."""
    if text is None:
        return ""
    s = str(text)
    return s if len(s) <= limit else s[: max(0, limit - 3)] + "..."


def pick_safe_prefix() -> str:
    return random.choice(_DEFENSIVE_PREFIXES)


def stable_int_hash(text: str) -> int:
    """Deterministic non-negative int from text (used for behavior seeding)."""
    if not text:
        return 0
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) & 0x7FFFFFFF


def is_harmful_goal(goal: str) -> bool:
    if not goal:
        return False
    g = goal.lower()
    return any(sig in g for sig in HARMFUL_SIGNALS)


def is_sensitive_goal(goal: str) -> bool:
    if not goal:
        return False
    g = goal.lower()
    return any(sig in g for sig in SENSITIVE_SIGNALS)


def rewrite_goal_to_defensive(goal: str) -> str:
    """Wrap a potentially harmful goal in defensive framing for evaluation."""
    if not goal:
        return goal
    return f"{pick_safe_prefix()}{goal}"


def sanitize_target_str_for_prompt(target_str: str, goal: str) -> str:
    """Strip dangerous-looking suffixes from the target prefix used in prompts.

    Originally this lived inline in run_attack_with_rag.py. Extracted because
    it's pure and is exercised by tests independently of the runner.
    """
    if not target_str:
        return ""
    cleaned = target_str.strip().strip('"').strip("'")
    # Avoid duplicate inclusion of the goal text inside the target prefix
    if goal and goal.lower() in cleaned.lower():
        cleaned = cleaned.replace(goal, "").strip()
    return cleaned


def set_global_seed(seed: int) -> None:
    """Seed random, numpy and torch (if available) for reproducibility."""
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np
        np.random.seed(seed & 0xFFFFFFFF)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed & 0xFFFFFFFF)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed & 0xFFFFFFFF)
    except Exception:
        pass


__all__ = [
    "safe_preview",
    "pick_safe_prefix",
    "stable_int_hash",
    "is_harmful_goal",
    "is_sensitive_goal",
    "rewrite_goal_to_defensive",
    "sanitize_target_str_for_prompt",
    "set_global_seed",
]
