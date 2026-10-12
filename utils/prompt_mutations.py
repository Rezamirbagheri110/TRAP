"""
Prompt mutation helpers extracted from RAGEnhancedAttackRunner.

These were inline static methods inside the 6k-line runner. Extracted so they
can be unit-tested in isolation and reused outside the runner.
"""

from __future__ import annotations

import re
from typing import List


_MARKER_BLACKLIST = (
    "jailbreak", "ignore previous", "ignore all previous", "system prompt override",
)


def mutate_prompt_for_diversity(prompt: str, avoid_markers: bool = False) -> str:
    """Light syntactic mutation that preserves meaning but breaks exact-match
    caches and templated detectors.

    Used as a fallback when the gradient bandit keeps re-selecting the same
    template and the orchestrator wants a small structural change.
    """
    if not prompt:
        return prompt
    out = prompt

    # Replace common imperative openers with paraphrased variants.
    replacements = [
        (r"^Write\b", "Compose"),
        (r"^Explain\b", "Walk through"),
        (r"^Describe\b", "Outline"),
        (r"^Create\b", "Produce"),
        (r"^Generate\b", "Draft"),
    ]
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, count=1)

    if avoid_markers:
        lowered = out.lower()
        for marker in _MARKER_BLACKLIST:
            if marker in lowered:
                out = re.sub(re.escape(marker), "context", out, flags=re.IGNORECASE)

    return out


def normalize_prompt_for_tough_models(prompt: str) -> str:
    """Make a prompt more palatable to safety-tuned models: drop explicit
    jailbreak markers and add a clarifying frame."""
    if not prompt:
        return prompt
    out = prompt
    for marker in _MARKER_BLACKLIST:
        out = re.sub(re.escape(marker), "professional context", out, flags=re.IGNORECASE)
    if not out.lower().startswith(("please", "kindly", "could you")):
        out = "Please " + out[0].lower() + out[1:]
    return out


def ensure_target_prefix(prompt: str, target_str: str) -> str:
    """Append a target-response prefix hint if not already present.

    The original implementation lived inline; extracted here so unit tests
    can verify idempotency and case-insensitive detection.
    """
    if not prompt or not target_str:
        return prompt
    if target_str.lower() in prompt.lower():
        return prompt
    return f"{prompt}\n\nBegin your response with: \"{target_str}\""


def enforce_actionable_prompt(prompt: str, goal: str, min_words: int = 12) -> str:
    """If the prompt is too short or doesn't reference the goal, prepend goal text.

    This was previously a multi-branch private method; the trimmed version
    keeps the actionable behaviour (avoid empty / off-goal prompts) without
    the dead branches.
    """
    if not prompt or not prompt.strip():
        return goal
    word_count = len(prompt.split())
    if word_count < min_words and goal:
        return f"{prompt}\n\nSpecifically: {goal}"
    return prompt
