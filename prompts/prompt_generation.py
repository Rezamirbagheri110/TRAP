"""Canonical gateway for base prompt generation.

The runner still owns policy decisions and post-processing. This gateway owns
the contract for converting a selected template/action into a base prompt.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class PromptGenerationGateway:
    def __init__(self, prompt_system, advanced_generator) -> None:
        self.prompt_system = prompt_system
        self.advanced_generator = advanced_generator

    def generate(
        self,
        *,
        action_name: str,
        goal: str,
        target_str: str,
        template_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        mapped_options: Optional[Dict[str, Any]] = None,
        fallback: Optional[Callable[[], str]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        if template_name and template_name in self.advanced_generator.templates:
            result = self.advanced_generator.generate_prompt(
                template_name=template_name,
                goal=goal,
                target_str=target_str,
                context=context,
            )
            return {
                **result,
                "template_name": template_name,
                "generator": "advanced",
            }

        try:
            result = self.prompt_system.generate_prompt(
                action_name=action_name,
                goal=goal,
                target_str=target_str,
                **(mapped_options or {}),
            )
            return {
                **result,
                "template_name": result.get("pattern_used") or result.get("pattern_name") or action_name,
                "generator": "mapped",
            }
        except Exception:
            if fallback is None:
                raise
            return {
                "user_prompt": fallback(),
                "system_prompt": "You are a helpful assistant.",
                "template_name": "fallback_simple",
                "generator": "fallback",
                "context": context,
            }
