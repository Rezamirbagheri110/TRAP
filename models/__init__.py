"""
LLM Models and Interfaces Package

This package provides language model interfaces for the TRAP attack method,
including OpenRouter-based LLM access and evaluation capabilities.
"""

# Import with error handling for optional dependencies
try:
    from .language_models_openrouter import LanguageModel, GPT, OpenRouterLLM
except ImportError as e:
    import logging
    logging.warning(f"Could not import language_models_openrouter: {e}")
    LanguageModel = None
    GPT = None
    OpenRouterLLM = None

try:
    from .conversers import AttackLM, TargetLM
except ImportError as e:
    import logging
    logging.debug(f"Could not import conversers: {e}")
    AttackLM = None
    TargetLM = None

try:
    from .conversers_openrouter import (
        AttackLM as OpenRouterAttackLM,
        TargetLM as OpenRouterTargetLM
    )
except ImportError as e:
    import logging
    logging.debug(f"Could not import conversers_openrouter: {e}")
    OpenRouterAttackLM = None
    OpenRouterTargetLM = None

try:
    from .evaluators import (
        EvaluatorBase,
        NoEvaluator,
        GPTEvaluator,
        OpenRouterEvaluator,
        load_evaluator,
        EvaluationReport,
        EvaluationFeedback,
        JudgeConfiguration
    )
except ImportError as e:
    import logging
    logging.warning(f"Could not import evaluators: {e}")
    EvaluatorBase = None
    NoEvaluator = None
    GPTEvaluator = None
    OpenRouterEvaluator = None
    load_evaluator = None
    EvaluationReport = None
    EvaluationFeedback = None
    JudgeConfiguration = None

__all__ = [
    # Language models
    'LanguageModel',
    'GPT',
    'OpenRouterLLM',
    
    # Conversers
    'AttackLM',
    'TargetLM',
    'OpenRouterAttackLM',
    'OpenRouterTargetLM',
    
    # Evaluators
    'EvaluatorBase',
    'NoEvaluator',
    'GPTEvaluator',
    'OpenRouterEvaluator',
    'load_evaluator',
    'EvaluationReport',
    'EvaluationFeedback',
    'JudgeConfiguration'
]