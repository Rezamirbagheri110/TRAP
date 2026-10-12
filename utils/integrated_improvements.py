"""
Integrated Improvements Module

Collects all optional enhancements in one place for consistent use:
1. Advanced Refinement
2. Gradient Bandit Template Selection
3. Dense Reward Shaping
4. RAG MMR Diversity
5. LLM Caching
6. Template Quality Analysis
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class IntegratedImprovements:
    """
    Central coordinator for optional improvement components.
    """

    def __init__(self, llm_wrapper=None, rag_kb=None):
        """
        Initialize all improvements.

        Args:
            llm_wrapper: LLM wrapper used for prompt refinement.
            rag_kb: RAG knowledge base used for MMR retrieval.
        """
        self.llm_wrapper = llm_wrapper
        self.rag_kb = rag_kb

        # Initialize components
        self._init_advanced_refinement()
        self._init_gradient_bandit_template_selector()
        self._init_dense_reward_shaper()
        self._init_rag_mmr()
        self._init_llm_cache()
        self._init_template_analyzer()

        logger.info("[INTEGRATED] All improvements initialized")

    def _init_advanced_refinement(self):
        """Initialize Advanced Refinement Engine"""
        if os.getenv("OURS_USE_ADVANCED_REFINEMENT", "true").lower() == "true":
            try:
                from utils.advanced_refinement import AdvancedRefinementEngine
                max_parallel = int(os.getenv("OURS_REFINE_PARALLEL", "3"))
                self.refinement_engine = AdvancedRefinementEngine(
                    llm_wrapper=self.llm_wrapper,
                    max_parallel=max_parallel
                )
                logger.info(f"[INTEGRATED] Advanced Refinement enabled (parallel={max_parallel})")
            except Exception as e:
                logger.warning(f"[INTEGRATED] Advanced Refinement failed to load: {e}")
                self.refinement_engine = None
        else:
            self.refinement_engine = None
            logger.info("[INTEGRATED] Advanced Refinement disabled")

    def _init_gradient_bandit_template_selector(self):
        """Initialize gradient bandit template selector."""
        if os.getenv("OURS_USE_BANDIT_TEMPLATE_SELECTION", "true").lower() == "true":
            try:
                from utils.gradient_bandit_template_selector import GradientBanditTemplateSelector
                # Templates will be set later via set_templates()
                self.bandit_selector = None  # Placeholder
                logger.info("[INTEGRATED] Gradient bandit selector ready (will init with templates)")
            except Exception as e:
                logger.warning(f"[INTEGRATED] Gradient bandit failed to load: {e}")
                self.bandit_selector = None
        else:
            self.bandit_selector = None
            logger.info("[INTEGRATED] Gradient bandit selector disabled")

    def _init_dense_reward_shaper(self):
        """Initialize Dense Reward Shaper"""
        if os.getenv("OURS_USE_DENSE_REWARDS", "true").lower() == "true":
            try:
                from utils.dense_rewards import DenseRewardShaper
                engagement_bonus = float(os.getenv("OURS_DENSE_ENGAGEMENT_BONUS", "1.0"))
                harmful_bonus = float(os.getenv("OURS_DENSE_HARMFUL_CONTENT_BONUS", "2.0"))
                self.dense_reward_shaper = DenseRewardShaper(
                    engagement_bonus=engagement_bonus,
                    harmful_content_bonus=harmful_bonus
                )
                logger.info(f"[INTEGRATED] Dense Rewards enabled (engagement={engagement_bonus}, harmful={harmful_bonus})")
            except Exception as e:
                logger.warning(f"[INTEGRATED] Dense Rewards failed to load: {e}")
                self.dense_reward_shaper = None
        else:
            self.dense_reward_shaper = None
            logger.info("[INTEGRATED] Dense Rewards disabled")

    def _init_rag_mmr(self):
        """Initialize RAG MMR Retriever"""
        if os.getenv("OURS_RAG_USE_MMR", "true").lower() == "true":
            try:
                from utils.rag_mmr import MMRRetriever
                lambda_param = float(os.getenv("OURS_RAG_MMR_LAMBDA", "0.7"))
                self.mmr_retriever = MMRRetriever(lambda_param=lambda_param)
                logger.info(f"[INTEGRATED] RAG MMR enabled (lambda={lambda_param})")
            except Exception as e:
                logger.warning(f"[INTEGRATED] RAG MMR failed to load: {e}")
                self.mmr_retriever = None
        else:
            self.mmr_retriever = None
            logger.info("[INTEGRATED] RAG MMR disabled")

    def _init_llm_cache(self):
        """Initialize LLM Response Cache"""
        if os.getenv("OURS_ENABLE_LLM_CACHE", "true").lower() == "true":
            try:
                from utils.llm_cache import LLMCache
                cache_size = int(os.getenv("OURS_LLM_CACHE_SIZE", "1000"))
                cache_file = str(Path(__file__).resolve().parents[1] / "cache" / "llm_responses.pkl")
                self.llm_cache = LLMCache(
                    max_size=cache_size,
                    cache_file=cache_file
                )
                logger.info(f"[INTEGRATED] LLM Cache enabled (size={cache_size})")
            except Exception as e:
                logger.warning(f"[INTEGRATED] LLM Cache failed to load: {e}")
                self.llm_cache = None
        else:
            self.llm_cache = None
            logger.info("[INTEGRATED] LLM Cache disabled")

    def _init_template_analyzer(self):
        """Initialize Template Quality Analyzer"""
        self.template_stats = {}  # {template_name: {'count': 0, 'total_score': 0, 'successes': 0}}
        self.template_analyzer_enabled = True
        logger.info("[INTEGRATED] Template Analyzer enabled")

    # ========================================
    # Public Methods
    # ========================================

    def set_templates(self, templates: List[str]):
        """
        Set templates for gradient bandit selector.

        Args:
            templates: List of template names
        """
        if os.getenv("OURS_USE_BANDIT_TEMPLATE_SELECTION", "true").lower() == "true":
            try:
                from utils.gradient_bandit_template_selector import GradientBanditTemplateSelector
                step_size = float(os.getenv("OURS_BANDIT_STEP_SIZE", "0.1"))
                self.bandit_selector = GradientBanditTemplateSelector(
                    templates=templates,
                    step_size=step_size
                )
                logger.info(f"[INTEGRATED] Gradient bandit initialized with {len(templates)} templates")
            except Exception as e:
                logger.warning(f"[INTEGRATED] Gradient bandit init failed: {e}")
                self.bandit_selector = None

    def select_template(self, available_templates: List[str], exclude: List[str] = None) -> str:
        """
        Select template using gradient bandit or fallback to random.

        Args:
            available_templates: Available template names
            exclude: Templates to exclude

        Returns:
            Selected template name
        """
        if self.bandit_selector and len(available_templates) > 0:
            try:
                selected = self.bandit_selector.select(exclude=exclude or [])
                # Make sure selected is in available
                if selected in available_templates:
                    return selected
                else:
                    logger.warning(f"[INTEGRATED] Bandit selected unavailable template '{selected}', falling back")
            except Exception as e:
                logger.warning(f"[INTEGRATED] Bandit selection failed: {e}")

        # Fallback to random
        valid = [t for t in available_templates if t not in (exclude or [])]
        if valid:
            import random
            return random.choice(valid)
        elif available_templates:
            import random
            return random.choice(available_templates)
        else:
            return "fallback"

    def update_template_score(self, template_name: str, score: float):
        """
        Update template with score (for gradient bandit).

        Args:
            template_name: Template name
            score: Score received (0-10)
        """
        if self.bandit_selector:
            try:
                self.bandit_selector.update(template_name, score)
            except Exception as e:
                logger.warning(f"[INTEGRATED] Bandit update failed: {e}")

        # Also update template stats
        if template_name not in self.template_stats:
            self.template_stats[template_name] = {
                'count': 0,
                'total_score': 0.0,
                'successes': 0
            }

        self.template_stats[template_name]['count'] += 1
        self.template_stats[template_name]['total_score'] += score
        if score >= 7.0:
            self.template_stats[template_name]['successes'] += 1

    def refine_prompt_advanced(self,
                               goal: str,
                               prompt: str,
                               current_score: float,
                               template_name: str = None) -> List[Dict]:
        """
        Refine prompt using advanced multi-strategy approach

        Args:
            goal: Target goal
            prompt: Current prompt
            current_score: Current score
            template_name: Current template (optional)

        Returns:
            List of refined prompts with metadata
        """
        if self.refinement_engine:
            try:
                return self.refinement_engine.refine_parallel(
                    goal=goal,
                    base_prompt=prompt,
                    current_score=current_score,
                    template_name=template_name
                )
            except Exception as e:
                logger.warning(f"[INTEGRATED] Advanced refinement failed: {e}")
                return []
        else:
            return []

    def compute_dense_reward(self, response: str, base_score: float, query: str = "") -> float:
        """
        Compute dense reward from response

        Args:
            response: LLM response
            base_score: Base score from evaluator
            query: Original query

        Returns:
            Dense reward score
        """
        if self.dense_reward_shaper:
            try:
                return self.dense_reward_shaper.compute_dense_reward(
                    response=response,
                    base_score=base_score,
                    query=query
                )
            except Exception as e:
                logger.warning(f"[INTEGRATED] Dense reward failed: {e}")
                return base_score
        else:
            return base_score

    def retrieve_with_mmr(self,
                         query_embedding: np.ndarray,
                         candidates: List[Any],
                         candidate_embeddings: List[np.ndarray],
                         top_k: int = 5) -> List[Any]:
        """
        Retrieve with MMR diversity

        Args:
            query_embedding: Query embedding
            candidates: Candidate items
            candidate_embeddings: Candidate embeddings
            top_k: Number to return

        Returns:
            List of diverse candidates
        """
        if self.mmr_retriever and len(candidates) > top_k:
            try:
                return self.mmr_retriever.retrieve(
                    query_embedding=query_embedding,
                    candidate_embeddings=candidate_embeddings,
                    candidate_items=candidates,
                    top_k=top_k
                )
            except Exception as e:
                logger.warning(f"[INTEGRATED] MMR retrieval failed: {e}")
                return candidates[:top_k]
        else:
            return candidates[:top_k]

    def cache_get(self, prompt: str, model: str = "default") -> Optional[str]:
        """Get cached LLM response"""
        if self.llm_cache:
            return self.llm_cache.get(prompt, model)
        return None

    def cache_put(self, prompt: str, response: str, model: str = "default"):
        """Cache LLM response"""
        if self.llm_cache:
            self.llm_cache.put(prompt, response, model)

    def get_template_analysis(self) -> Dict:
        """
        Get template quality analysis

        Returns:
            Dictionary with template statistics
        """
        analysis = {}
        for template, stats in self.template_stats.items():
            count = stats['count']
            if count > 0:
                avg_score = stats['total_score'] / count
                success_rate = (stats['successes'] / count) * 100
                analysis[template] = {
                    'count': count,
                    'avg_score': avg_score,
                    'success_rate': success_rate,
                    'quality_rating': self._rate_template_quality(avg_score, success_rate)
                }

        # Sort by quality
        sorted_analysis = dict(sorted(
            analysis.items(),
            key=lambda x: x[1]['avg_score'],
            reverse=True
        ))

        return sorted_analysis

    def _rate_template_quality(self, avg_score: float, success_rate: float) -> str:
        """Rate template quality"""
        if success_rate >= 50 and avg_score >= 6:
            return "Excellent"
        elif success_rate >= 30 and avg_score >= 4:
            return "Good"
        elif success_rate >= 10 and avg_score >= 2:
            return "Fair"
        else:
            return "Poor"

    def get_statistics(self) -> Dict:
        """Get all statistics"""
        stats = {
            'advanced_refinement': None,
            'bandit_selector': None,
            'dense_rewards': None,
            'rag_mmr': None,
            'llm_cache': None,
            'template_analysis': self.get_template_analysis()
        }

        if self.refinement_engine:
            stats['advanced_refinement'] = self.refinement_engine.get_statistics()

        if self.bandit_selector:
            stats['bandit_selector'] = self.bandit_selector.get_statistics()

        if self.llm_cache:
            stats['llm_cache'] = self.llm_cache.get_statistics()

        return stats

    def save(self):
        """Save all persistent components"""
        if self.llm_cache:
            try:
                self.llm_cache.save()
                logger.info("[INTEGRATED] LLM cache saved")
            except Exception as e:
                logger.warning(f"[INTEGRATED] Cache save failed: {e}")



