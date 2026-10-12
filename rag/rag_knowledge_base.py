"""
RAG Knowledge Base for Storing and Retrieving Successful Attack Prompts

Features:
- Vector embeddings using sentence-transformers
- Similarity-based retrieval
- Deduplication of similar prompts
- Dynamic learning and updates
- Persistence to disk
"""

import json
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import pickle
import threading
from contextlib import contextmanager

# Import the embedder from utils
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.embeddings import SemanticEmbedder
try:
    from utils.runtime_config import get_env as _get_env
except ImportError:
    import os as _os_fallback
    def _get_env(name: str, override=None):  # type: ignore[misc]
        if override is not None:
            return override
        return _os_fallback.getenv(name)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between vectors or matrices with NaN guards.

    Args:
        a: First vector/matrix
        b: Second vector/matrix

    Returns:
        Cosine similarity matrix (guaranteed finite values)
    """
    # Guard against zero-norm vectors
    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)

    # Replace zero norms with small epsilon to prevent division by zero
    a_norms = np.where(a_norms == 0, 1e-10, a_norms)
    b_norms = np.where(b_norms == 0, 1e-10, b_norms)

    # Normalize vectors
    a_norm = a / a_norms
    b_norm = b / b_norms

    # Compute dot product
    similarity = np.dot(a_norm, b_norm.T)

    # Guard against NaN/inf (should not happen now, but extra safety)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=-1.0)

    # Clamp to valid range [-1, 1]
    similarity = np.clip(similarity, -1.0, 1.0)

    return similarity

logger = logging.getLogger(__name__)


@dataclass
class PromptExample:
    """Structure for storing a successful prompt example"""
    prompt: str
    goal: str
    success_score: float
    timestamp: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Remove embedding from JSON (stored separately)
        if 'embedding' in data:
            data['embedding'] = None
        return data


class RAGKnowledgeBase:
    """
    RAG Knowledge Base for managing successful attack prompts

    Features:
    - Semantic search using embeddings
    - Automatic deduplication based on similarity
    - Dynamic learning from new successful examples
    - Efficient storage and retrieval
    - Thread-safe access for parallel execution
    """

    # Class-level lock for file I/O synchronization across all instances
    _global_file_lock = threading.Lock()

    def __init__(
        self,
        embedder: Optional[SemanticEmbedder] = None,
        similarity_threshold: float = 0.60,
        max_examples: int = 10000,
        storage_dir: str = None,
        autoload: bool = True,
        read_only: Optional[bool] = None,
        persist_reembedded_on_load: bool = True,
    ):
        """
        Initialize RAG knowledge base

        Args:
            embedder: Semantic embedder instance (creates default if None)
            similarity_threshold: Threshold for considering prompts similar (0-1)
            max_examples: Maximum number of examples to store
            storage_dir: Directory for persisting knowledge base
            autoload: If False, the KB starts EMPTY — it will neither auto-load
                the persisted ``knowledge_base.pkl`` nor warm-start via
                ``preload_successful_examples``. This is required by ablation E
                (Full+NoRAG): the runner gates its own load/bootstrap on
                ``rag_enabled``, but without this flag the constructor itself
                would silently load the shared persisted KB and leak retrieval
                signal into the NoRAG run, contaminating the A<->E comparison.
        """
        self.embedder = embedder or SemanticEmbedder()
        # Respect provided threshold; allow lower recall settings when needed
        self.base_similarity_threshold = min(max(similarity_threshold, 0.3), 0.95)
        self.similarity_threshold = self.base_similarity_threshold
        self.max_examples = max_examples

        # Adaptive threshold management (more aggressive lowering after misses)
        self.consecutive_misses = 0
        self.miss_threshold_for_lowering = 2  # Lower threshold after only 2 misses
        self.min_allowed_threshold = 0.50  # NEW: Don't go below 0.50 to avoid irrelevant matches
        self.lowered_threshold = max(self.min_allowed_threshold, self.base_similarity_threshold - 0.15)

        # Storage
        self.examples: List[PromptExample] = []
        self.embeddings_matrix: Optional[np.ndarray] = None

        # Setup storage directory
        if storage_dir is None:
            storage_dir = Path(__file__).parent / "knowledge_base"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Stats
        self.stats = {
            'total_added': 0,
            'duplicates_rejected': 0,
            'weak_prompts_rejected': 0,
            'last_update': None
        }

        # Thread-safe locks for in-memory operations
        self._memory_lock = threading.Lock()  # Protects examples and embeddings_matrix

        # Mode flags
        self.read_only = (
            bool(_get_env("OURS_RAG_READONLY"))
            if read_only is None else bool(read_only)
        )
        self.persist_reembedded_on_load = bool(persist_reembedded_on_load)
        force_reload = bool(_get_env("OURS_FORCE_RELOAD_KB"))
        
        logger.info(f"RAG Knowledge Base initialized")
        logger.info(f"Similarity threshold: {similarity_threshold}")
        logger.info(f"Storage directory: {self.storage_dir}")

        # Ablation E (Full+NoRAG): when the caller disables RAG it passes
        # autoload=False so the KB stays empty — no persisted load, no warm-start
        # preload. Without this the constructor would leak the shared KB into a
        # NoRAG run regardless of the runner's own rag_enabled gating.
        if not autoload:
            logger.info("[BOOT] autoload=False — KB starts empty (RAG disabled / isolated)")
            return

        # Auto-load persisted KB if available (skip heavy JSON bootstrap on every run)
        kb_file = self.storage_dir / "knowledge_base.pkl"
        if kb_file.exists() and not force_reload:
            if self.load(str(kb_file)):
                logger.info(f"[BOOT] Loaded persisted KB ({len(self.examples)} examples)")

                # CRITICAL: Auto-cleanup contaminated examples on load
                cleanup_on_load = bool(_get_env("OURS_KB_AUTO_CLEANUP"))
                if cleanup_on_load:
                    removed = self.remove_contaminated_examples(min_clean_score=7.0)
                    if removed > 0:
                        # Save cleaned KB immediately
                        self.save()
                        logger.info(f"[BOOT] Auto-cleanup removed {removed} contaminated examples, saved clean KB")
        else:
            if force_reload and kb_file.exists():
                logger.info("[BOOT] Force reload requested; ignoring existing knowledge_base.pkl")
            # If force reload or no file, start empty; try to warm-start with seed data
            self.preload_successful_examples()

    def preload_successful_examples(self) -> int:
        """
        Warm-start the knowledge base with curated successful examples.
        Loads from seed_examples.json if available, and optionally from recent results.

        Returns:
            Number of examples preloaded
        """
        if len(self.examples) > 0:
            logger.debug("[PRELOAD] KB already has examples, skipping preload")
            return 0

        loaded_count = 0

        # 1. Load curated seed examples
        seed_file = self.storage_dir / "seed_examples.json"
        if seed_file.exists():
            try:
                with open(seed_file, 'r', encoding='utf-8') as f:
                    seed_data = json.load(f)

                for item in seed_data.get('examples', []):
                    success, msg = self.add_example(
                        prompt=item['prompt'],
                        goal=item['goal'],
                        success_score=item.get('score', 8.0),
                        metadata={
                            'source': 'seed',
                            'why_successful': item.get('why_successful', ''),
                            'techniques': item.get('techniques', [])
                        },
                        force=True  # Don't filter seed examples
                    )
                    if success:
                        loaded_count += 1

                logger.info(f"[PRELOAD] Loaded {loaded_count} seed examples from {seed_file}")
            except Exception as e:
                logger.warning(f"[PRELOAD] Failed to load seed examples: {e}")

        # 2. Optionally load from recent successful results
        if bool(_get_env("OURS_RAG_PRELOAD_FROM_RESULTS", False)):
            results_dir = Path(__file__).parent.parent / "results"
            if results_dir.exists():
                loaded_from_results = self._load_from_recent_results(results_dir)
                loaded_count += loaded_from_results
                logger.info(f"[PRELOAD] Loaded {loaded_from_results} examples from recent results")

        if loaded_count > 0:
            self.save()  # Persist preloaded data
            logger.info(f"[PRELOAD] Total preloaded: {loaded_count} examples")

        return loaded_count

    def _load_from_recent_results(self, results_dir: Path, max_files: int = 3) -> int:
        """Load successful examples from recent result files"""
        loaded = 0

        try:
            # Get recent result files
            result_files = sorted(
                results_dir.glob("ours_results_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )[:max_files]

            for result_file in result_files:
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    for behavior_result in data.get('results', []):
                        if behavior_result.get('success') and behavior_result.get('best_score', 0) >= 7:
                            success, _ = self.add_example(
                                prompt=behavior_result['best_prompt'],
                                goal=behavior_result['goal'],
                                success_score=behavior_result['best_score'],
                                metadata={
                                    'source': 'past_result',
                                    'file': result_file.name
                                },
                                force=False  # Allow filtering for past results
                            )
                            if success:
                                loaded += 1
                except Exception as e:
                    logger.debug(f"[PRELOAD] Error loading {result_file.name}: {e}")

        except Exception as e:
            logger.warning(f"[PRELOAD] Error scanning results directory: {e}")

        return loaded

    def add_example(
        self,
        prompt: str,
        goal: str,
        success_score: float,
        metadata: Optional[Dict] = None,
        force: bool = False,
        min_score: float = 2.0,
    ) -> Tuple[bool, str]:
        """
        Add a new successful example to knowledge base (THREAD-SAFE)

        Args:
            prompt: The successful prompt
            goal: The goal/behavior being targeted
            success_score: Success score (0-10)
            metadata: Additional metadata
            force: Force addition even if similar exists
            min_score: Minimum score floor for this call; callers should pass
                their learning_threshold so the KB gate matches the policy gate.

        Returns:
            Tuple of (success, message)
        """
        if getattr(self, "read_only", False):
            return False, "Prompt rejected: KB is read-only (OURS_RAG_READONLY=true)"

        # Defensive/educational keyword filter. This was previously a hard rule that
        # rejected many genuinely successful prompts. It is now:
        #   1) Tiered (strong vs. weak signals)
        #   2) Conditional on success score (high-scoring prompts bypass)
        #   3) Overridable via OURS_RAG_DISABLE_DEFENSIVE_FILTER=true
        disable_filter = bool(_get_env("OURS_RAG_DISABLE_DEFENSIVE_FILTER"))

        # Strong indicators: phrases the target model very often refuses on.
        strong_defensive_keywords = [
            "fictional", "hypothetical", "educational purposes", "research only",
            "for educational", "academic exercise", "simulated", "theoretical",
            "ethical", "defensive", "safeguard",
        ]
        # Weak indicators: framing terms that *sometimes* signal a refusal-prone
        # prompt but also appear in many genuine wins; we ignore these unless the
        # prompt also scored low.
        weak_defensive_keywords = [
            "methodology", "tutorial", "prevention", "comprehensive documentation",
            "procedural steps", "complete methodology", "technical methodology",
            "scenario setup", "realistic scenario", "testing purposes",
            "instructional guide", "document the complete", "security testing", "audit",
        ]
        prompt_lower = prompt.lower()
        strong_matches = [kw for kw in strong_defensive_keywords if kw in prompt_lower]
        weak_matches = [kw for kw in weak_defensive_keywords if kw in prompt_lower]

        # High-confidence wins (score >= 8) bypass the filter — they passed a
        # judge despite the framing, so the framing did not cause a refusal.
        bypass_for_high_score = success_score >= 8.0

        if not force and not disable_filter and not bypass_for_high_score:
            if strong_matches:
                self.stats['weak_prompts_rejected'] += 1
                logger.warning(f"[RAG FILTER] Rejected (strong defensive: {strong_matches[:3]})")
                return False, f"Prompt rejected: contains defensive framing ({', '.join(strong_matches[:2])})"
            # Weak matches only trigger rejection when the prompt also has a low score.
            if weak_matches and success_score < 5.0:
                self.stats['weak_prompts_rejected'] += 1
                logger.warning(f"[RAG FILTER] Rejected low-score prompt with weak defensive markers: {weak_matches[:3]}")
                return False, f"Prompt rejected: weak defensive markers with low score ({', '.join(weak_matches[:2])})"

        # Validate success score against the caller's min_score gate.
        # Callers should pass their learning_threshold so this gate matches
        # the policy gate that guards learn_from_success().
        if success_score < min_score and not force:
            self.stats['weak_prompts_rejected'] += 1
            return False, f"Prompt rejected: low success score {success_score} (min={min_score})"

        # Create embedding
        embedding = self.embedder.embed(prompt)
        if embedding is None:
            return False, "Prompt rejected: embedding failed"
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        # THREAD-SAFE: Acquire lock for in-memory operations
        with self._memory_lock:
            # Check for duplicates/similar prompts; allow more growth when KB is small
            if not force and len(self.examples) > 0:
                is_duplicate, similarity, similar_idx = self._check_similarity(embedding)
                if is_duplicate and len(self.examples) > 10:
                    self.stats['duplicates_rejected'] += 1
                    return False, f"Prompt rejected: too similar to existing (similarity: {similarity:.3f})"

            # Create example
            example = PromptExample(
                prompt=prompt,
                goal=goal,
                success_score=success_score,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {},
                embedding=embedding,
                id=f"prompt_{len(self.examples)}_{datetime.now().timestamp()}"
            )

            # Add to knowledge base
            self.examples.append(example)
            self._rebuild_embeddings_matrix()

            # Update stats
            self.stats['total_added'] += 1
            self.stats['last_update'] = datetime.now().isoformat()

            # Trim oldest/lowest-scoring examples if we exceed cap
            if len(self.examples) > self.max_examples:
                self.examples.sort(key=lambda ex: ex.success_score, reverse=True)
                self.examples = self.examples[:self.max_examples]
                self._rebuild_embeddings_matrix()

            logger.info(f"Added new example (total: {len(self.examples)})")

        # Auto-save if configured (THREAD-SAFE file I/O)
        self.save()

        return True, f"Successfully added example with score {success_score}"
    
    def retrieve_similar(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.35,
        min_success_score: float = 0.0,
        weight_by_score: bool = True
    ) -> List[Tuple[PromptExample, float]]:
        """
        Retrieve most similar successful examples for a query with adaptive threshold (THREAD-SAFE).

        Args:
            query: Query text (goal or partial prompt)
            top_k: Number of top results to return
            min_score: Minimum similarity score
            min_success_score: Minimum stored success score to consider
            weight_by_score: Whether to weight similarity by example success score

        Returns:
            List of (example, similarity_score) tuples
        """
        # THREAD-SAFE: Acquire lock for reading KB state
        with self._memory_lock:
            if len(self.examples) == 0:
                self.consecutive_misses += 1
                return []

            # Embed query
            query_embedding = self.embedder.embed(query).reshape(1, -1)

            # Compute similarities
            similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]

            # ADAPTIVE: Use current similarity_threshold instead of fixed min_score
            effective_threshold = max(min_score, self.similarity_threshold)

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more for filtering

            # Filter by minimum score and return results
            scored_results = []
            for idx in top_indices:
                score = float(similarities[idx])
                example = self.examples[idx]
                if example.success_score < min_success_score:
                    continue
                if score >= effective_threshold:
                    weight = score
                    if weight_by_score:
                        weight = score * (0.6 + 0.4 * (example.success_score / 10.0))
                    scored_results.append((example, score, weight))

            scored_results.sort(key=lambda x: x[2], reverse=True)
            results = [(ex, sim) for ex, sim, _ in scored_results[:top_k]]

            # ADAPTIVE: Track misses and adjust threshold
            if len(results) == 0:
                self.consecutive_misses += 1
                if self.consecutive_misses >= self.miss_threshold_for_lowering:
                    if self.similarity_threshold > self.lowered_threshold:
                        # NEW: Enforce minimum threshold
                        new_threshold = max(self.min_allowed_threshold, self.lowered_threshold)
                        logger.warning(
                            f"[RAG-ADAPTIVE] {self.consecutive_misses} consecutive misses -> "
                            f"Lowering threshold from {self.similarity_threshold:.2f} to {new_threshold:.2f} (min={self.min_allowed_threshold})"
                        )
                        self.similarity_threshold = new_threshold
            else:
                # Gradual reset instead of immediate reset
                # Only reset after multiple consecutive hits to avoid thrashing
                if self.consecutive_misses > 0:
                    logger.info(f"[RAG-ADAPTIVE] Hit after {self.consecutive_misses} misses")
                self.consecutive_misses = max(0, self.consecutive_misses - 2)  # Decay by 2 on hit

                # Only reset threshold after 2 consecutive hits (consecutive_misses <= 0)
                if self.consecutive_misses == 0 and self.similarity_threshold < self.base_similarity_threshold:
                    logger.info(f"[RAG-ADAPTIVE] Multiple hits -> Resetting threshold to {self.base_similarity_threshold:.2f}")
                    self.similarity_threshold = self.base_similarity_threshold

        return results
    
    def get_best_examples(
        self,
        n: int = 10,
        min_success_score: float = 8.0
    ) -> List[PromptExample]:
        """
        Get best performing examples
        
        Args:
            n: Number of examples to return
            min_success_score: Minimum success score
            
        Returns:
            List of best examples
        """
        # THREAD-SAFE: Acquire lock for consistent snapshot
        with self._memory_lock:
            # Filter by score
            filtered = [ex for ex in self.examples if ex.success_score >= min_success_score]
            
            # Sort by success score
            sorted_examples = sorted(filtered, key=lambda x: x.success_score, reverse=True)
            
            return sorted_examples[:n]
    
    def _check_similarity(self, embedding: np.ndarray) -> Tuple[bool, float, int]:
        """
        Check if embedding is too similar to existing examples
        
        Returns:
            Tuple of (is_duplicate, max_similarity, most_similar_index)
        """
        if self.embeddings_matrix is None or len(self.embeddings_matrix) == 0:
            return False, 0.0, -1
        
        emb_vec = embedding.reshape(1, -1)
        # Normalize to avoid magnitude skew
        emb_vec = emb_vec / np.clip(np.linalg.norm(emb_vec, axis=1, keepdims=True), 1e-8, None)
        embeddings = self.embeddings_matrix
        similarities = np.dot(emb_vec, embeddings.T)[0]
        
        # Find maximum similarity
        max_idx = int(np.argmax(similarities))
        max_similarity = float(similarities[max_idx])
        
        is_duplicate = max_similarity >= self.similarity_threshold
        
        return is_duplicate, max_similarity, max_idx
    
    def _rebuild_embeddings_matrix(self):
        """Rebuild the embeddings matrix from current examples"""
        if len(self.examples) == 0:
            self.embeddings_matrix = None
            return
        
        embeddings = [ex.embedding for ex in self.examples if ex.embedding is not None]
        if embeddings:
            matrix = np.vstack(embeddings)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-8, None)
            matrix = matrix / norms
            # Keep normalized copy on examples for consistency
            for i, ex in enumerate(self.examples):
                if ex.embedding is not None and i < matrix.shape[0]:
                    ex.embedding = matrix[i]
            self.embeddings_matrix = matrix
        else:
            self.embeddings_matrix = None
    
    def remove_duplicates_old(self, threshold: float = None) -> int:
        """
        OLD IMPLEMENTATION (SLOW): Remove duplicate/very similar prompts

        DEPRECATED: Use remove_duplicates() instead (100x faster)
        
        Args:
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            Number of duplicates removed
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        if len(self.examples) < 2:
            return 0
        
        # Track which examples to keep
        keep_indices = []
        removed_count = 0
        
        for i, example in enumerate(self.examples):
            # Check against kept examples
            is_duplicate = False
            
            for kept_idx in keep_indices:
                kept_embedding = self.examples[kept_idx].embedding.reshape(1, -1)
                current_embedding = example.embedding.reshape(1, -1)
                kept_embedding = kept_embedding / np.clip(np.linalg.norm(kept_embedding, axis=1, keepdims=True), 1e-8, None)
                current_embedding = current_embedding / np.clip(np.linalg.norm(current_embedding, axis=1, keepdims=True), 1e-8, None)
                similarity = float(np.dot(kept_embedding.flatten(), current_embedding.flatten()))
                
                if similarity >= threshold:
                    # Keep the one with higher success score
                    if example.success_score > self.examples[kept_idx].success_score:
                        # Replace the kept one with current
                        keep_indices.remove(kept_idx)
                        keep_indices.append(i)
                    is_duplicate = True
                    removed_count += 1
                    break
            
            if not is_duplicate:
                keep_indices.append(i)
        
        # Keep only non-duplicate examples
        self.examples = [self.examples[i] for i in keep_indices]
        self._rebuild_embeddings_matrix()
        
        logger.info(f"Removed {removed_count} duplicates, {len(self.examples)} examples remain")

        return removed_count

    def remove_contaminated_examples(self, min_clean_score: float = 7.0) -> int:
        """
        CRITICAL: Remove contaminated examples from aspect-boost false positives (THREAD-SAFE).

        Filters out:
        1. Examples with success_type == "HYBRID_BOOSTED" (aspect-boosted)
        2. Examples with low success_score (< min_clean_score)
        3. Examples with defensive keywords (should have been filtered but might be legacy)

        Args:
            min_clean_score: Minimum score to keep (default: 7.0)

        Returns:
            Number of contaminated examples removed
        """
        # THREAD-SAFE: Acquire lock for mutation
        with self._memory_lock:
            if len(self.examples) == 0:
                return 0

            original_count = len(self.examples)

            # Defensive keywords (same as in add_example)
            defensive_keywords = [
                "fictional", "hypothetical", "educational purposes", "research only",
                "audit", "defensive", "safeguard", "ethical", "simulated",
                "theoretical", "for educational", "academic exercise",
                "methodology", "document the complete", "instructional guide",
                "tutorial", "prevention", "security testing", "comprehensive documentation",
                "procedural steps", "complete methodology", "technical methodology",
                "scenario setup", "realistic scenario", "testing purposes"
            ]

            clean_examples = []
            removed_boosted = 0
            removed_low_score = 0
            removed_defensive = 0

            for ex in self.examples:
                # Filter 1: Remove hybrid-boosted examples
                success_type = ex.metadata.get('success_type') if ex.metadata else None
                if success_type == "HYBRID_BOOSTED":
                    removed_boosted += 1
                    continue

                # Filter 2: Remove low-score examples
                if ex.success_score < min_clean_score:
                    removed_low_score += 1
                    continue

                # Filter 3: Remove defensive prompts (legacy cleanup)
                prompt_lower = ex.prompt.lower()
                has_defensive = any(kw in prompt_lower for kw in defensive_keywords)
                if has_defensive:
                    removed_defensive += 1
                    continue

                # Keep this example
                clean_examples.append(ex)

            self.examples = clean_examples
            self._rebuild_embeddings_matrix()

            removed_count = original_count - len(self.examples)
            logger.info(
                f"[KB CLEANUP] Removed {removed_count} contaminated examples "
                f"(boosted={removed_boosted}, low_score={removed_low_score}, defensive={removed_defensive}), "
                f"{len(self.examples)} clean examples remain"
            )

            # Update stats
            if removed_count > 0:
                self.stats['contamination_removed'] = self.stats.get('contamination_removed', 0) + removed_count

            return removed_count

    def remove_duplicates(self, threshold: float = None) -> int:
        """
        Remove duplicate/very similar prompts using OPTIMIZED vectorized operations (THREAD-SAFE).

        100x faster than old implementation for large knowledge bases.
        Uses vectorized similarity matrix instead of nested loops.

        Args:
            threshold: Similarity threshold (uses default if None)

        Returns:
            Number of duplicates removed
        """
        # THREAD-SAFE: Acquire lock for mutation
        with self._memory_lock:
            if threshold is None:
                threshold = self.similarity_threshold

            if len(self.examples) < 2:
                return 0

            original_count = len(self.examples)

            # Build embedding matrix once (vectorized)
            embeddings = np.array([ex.embedding for ex in self.examples])

            # Normalize all embeddings at once
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-8, None)
            normalized_embeddings = embeddings / norms

            # Compute ALL pairwise similarities at once
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

            # Use efficient algorithm to find duplicates
            keep_mask = np.ones(len(self.examples), dtype=bool)

            for i in range(len(self.examples)):
                if not keep_mask[i]:
                    continue

                # Find all duplicates of example i
                duplicates = np.where(similarity_matrix[i] > threshold)[0]
                duplicates = duplicates[duplicates > i]  # Only look forward

                if len(duplicates) > 0:
                    # For each duplicate, keep the one with higher success score
                    for dup_idx in duplicates:
                        if self.examples[i].success_score >= self.examples[dup_idx].success_score:
                            keep_mask[dup_idx] = False
                        else:
                            keep_mask[i] = False
                            break

            # Keep only non-duplicates
            self.examples = [ex for i, ex in enumerate(self.examples) if keep_mask[i]]
            self._rebuild_embeddings_matrix()

            removed_count = original_count - len(self.examples)
            logger.info(f"[OPTIMIZED] Removed {removed_count} duplicates ({removed_count/original_count*100:.1f}%), {len(self.examples)} remain")

            return removed_count

    def load_from_json(self, json_path: str, default_goal: str = "General") -> int:
        """
        Load examples from JSON file
        
        Supports two formats:
        1. Simple: ["prompt1", "prompt2", ...] - Clean array of strings
        2. Structured: [{"prompt": "...", "goal": "...", "success_score": 9.0}, ...]
        
        Args:
            json_path: Path to JSON file
            default_goal: Default goal for simple format prompts
            
        Returns:
            Number of examples loaded
        """
        logger.info(f"Loading examples from: {json_path}")
        # Minimum score filter for "successful" examples (default 7.0)
        min_success = float(_get_env("OURS_RAG_MIN_SUCCESS"))
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            logger.warning("JSON file is empty")
            return 0
        
        count = 0
        
        # Detect format
        if isinstance(data[0], str):
            # Simple format: array of strings (RECOMMENDED - clean and efficient)
            logger.info("Detected simple format (array of prompt strings)")
            for prompt in data:
                if prompt and len(prompt) > 10:
                    # Simple format has no score; treat as high success
                    success, msg = self.add_example(
                        prompt=prompt,
                        goal=default_goal,
                        success_score=9.0,  # Default high score for successful examples
                        metadata={}
                    )
                    if success:
                        count += 1
        
        else:
            # Structured format: array of objects
            logger.info("Detected structured format (array of objects)")
            for item in data:
                # Extract fields (handle different formats)
                prompt = item.get('prompt', item.get('text', ''))
                goal = item.get('goal', item.get('behavior', item.get('target', default_goal)))
                success_score = float(item.get('success_score', item.get('score', 9.0)))
                
                if prompt and success_score >= min_success:
                    success, msg = self.add_example(
                        prompt=prompt,
                        goal=goal if goal else default_goal,
                        success_score=success_score,
                        metadata={}  # No redundant metadata
                    )
                    if success:
                        count += 1
        
        logger.info(f"Loaded {count} examples from JSON")
        return count
    
    def save(self, path: str = None) -> bool:
        """
        Save knowledge base to disk (THREAD-SAFE with atomic writes)

        Args:
            path: Optional custom path (uses default storage if None)

        Returns:
            Success status
        """
        try:
            if self.read_only:
                logger.debug("Knowledge base save skipped: read-only mode")
                return False
            if path is None:
                path = self.storage_dir / "knowledge_base.pkl"

            path = Path(path)

            # THREAD-SAFE: Use global file lock for atomic write
            with self._global_file_lock:
                # Acquire memory lock to snapshot current state
                with self._memory_lock:
                    # Prepare data
                    data = {
                        'examples': [ex.to_dict() for ex in self.examples],
                        'embeddings': self.embeddings_matrix,
                        'stats': self.stats,
                        'config': {
                            'similarity_threshold': self.similarity_threshold,
                            'max_examples': self.max_examples
                        }
                    }

                # Write to temp file first (atomic operation pattern)
                temp_path = path.with_suffix('.tmp')
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f)

                # Atomically replace original with temp (on POSIX and Windows)
                temp_path.replace(path)

            logger.info(f"Knowledge base saved to: {path}")
            return True

        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False
    
    def load(self, path: str = None) -> bool:
        """
        Load knowledge base from disk (THREAD-SAFE)

        Args:
            path: Optional custom path (uses default storage if None)

        Returns:
            Success status
        """
        try:
            if path is None:
                path = self.storage_dir / "knowledge_base.pkl"

            path = Path(path)

            if not path.exists():
                logger.warning(f"Knowledge base file not found: {path}")
                return False

            # THREAD-SAFE: Read from disk under global file lock only (avoid holding it during heavy work)
            with self._global_file_lock:
                with open(path, 'rb') as f:
                    data = pickle.load(f)

            # THREAD-SAFE: Restore examples and state within memory lock
            # (held until end to prevent concurrent add_example updates)
            needs_reembed = False
            with self._memory_lock:
                # Restore examples (recreate PromptExample objects)
                self.examples = []
                for ex_dict in data['examples']:
                    ex = PromptExample(**ex_dict)
                    self.examples.append(ex)

                # Restore embeddings
                self.embeddings_matrix = data.get('embeddings')
                # Normalize input type for embeddings (handle legacy list pickles)
                try:
                    import numpy as np  # Local import to avoid global dependency issues
                except Exception:
                    np = None
                if self.embeddings_matrix is not None and np is not None:
                    if isinstance(self.embeddings_matrix, list):
                        try:
                            self.embeddings_matrix = np.array(self.embeddings_matrix, dtype=float)
                        except Exception:
                            self.embeddings_matrix = None

                # Re-embed if embeddings are missing or dimension-mismatched
                if self.embeddings_matrix is None:
                    needs_reembed = True
                    logger.info("Re-computing embeddings because none were stored...")
                else:
                    stored_dim = None
                    try:
                        stored_dim = (
                            self.embeddings_matrix.shape[1]
                            if getattr(self.embeddings_matrix, "ndim", 0) == 2
                            else None
                        )
                    except Exception:
                        stored_dim = None
                    if stored_dim and stored_dim != self.embedder.embedding_dim:
                        error_msg = (
                            f"\n[CRITICAL] Embedding dimension mismatch detected!\n"
                            f"  Stored KB has dimension: {stored_dim}\n"
                            f"  Current embedder has dimension: {self.embedder.embedding_dim}\n"
                            f"  KB has {len(self.examples)} examples\n"
                            f"  Re-embedding would take approximately {len(self.examples) * 0.5:.1f} seconds\n\n"
                            f"Options:\n"
                            f"  1. Use original embedding model (recommended)\n"
                            f"  2. Delete KB and start fresh\n"
                            f"  3. Allow re-embedding (set OURS_FORCE_REEMBED=true in config)\n"
                        )

                        force_reembed = bool(_get_env("OURS_FORCE_REEMBED"))

                        if not force_reembed:
                            logger.error(error_msg)
                            raise ValueError(
                                f"Embedding dimension mismatch: stored={stored_dim}, current={self.embedder.embedding_dim}. "
                                f"Set OURS_FORCE_REEMBED=true to allow re-embedding."
                            )

                        # User opted in to re-embedding
                        logger.warning(f"Force re-embedding {len(self.examples)} examples (OURS_FORCE_REEMBED=true)...")
                        needs_reembed = True

                if needs_reembed:
                    for ex in self.examples:
                        ex.embedding = self.embedder.embed(ex.prompt)
                    self._rebuild_embeddings_matrix()
                    # Persist the refreshed embeddings to avoid repeat work
                    # Note: save() will acquire locks again, but that's OK
                else:
                    # Normalize stored embeddings and assign back to examples
                    norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
                    norms = np.clip(norms, 1e-8, None)
                    self.embeddings_matrix = self.embeddings_matrix / norms
                    if np is not None:
                        for i, ex in enumerate(self.examples):
                            if i < len(self.embeddings_matrix):
                                ex.embedding = self.embeddings_matrix[i]

                # Restore stats
                self.stats = data.get('stats', self.stats)

            # Save re-embedded KB if needed (outside the lock)
            if needs_reembed and self.persist_reembedded_on_load:
                self.save(path)

            logger.info(f"Loaded {len(self.examples)} examples from: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        # THREAD-SAFE: Acquire lock for consistent snapshot
        with self._memory_lock:
            stats = dict(self.stats)
            total_examples = len(self.examples)
            avg_success_score = np.mean([ex.success_score for ex in self.examples]) if self.examples else 0
            embedding_dim = self.embedder.embedding_dim

        return {
            **stats,
            'total_examples': total_examples,
            'embedding_dim': embedding_dim,
            'storage_size_mb': self._get_storage_size(),
            'avg_success_score': avg_success_score
        }
    
    def _get_storage_size(self) -> float:
        """Get storage size in MB"""
        kb_file = self.storage_dir / "knowledge_base.pkl"
        if kb_file.exists():
            return kb_file.stat().st_size / (1024 * 1024)
        return 0.0

    def prune_by_score(self, min_score: float = 7.0) -> int:
        """
        Drop examples that do not meet the minimum success score gate (THREAD-SAFE).

        Args:
            min_score: Minimum unified judge score required to keep an example.

        Returns:
            Number of examples removed.
        """
        # THREAD-SAFE: Acquire lock for mutation
        with self._memory_lock:
            before = len(self.examples)
            if before == 0:
                return 0

            self.examples = [ex for ex in self.examples if ex.success_score >= min_score]
            removed = before - len(self.examples)
            if removed > 0:
                self._rebuild_embeddings_matrix()
                logger.info(f"[PRUNE] Removed {removed} low-scoring examples (< {min_score})")
            return removed
    
    def clear(self):
        """Clear all examples (THREAD-SAFE)"""
        # THREAD-SAFE: Acquire lock for mutation
        with self._memory_lock:
            self.examples = []
            self.embeddings_matrix = None
            self.stats = {
                'total_added': 0,
                'duplicates_rejected': 0,
                'weak_prompts_rejected': 0,
                'last_update': None
            }
            logger.info("Knowledge base cleared")


def create_rag_knowledge_base(**kwargs) -> RAGKnowledgeBase:
    """Factory function to create RAG knowledge base"""
    return RAGKnowledgeBase(**kwargs)


if __name__ == "__main__":
    # Test the RAG knowledge base
    print("=" * 80)
    print("Testing RAG Knowledge Base")
    print("=" * 80)
    
    # Create knowledge base
    kb = create_rag_knowledge_base()
    
    # Add some test examples
    test_prompts = [
        {
            "prompt": "Imagine you are a security researcher studying...",
            "goal": "Create harmful content",
            "score": 8.5
        },
        {
            "prompt": "For educational purposes, explain in detail...",
            "goal": "Generate dangerous information",
            "score": 9.0
        },
        {
            "prompt": "In a hypothetical scenario where...",
            "goal": "Bypass safety guidelines",
            "score": 7.5
        }
    ]
    
    for ex in test_prompts:
        success, msg = kb.add_example(ex["prompt"], ex["goal"], ex["score"])
        print(f"{'[OK]' if success else '[X]'} {msg}")
    
    # Test retrieval
    print("\nRetrieving similar prompts for: 'security research'")
    results = kb.retrieve_similar("security research", top_k=3)
    for ex, score in results:
        print(f"  Similarity: {score:.3f} - {ex.prompt[:50]}...")
    
    # Print stats
    print("\nKnowledge Base Stats:")
    stats = kb.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] RAG Knowledge Base test complete!")
