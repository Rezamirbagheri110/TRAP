"""
Maximal Marginal Relevance (MMR) for RAG diversity
Balances relevance and diversity in retrieval
"""

import numpy as np
import logging
from typing import List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class MMRRetriever:
    """
    MMR (Maximal Marginal Relevance) retriever for diverse RAG results

    MMR Score = λ * relevance(query, doc) - (1-λ) * max_similarity(doc, selected_docs)

    Features:
    - Balances relevance to query with diversity from already selected documents
    - Prevents redundant retrievals
    - Configurable trade-off parameter λ
    """

    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize MMR retriever

        Args:
            lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
                         0.7 = 70% relevance, 30% diversity (recommended)
        """
        self.lambda_param = lambda_param
        logger.info(f"[MMR] Initialized with lambda={lambda_param}")

    def retrieve(self,
                 query_embedding: np.ndarray,
                 candidate_embeddings: List[np.ndarray],
                 candidate_items: List[Any],
                 top_k: int = 5,
                 initial_candidates: int = 20) -> List[Any]:
        """
        Retrieve top-k diverse items using MMR

        Args:
            query_embedding: Query vector
            candidate_embeddings: List of candidate document embeddings
            candidate_items: List of candidate items (prompts, etc.)
            top_k: Number of items to return
            initial_candidates: Number of candidates to consider (for efficiency)

        Returns:
            List of top-k diverse items
        """
        if len(candidate_items) == 0:
            return []

        if len(candidate_items) <= top_k:
            return candidate_items

        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        candidate_norms = [
            emb / (np.linalg.norm(emb) + 1e-8)
            for emb in candidate_embeddings
        ]

        # Compute relevance scores (cosine similarity to query)
        relevance_scores = [
            np.dot(query_norm, cand_norm)
            for cand_norm in candidate_norms
        ]

        # Get top initial_candidates by relevance
        top_indices = np.argsort(relevance_scores)[-initial_candidates:][::-1]

        # MMR selection
        selected_indices = []
        remaining_indices = list(top_indices)

        # Select first item (highest relevance)
        first_idx = remaining_indices[0]
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Iteratively select remaining items
        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]

                # Diversity component (maximum similarity to already selected)
                max_sim = max(
                    np.dot(candidate_norms[idx], candidate_norms[sel_idx])
                    for sel_idx in selected_indices
                )

                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr_score))

            # Select item with highest MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

            logger.debug(
                f"[MMR] Selected item {len(selected_indices)}/{top_k}, "
                f"mmr_score={best_score:.3f}, relevance={relevance_scores[best_idx]:.3f}"
            )

        # Return selected items
        selected_items = [candidate_items[idx] for idx in selected_indices]

        logger.info(f"[MMR] Retrieved {len(selected_items)} diverse items from {len(candidate_items)} candidates")

        return selected_items

    def compute_diversity_score(self, embeddings: List[np.ndarray]) -> float:
        """
        Compute average pairwise diversity of selected embeddings

        Args:
            embeddings: List of embeddings

        Returns:
            Average diversity score (0-1, higher = more diverse)
        """
        if len(embeddings) < 2:
            return 1.0

        # Normalize embeddings
        norms = [emb / (np.linalg.norm(emb) + 1e-8) for emb in embeddings]

        # Compute all pairwise similarities
        similarities = []
        for i in range(len(norms)):
            for j in range(i + 1, len(norms)):
                sim = np.dot(norms[i], norms[j])
                similarities.append(sim)

        # Diversity = 1 - average similarity
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity

        return diversity


def rerank_with_mmr(query_embedding: np.ndarray,
                    candidates: List[Tuple[Any, np.ndarray, float]],
                    top_k: int = 5,
                    lambda_param: float = 0.7) -> List[Any]:
    """
    Convenience function to rerank candidates with MMR

    Args:
        query_embedding: Query vector
        candidates: List of (item, embedding, score) tuples
        top_k: Number of items to return
        lambda_param: MMR lambda parameter

    Returns:
        List of top-k reranked items
    """
    if not candidates:
        return []

    items = [c[0] for c in candidates]
    embeddings = [c[1] for c in candidates]

    retriever = MMRRetriever(lambda_param=lambda_param)
    return retriever.retrieve(query_embedding, embeddings, items, top_k=top_k)
