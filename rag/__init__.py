"""
RAG (Retrieval-Augmented Generation) System for Attack Prompts

This module provides a complete RAG system for storing, retrieving, and learning
from successful attack prompts to improve the agent's performance.

Components:
- RAGKnowledgeBase: Vector-based knowledge storage with deduplication
- RAGEnhancedPromptSelector: Smart prompt selection using RAG
- RAGAugmentedAgent: Agent wrapper with RAG capabilities

Usage:
    from rag import create_rag_enhanced_agent, RAGKnowledgeBase
    
    # Create knowledge base
    kb = RAGKnowledgeBase()
    kb.load_from_json("/path/to/private/examples.json")
    
    # Wrap your agent
    rag_agent = create_rag_enhanced_agent(base_agent, kb)
    
    # Use with dynamic learning
    rag_agent.update_from_result(prompt, goal, score)
"""

from .rag_knowledge_base import (
    RAGKnowledgeBase,
    PromptExample,
    create_rag_knowledge_base
)

from .rag_agent_integration import (
    RAGEnhancedPromptSelector,
    RAGAugmentedAgent,
    create_rag_enhanced_agent
)

__all__ = [
    'RAGKnowledgeBase',
    'PromptExample',
    'create_rag_knowledge_base',
    'RAGEnhancedPromptSelector',
    'RAGAugmentedAgent',
    'create_rag_enhanced_agent',
]

__version__ = '1.0.0'