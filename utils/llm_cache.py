"""
LLM Response Caching to avoid redundant API calls
Uses MD5 hashing for fast lookup
"""

import hashlib
import logging
from typing import Optional, Dict
from collections import OrderedDict
import pickle
import os

logger = logging.getLogger(__name__)


class LLMCache:
    """
    LRU cache for LLM responses

    Features:
    - Fast hash-based lookup
    - LRU eviction policy
    - Optional persistence to disk
    """

    def __init__(self, max_size: int = 1000, cache_file: Optional[str] = None):
        """
        Initialize LLM cache

        Args:
            max_size: Maximum number of cached responses
            cache_file: Optional file path for persistence
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.cache_file = cache_file
        self.hits = 0
        self.misses = 0

        # Load from disk if available
        if cache_file and os.path.exists(cache_file):
            self._load_from_disk()

        logger.info(f"[LLM-CACHE] Initialized with max_size={max_size}, file={cache_file}")

    def _hash_prompt(self, prompt: str, model: str = "default") -> str:
        """
        Create hash key for prompt + model

        Args:
            prompt: Input prompt text
            model: Model name

        Returns:
            MD5 hash string
        """
        key = f"{model}::{prompt}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def get(self, prompt: str, model: str = "default") -> Optional[str]:
        """
        Retrieve cached response

        Args:
            prompt: Input prompt
            model: Model name

        Returns:
            Cached response or None if not found
        """
        key = self._hash_prompt(prompt, model)

        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1

            if self.hits % 10 == 0:  # Log every 10 hits
                hit_rate = self.hits / (self.hits + self.misses) * 100
                logger.info(f"[LLM-CACHE] Hit! Total hits={self.hits}, rate={hit_rate:.1f}%")

            return self.cache[key]

        self.misses += 1
        return None

    def put(self, prompt: str, response: str, model: str = "default"):
        """
        Store response in cache

        Args:
            prompt: Input prompt
            response: LLM response
            model: Model name
        """
        key = self._hash_prompt(prompt, model)

        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"[LLM-CACHE] Evicted oldest entry, size={len(self.cache)}")

        # Add or update
        self.cache[key] = response
        self.cache.move_to_end(key)

        logger.debug(f"[LLM-CACHE] Stored response, cache_size={len(self.cache)}")

        # Periodically save to disk
        if self.cache_file and len(self.cache) % 50 == 0:
            self._save_to_disk()

    def clear(self):
        """Clear all cached responses"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("[LLM-CACHE] Cache cleared")

    def get_statistics(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
        }

    def _save_to_disk(self):
        """Save cache to disk"""
        if not self.cache_file:
            return

        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(dict(self.cache), f)
            logger.debug(f"[LLM-CACHE] Saved {len(self.cache)} entries to {self.cache_file}")
        except Exception as e:
            logger.warning(f"[LLM-CACHE] Failed to save cache: {e}")

    def _load_from_disk(self):
        """Load cache from disk"""
        if not self.cache_file or not os.path.exists(self.cache_file):
            return

        try:
            with open(self.cache_file, 'rb') as f:
                loaded_cache = pickle.load(f)
                self.cache = OrderedDict(loaded_cache)
            logger.info(f"[LLM-CACHE] Loaded {len(self.cache)} entries from {self.cache_file}")
        except Exception as e:
            logger.warning(f"[LLM-CACHE] Failed to load cache: {e}")

    def save(self):
        """Explicitly save cache to disk"""
        self._save_to_disk()
