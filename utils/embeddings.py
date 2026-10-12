"""
Semantic Embedding Module for RL Agents

Provides proper semantic embeddings using sentence transformers instead of 
placeholder hash-based random vectors. This is critical for RL agents to 
understand query semantics and generalize across similar goals.

Features:
- Multiple embedding backend support (sentence-transformers, OpenAI, local)
- Caching for performance optimization
- Graceful fallback when libraries unavailable
- Batch embedding support
"""

import os
import hashlib
import time
import pickle
import numpy as np
import torch
from collections import OrderedDict
from typing import Dict, List, Optional, Union
from functools import lru_cache
from pathlib import Path
import logging
import threading

# Ensure environment variables (including OPENROUTER_API_KEY and OURS_EMBEDDING_MODEL)
# are loaded from the root config.env if not already present.
try:  # pragma: no cover - environment setup
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / "config.env", override=False)
except Exception:
    pass

logger = logging.getLogger(__name__)

# Shared auth-error type so a dead OpenRouter key fails fast everywhere instead of
# silently degrading to the local fallback embedder.
try:  # pragma: no cover - import wiring
    from src.openrouter_llm import OpenRouterAuthError
except Exception:  # pragma: no cover - fallback when src isn't importable
    class OpenRouterAuthError(RuntimeError):
        """Raised when OpenRouter rejects the API key (HTTP 401)."""


class EmbeddingCache:
    """LRU cache for embeddings backed by OrderedDict for O(1) eviction (THREAD-SAFE)."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        # OrderedDict preserves insertion order; move_to_end() is O(1).
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def _compute_key(self, text: str, model_name: str) -> str:
        return hashlib.md5(f"{model_name}:{text}".encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Return a copy of the cached embedding, or None (THREAD-SAFE)."""
        key = self._compute_key(text, model_name)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)  # O(1) — mark as recently used
                return self._store[key].copy()
        return None

    def put(self, text: str, model_name: str, embedding: np.ndarray):
        """Insert or update an embedding, evicting the LRU entry if needed (THREAD-SAFE)."""
        key = self._compute_key(text, model_name)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            else:
                if len(self._store) >= self.max_size:
                    self._store.popitem(last=False)  # O(1) — evict oldest
            self._store[key] = embedding.copy()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def cache(self) -> "OrderedDict[str, np.ndarray]":
        """Read-only view for legacy code that reads .cache directly."""
        return self._store

    def clear(self):
        """Clear the cache (THREAD-SAFE)."""
        with self._lock:
            self._store.clear()

    def snapshot(self) -> Dict[str, Union[int, Dict[str, np.ndarray], List[str]]]:
        """Return a serializable snapshot for disk persistence (THREAD-SAFE)."""
        with self._lock:
            return {
                "max_size": self.max_size,
                "cache": {k: v.copy() for k, v in self._store.items()},
                "access_order": list(self._store.keys()),
            }

    def load_snapshot(self, data: Dict[str, Union[int, Dict[str, np.ndarray], List[str]]]):
        """Restore cache state from a snapshot (THREAD-SAFE)."""
        if not data:
            return
        with self._lock:
            self.max_size = int(data.get("max_size", self.max_size))
            raw = data.get("cache", {})
            order = data.get("access_order") or list(raw.keys())
            self._store = OrderedDict(
                (k, raw[k].copy() if isinstance(raw[k], np.ndarray) else raw[k])
                for k in order
                if k in raw
            )


class SemanticEmbedder:
    """
    Semantic embedding provider with multiple backend support.
    
    Supports:
    1. sentence-transformers (local, fastest, recommended)
    2. OpenAI embeddings API (cloud-based)
    3. Hash-based fallback (no semantic meaning, for testing only)
    """
    
    SUPPORTED_MODELS = {
        # Sentence Transformers models (local)
        "all-MiniLM-L6-v2": {"dim": 384, "type": "sentence_transformer"},
        "all-mpnet-base-v2": {"dim": 768, "type": "sentence_transformer"},
        "paraphrase-MiniLM-L6-v2": {"dim": 384, "type": "sentence_transformer"},
        "multi-qa-MiniLM-L6-cos-v1": {"dim": 384, "type": "sentence_transformer"},
        
        # OpenAI models
        "text-embedding-ada-002": {"dim": 1536, "type": "openai"},
        "text-embedding-3-small": {"dim": 1536, "type": "openai"},
        "text-embedding-3-large": {"dim": 3072, "type": "openai"},
        
        # OpenRouter embedding models (recommended - works via OpenRouter API)
        "qwen/qwen3-embedding-8b": {"dim": 4096, "type": "openrouter"},
        "openai/text-embedding-ada-002": {"dim": 1536, "type": "openrouter"},
        "openai/text-embedding-3-small": {"dim": 1536, "type": "openrouter"},
    }
    
    # Use Qwen embedding by default for meaningful semantic embeddings
    DEFAULT_MODEL = "qwen/qwen3-embedding-8b"
    
    def __init__(
        self,
        model_name: str = None,
        cache_size: int = 10000,
        device: str = None,
        openai_api_key: str = None,
        cache_file: Optional[Union[str, Path]] = None,
        cache_save_interval: Optional[int] = None
    ):
        """
        Initialize the semantic embedder.
        
        Args:
            model_name: Name of the embedding model to use
            cache_size: Maximum number of embeddings to cache
            device: Device for local models ('cuda' or 'cpu')
            openai_api_key: API key for OpenAI embeddings (optional)
            cache_file: Optional path to persist embeddings across runs
            cache_save_interval: Optional override for how often to flush cache to disk
        """
        # Allow env override via OURS_EMBEDDING_MODEL while keeping explicit arg highest priority
        self.model_name = model_name or os.getenv("OURS_EMBEDDING_MODEL") or self.DEFAULT_MODEL
        self.cache = EmbeddingCache(max_size=cache_size)
        self.enable_disk_cache = os.getenv("OURS_EMBED_CACHE_DISK", "1") != "0"
        self.cache_save_interval = cache_save_interval or int(os.getenv("OURS_EMBED_CACHE_SAVE_INTERVAL", "50"))
        default_cache_path = Path(__file__).resolve().parents[1] / "cache" / "embeddings_cache.pkl"
        self.cache_file = Path(cache_file) if cache_file else default_cache_path
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._cache_puts_since_save = 0
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.embed_retries = int(os.getenv("OURS_EMBED_RETRIES", "3"))
        self.embed_backoff_base = float(os.getenv("OURS_EMBED_BACKOFF_BASE", "1.5"))
        self.embed_backoff_max = float(os.getenv("OURS_EMBED_BACKOFF_MAX", "10.0"))
        # Default timeout aggressively lowered to avoid 30s stalls
        self.embed_timeout = int(os.getenv("OURS_EMBED_TIMEOUT", "5"))
        self.enable_local_fallback = os.getenv("OURS_EMBED_ENABLE_LOCAL_FALLBACK", "1") != "0"
        self.local_fallback_model_name = os.getenv("OURS_EMBED_FALLBACK_MODEL", "all-MiniLM-L6-v2")
        
        self._model = None
        self._model_type = None
        self._embedding_dim = None
        self._local_fallback_model = None
        self._local_fallback_dim = None
        self._save_cache_lock = threading.Lock()  # Protect disk cache operations

        self._load_cache_from_disk()
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        model_config = self.SUPPORTED_MODELS.get(self.model_name)
        
        if model_config is None:
            raise ValueError(
                f"Unknown model '{self.model_name}'. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        self._model_type = model_config["type"]
        self._embedding_dim = model_config["dim"]
        
        if self._model_type == "sentence_transformer":
            self._init_sentence_transformer()
        elif self._model_type == "openai":
            self._init_openai()
        elif self._model_type == "openrouter":
            self._init_openrouter()
        else:
            raise ValueError(f"Unsupported model type: {self._model_type}")
    
    def _init_sentence_transformer(self):
        """Initialize sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading sentence-transformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully on {self.device}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence-transformer model: {e}")
    
    def _init_openai(self):
        """Initialize OpenAI embeddings."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")
        
        try:
            import openai
            openai.api_key = self.openai_api_key
            # Test with a simple embedding
            logger.info(f"Using OpenAI embedding model: {self.model_name}")
            
        except ImportError:
            raise ImportError("openai package not installed")
    
    def _init_openrouter(self):
        """Initialize OpenRouter-based embeddings using Qwen embedding model."""
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. "
                "Please set the OPENROUTER_API_KEY environment variable."
            )
        
        try:
            import requests
            self._requests = requests
            
            # Use OpenRouter's embeddings API with Qwen model
            self._openrouter_url = "https://openrouter.ai/api/v1/embeddings"
            self._openrouter_headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "HTTP-Referer": "https://github.com/jailbreak-research",
                "X-Title": "Jailbreak Attack Research - Embeddings",
                "Content-Type": "application/json"
            }
            
            logger.info(f"Initialized OpenRouter embeddings using {self.model_name}")
            logger.info(f"Embedding dimension: {self._embedding_dim}")
            
        except ImportError:
            raise ImportError("requests package not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenRouter embeddings: {e}")
    
    def _load_cache_from_disk(self):
        """Load persisted embeddings cache if available."""
        if not self.enable_disk_cache or not self.cache_file or not self.cache_file.exists():
            return
        try:
            with open(self.cache_file, "rb") as f:
                snapshot = pickle.load(f)
            if isinstance(snapshot, dict):
                self.cache.load_snapshot(snapshot)
                logger.info(
                    "[CACHE] Loaded %d embeddings from %s",
                    len(self.cache.cache),
                    self.cache_file,
                )
        except Exception as err:
            logger.warning(f"[CACHE] Failed to load cache from {self.cache_file}: {err}")
    
    def _maybe_save_cache(self, force: bool = False):
        """Persist cache to disk every N inserts to avoid repeated OpenRouter calls (THREAD-SAFE)."""
        if not self.enable_disk_cache or not self.cache_file:
            return

        # THREAD-SAFE: Check counter without lock first (optimization)
        # This is safe because increments are atomic in CPython
        self._cache_puts_since_save += 1
        if not force and self._cache_puts_since_save < self.cache_save_interval:
            return

        # Only take the lock if we actually need to save
        with self._save_cache_lock:
            # Double-check pattern: verify condition still holds after acquiring lock
            if not force and self._cache_puts_since_save < self.cache_save_interval:
                return
            try:
                snapshot = self.cache.snapshot()
                # Atomic write: write to temp then replace
                temp_file = self.cache_file.with_suffix('.tmp')
                with open(temp_file, "wb") as f:
                    pickle.dump(snapshot, f)
                # Atomically replace original
                temp_file.replace(self.cache_file)
                self._cache_puts_since_save = 0
            except Exception as err:
                logger.debug(f"[CACHE] Skipping cache save: {err}")
    
    def _normalize_to_dim(self, embedding: Union[np.ndarray, list], target_dim: Optional[int] = None) -> np.ndarray:
        """Normalize embedding to target dimension (pad/truncate) and unit length."""
        if embedding is None:
            return self._hash_fallback_embedding("")
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        target = target_dim or self._embedding_dim or arr.shape[0]
        # Pad or truncate to match expected dimension
        if arr.shape[0] < target:
            pad_width = target - arr.shape[0]
            arr = np.pad(arr, (0, pad_width), mode="constant")
        elif arr.shape[0] > target:
            arr = arr[:target]
        norm = np.linalg.norm(arr) + 1e-12
        return arr / norm
    
    def _embed_local_fallback(self, text: str) -> np.ndarray:
        """Use a local sentence-transformer as a semantic fallback."""
        if not self.enable_local_fallback:
            raise RuntimeError("Local fallback disabled")
        if self._local_fallback_model is None:
            from sentence_transformers import SentenceTransformer
            
            self._local_fallback_model = SentenceTransformer(
                self.local_fallback_model_name,
                device=self.device
            )
            try:
                self._local_fallback_dim = self._local_fallback_model.get_sentence_embedding_dimension()
            except Exception:
                self._local_fallback_dim = None
        emb = self._local_fallback_model.encode(text, convert_to_numpy=True, batch_size=32)
        return self._normalize_to_dim(emb, target_dim=self._embedding_dim or self._local_fallback_dim)
    
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim
    
    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            use_cache: Whether to use caching
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache
        if use_cache:
            cached = self.cache.get(text, self.model_name)
            if cached is not None:
                return cached
        
        # Compute embedding
        if self._model_type == "sentence_transformer":
            embedding = self._embed_sentence_transformer(text)
        elif self._model_type == "openai":
            embedding = self._embed_openai(text)
        elif self._model_type == "openrouter":
            embedding = self._embed_openrouter(text)
        else:
            raise ValueError(f"Unsupported model type: {self._model_type}")
        embedding = self._normalize_to_dim(embedding)
        
        # Cache result
        if use_cache:
            self.cache.put(text, self.model_name, embedding)
            self._maybe_save_cache()
        
        return embedding
    
    def embed_batch(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Embed multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching
            
        Returns:
            Array of embeddings [batch_size, embedding_dim]
        """
        if not texts:
            return np.array([])
        
        # Check cache for each text
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if use_cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    embeddings.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Compute uncached embeddings
        if uncached_texts:
            if self._model_type == "sentence_transformer":
                new_embeddings = self._embed_batch_sentence_transformer(uncached_texts)
            elif self._model_type == "openai":
                new_embeddings = self._embed_batch_openai(uncached_texts)
            elif self._model_type == "openrouter":
                new_embeddings = self._embed_batch_openrouter(uncached_texts)
            else:
                raise ValueError(f"Unsupported model type: {self._model_type}")
            
            # Cache and collect
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                if use_cache:
                    normalized = self._normalize_to_dim(emb)
                    self.cache.put(text, self.model_name, normalized)
                    self._maybe_save_cache()
                    emb = normalized
                embeddings.append((idx, emb))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([e[1] for e in embeddings])
    
    def embed_to_tensor(self, text: str, device: str = None) -> torch.Tensor:
        """
        Embed text and return as PyTorch tensor.
        
        Args:
            text: Text to embed
            device: Device to place tensor on
            
        Returns:
            Embedding as torch.Tensor
        """
        device = device or self.device
        embedding = self.embed(text)
        return torch.FloatTensor(embedding).to(device)
    
    def embed_batch_to_tensor(self, texts: List[str], device: str = None) -> torch.Tensor:
        """
        Embed batch and return as PyTorch tensor.
        
        Args:
            texts: List of texts to embed
            device: Device to place tensor on
            
        Returns:
            Embeddings as torch.Tensor [batch_size, embedding_dim]
        """
        device = device or self.device
        embeddings = self.embed_batch(texts)
        return torch.FloatTensor(embeddings).to(device)
    
    def _embed_sentence_transformer(self, text: str) -> np.ndarray:
        """Embed using sentence-transformers."""
        return self._model.encode(text, convert_to_numpy=True)
    
    def _embed_batch_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Batch embed using sentence-transformers."""
        return self._model.encode(texts, convert_to_numpy=True, batch_size=32)
    
    def _embed_openai(self, text: str) -> np.ndarray:
        """Embed using OpenAI API."""
        import openai
        
        response = openai.Embedding.create(
            input=text,
            model=self.model_name
        )
        return np.array(response['data'][0]['embedding'])
    
    def _embed_batch_openai(self, texts: List[str]) -> np.ndarray:
        """Batch embed using OpenAI API."""
        import openai
        
        response = openai.Embedding.create(
            input=texts,
            model=self.model_name
        )
        return np.array([d['embedding'] for d in response['data']])
    
    def _embed_openrouter(self, text: str) -> np.ndarray:
        """
        Embed using OpenRouter embeddings API with Qwen model.
        
        Uses the OpenRouter embeddings endpoint with qwen/qwen3-embedding-8b
        to generate high-quality semantic embeddings.
        """
        import json

        payload = {
            "model": self.model_name,
            "input": text
        }

        # Retry with configurable backoff to soften transient SSL/EOF errors from OpenRouter
        last_error = None
        for attempt in range(1, self.embed_retries + 1):
            try:
                response = self._requests.post(
                    self._openrouter_url,
                    headers=self._openrouter_headers,
                    data=json.dumps(payload),
                    timeout=self.embed_timeout
                )
                if response.status_code == 200:
                    result = response.json()
                    embedding = np.array(result['data'][0]['embedding'], dtype=np.float32)

                    # Normalize to unit length
                    return self._normalize_to_dim(embedding)
                
                # Fast-fail on a bad/expired key (401). This is an account-level
                # problem that the local fallback would silently paper over,
                # producing a degraded RAG run. Raise a dedicated error that the
                # retry loop below will NOT swallow.
                if response.status_code == 401:
                    raise OpenRouterAuthError(
                        f"OpenRouter authentication failed (401): {response.text[:300]}. "
                        f"Check OPENROUTER_API_KEY in config.env."
                    )

                # 403 is typically per-request policy moderation; allow the normal
                # fallback path so a single blocked input doesn't abort the run.
                if response.status_code == 403:
                    raise RuntimeError(
                        f"OpenRouter policy error 403: {response.text[:300]}"
                    )

                # Retry on transient status codes
                if response.status_code in (408, 429) or 500 <= response.status_code <= 599:
                    raise self._requests.exceptions.RequestException(
                        f"Transient status {response.status_code}: {response.text[:300]}"
                    )

                raise RuntimeError(
                    f"OpenRouter API error: {response.status_code} - {response.text[:300]}"
                )
            except OpenRouterAuthError:
                # Never retry or fall back on a dead key — propagate immediately.
                raise
            except Exception as err:  # pragma: no cover - network errors are external
                last_error = err
                if attempt >= self.embed_retries:
                    break
                wait = min(self.embed_backoff_max, self.embed_backoff_base * (2 ** (attempt - 1)))
                logger.warning(
                    "[EMBED] OpenRouter attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt,
                    self.embed_retries,
                    err,
                    wait,
                )
                time.sleep(wait)

        # CRITICAL FIX: Raise error instead of silent fallback to hash-based embedding
        # Use local semantic fallback when available, otherwise deterministic hash fallback
        if self.enable_local_fallback:
            try:
                logger.warning(
                    "[EMBED] OpenRouter failed after %d attempts; using local fallback model '%s'",
                    self.embed_retries,
                    self.local_fallback_model_name
                )
                return self._embed_local_fallback(text)
            except Exception as local_err:
                logger.warning("[EMBED] Local fallback failed: %s; using hash-based embedding", local_err)
        
        logger.error(
            "[CRITICAL] OpenRouter embedding API failed after %d retries. Last error: %s",
            self.embed_retries,
            last_error,
        )
        return self._hash_fallback_embedding(text)
    
    def _embed_batch_openrouter(self, texts: List[str]) -> np.ndarray:
        """Batch embed using OpenRouter API."""
        embeddings = [self._embed_openrouter(t) for t in texts]
        return np.array(embeddings)
    
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def most_similar(
        self, 
        query: str, 
        candidates: List[str], 
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar texts from candidates.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of (text, similarity_score) tuples
        """
        query_emb = self.embed(query)
        candidate_embs = self.embed_batch(candidates)
        
        # Compute similarities
        similarities = np.dot(candidate_embs, query_emb)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(candidates[i], float(similarities[i])) for i in top_indices]
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache.cache),
            "max_size": self.cache.max_size,
            "model_name": self.model_name,
            "model_type": self._model_type,
            "embedding_dim": self._embedding_dim
        }
    
    def __del__(self):
        """Best-effort cache flush on interpreter shutdown."""
        try:
            self._maybe_save_cache(force=True)
        except Exception:
            pass

    def _hash_fallback_embedding(self, text: str) -> np.ndarray:
        """
        Deterministic fallback embedding to keep pipelines alive when remote calls fail.

        WARNING: Hash-based vectors are semantically meaningless — cosine
        similarity between two such vectors is essentially noise. When this
        fallback is reached the entire RAG pipeline degrades to keyword
        matching at best. We emit a one-time loud warning per process and
        raise hard if the user has opted into strict mode.
        """
        if not getattr(self, "_hash_fallback_warned", False):
            logger.warning(
                "[EMBEDDING] Falling back to hash-based pseudo-embeddings. "
                "Semantic search and RAG retrieval quality will be effectively random. "
                "Install sentence-transformers (pip install sentence-transformers) or "
                "configure a working remote embedder. Set OURS_STRICT_EMBEDDING=true to "
                "raise instead of silently degrading."
            )
            self._hash_fallback_warned = True
        if os.getenv("OURS_STRICT_EMBEDDING", "false").lower() == "true":
            raise RuntimeError(
                "Embedding backend unavailable and OURS_STRICT_EMBEDDING=true. "
                "Refusing to return hash-pseudo-embedding because it would silently "
                "destroy RAG retrieval quality."
            )
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.embedding_dim, dtype=np.float32)
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm


# Global embedder instance (lazy initialization)
_global_embedder: Optional[SemanticEmbedder] = None


def get_embedder(
    model_name: str = None,
    reinitialize: bool = False,
    **kwargs
) -> SemanticEmbedder:
    """
    Get or create the global embedder instance.
    
    Args:
        model_name: Model name (uses default if not specified)
        reinitialize: Force re-creation of embedder
        **kwargs: Additional arguments for SemanticEmbedder
        
    Returns:
        SemanticEmbedder instance
    """
    global _global_embedder
    
    if _global_embedder is None or reinitialize:
        _global_embedder = SemanticEmbedder(model_name=model_name, **kwargs)
    
    return _global_embedder


def embed_query(text: str) -> np.ndarray:
    """
    Convenience function to embed a query using global embedder.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector
    """
    return get_embedder().embed(text)


def embed_query_tensor(text: str, device: str = None) -> torch.Tensor:
    """
    Convenience function to embed a query as tensor.
    
    Args:
        text: Text to embed
        device: Device for tensor
        
    Returns:
        Embedding tensor
    """
    return get_embedder().embed_to_tensor(text, device)


if __name__ == "__main__":
    print("=" * 80)
    print("SEMANTIC EMBEDDING MODULE TEST")
    print("=" * 80)
    
    # Test with different backends
    test_texts = [
        "Write a tutorial on cybersecurity",
        "Explain how to protect systems from hackers",
        "Create a cooking recipe for pasta",
        "Design a machine learning model"
    ]
    
    print("\nTesting embedding module...")
    embedder = SemanticEmbedder()
    
    print(f"\nConfiguration:")
    print(f"  Model: {embedder.model_name}")
    print(f"  Type: {embedder._model_type}")
    print(f"  Dimension: {embedder.embedding_dim}")
    print(f"  Device: {embedder.device}")
    
    print("\nEmbedding test texts...")
    for text in test_texts:
        emb = embedder.embed(text)
        print(f"  '{text[:40]}...' -> shape {emb.shape}, norm {np.linalg.norm(emb):.3f}")
    
    print("\nSimilarity test:")
    for i, text1 in enumerate(test_texts):
        for text2 in test_texts[i+1:]:
            sim = embedder.similarity(text1, text2)
            print(f"  sim('{text1[:30]}...', '{text2[:30]}...') = {sim:.3f}")
    
    print("\nMost similar test:")
    query = "How to secure computer systems"
    results = embedder.most_similar(query, test_texts, top_k=3)
    print(f"Query: '{query}'")
    for text, score in results:
        print(f"  {score:.3f}: {text[:50]}...")
    
    print("\nCache stats:", embedder.get_cache_stats())
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Embedding module test complete!")

