"""Production-Grade Hashing and Multi-Tiered Caching Engine for LLM & RAG APIs.

Features:
1. Deterministic Prompt Canonicalization & Composite SHA-256 Key Derivation (CacheKeyBuilder)
2. Thread-Safe In-Memory LRU Cache with TTL Expiration & Observability Metrics (LRUTTLCache / InMemoryCache)
3. Distributed Redis Cache Support with Graceful Fallback (RedisCache)
4. Multi-Tiered L1/L2 Hybrid Cache (TieredCache)
5. Semantic Similarity Cache (SemanticCache)
"""

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("production_api.cache")


# =====================================================================
# 1. Hashing & Deterministic Key Builder
# =====================================================================

class CacheKeyBuilder:
    """Builds deterministic, collision-resistant composite cache keys for LLM queries."""

    @staticmethod
    def normalize_prompt(prompt: str) -> str:
        """Canonicalize user prompt: normalize whitespace, lowercase, and trim punctuation."""
        if not prompt:
            return ""
        # Collapse multiple whitespaces/tabs/newlines to single space
        normalized = re.sub(r"\s+", " ", prompt.strip()).lower()
        # Strip trailing non-semantic punctuation
        normalized = re.sub(r"[?!.,;:\s]+$", "", normalized)
        return normalized

    @classmethod
    def build_key(
        cls,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
        system_prompt: str = "",
        tenant_id: str = "default",
        prefix: str = "llm:cache",
    ) -> str:
        """Derive composite SHA-256 hash key from query parameters."""
        canonical_prompt = cls.normalize_prompt(prompt)
        sys_hash = hashlib.sha256(system_prompt.strip().encode("utf-8")).hexdigest()[:8] if system_prompt else "none"

        composite_payload = {
            "tenant": tenant_id,
            "model": model.lower(),
            "temp": round(temperature, 2),
            "sys_hash": sys_hash,
            "prompt": canonical_prompt,
        }

        serialized = json.dumps(composite_payload, sort_keys=True)
        hash_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"{prefix}:{hash_digest}"


# =====================================================================
# 2. Cache Models & Metrics Tracker
# =====================================================================

@dataclass
class CacheEntry:
    """Represents a cached response with lifecycle timestamps and hit count."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hits: int = 0

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if entry has passed its TTL expiration timestamp."""
        if self.expires_at is None:
            return False
        current_time = now if now is not None else time.time()
        return current_time > self.expires_at


@dataclass
class CacheMetrics:
    """Tracks cache performance and hit rates for observability."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate_percent(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return round((self.hits / self.total_requests) * 100.0, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate_percent": self.hit_rate_percent,
        }


# =====================================================================
# 3. Thread-Safe In-Memory LRU Cache with TTL
# =====================================================================

class LRUTTLCache:
    """Thread-safe in-memory cache supporting Least Recently Used (LRU) eviction, TTL, and metrics."""

    def __init__(self, maxsize: int = 1000, default_ttl: Optional[int] = 3600):
        self.maxsize = maxsize
        self.default_ttl = default_ttl
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self.metrics = CacheMetrics()

    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Retrieve cached value by key. Returns default on cache miss or expiration."""
        with self._lock:
            if key not in self._store:
                self.metrics.misses += 1
                return default

            entry = self._store[key]
            now = time.time()

            # Check expiration
            if entry.is_expired(now):
                del self._store[key]
                self.metrics.misses += 1
                self.metrics.evictions += 1
                return default

            # Refresh position in LRU order and increment hits
            self._store.move_to_end(key)
            entry.hits += 1
            self.metrics.hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store key-value pair with optional custom TTL in seconds."""
        with self._lock:
            effective_ttl = ttl if ttl is not None else self.default_ttl
            expires_at = (time.time() + effective_ttl) if effective_ttl and effective_ttl > 0 else None

            # If key exists, update and move to end
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key].value = value
                self._store[key].expires_at = expires_at
                return

            # Check if capacity reached, evict oldest (LRU)
            if len(self._store) >= self.maxsize:
                self._store.popitem(last=False)
                self.metrics.evictions += 1

            self._store[key] = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                expires_at=expires_at,
            )

    def delete(self, key: str) -> bool:
        """Delete specific key from cache."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Flush all cache entries."""
        with self._lock:
            self._store.clear()

    def cleanup_expired(self) -> int:
        """Actively sweep and evict all expired entries."""
        with self._lock:
            now = time.time()
            expired_keys = [k for k, v in self._store.items() if v.is_expired(now)]
            for key in expired_keys:
                del self._store[key]
                self.metrics.evictions += 1
            return len(expired_keys)

    def size(self) -> int:
        """Return current number of items in cache."""
        with self._lock:
            return len(self._store)

    def __len__(self) -> int:
        return self.size()

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational cache statistics."""
        with self._lock:
            stats = self.metrics.to_dict()
            stats["current_size"] = self.size()
            stats["maxsize"] = self.maxsize
            return stats


# Backward-compatible alias
InMemoryCache = LRUTTLCache


# =====================================================================
# 4. Distributed Redis Cache (with Graceful Fallback)
# =====================================================================

class RedisCache:
    """Redis-backed distributed cache with connection pooling and JSON serialization."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        prefix: str = "quantum:llm:",
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_ttl = default_ttl
        self.prefix = prefix
        self._client = None
        self._is_connected = False
        self._init_client()

    def _init_client(self) -> None:
        """Attempt to initialize Redis connection."""
        try:
            import redis
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=1.5,
                socket_connect_timeout=1.5,
            )
            # Test ping
            self._client.ping()
            self._is_connected = True
            logger.info("Connected to Redis cache at %s", self.redis_url)
        except Exception as err:
            self._is_connected = False
            self._client = None
            logger.debug("Redis cache not available (fallback active): %s", err)

    def is_available(self) -> bool:
        """Check if Redis backend is active."""
        return self._is_connected and self._client is not None

    def get(self, key: str) -> Optional[Any]:
        """Get deserialized JSON value from Redis."""
        if not self.is_available():
            return None
        try:
            raw = self._client.get(f"{self.prefix}{key}")
            return json.loads(raw) if raw else None
        except Exception as err:
            logger.warning("Redis GET error: %s", err)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store JSON-serialized value in Redis with TTL."""
        if not self.is_available():
            return False
        try:
            serialized = json.dumps(value)
            effective_ttl = ttl or self.default_ttl
            self._client.setex(f"{self.prefix}{key}", effective_ttl, serialized)
            return True
        except Exception as err:
            logger.warning("Redis SET error: %s", err)
            return False

    def clear(self) -> None:
        """Flush keys matching prefix in Redis."""
        if not self.is_available():
            return
        try:
            keys = self._client.keys(f"{self.prefix}*")
            if keys:
                self._client.delete(*keys)
        except Exception as err:
            logger.warning("Redis CLEAR error: %s", err)


# =====================================================================
# 5. Multi-Tier Hybrid Cache (L1 Local Memory + L2 Redis)
# =====================================================================

class TieredCache:
    """Two-level caching strategy: L1 (in-memory LRU) -> L2 (Redis)."""

    def __init__(
        self,
        l1_cache: Optional[LRUTTLCache] = None,
        l2_cache: Optional[RedisCache] = None,
    ):
        self.l1 = l1_cache or LRUTTLCache(maxsize=1000, default_ttl=1800)
        self.l2 = l2_cache or RedisCache(default_ttl=7200)

    def get(self, key: str) -> Optional[Any]:
        """Read from L1 memory first; on miss, check L2 and promote to L1."""
        # 1. Check L1
        val = self.l1.get(key)
        if val is not None:
            return val

        # 2. Check L2
        if self.l2.is_available():
            val = self.l2.get(key)
            if val is not None:
                # Promote to L1
                self.l1.set(key, val)
                return val

        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Write to both L1 and L2 caches."""
        self.l1.set(key, value, ttl=ttl)
        if self.l2.is_available():
            self.l2.set(key, value, ttl=ttl)

    def clear(self) -> None:
        """Clear both L1 and L2 caches."""
        self.l1.clear()
        if self.l2.is_available():
            self.l2.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Get combined cache stats."""
        stats = self.l1.get_metrics()
        stats["l2_connected"] = self.l2.is_available()
        return stats


# =====================================================================
# 6. Semantic Cache (Embedding Vector Similarity)
# =====================================================================

@dataclass
class SemanticCacheItem:
    prompt: str
    embedding: List[float]
    response: Any
    created_at: float = field(default_factory=time.time)


class SemanticCache:
    """Semantic cache matching user queries by embedding cosine similarity."""

    def __init__(
        self,
        similarity_threshold: float = 0.94,
        max_entries: int = 500,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.embedding_fn = embedding_fn
        self._entries: List[SemanticCacheItem] = []
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two float vectors."""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = math.sqrt(sum(a * a for a in v1))
        norm_v2 = math.sqrt(sum(b * b for b in v2))
        if norm_v1 == 0.0 or norm_v2 == 0.0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def lookup(
        self,
        prompt: str,
        query_embedding: Optional[List[float]] = None,
    ) -> Optional[Tuple[Any, float]]:
        """Find cached response if embedding similarity >= threshold."""
        with self._lock:
            emb = query_embedding
            if emb is None and self.embedding_fn:
                emb = self.embedding_fn(prompt)

            if not emb or not self._entries:
                self.misses += 1
                return None

            best_match: Optional[SemanticCacheItem] = None
            highest_sim = -1.0

            for item in self._entries:
                sim = self.cosine_similarity(emb, item.embedding)
                if sim > highest_sim:
                    highest_sim = sim
                    best_match = item

            if highest_sim >= self.similarity_threshold and best_match:
                self.hits += 1
                return best_match.response, round(highest_sim, 4)

            self.misses += 1
            return None

    def store(
        self,
        prompt: str,
        response: Any,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Store prompt, embedding, and response in semantic cache."""
        with self._lock:
            emb = embedding
            if emb is None and self.embedding_fn:
                emb = self.embedding_fn(prompt)

            if not emb:
                return

            if len(self._entries) >= self.max_entries:
                self._entries.pop(0)  # Evict oldest entry

            self._entries.append(
                SemanticCacheItem(
                    prompt=prompt,
                    embedding=emb,
                    response=response,
                )
            )

    def clear(self) -> None:
        """Flush semantic cache."""
        with self._lock:
            self._entries.clear()


# =====================================================================
# 7. Global Singletons & Convenience Helpers
# =====================================================================

default_cache = LRUTTLCache(maxsize=1000, default_ttl=3600)
tiered_cache = TieredCache(l1_cache=default_cache)
semantic_cache = SemanticCache(similarity_threshold=0.94)

# Global default instance for easy import
cache = default_cache


def get_cached_response(
    prompt: str,
    model: str = "default",
    temperature: float = 0.7,
    system_prompt: str = "",
    tenant_id: str = "default",
) -> Optional[Any]:
    """Helper to query cache using deterministic composite key."""
    key = CacheKeyBuilder.build_key(
        prompt=prompt,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        tenant_id=tenant_id,
    )
    return tiered_cache.get(key)


def set_cached_response(
    prompt: str,
    value: Any,
    model: str = "default",
    temperature: float = 0.7,
    system_prompt: str = "",
    tenant_id: str = "default",
    ttl: Optional[int] = None,
) -> None:
    """Helper to write response to cache with deterministic composite key."""
    key = CacheKeyBuilder.build_key(
        prompt=prompt,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        tenant_id=tenant_id,
    )
    tiered_cache.set(key, value, ttl=ttl)
