"""Comprehensive unit tests for production caching, hashing, TTL, LRU, and metrics."""

import time
import pytest
from production_api.cache import (
    CacheKeyBuilder,
    CacheMetrics,
    InMemoryCache,
    LRUTTLCache,
    SemanticCache,
    TieredCache,
    get_cached_response,
    set_cached_response,
)


# ==========================================
# 1. Backwards Compatibility & Basic Ops
# ==========================================

def test_in_memory_cache_basic_ops():
    """Verify standard in-memory cache get, set, delete, and clear."""
    cache = InMemoryCache()
    assert cache.get("key1") is None
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.delete("key1") is True
    assert cache.get("key1") is None

    cache.set("key2", "value2")
    cache.clear()
    assert cache.get("key2") is None
    assert cache.size() == 0


# ==========================================
# 2. Key Builder & Canonical Hashing Tests
# ==========================================

def test_cache_key_builder_normalization():
    """Prompt whitespace and casing are canonicalized deterministically."""
    k1 = CacheKeyBuilder.build_key("  Hello World?  ", model="gpt-4o", temperature=0.7)
    k2 = CacheKeyBuilder.build_key("hello world", model="gpt-4o", temperature=0.7)
    assert k1 == k2

    # Different model or temperature produces different keys
    k3 = CacheKeyBuilder.build_key("hello world", model="claude-3-5", temperature=0.7)
    k4 = CacheKeyBuilder.build_key("hello world", model="gpt-4o", temperature=0.2)
    assert k1 != k3
    assert k1 != k4


# ==========================================
# 3. LRU Eviction & Capacity Limits
# ==========================================

def test_lru_eviction():
    """Cache evicts Least Recently Used item when capacity is reached."""
    cache = LRUTTLCache(maxsize=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    # Access "a" to make "b" the oldest unaccessed
    assert cache.get("a") == 1

    # Insert "d" -> "b" should be evicted
    cache.set("d", 4)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4
    assert cache.size() == 3


# ==========================================
# 4. TTL Expiration Tests
# ==========================================

def test_ttl_expiration():
    """Entries expire after their TTL window."""
    cache = LRUTTLCache(maxsize=10, default_ttl=1)  # 1 second default TTL
    cache.set("ephemeral", "data", ttl=1)

    assert cache.get("ephemeral") == "data"
    time.sleep(1.1)
    assert cache.get("ephemeral") is None


def test_cleanup_expired():
    """Active cleanup sweeps expired items."""
    cache = LRUTTLCache(maxsize=10)
    cache.set("temp1", "val1", ttl=1)
    cache.set("temp2", "val2", ttl=10)

    time.sleep(1.1)
    evicted_count = cache.cleanup_expired()
    assert evicted_count == 1
    assert cache.get("temp1") is None
    assert cache.get("temp2") == "val2"


# ==========================================
# 5. Cache Metrics & Observability Tests
# ==========================================

def test_cache_metrics():
    """Verify metrics calculation for hits, misses, and hit rate percent."""
    cache = LRUTTLCache(maxsize=10)
    cache.set("k", "v")

    # 1 hit, 1 miss
    _ = cache.get("k")
    _ = cache.get("nonexistent")

    stats = cache.get_metrics()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["total_requests"] == 2
    assert stats["hit_rate_percent"] == 50.0


# ==========================================
# 6. Semantic Cache Tests
# ==========================================

def test_semantic_cache():
    """Semantic cache matches vectors within cosine similarity threshold."""
    sem_cache = SemanticCache(similarity_threshold=0.90)

    # Store query with embedding [1.0, 0.0]
    sem_cache.store(
        prompt="How to scale Kubernetes cluster?",
        response="Use HPA and node autoscalers.",
        embedding=[1.0, 0.0],
    )

    # Query with nearly identical embedding [0.98, 0.19]
    hit = sem_cache.lookup(
        prompt="How to autoscale Kubernetes nodes?",
        query_embedding=[0.98, 0.19],
    )
    assert hit is not None
    response, score = hit
    assert response == "Use HPA and node autoscalers."
    assert score >= 0.90

    # Query with orthogonal/unrelated embedding [0.0, 1.0] -> miss
    miss = sem_cache.lookup(
        prompt="Office coffee machine recipe",
        query_embedding=[0.0, 1.0],
    )
    assert miss is None


# ==========================================
# 7. Tiered Cache & Helper Functions
# ==========================================

def test_convenience_helpers():
    """Verify get_cached_response and set_cached_response helpers."""
    prompt = "What is the AWS instance type for compute pool?"
    answer = "c6i.4xlarge"

    set_cached_response(prompt, answer, model="gpt-4o")
    retrieved = get_cached_response(prompt, model="gpt-4o")
    assert retrieved == answer

    # Miss for different model
    assert get_cached_response(prompt, model="claude-3-5") is None
