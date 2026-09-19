"""Cached Embeddings and Storage Backends for LangChain RAG.

Provides CacheBackedEmbeddings with multiple storage backends (LocalFileStore,
SQLiteStore, InMemoryStore) to eliminate redundant embedding calculations,
slash API token costs, and accelerate document re-indexing from seconds to microseconds.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore, ByteStore, InMemoryByteStore


class LocalFileByteStore(ByteStore):
    """File-system persistent ByteStore for embedding vectors.

    Stores values as binary files in a designated directory using hashed keys
    to prevent filesystem path invalid character errors.
    """

    def __init__(self, root_path: str | Path) -> None:
        """Initialize LocalFileByteStore.

        Args:
            root_path: Directory path where cached binary files are stored.
        """
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Derive a safe filesystem path from key."""
        safe_key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root_path / safe_key

    def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        """Get the values associated with the given keys."""
        results: list[bytes | None] = []
        for key in keys:
            path = self._get_path(key)
            if path.is_file():
                try:
                    results.append(path.read_bytes())
                except OSError:
                    results.append(None)
            else:
                results.append(None)
        return results

    def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        """Set the values for the given keys."""
        for key, value in key_value_pairs:
            path = self._get_path(key)
            try:
                path.write_bytes(value)
            except OSError as exc:
                raise IOError(f"Failed to write cache entry for {key}: {exc}") from exc

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their values."""
        for key in keys:
            path = self._get_path(key)
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass

    def yield_keys(self, prefix: str | None = None) -> Iterator[str]:
        """Yield all keys in the store."""
        # Yields files in root directory
        for p in self.root_path.glob("*"):
            if p.is_file():
                yield p.name

    def clear(self) -> int:
        """Clear all entries in the cache directory. Returns count of deleted files."""
        count = 0
        for p in self.root_path.glob("*"):
            if p.is_file():
                try:
                    p.unlink()
                    count += 1
                except OSError:
                    pass
        return count


class SQLiteByteStore(ByteStore):
    """High-performance ACID SQLite ByteStore for embedding vectors."""

    def __init__(self, db_path: str | Path) -> None:
        """Initialize SQLiteByteStore."""
        self.db_path = str(db_path)
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10.0)

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON embedding_cache(key)")
            conn.commit()

    def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        """Get values for the given keys in a single query."""
        if not keys:
            return []
        key_map: dict[str, bytes] = {}
        with self._get_connection() as conn:
            placeholders = ",".join("?" for _ in keys)
            cursor = conn.execute(
                f"SELECT key, value FROM embedding_cache WHERE key IN ({placeholders})",
                list(keys),
            )
            for k, v in cursor.fetchall():
                key_map[k] = v
        return [key_map.get(k) for k in keys]

    def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        """Set keys and values atomically."""
        if not key_value_pairs:
            return
        now = time.time()
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (key, value, created_at) VALUES (?, ?, ?)",
                [(k, v, now) for k, v in key_value_pairs],
            )
            conn.commit()

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete keys."""
        if not keys:
            return
        with self._get_connection() as conn:
            placeholders = ",".join("?" for _ in keys)
            conn.execute(
                f"DELETE FROM embedding_cache WHERE key IN ({placeholders})",
                list(keys),
            )
            conn.commit()

    def yield_keys(self, prefix: str | None = None) -> Iterator[str]:
        """Yield all keys in the store."""
        with self._get_connection() as conn:
            if prefix:
                cursor = conn.execute(
                    "SELECT key FROM embedding_cache WHERE key LIKE ?",
                    (f"{prefix}%",),
                )
            else:
                cursor = conn.execute("SELECT key FROM embedding_cache")
            for row in cursor.fetchall():
                yield row[0]

    def count(self) -> int:
        """Return total cached entries count."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embedding_cache")
            row = cursor.fetchone()
            return row[0] if row else 0

    def clear(self) -> int:
        """Clear all entries."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM embedding_cache")
            conn.commit()
            return cursor.rowcount


class CacheBackedEmbeddings(Embeddings):
    """Interface for embedding models with a backing key-value byte store.

    Caches embedding calculations using document SHA-256 hashes and namespaces.
    When embedding documents or queries, the cache is consulted first. Only missing
    embeddings are generated by the underlying model and subsequently saved to the store.
    """

    def __init__(
        self,
        underlying_embeddings: Embeddings,
        document_embedding_store: ByteStore,
        namespace: str = "",
        query_embedding_store: ByteStore | None = None,
    ) -> None:
        """Initialize CacheBackedEmbeddings.

        Args:
            underlying_embeddings: The underlying LangChain Embeddings model.
            document_embedding_store: ByteStore for document chunk embeddings.
            namespace: Model-specific namespace prefix (e.g., 'all-minilm-l6-v2').
            query_embedding_store: Optional separate ByteStore for query embeddings.
        """
        self.underlying_embeddings = underlying_embeddings
        self.document_embedding_store = document_embedding_store
        self.query_embedding_store = query_embedding_store or document_embedding_store
        self.namespace = namespace or getattr(
            underlying_embeddings, "model_name", underlying_embeddings.__class__.__name__
        )

        # Telemetry & Performance Tracking
        self.hits: int = 0
        self.misses: int = 0
        self.time_saved_ms: float = 0.0

    @classmethod
    def from_bytes_store(
        cls,
        underlying_embeddings: Embeddings,
        document_embedding_store: ByteStore,
        namespace: str = "",
        query_embedding_store: ByteStore | None = None,
    ) -> CacheBackedEmbeddings:
        """Factory method matching LangChain standard convention."""
        return cls(
            underlying_embeddings=underlying_embeddings,
            document_embedding_store=document_embedding_store,
            namespace=namespace,
            query_embedding_store=query_embedding_store,
        )

    def _get_key(self, text: str, prefix: str = "doc") -> str:
        """Generate a namespaced cache key from text content."""
        hasher = hashlib.sha256(text.encode("utf-8"))
        digest = hasher.hexdigest()
        return f"{self.namespace}:{prefix}:{digest}"

    def _encode_vector(self, vector: list[float]) -> bytes:
        """Fast binary packing for float vectors."""
        # Packing as 32-bit floats for compact binary storage (4 bytes per dimension)
        return struct.pack(f"{len(vector)}f", *vector)

    def _decode_vector(self, data: bytes) -> list[float]:
        """Unpack binary bytes into float vector."""
        count = len(data) // 4
        return list(struct.unpack(f"{count}f", data))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents, utilizing cache whenever possible.

        1. Checks store for all requested document hashes.
        2. Segregates cache hits vs cache misses.
        3. Invokes underlying embedding model ONLY for missing texts.
        4. Saves newly computed vectors into the byte store.
        5. Returns full ordered list of embedding vectors.
        """
        if not texts:
            return []

        keys = [self._get_key(t, prefix="doc") for t in texts]
        cached_vectors = self.document_embedding_store.mget(keys)

        missing_indices: list[int] = []
        missing_texts: list[str] = []
        result_vectors: list[list[float] | None] = [None] * len(texts)

        for idx, (raw_val, text) in enumerate(zip(cached_vectors, texts)):
            if raw_val is not None:
                self.hits += 1
                result_vectors[idx] = self._decode_vector(raw_val)
            else:
                self.misses += 1
                missing_indices.append(idx)
                missing_texts.append(text)

        # If there are misses, compute them via the underlying model
        if missing_texts:
            t0 = time.perf_counter()
            new_vectors = self.underlying_embeddings.embed_documents(missing_texts)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Store the newly computed embeddings
            pairs_to_set: list[tuple[str, bytes]] = []
            for miss_idx, text_idx in enumerate(missing_indices):
                vec = new_vectors[miss_idx]
                result_vectors[text_idx] = vec
                encoded = self._encode_vector(vec)
                pairs_to_set.append((keys[text_idx], encoded))

            self.document_embedding_store.mset(pairs_to_set)

        return [v for v in result_vectors if v is not None]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query, utilizing query cache if available."""
        key = self._get_key(text, prefix="query")
        raw_val = self.query_embedding_store.mget([key])[0]

        if raw_val is not None:
            self.hits += 1
            return self._decode_vector(raw_val)

        self.misses += 1
        t0 = time.perf_counter()
        vector = self.underlying_embeddings.embed_query(text)
        self.query_embedding_store.mset([(key, self._encode_vector(vector))])
        return vector

    def get_stats(self) -> dict[str, Any]:
        """Return cache performance statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100.0) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate_pct": round(hit_rate, 2),
            "namespace": self.namespace,
        }

    def reset_stats(self) -> None:
        """Reset telemetry counters."""
        self.hits = 0
        self.misses = 0


def create_cached_embeddings(
    underlying_embeddings: Embeddings | None = None,
    cache_dir: str | Path = ".cache/embeddings",
    store_type: str = "sqlite",
    namespace: str = "",
) -> CacheBackedEmbeddings:
    """Factory to easily create a CacheBackedEmbeddings instance.

    Args:
        underlying_embeddings: Base embedding model (defaults to LocalEmbeddings).
        cache_dir: Directory where cache is persisted.
        store_type: 'sqlite' (default), 'file', or 'memory'.
        namespace: Cache isolation namespace.

    Returns:
        Configured CacheBackedEmbeddings instance.
    """
    from langchain_rag.embeddings import get_embeddings

    base_model = underlying_embeddings or get_embeddings("local")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if store_type == "sqlite":
        store: ByteStore = SQLiteByteStore(cache_path / "embeddings.sqlite")
    elif store_type == "file":
        store = LocalFileByteStore(cache_path / "files")
    elif store_type == "memory":
        store = InMemoryByteStore()
    else:
        raise ValueError(f"Unknown store_type '{store_type}'. Choose 'sqlite', 'file', or 'memory'.")

    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=base_model,
        document_embedding_store=store,
        namespace=namespace,
    )
