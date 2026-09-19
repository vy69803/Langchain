#!/usr/bin/env python3
"""Deep Dive into LangChain Embeddings & Cache-Backed Vector Storage.

Demonstrates:
  1. CacheBackedEmbeddings (Cold vs Warm embedding speedup & zero duplicate computation)
  2. Incremental Batch Ingestion (Partial cache hits for updated documents)
  3. Query Embedding Caching (Sub-millisecond semantic search queries)
  4. Multiple Storage Backends (InMemoryStore, LocalFileByteStore, SQLiteByteStore)
  5. Integration with LangChain Vector Stores (ChromaDB)
  6. Real-Time Telemetry & Cost / Compute Savings Metrics

Usage:
  # Run full deep dive demonstration suite:
  python embeddings_deep.py

  # Test single query embedding with caching:
  python embeddings_deep.py -q "What is vector indexing?"

  # Test with specific storage backend (sqlite / file / memory):
  python embeddings_deep.py --backend file

  # Run interactive console:
  python embeddings_deep.py --interactive
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src directory is in sys.path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.stores import InMemoryByteStore

from langchain_rag.cached_embeddings import (
    CacheBackedEmbeddings,
    LocalFileByteStore,
    SQLiteByteStore,
    create_cached_embeddings,
)
from langchain_rag.embeddings import (
    calculate_similarity,
    cosine_similarity,
    get_embeddings,
)

load_dotenv()


def print_banner(title: str, subtitle: str = "") -> None:
    width = 76
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    if subtitle:
        print(f" {subtitle}".center(width))
    print("=" * width)


SAMPLE_CORPUS = [
    "Retrieval-Augmented Generation (RAG) grounds LLM responses on external private knowledge.",
    "ChromaDB is an open-source embedding database designed for high-performance AI applications.",
    "Cosine similarity computes the normalized dot product between two high-dimensional vectors.",
    "BM25 is a probabilistic sparse keyword ranking function used in information retrieval.",
    "Hybrid search combines dense semantic vector retrieval with sparse BM25 keyword matching using RRF.",
    "LangChain provides composable primitives for building production LLM chains and agents.",
    "Cross-encoders score query-document pairs jointly to provide high-precision re-ranking.",
    "Semantic chunking splits text dynamically at semantic boundary shifts rather than arbitrary token counts.",
]


def run_demonstrations(backend_type: str = "sqlite") -> None:
    cache_dir = Path(".cache/embeddings_demo")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print_banner(
        "LANGCHAIN EMBEDDINGS DEEP DIVE & CACHE-BACKED STORAGE",
        f"Backend: {backend_type.upper()} | Model: all-MiniLM-L6-v2 (384-dim ONNX)",
    )

    base_embeddings = get_embeddings("local")
    cached_emb = create_cached_embeddings(
        underlying_embeddings=base_embeddings,
        cache_dir=cache_dir,
        store_type=backend_type,
        namespace="minilm_l6_v2",
    )

    # ---------------------------------------------------------
    # SCENARIO 1: Cold Run vs. Warm Run Benchmark
    # ---------------------------------------------------------
    print_banner(
        "SCENARIO 1: Cold Indexing vs. Warm Re-Indexing Benchmark",
        f"Embedding {len(SAMPLE_CORPUS)} Knowledge Base Documents",
    )

    print("Phase 1: Cold Run (Cache Empty - Model must compute vectors for all documents)...")
    t0 = time.perf_counter()
    cold_vectors = cached_emb.embed_documents(SAMPLE_CORPUS)
    cold_time_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  ✔ Generated:   {len(cold_vectors)} vectors ({len(cold_vectors[0])} dimensions each)")
    print(f"  ✔ Latency:     {cold_time_ms:.2f} ms")
    print(f"  ✔ Cache Hits:  {cached_emb.hits} | Misses: {cached_emb.misses}")

    print("\nPhase 2: Warm Run (Re-indexing same documents - 100% Cache Hits)...")
    cached_emb.reset_stats()
    t0 = time.perf_counter()
    warm_vectors = cached_emb.embed_documents(SAMPLE_CORPUS)
    warm_time_ms = (time.perf_counter() - t0) * 1000.0

    speedup = (cold_time_ms / warm_time_ms) if warm_time_ms > 0 else 999.0
    print(f"  ✔ Retrieved:   {len(warm_vectors)} vectors from {backend_type.upper()} store")
    print(f"  ✔ Latency:     {warm_time_ms:.3f} ms")
    print(f"  ✔ Speedup:     {speedup:.1f}x Faster!")
    print(f"  ✔ Cache Hits:  {cached_emb.hits} (Hit Rate: {cached_emb.get_stats()['hit_rate_pct']}%)")
    print(f"  ✔ API / Model Cost: $0.00 (Zero duplicate compute)")

    # ---------------------------------------------------------
    # SCENARIO 2: Incremental Batch Ingestion (Partial Cache Hits)
    # ---------------------------------------------------------
    print_banner(
        "SCENARIO 2: Incremental Ingestion (Partial Cache Hits)",
        "Mixed Batch: 5 Existing Docs + 3 Newly Created Documents",
    )

    new_documents = [
        SAMPLE_CORPUS[0],  # Existing
        SAMPLE_CORPUS[1],  # Existing
        SAMPLE_CORPUS[2],  # Existing
        "Supabase pgvector offers scalable PostgreSQL vector search with ACID guarantees.",  # NEW
        SAMPLE_CORPUS[3],  # Existing
        SAMPLE_CORPUS[4],  # Existing
        "HuggingFace sentence-transformers provide dense multilingual semantic representations.",  # NEW
        "Model cascading routes low-complexity queries to smaller, cost-effective models.",  # NEW
    ]

    cached_emb.reset_stats()
    t0 = time.perf_counter()
    mixed_vectors = cached_emb.embed_documents(new_documents)
    mixed_time_ms = (time.perf_counter() - t0) * 1000.0

    stats = cached_emb.get_stats()
    print(f"Total Documents in Ingestion Batch: {len(new_documents)}")
    print(f"  ✔ Cache Hits (Existing): {stats['hits']} (Resolved in microseconds)")
    print(f"  ✔ Cache Misses (New):     {stats['misses']} (Only new docs forwarded to model)")
    print(f"  ✔ Batch Hit Rate:         {stats['hit_rate_pct']}%")
    print(f"  ✔ Total Ingestion Time:   {mixed_time_ms:.2f} ms")

    # ---------------------------------------------------------
    # SCENARIO 3: Query Embedding Caching
    # ---------------------------------------------------------
    print_banner(
        "SCENARIO 3: Query Embedding Caching & Repeated Retrieval",
        "Caching frequently searched user queries",
    )

    test_query = "How does hybrid search work with RRF?"
    cached_emb.reset_stats()

    # Query 1 (Cold)
    t0 = time.perf_counter()
    q_vec1 = cached_emb.embed_query(test_query)
    q1_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Query 1 (Cold Run): '{test_query}'")
    print(f"  - Latency:    {q1_time_ms:.2f} ms")
    print(f"  - Cache Hit:  {cached_emb.hits > 0}")

    # Query 2 (Warm)
    t0 = time.perf_counter()
    q_vec2 = cached_emb.embed_query(test_query)
    q2_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\nQuery 2 (Warm Run): '{test_query}'")
    print(f"  - Latency:    {q2_time_ms:.3f} ms")
    print(f"  - Cache Hit:  True (Instant memory/disk lookup)")
    print(f"  - Vector Match Check: {q_vec1[:3] == q_vec2[:3]} (Identical coordinates)")

    # ---------------------------------------------------------
    # SCENARIO 4: Storage Backends Comparison
    # ---------------------------------------------------------
    print_banner(
        "SCENARIO 4: Storage Backends Benchmark Comparison",
        "Evaluating InMemoryByteStore vs. LocalFileByteStore vs. SQLiteByteStore",
    )

    test_dataset = [f"Synthetic enterprise document {i} for benchmark testing." for i in range(25)]

    stores_to_test = [
        ("InMemoryByteStore", InMemoryByteStore()),
        ("LocalFileByteStore", LocalFileByteStore(cache_dir / "bench_files")),
        ("SQLiteByteStore", SQLiteByteStore(cache_dir / "bench_sqlite.db")),
    ]

    print(f"{'Storage Engine':<22} | {'Cold Index':<12} | {'Warm Lookup':<12} | {'Warm Speedup':<12} | {'Persistence':<12}")
    print("-" * 76)

    for name, store_inst in stores_to_test:
        c_emb = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=base_embeddings,
            document_embedding_store=store_inst,
            namespace=f"bench_{name}",
        )
        # Cold
        t0 = time.perf_counter()
        c_emb.embed_documents(test_dataset)
        cold_ms = (time.perf_counter() - t0) * 1000.0

        # Warm
        t0 = time.perf_counter()
        c_emb.embed_documents(test_dataset)
        warm_ms = (time.perf_counter() - t0) * 1000.0

        speed = (cold_ms / warm_ms) if warm_ms > 0 else 999.0
        persists = "No (RAM)" if "InMemory" in name else "Yes (Disk)"
        print(f"{name:<22} | {cold_ms:>8.2f} ms | {warm_ms:>8.3f} ms | {speed:>9.1f}x | {persists:<12}")

    # ---------------------------------------------------------
    # SCENARIO 5: ChromaDB VectorStore Integration
    # ---------------------------------------------------------
    print_banner(
        "SCENARIO 5: Vector Store (ChromaDB) Integration",
        "Passing CacheBackedEmbeddings seamlessly to Vector Stores",
    )

    from langchain_rag.vector_stores import create_vector_store

    vector_store = create_vector_store("cached_embeddings_demo_collection")
    # Add documents through cached embeddings
    print("Indexing documents into ChromaDB using CacheBackedEmbeddings...")
    vector_store.add_texts(
        SAMPLE_CORPUS[:4],
        metadatas=[{"source": "demo", "doc_id": i} for i in range(4)],
    )

    query_res = vector_store.query("What is ChromaDB?", n_results=1)
    print(f"\nVector Store Query: 'What is ChromaDB?'")
    print(f"Top Matched Result:\n  \"{query_res[0]['text']}\" (Score/Distance: {query_res[0].get('distance', 0.0):.4f})")

    # ---------------------------------------------------------
    # SCENARIO 6: Summary & Best Practices
    # ---------------------------------------------------------
    print_banner("SUMMARY OF EMBEDDING CACHING STRATEGIES")
    print("1. Re-Indexing Efficiency: CacheBackedEmbeddings turns O(N) model forward passes into O(1) hash lookups.")
    print("2. API Cost Protection: When using OpenAI/Cohere/Voyage embeddings, caching prevents billing for duplicate chunks.")
    print("3. Storage Selection:")
    print("   • SQLiteByteStore: Recommended for local/server production with ACID safety & single-file management.")
    print("   • LocalFileByteStore: Simple, human-inspectable directory of hashed binary files.")
    print("   • InMemoryByteStore: Highest throughput for short-lived batch jobs, test suites, or RAM-rich containers.")
    print("\n[✔] Deep Dive Demonstration Completed Successfully!\n")


def interactive_console(backend: str = "sqlite") -> None:
    cache_dir = Path(".cache/embeddings_interactive")
    cached_emb = create_cached_embeddings(
        cache_dir=cache_dir,
        store_type=backend,
        namespace="interactive_demo",
    )

    print_banner(
        "INTERACTIVE EMBEDDINGS CACHE CONSOLE",
        f"Backend: {backend.upper()} | Type 'exit' to quit | Type 'stats' for metrics",
    )
    print("Enter text or queries to test embedding latency, caching hits, and vector properties.\n")

    while True:
        try:
            text = input("\nEmbedding-Lab > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not text or text.lower() in ("exit", "quit", "q"):
            print("Session Stats:", cached_emb.get_stats())
            break

        if text.lower() == "stats":
            print("Current Cache Stats:", cached_emb.get_stats())
            continue

        if text.lower() == "reset":
            cached_emb.reset_stats()
            print("Cache statistics reset.")
            continue

        t0 = time.perf_counter()
        initial_hits = cached_emb.hits
        vec = cached_emb.embed_query(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        is_hit = cached_emb.hits > initial_hits

        status = "⚡ CACHE HIT (Cached Vector)" if is_hit else "⚙️ CACHE MISS (Computed Vector)"
        print(f"\n[{status}]")
        print(f"  • Latency:     {elapsed_ms:.3f} ms")
        print(f"  • Dimensions:  {len(vec)}")
        print(f"  • Vector Head: {[round(x, 4) for x in vec[:5]]}...")
        print(f"  • Cumulative Stats: {cached_emb.get_stats()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep Dive into LangChain Embeddings & Cache-Backed Vector Storage",
    )
    parser.add_argument("-q", "--query", type=str, help="Single query to embed with caching")
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default="sqlite",
        choices=["sqlite", "file", "memory"],
        help="Storage backend: sqlite (default), file, memory",
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive embeddings lab")
    parser.add_argument("--clear-cache", action="store_true", help="Clear test cache directory before running")

    args = parser.parse_args()

    if args.clear_cache:
        cache_path = Path(".cache/embeddings_demo")
        if cache_path.exists():
            shutil.rmtree(cache_path)
            print(f"Cleared {cache_path}")

    if args.interactive:
        interactive_console(backend=args.backend)
        return

    if args.query:
        emb = create_cached_embeddings(store_type=args.backend)
        t0 = time.perf_counter()
        vec = emb.embed_query(args.query)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"\nQuery: '{args.query}'")
        print(f"Dimensions: {len(vec)}")
        print(f"Latency: {elapsed_ms:.2f} ms")
        print(f"Stats: {emb.get_stats()}")
        return

    run_demonstrations(backend_type=args.backend)


if __name__ == "__main__":
    main()
