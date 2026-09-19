#!/usr/bin/env python3
"""Production Hybrid Search: Dense Vector + Sparse BM25 Fusion.

Combines ChromaDB vector embeddings (semantic similarity) with Okapi BM25
(lexical/keyword precision) using Reciprocal Rank Fusion (RRF) and Convex
Score Fusion.

Usage:
  # Run comprehensive test suite & comparison demos:
  python prod_hybrid_search.py

  # Search with a custom query:
  python prod_hybrid_search.py -q "ERR-9021-TOKEN-REVOKED auth failure"

  # Compare Dense vs Sparse vs Hybrid side-by-side:
  python prod_hybrid_search.py -q "RFC 6749" --mode compare

  # Interactive search console:
  python prod_hybrid_search.py --interactive
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src directory is in sys.path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
from langchain_rag.hybrid_search import (
    BM25Index,
    HybridSearchEngine,
    create_hybrid_search_engine,
)
from langchain_rag.llm import get_llm

load_dotenv()

# Real-world enterprise sample dataset highlighting the strengths of both search types
SAMPLE_ENTERPRISE_KB = [
    {
        "text": "Error Code ERR-9021-TOKEN-REVOKED: The client OAuth 2.0 bearer token was invalidated by the security STS service.",
        "metadata": {"category": "auth", "system": "identity-gateway", "severity": "high"},
    },
    {
        "text": "RFC 6749 specifies the OAuth 2.0 Authorization Framework, defining access token issuance and authorization flows.",
        "metadata": {"category": "spec", "system": "standards", "severity": "info"},
    },
    {
        "text": "When users encounter expired login credentials or authentication timeouts, redirect them to the single sign-on portal.",
        "metadata": {"category": "auth", "system": "sso-portal", "severity": "medium"},
    },
    {
        "text": "Microservices communicate asynchronously using Apache Kafka event topics with snappy compression enabled.",
        "metadata": {"category": "infra", "system": "messaging", "severity": "low"},
    },
    {
        "text": "PostgreSQL database query optimization: Create composite B-Tree indexes on tenant_id and created_at timestamps.",
        "metadata": {"category": "database", "system": "postgres", "severity": "low"},
    },
    {
        "text": "Thinking Machines: Inkling is an ultra-fast instruction-tuned language model accessible via OpenRouter.",
        "metadata": {"category": "ai", "system": "openrouter", "severity": "info"},
    },
    {
        "text": "ChromaDB is a lightweight open-source vector database designed for high-performance similarity search in RAG pipelines.",
        "metadata": {"category": "ai", "system": "vector-store", "severity": "info"},
    },
    {
        "text": "Incident Playbook for ERR-9021-TOKEN-REVOKED: Flush Redis session cache, rotate client secret, and request new JWT.",
        "metadata": {"category": "auth", "system": "identity-gateway", "severity": "critical"},
    },
]


def print_header(title: str, subtitle: str = "") -> None:
    width = 72
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    if subtitle:
        print(f" {subtitle}".center(width))
    print("=" * width)


def format_doc_preview(text: str, max_len: int = 85) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def display_results_table(title: str, results: list[dict[str, Any]], score_key: str = "score") -> None:
    print(f"\n--- {title} (Total: {len(results)}) ---")
    if not results:
        print("  [No matching documents found]")
        return

    header = f"  {'Rank':<5} | {'Score':<10} | {'Doc ID':<12} | {'Content'}"
    divider = "  " + "-" * 78
    print(header)
    print(divider)

    for i, r in enumerate(results, 1):
        score_val = r.get(score_key, 0.0)
        score_str = f"{score_val:.4f}" if isinstance(score_val, (int, float)) else str(score_val)
        preview = format_doc_preview(r["text"], max_len=52)
        doc_id = r.get("id", f"doc_{i}")
        print(f"  #{i:<4} | {score_str:<10} | {doc_id:<12} | {preview}")


def display_hybrid_comparison(engine: HybridSearchEngine, query: str, k: int = 3) -> None:
    print(f"\nQuery: \"{query}\"")
    print("-" * 72)

    dense_res = engine.search_dense(query, k=k)
    sparse_res = engine.search_sparse(query, k=k)
    hybrid_rrf = engine.search(query, k=k, fusion_mode="rrf")
    hybrid_weighted = engine.search(query, k=k, fusion_mode="weighted", alpha=0.5)

    display_results_table("1. Dense Vector Search Only (ChromaDB / Semantic)", dense_res, score_key="dense_score")
    display_results_table("2. Sparse Lexical Search Only (BM25 / Keywords)", sparse_res, score_key="sparse_score")
    display_results_table("3. Hybrid Search (Reciprocal Rank Fusion - RRF)", hybrid_rrf, score_key="hybrid_score")
    display_results_table("4. Hybrid Search (Convex Weighted Fusion - alpha=0.5)", hybrid_weighted, score_key="hybrid_score")


def run_demonstration_suite(engine: HybridSearchEngine) -> None:
    print_header(
        "PRODUCTION HYBRID SEARCH DEMONSTRATION",
        "Semantic Vector (ChromaDB) + Lexical Keyword (Okapi BM25) Fusion",
    )

    # Reset collection for a clean, deterministic demonstration
    engine.reset()

    print("\n[Step 1] Ingesting enterprise knowledge base into Hybrid Engine...")
    texts = [item["text"] for item in SAMPLE_ENTERPRISE_KB]
    metas = [item["metadata"] for item in SAMPLE_ENTERPRISE_KB]
    ids = [f"kb_item_{i+1}" for i in range(len(texts))]

    assigned = engine.add_texts(texts, ids=ids, metadatas=metas)
    print(f"  Indexed {len(assigned)} documents. Total in store: {engine.count()}")

    # --- Scenario 1: Exact error code / acronym (Sparse wins over Dense) ---
    print_header(
        "SCENARIO 1: Exact Technical Identifier / Error Code",
        "Query: 'ERR-9021-TOKEN-REVOKED'",
    )
    print("Why BM25 is crucial: Dense embeddings often compress or dilute arbitrary")
    print("alphanumeric codes, whereas BM25 computes exact lexical matches with high IDF.")
    display_hybrid_comparison(engine, "ERR-9021-TOKEN-REVOKED", k=3)

    # --- Scenario 2: Conceptual / Semantic (Dense wins over Sparse) ---
    print_header(
        "SCENARIO 2: Conceptual & Synonym Query (Zero Keyword Overlap)",
        "Query: 'handling expired login credentials and authentication timeouts'",
    )
    print("Why Vector Search is crucial: Queries with different vocabulary but identical")
    print("underlying meaning are captured via high-dimensional vector embeddings.")
    display_hybrid_comparison(engine, "handling expired login credentials and authentication timeouts", k=3)

    # --- Scenario 3: Mixed Query (Hybrid RRF achieves best of both worlds) ---
    print_header(
        "SCENARIO 3: Mixed Technical & Conceptual Query",
        "Query: 'How to recover and rotate secrets after ERR-9021-TOKEN-REVOKED?'",
    )
    print("Why Hybrid RRF shines: Combines exact keyword matches with semantic intent,")
    print("ensuring the specific incident playbook is ranked #1.")
    display_hybrid_comparison(engine, "How to recover and rotate secrets after ERR-9021-TOKEN-REVOKED?", k=3)

    # --- Scenario 4: Metadata Filtering ---
    print_header(
        "SCENARIO 4: Metadata Filtered Hybrid Search",
        "Filter: category = 'auth', Query: 'token'",
    )
    filtered = engine.search("token", k=3, where={"category": "auth"})
    display_results_table("Hybrid Search with {'category': 'auth'} filter", filtered, score_key="hybrid_score")

    # --- Scenario 5: LangChain BaseRetriever Integration ---
    print_header(
        "SCENARIO 5: LangChain BaseRetriever Interface",
        "Converting HybridSearchEngine to LangChain Retriever",
    )
    retriever = engine.as_retriever(k=2, fusion_mode="rrf")
    lc_docs = retriever.invoke("OAuth 2.0 specification")
    print(f"  Retriever returned {len(lc_docs)} LangChain Document objects:")
    for i, doc in enumerate(lc_docs, 1):
        print(f"    [{i}] (RRF Score: {doc.metadata.get('hybrid_score')}): {doc.page_content}")

    # --- Scenario 6: End-to-End LLM Answer Generation ---
    print_header(
        "SCENARIO 6: Grounded QA Generation via Hybrid Context",
        "Query: 'What are the exact recovery steps for ERR-9021-TOKEN-REVOKED?'",
    )
    rag_query = "What are the exact recovery steps for ERR-9021-TOKEN-REVOKED?"
    top_hits = engine.search(rag_query, k=2, fusion_mode="rrf")
    context_str = "\n".join([f"- {h['text']}" for h in top_hits])

    print(f"Context retrieved via Hybrid Search:\n{context_str}\n")
    try:
        llm = get_llm()
        prompt = (
            "You are a helpful IT assistant. Answer the question using ONLY the provided context.\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {rag_query}\n\n"
            "Helpful Answer:"
        )
        print("Invoking Thinking Machines model...")
        response = llm.invoke(prompt)
        print(f"\nModel Answer:\n{response.content}\n")
    except Exception as e:
        print(f"[Notice] LLM call skipped or API key not present: {e}")
        print("Hybrid search context was successfully retrieved and formatted for generation.")

    print("\n" + "=" * 72)
    print(" [✔] Production Hybrid Search Suite Completed Successfully!".center(72))
    print("=" * 72 + "\n")


def interactive_mode(engine: HybridSearchEngine) -> None:
    print_header("INTERACTIVE HYBRID SEARCH CONSOLE", "Type 'exit' or 'quit' to exit")
    print(f"Current documents indexed: {engine.count()}")

    while True:
        try:
            query = input("\nHybrid Search > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query or query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        display_hybrid_comparison(engine, query, k=3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production Hybrid Search CLI (Dense ChromaDB + Sparse BM25 Fusion)",
    )
    parser.add_argument("-q", "--query", type=str, help="Search query string")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["rrf", "weighted", "compare", "dense", "sparse"],
        default="rrf",
        help="Search / Fusion mode (default: rrf)",
    )
    parser.add_argument("-k", type=int, default=3, help="Number of results to retrieve (default: 3)")
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.5,
        help="Dense weight (0.0 to 1.0) for weighted fusion (default: 0.5)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to an optional text or markdown file to index before searching",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch interactive search REPL console",
    )

    args = parser.parse_args()

    engine = create_hybrid_search_engine(collection_name="prod_hybrid_demo")

    # If indexing a file
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File '{file_path}' does not exist.")
            sys.exit(1)
        print(f"Indexing file: {file_path}...")
        ids = engine.index_file(file_path)
        print(f"Indexed {len(ids)} chunk(s). Total docs: {engine.count()}")

    # Ensure sample KB is indexed if running single query or interactive mode without documents
    if (args.query or args.interactive) and engine.count() == 0 and not args.file:
        texts = [item["text"] for item in SAMPLE_ENTERPRISE_KB]
        metas = [item["metadata"] for item in SAMPLE_ENTERPRISE_KB]
        engine.add_texts(texts, metadatas=metas)

    # Interactive mode
    if args.interactive:
        interactive_mode(engine)
        return

    # Specific query execution
    if args.query:
        if args.mode == "compare":
            display_hybrid_comparison(engine, args.query, k=args.k)
        elif args.mode == "dense":
            results = engine.search_dense(args.query, k=args.k)
            display_results_table("Dense Search Results", results, score_key="dense_score")
        elif args.mode == "sparse":
            results = engine.search_sparse(args.query, k=args.k)
            display_results_table("Sparse (BM25) Search Results", results, score_key="sparse_score")
        else:
            results = engine.search(
                args.query,
                k=args.k,
                alpha=args.alpha,
                fusion_mode=args.mode,
            )
            display_results_table(
                f"Hybrid Search Results ({args.mode.upper()})",
                results,
                score_key="hybrid_score",
            )
        return

    # Default run: comprehensive demonstration
    run_demonstration_suite(engine)


if __name__ == "__main__":
    main()
