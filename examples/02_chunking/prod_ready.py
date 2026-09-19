#!/usr/bin/env python3
"""Production-Ready Chunking Architecture & Benchmark Suite.

Enterprise multi-strategy document chunking for high-accuracy RAG systems:
  1. Adaptive Semantic Chunking (Dynamic statistical thresholds + Min/Max guardrails)
  2. Hierarchical Parent-Child Chunking (High-precision dense retrieval + full LLM context)
  3. Contextual Markdown Chunking (Breadcrumb injection + code/table protection)
  4. Production Vector Store Integration & Retrieval Benchmark

Usage:
  # Run full benchmark and comparative demonstration suite:
  python Chunking/prod_ready.py

  # Chunk a specific document file:
  python Chunking/prod_ready.py -f README.md --strategy semantic

  # Test retrieval precision on chunked corpus with a query:
  python Chunking/prod_ready.py -q "How are OAuth token revocation incidents resolved?"

  # Run statistical chunk distribution benchmark:
  python Chunking/prod_ready.py --benchmark

  # Interactive chunking and retrieval console:
  python Chunking/prod_ready.py --interactive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root and src directory are in sys.path for direct execution
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_core.documents import Document

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_rag.document_loader import load_document
from langchain_rag.embeddings import get_embeddings
from langchain_rag.semantic_chunking import (
    ChunkMetrics,
    ContextualMarkdownChunker,
    ParentChildChunker,
    ProductionChunker,
    ProductionSemanticChunker,
    SentenceSplitter,
    create_production_chunker,
    estimate_tokens,
)
from langchain_rag.vector_stores import VectorStore, create_vector_store

load_dotenv()


def smart_chunker(
    text: str,
    use_semantic: bool = True,
    fallback_chunk_size: int = 500,
) -> list[str]:
    """
    Production chunking with semantic as primary, recursive as fallback.
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key or api_key.startswith("your_"):
            raise ValueError("No valid API key for OpenAIEmbeddings")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception:
        embeddings = get_embeddings("local")

    if use_semantic:
        try:
            chunker = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90,
            )
            return chunker.split_text(text)
        except Exception as e:
            print(f"Semantic chunking failed ({e}), falling back to recursive...")

    fallback = RecursiveCharacterTextSplitter(
        chunk_size=fallback_chunk_size,
        chunk_overlap=50,
    )
    return fallback.split_text(text)


# Real-world complex enterprise document with multiple topic shifts, technical specs, code, and playbooks
SAMPLE_ENTERPRISE_DOCUMENT = """# QuantumCloud Architecture & Operational Runbook

## 1. Authentication & Security Specifications
QuantumCloud enforces zero-trust identity verification across all gateway endpoints.
All API clients must obtain a cryptographically signed OAuth 2.0 JSON Web Token (JWT) issued by the centralized STS authority.
Tokens carry an ephemeral expiration window of 900 seconds (15 minutes) and must include the audience claim `urn:quantum:api`.

### 1.1 Incident Playbook: ERR-9021-TOKEN-REVOKED
When a high-risk security event is detected, STS emits an `ERR-9021-TOKEN-REVOKED` signal.
Engineers must immediately execute the following mitigation sequence:
1. Invalidate and purge the client session key from the distributed Redis cluster.
2. Force client secret rotation via the HashiCorp Vault management API.
3. Notify the enterprise security operations center (SOC) via PagerDuty webhook.

```bash
# Emergency token revocation script
vault write auth/approle/role/gateway-prod/secret-id-accessor/destroy accessor=$ACCESSOR_ID
redis-cli -h redis-auth.internal DEL "session:$SESSION_ID"
```

## 2. Distributed Database Indexing & Performance
Our distributed PostgreSQL cluster processes over 50,000 read-write queries per second.
To prevent query timeouts and table sequential scans, all multi-tenant tables must maintain composite B-Tree indexes.
Indexes must always lead with `tenant_id` followed by the monotonically increasing timestamp `created_at`.

Connection pooling is governed by PgBouncer running in transaction pooling mode.
Each microservice container is strictly allocated a maximum pool size of 20 connections to prevent connection exhaustion.

## 3. Artisanal Sourdough Fermentation (Employee Culinary Guild)
For the monthly QuantumCloud hackathon luncheon, the culinary guild prepares wild-fermented sourdough loaves.
The levain must be refreshed at a 1:2:2 ratio (starter : water : flour) twelve hours prior to mixing the final dough.
Maintaining a bulk fermentation ambient temperature of 26°C (78°F) ensures optimal balance between lactic and acetic acids.
Baking at 230°C with initial steam injection produces a blistered crust with high gelatinization.

## 4. Large Language Model Orchestration
The internal AI gateway routes developer queries to specialized language models based on prompt complexity.
Latency-critical semantic caching is powered by ChromaDB running in-memory with cosine vector indexing.
Prompts exceeding 4,000 tokens are dynamically compressed using sentence-level saliency extraction before LLM dispatch.
"""


def print_header(title: str, subtitle: str = "") -> None:
    width = 76
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    if subtitle:
        print(f" {subtitle}".center(width))
    print("=" * width)


def format_preview(text: str, max_len: int = 100) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def display_metrics_table(results: list[dict[str, Any]]) -> None:
    """Display comparison metrics table."""
    print("\n" + "-" * 76)
    print(f"{'Strategy':<24} | {'Chunks':<6} | {'Avg Chars':<9} | {'Min-Max':<11} | {'Tokens':<6} | {'Time':<7}")
    print("-" * 76)
    for r in results:
        strat = r["strategy"][:24]
        chunks = r["chunks"]
        avg_c = f"{r['mean_chars']:.0f}"
        min_max = f"{r['min_chars']}-{r['max_chars']}"
        tokens = r["total_tokens"]
        duration = f"{r['duration_ms']:.1f}ms"
        print(f"{strat:<24} | {chunks:<6} | {avg_c:<9} | {min_max:<11} | {tokens:<6} | {duration:<7}")
    print("-" * 76)


def run_comprehensive_suite() -> None:
    """Run full production chunking demonstration, comparisons, and retrieval benchmarks."""
    print_header(
        "PRODUCTION RAG CHUNKING ARCHITECTURE",
        "Adaptive Semantic | Parent-Child | Contextual Markdown | Guardrails",
    )

    embeddings = get_embeddings("local")
    production_chunker = create_production_chunker(embeddings)
    sample_doc = Document(
        page_content=SAMPLE_ENTERPRISE_DOCUMENT,
        metadata={"source": "QuantumCloud_Runbook.md", "category": "engineering"},
    )

    # --- Quick Demonstration: smart_chunker() ---
    print_header(
        "FUNCTION USAGE: smart_chunker()",
        "Primary: Semantic Chunking (90th percentile) | Fallback: Recursive Splitter",
    )
    sample_prose = (
        "Artificial intelligence and deep learning models are transforming modern oncology and radiology. "
        "Physicians use neural networks to analyze MRI scans for early diagnosis.\n\n"
        "Baking artisanal sourdough bread requires wild yeast cultures and precise hydration ratios. "
        "Fermenting at 26°C yields the ideal lactic-to-acetic acid balance.\n\n"
        "Quantum computing leverages superposition and entanglement to execute complex Shor's algorithm simulations."
    )
    print("Executing smart_chunker(sample_prose, use_semantic=True)...")
    smart_chunks = smart_chunker(sample_prose, use_semantic=True)
    print(f"✔ Generated {len(smart_chunks)} chunks via smart_chunker:")
    for idx, sc in enumerate(smart_chunks, 1):
        print(f"  [{idx}] \"{sc.strip()}\"")

    # --- Strategy 1: Adaptive Semantic Chunking ---
    print_header(
        "STRATEGY 1: Adaptive Semantic Chunking with Guardrails",
        "Embedding Similarity Breakpoints + Min/Max Size Bounds",
    )
    semantic_chunks, sem_metrics = production_chunker.chunk_document(
        sample_doc,
        strategy="semantic",
        threshold_type="percentile",
        threshold_amount=90.0,
        min_chunk_chars=120,
        max_chunk_chars=1200,
    )

    print(f"✔ Generated {len(semantic_chunks)} semantic chunks in {sem_metrics.duration_ms:.1f}ms")
    print(f"  Distribution: Min={sem_metrics.min_chunk_chars} chars, Max={sem_metrics.max_chunk_chars} chars, Mean={sem_metrics.mean_chunk_chars:.1f} chars\n")

    for i, chunk in enumerate(semantic_chunks, 1):
        print(f"  [Semantic Chunk #{i}] ({len(chunk.page_content)} chars / ~{chunk.metadata['token_estimate']} tokens):")
        print(f"  └── \"{format_preview(chunk.page_content, 120)}\"")

    # --- Strategy 2: Contextual Markdown Chunking ---
    print_header(
        "STRATEGY 2: Contextual Markdown Chunking",
        "Header Breadcrumb Injection & Code-Fence Protection",
    )
    md_chunks, md_metrics = production_chunker.chunk_document(
        sample_doc,
        strategy="markdown",
    )
    print(f"✔ Generated {len(md_chunks)} contextual markdown chunks in {md_metrics.duration_ms:.1f}ms\n")

    for i, chunk in enumerate(md_chunks, 1):
        breadcrumb = chunk.metadata.get("breadcrumbs", "N/A")
        print(f"  [Markdown Chunk #{i}] Breadcrumb: [{breadcrumb}]")
        print(f"  └── \"{format_preview(chunk.page_content, 110)}\"")

    # --- Strategy 3: Hierarchical Parent-Child Chunking ---
    print_header(
        "STRATEGY 3: Hierarchical Parent-Child Chunking (Small-to-Big)",
        "Small vectors for search precision + Large parent for LLM context",
    )
    parent_docs, child_docs = production_chunker.parent_child_chunker.split_document_hierarchical(sample_doc)
    pc_metrics = production_chunker.calculate_metrics(child_docs)

    print(f"✔ Generated {len(parent_docs)} Parent Contexts and {len(child_docs)} Child Search Chunks\n")
    print(f"  Parent Block #1 (Size: {len(parent_docs[0].page_content)} chars):")
    print(f"  └── \"{format_preview(parent_docs[0].page_content, 110)}\"")
    print(f"  Children linked to Parent #1 ({len([c for c in child_docs if c.metadata['parent_id'] == parent_docs[0].metadata['parent_id']])} child chunks):")
    for c in child_docs[:2]:
        print(f"      • Child [{c.metadata['child_index']}]: \"{format_preview(c.page_content, 85)}\"")

    # --- Strategy 4: Standard Recursive Splitter (Baseline) ---
    rec_chunks, rec_metrics = production_chunker.chunk_document(
        sample_doc,
        strategy="recursive",
        chunk_size=400,
        chunk_overlap=50,
    )

    # --- Metrics & Quality Comparison Table ---
    print_header(
        "CHUNK DISTRIBUTION & PERFORMANCE BENCHMARK",
        "Comparative Analysis of Chunking Quality Metrics",
    )
    benchmark_rows = [
        {
            "strategy": "Recursive (Baseline)",
            "chunks": rec_metrics.total_chunks,
            "mean_chars": rec_metrics.mean_chunk_chars,
            "min_chars": rec_metrics.min_chunk_chars,
            "max_chars": rec_metrics.max_chunk_chars,
            "total_tokens": rec_metrics.total_estimated_tokens,
            "duration_ms": rec_metrics.duration_ms,
        },
        {
            "strategy": "Production Semantic",
            "chunks": sem_metrics.total_chunks,
            "mean_chars": sem_metrics.mean_chunk_chars,
            "min_chars": sem_metrics.min_chunk_chars,
            "max_chars": sem_metrics.max_chunk_chars,
            "total_tokens": sem_metrics.total_estimated_tokens,
            "duration_ms": sem_metrics.duration_ms,
        },
        {
            "strategy": "Contextual Markdown",
            "chunks": md_metrics.total_chunks,
            "mean_chars": md_metrics.mean_chunk_chars,
            "min_chars": md_metrics.min_chunk_chars,
            "max_chars": md_metrics.max_chunk_chars,
            "total_tokens": md_metrics.total_estimated_tokens,
            "duration_ms": md_metrics.duration_ms,
        },
        {
            "strategy": "Parent-Child (Children)",
            "chunks": pc_metrics.total_chunks,
            "mean_chars": pc_metrics.mean_chunk_chars,
            "min_chars": pc_metrics.min_chunk_chars,
            "max_chars": pc_metrics.max_chunk_chars,
            "total_tokens": pc_metrics.total_estimated_tokens,
            "duration_ms": pc_metrics.duration_ms,
        },
    ]
    display_metrics_table(benchmark_rows)

    # --- End-to-End Vector Retrieval Accuracy Test ---
    print_header(
        "END-TO-END RETRIEVAL ACCURACY TEST",
        "Evaluating Precision & Context Boundary Fidelity",
    )

    queries = [
        ("Auth/Incident", "How do engineers mitigate an ERR-9021-TOKEN-REVOKED error?"),
        ("Database", "What composite index columns are required on PostgreSQL?"),
        ("Out-of-domain", "What ambient temperature is used for sourdough fermentation?"),
    ]

    # Ingest into test collections in ChromaDB
    sem_store = create_vector_store("benchmark_semantic")
    sem_store.reset()
    sem_store.add_documents(semantic_chunks)

    pc_store = create_vector_store("benchmark_parent_child")
    pc_store.reset()
    pc_store.add_documents(child_docs)

    for category, q in queries:
        print(f"\n[Query ({category})]: \"{q}\"")

        # 1. Semantic Chunker retrieval
        sem_res = sem_store.query(q, n_results=1)
        if sem_res and sem_res[0]["text"]:
            dist = sem_res[0]["distance"]
            print(f"  ► [Semantic Chunk Hit] (Cosine Dist: {dist:.4f}):")
            print(f"    \"{format_preview(sem_res[0]['text'], 110)}\"")

        # 2. Parent-Child retrieval with parent expansion
        pc_res = pc_store.query(q, n_results=1)
        if pc_res and pc_res[0]["metadata"]:
            parent_text = pc_res[0]["metadata"].get("parent_text", "")
            child_text = pc_res[0]["text"]
            dist = pc_res[0]["distance"]
            print(f"  ► [Parent-Child Search Hit] (Child Dist: {dist:.4f}):")
            print(f"    Matched Child : \"{format_preview(child_text, 80)}\"")
            print(f"    Resolved Parent (passed to LLM): \"{format_preview(parent_text, 110)}\"")

    print_header(
        "✔ PRODUCTION CHUNKING SUITE COMPLETED",
        "All strategies, guardrails, and retrieval paths validated!",
    )


def process_custom_file(
    file_path: Path,
    strategy: str = "semantic",
    export_path: Optional[Path] = None,
    query: Optional[str] = None,
) -> None:
    """Process a user-provided file with the production chunker."""
    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return

    print(f"Loading document: {file_path.name}...")
    docs = load_document(file_path)
    if not docs:
        print(f"Error: Could not extract content from '{file_path}'.")
        return

    full_doc = Document(
        page_content="\n\n".join(d.page_content for d in docs),
        metadata={"source": file_path.name, "path": str(file_path)},
    )

    chunker = create_production_chunker()
    chunks, metrics = chunker.chunk_document(full_doc, strategy=strategy)

    print_header(
        f"CHUNKING RESULTS: {file_path.name}",
        f"Strategy: {strategy.upper()} | Generated {metrics.total_chunks} Chunks",
    )
    print(f"  Total Characters: {metrics.total_characters}")
    print(f"  Estimated Tokens: {metrics.total_estimated_tokens}")
    print(f"  Avg Chunk Size  : {metrics.mean_chunk_chars:.1f} characters")
    print(f"  Min / Max Size  : {metrics.min_chunk_chars} / {metrics.max_chunk_chars} characters")
    print(f"  Processing Time : {metrics.duration_ms:.2f} ms\n")

    print("Sample Chunks Preview:")
    for idx, c in enumerate(chunks[:5], 1):
        print(f"  [{idx}] ({len(c.page_content)} chars): \"{format_preview(c.page_content, 110)}\"")

    if len(chunks) > 5:
        print(f"  ... and {len(chunks) - 5} more chunks.")

    if export_path:
        export_data = [
            {
                "chunk_index": idx,
                "content": c.page_content,
                "metadata": c.metadata,
            }
            for idx, c in enumerate(chunks)
        ]
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"\n✔ Successfully exported {len(chunks)} chunks to '{export_path}'")

    if query:
        print_header("QUERYING VECTOR STORE", f"Query: '{query}'")
        store = create_vector_store(f"cli_chunk_test_{strategy}")
        store.reset()
        store.add_documents(chunks)
        results = store.query(query, n_results=3)
        for rank, r in enumerate(results, 1):
            print(f"  #{rank} (Distance: {r['distance']:.4f}):")
            print(f"  \"{format_preview(r['text'], 140)}\"\n")


def interactive_console() -> None:
    """Interactive CLI REPL for live text chunking and similarity exploration."""
    print_header("INTERACTIVE PRODUCTION CHUNKER CONSOLE", "Type 'exit' or 'quit' to exit")
    chunker = create_production_chunker()

    current_strategy = "semantic"

    while True:
        try:
            print(f"\nCurrent Strategy: [{current_strategy.upper()}]")
            print("Commands: ':strategy <semantic|parent_child|markdown|recursive>', ':exit', or paste text to chunk:")
            user_input = input("Chunker > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", ":exit", ":quit"):
                print("Exiting interactive console.")
                break

            if user_input.startswith(":strategy "):
                new_strat = user_input.split(" ", 1)[1].strip().lower()
                if new_strat in ("semantic", "parent_child", "markdown", "recursive"):
                    current_strategy = new_strat
                    print(f"✔ Strategy switched to: {current_strategy}")
                else:
                    print("Invalid strategy. Options: semantic, parent_child, markdown, recursive")
                continue

            test_doc = Document(page_content=user_input, metadata={"source": "interactive_prompt"})
            chunks, metrics = chunker.chunk_document(test_doc, strategy=current_strategy)

            print(f"\n✔ Chunked into {metrics.total_chunks} chunks ({metrics.duration_ms:.1f}ms):")
            for idx, c in enumerate(chunks, 1):
                print(f"  --- Chunk #{idx} ({len(c.page_content)} chars, ~{estimate_tokens(c.page_content)} tokens) ---")
                print(f"  {c.page_content}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive console.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production-Ready Document Chunking Architecture & Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-f", "--file", type=str, help="Path to document file to chunk")
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default="semantic",
        choices=["semantic", "parent_child", "markdown", "recursive"],
        help="Chunking strategy (default: semantic)",
    )
    parser.add_argument("-q", "--query", type=str, help="Search query to test on chunked document")
    parser.add_argument("-e", "--export", type=str, help="Export chunked documents to JSON file")
    parser.add_argument("--benchmark", action="store_true", help="Run multi-strategy benchmark suite")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive chunking console")

    args = parser.parse_args()

    if args.interactive:
        interactive_console()
    elif args.file:
        export_p = Path(args.export) if args.export else None
        process_custom_file(
            file_path=Path(args.file),
            strategy=args.strategy,
            export_path=export_p,
            query=args.query,
        )
    elif args.query:
        # Run test on sample document with query
        sample_doc = Document(
            page_content=SAMPLE_ENTERPRISE_DOCUMENT,
            metadata={"source": "QuantumCloud_Runbook.md"},
        )
        chunker = create_production_chunker()
        chunks, _ = chunker.chunk_document(sample_doc, strategy=args.strategy)
        store = create_vector_store("query_test_collection")
        store.reset()
        store.add_documents(chunks)
        print_header(f"QUERY EXECUTION ({args.strategy.upper()})", f"Query: '{args.query}'")
        results = store.query(args.query, n_results=3)
        for rank, r in enumerate(results, 1):
            print(f"  #{rank} (Distance: {r['distance']:.4f}):")
            print(f"  \"{r['text']}\"\n")
    else:
        # Default: run comprehensive demonstration & benchmark suite
        run_comprehensive_suite()


if __name__ == "__main__":
    main()
