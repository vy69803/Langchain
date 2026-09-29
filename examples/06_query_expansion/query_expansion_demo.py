#!/usr/bin/env python3
"""Query Expansion, Query Rewriting, and HyDE Demonstration.

Demonstrates:
  1. Multi-Query Expansion with Reciprocal Rank Fusion (RRF).
  2. Query Rewriting with Conversation Context (Pronoun/Coreference Resolution).
  3. Hypothetical Document Embeddings (HyDE).
  4. Query Decomposition for Multi-Hop / Comparative Queries.
  5. Multi-Query Expanded Hybrid Search (ChromaDB Dense + BM25 Sparse + RRF).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src/ is on sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from langchain_rag.hybrid_search import create_hybrid_search_engine
from langchain_rag.query_expansion import (
    create_hyde_retriever,
    create_multi_query_retriever,
    create_query_expander,
)
from langchain_rag.rag_pipeline import create_rag_pipeline


SAMPLE_DOCS = [
    (
        "GitLab Duo Enterprise is an AI add-on priced at $39 per user/month billed annually. "
        "It includes Code Suggestions, Chat, Root Cause Analysis, Vulnerability Explanation, and Test Generation."
    ),
    (
        "GitLab Duo Pro is targeted at developer productivity and costs $19 per user/month. "
        "It includes Code Suggestions and Chat within supported IDEs like VS Code and JetBrains."
    ),
    (
        "Error ERR-9021-TOKEN-REVOKED: The personal access token or OAuth token was invalidated by administrator "
        "or expired due to rotation policies. Team members should regenerate credentials in user profile."
    ),
    (
        "GitLab Runner executes CI/CD jobs in Docker containers, Kubernetes pods, or bare-metal machines. "
        "Runners poll the GitLab coordinator via HTTP/HTTPS for queued job payloads."
    ),
    (
        "PostgreSQL connection pooling is handled by PgBouncer to avoid port exhaustion and database memory saturation."
    ),
]


def main() -> None:
    print("=" * 70)
    print("      QUERY EXPANSION, QUERY REWRITING & HyDE DEMONSTRATION      ")
    print("=" * 70)

    # 1. Initialize Pipeline & Knowledge Base
    print("\n[1] Indexing sample knowledge documents into ephemeral Vector Store...")
    pipeline = create_rag_pipeline(collection_name="demo_query_transformations")
    pipeline.index_texts(SAMPLE_DOCS)
    print(f"    Indexed {len(SAMPLE_DOCS)} documents successfully.")

    # 2. Query Expander Instance
    expander = create_query_expander()

    # 3. Query Rewriting Demo (with conversational coreference resolution)
    print("\n" + "-" * 70)
    print("[2] Demonstration: Conversational Query Rewriting")
    print("-" * 70)
    chat_history = [
        {"role": "user", "content": "Tell me about GitLab Duo Enterprise features."},
        {"role": "assistant", "content": "It offers code suggestions, chat, and root cause analysis."},
    ]
    raw_followup = "What does it cost and what is included in that price?"
    print(f"  User Followup Query: '{raw_followup}'")
    print(f"  Chat History Context: User was discussing 'GitLab Duo Enterprise'")

    try:
        rewritten = expander.rewrite_query(raw_followup, chat_history=chat_history)
        print(f"  Rewritten Query     : '{rewritten}'")
    except Exception as e:
        print(f"  [LLM Notice]: {e}")
        rewritten = raw_followup

    # Retrieve with rewritten query
    rewrite_results = pipeline.retrieve(query=raw_followup, k=2, query_transform="rewrite", chat_history=chat_history)
    print(f"  Retrieved Chunk Top Match:\n    {rewrite_results[0]['text'][:120]}...")

    # 4. Multi-Query Expansion Demo (RRF Fusion)
    print("\n" + "-" * 70)
    print("[3] Demonstration: Multi-Query Expansion + Reciprocal Rank Fusion")
    print("-" * 70)
    vague_query = "fix revoked token login error"
    print(f"  Initial Query: '{vague_query}'")

    mq_retriever = create_multi_query_retriever(retriever=pipeline.vector_store, num_queries=3)
    fused_hits, generated_queries = mq_retriever.retrieve(
        query=vague_query,
        k=2,
        return_expansion_details=True,
    )

    print("  Expanded Query Variations:")
    for idx, q_var in enumerate(generated_queries, start=1):
        print(f"    ({idx}) {q_var}")

    print(f"\n  Top Fused Result (RRF):")
    for r in fused_hits:
        print(f"    - Rank #{r['fused_rank']} [RRF Score: {r['fused_score']} | Matched by {r['query_count']} queries]:")
        print(f"      {r['text'][:120]}...")

    # 5. HyDE (Hypothetical Document Embeddings) Demo
    print("\n" + "-" * 70)
    print("[4] Demonstration: HyDE (Hypothetical Document Embeddings)")
    print("-" * 70)
    hyde_query = "What is the monthly cost for enterprise AI features?"
    print(f"  Query: '{hyde_query}'")

    hyde_retriever = create_hyde_retriever(vector_retriever=pipeline.vector_store)
    hyde_hits, hypothetical_doc = hyde_retriever.retrieve(
        query=hyde_query,
        k=1,
        return_hypothetical_doc=True,
    )
    print(f"  Generated Hypothetical Document Passage:\n    \"{hypothetical_doc[:160]}...\"")
    print(f"\n  Retrieved Real Match via HyDE Embedding:\n    \"{hyde_hits[0]['text'][:140]}...\"")

    # 6. Combined: Multi-Query + Hybrid Search (Dense ChromaDB + Sparse BM25 + RRF)
    print("\n" + "-" * 70)
    print("[5] Ultimate Pipeline: Multi-Query Expansion + Dense (Chroma) + Sparse (BM25)")
    print("-" * 70)
    hybrid_engine = create_hybrid_search_engine(collection_name="demo_multiquery_hybrid")
    hybrid_engine.add_texts(SAMPLE_DOCS)

    test_prompt = "how do runner jobs communicate with gitlab"
    print(f"  Query: '{test_prompt}'")
    hybrid_expanded_hits = hybrid_engine.search_expanded(test_prompt, num_queries=2, k=2)

    for r in hybrid_expanded_hits:
        print(f"  [Fused Rank #{r['fused_rank']} | Score: {r['fused_score']}]:")
        print(f"    {r['text'][:140]}...")

    print("\n" + "=" * 70)
    print("      DEMONSTRATION COMPLETED SUCCESSFULLY      ")
    print("=" * 70)


if __name__ == "__main__":
    main()
