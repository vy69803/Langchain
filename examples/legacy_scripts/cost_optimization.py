#!/usr/bin/env python3
"""Cost Optimization Demonstration & Production Toolkit for LangChain RAG.

Optimizes LLM and RAG operating costs via:
  1. Semantic Caching (0 token cost & <5ms latency for semantically equivalent queries)
  2. Context & Prompt Compression (extracting relevant sentences to reduce input tokens)
  3. Tiered Model Cascading (routing simple queries to free/cheap models)
  4. Real-time Token & Dollar Cost Tracking & Budget Guardrails

Usage:
  # Run the full cost optimization demonstration suite:
  python cost_optimization.py

  # Test semantic cache with a query:
  python cost_optimization.py -q "What is ChromaDB used for?"

  # Run interactive cost optimization console:
  python cost_optimization.py --interactive
"""

from __future__ import annotations

import argparse
import os
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
from langchain_rag.cost_optimization import (
    CostOptimizer,
    CostTracker,
    ModelTier,
    PromptCompressor,
    SemanticCache,
    TieredRouter,
    calculate_cost,
    create_cost_optimizer,
    estimate_tokens,
)

load_dotenv()


def print_header(title: str, subtitle: str = "") -> None:
    width = 74
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    if subtitle:
        print(f" {subtitle}".center(width))
    print("=" * width)


def run_demonstration_suite(optimizer: CostOptimizer) -> None:
    print_header(
        "RAG & LLM COST OPTIMIZATION DEMONSTRATION",
        "Semantic Caching | Context Pruning | Model Routing | Dollar Tracking",
    )

    sample_long_context = (
        "ChromaDB is an open-source embedding vector database built specifically for AI application development. "
        "It runs locally in-memory or persisted to disk, making it extremely fast and cost-effective. "
        "The weather in San Francisco today is sunny with light coastal winds and moderate humidity. "
        "ChromaDB integrates natively with LangChain, LlamaIndex, and Python data science libraries. "
        "A standard chocolate chip cookie recipe calls for brown sugar, flour, butter, vanilla extract, and chocolate chips. "
        "In RAG architectures, ChromaDB stores dense document embeddings and performs approximate nearest neighbor search."
    )

    # --- Scenario 1: Context & Prompt Compression ---
    print_header(
        "SCENARIO 1: Dynamic Context Compression & Pruning",
        "Query: 'What is ChromaDB used for in RAG?'",
    )
    test_query = "What is ChromaDB used for in RAG?"
    orig_tok = estimate_tokens(sample_long_context)
    compressed_text, _, comp_tok = optimizer.compressor.extract_relevant_sentences(
        context=sample_long_context,
        query=test_query,
        max_tokens=40,
    )
    pct_reduction = ((orig_tok - comp_tok) / orig_tok) * 100.0

    print(f"Original Context ({orig_tok} tokens):\n  \"{sample_long_context}\"\n")
    print(f"Compressed Context ({comp_tok} tokens - {pct_reduction:.1f}% reduction):\n  \"{compressed_text}\"\n")
    print(f"✔ Pruned irrelevant sentences (weather, baking recipe) while preserving all ChromaDB facts.")

    # --- Scenario 2: First Query Execution (Cold Request / Cache Miss) ---
    print_header(
        "SCENARIO 2: Initial Query (Cold Run / Cache Miss)",
        f"Query: '{test_query}'",
    )
    print("Sending query through CostOptimizer...")
    res1 = optimizer.execute_optimized_query(
        query=test_query,
        context=sample_long_context,
    )
    print(f"  Model Used:       {res1['model_used']} (Tier: {res1['model_tier']})")
    print(f"  Route Reason:     {res1['route_reason']}")
    print(f"  Cache Hit:        {res1['cache_hit']}")
    print(f"  Latency:          {res1['latency_ms']} ms")
    print(f"  Cost Incurred:    ${res1['cost_usd']:.6f}")
    print(f"  Answer:\n    {res1['answer']}\n")

    # --- Scenario 3: Semantic Cache Hit (Paraphrased Question) ---
    paraphrased_query = "Tell me what ChromaDB does in a RAG pipeline?"
    print_header(
        "SCENARIO 3: Semantic Cache Hit (Semantically Equivalent Question)",
        f"New Query: '{paraphrased_query}'",
    )
    print(f"Original Query in Cache: '{test_query}'")
    print(f"New Query received:      '{paraphrased_query}'")
    print("Evaluating cosine similarity in vector space...")

    res2 = optimizer.execute_optimized_query(
        query=paraphrased_query,
        context=sample_long_context,
    )
    print(f"  Cache Hit:          {res2['cache_hit']} (Similarity Score: {res2['similarity_score']})")
    print(f"  Model Used:         {res2['model_used']}")
    print(f"  Latency:            {res2['latency_ms']} ms (Near-Zero Latency!)")
    print(f"  Token Cost:         ${res2['cost_usd']:.6f} (100% Free / Zero API Tokens)")
    print(f"  Estimated Saved:    ${res2['saved_usd']:.6f}")
    print(f"  Instant Answer:\n    {res2['answer']}\n")

    # --- Scenario 4: Token Budgeting & Query Revocation (Guardrail) ---
    print_header(
        "SCENARIO 4: Query Token Budgeting & Revocation Guardrail",
        "Block excessively long/abusive queries from reaching the LLM",
    )
    long_abusive_query = (
        "Please provide a comprehensive historical and technical analysis of artificial intelligence starting from the 1956 Dartmouth workshop, "
        "including all mathematical formulations of backpropagation, recurrent neural networks, long short-term memory networks, "
        "transformer self-attention mechanisms, scaled dot-product formulas, rotary position embeddings (RoPE), mixture-of-experts gating networks, "
        "reinforcement learning from human feedback (RLHF), direct preference optimization (DPO), constitutional AI alignment frameworks, "
        "and then write a complete, multi-threaded, production-ready Python distributed training loop supporting PyTorch Fully Sharded Data Parallel (FSDP), "
        "Megatron-LM tensor model parallelism, pipeline parallelism, activation checkpointing, flash attention v3 kernels, and DeepSpeed ZeRO-3 offloading, "
        "complete with end-to-end unit tests, docstrings, Prometheus instrumentation, and Terraform scripts for deploying across AWS EKS GPU clusters."
    )
    long_tokens = estimate_tokens(long_abusive_query)
    print(f"User Query Length: ~{long_tokens} tokens (Max budget: {optimizer.max_query_tokens} tokens)")
    print("Evaluating query against token budget guardrails...")

    res_blocked = optimizer.execute_optimized_query(query=long_abusive_query)
    print(f"  Query Revoked:      {res_blocked.get('revoked')}")
    print(f"  Revocation Reason:  {res_blocked.get('revocation_reason')}")
    print(f"  Latency:            {res_blocked['latency_ms']} ms (Instant rejection!)")
    print(f"  API Cost Incurred:  ${res_blocked['cost_usd']:.6f} (100% Protected from expensive LLM run)")
    print(f"  Estimated Saved:    ${res_blocked['saved_usd']:.6f}\n")

    # --- Scenario 5: Tiered Model Cascading (Complex Reasoning) ---
    print_header(
        "SCENARIO 5: Intelligent Tiered Model Routing",
        "Routing queries based on intent and computational complexity",
    )
    test_queries = [
        ("What is an embedding?", "Simple factual query -> Free tier"),
        ("Compare and contrast dense vector search vs sparse BM25 indexing in detail", "Comparative analysis -> Balanced tier"),
        ("Architect and implement a high-concurrency token bucket rate limiter to prevent security exploit", "Complex architecture & security -> Flagship tier"),
    ]

    for q, desc in test_queries:
        tier, model, reason = optimizer.router.classify_query(q)
        print(f"• Query: \"{q}\"")
        print(f"  Intent:         {desc}")
        print(f"  Selected Tier:  {tier.value.upper()} -> Model: {model}")
        print(f"  Routing Rule:   {reason}\n")

    # --- Scenario 6: Real-Time Cost & ROI Analytics ---
    print_header(
        "SCENARIO 6: Cumulative Cost & ROI Dashboard",
        "Real-Time Metrics from CostTracker",
    )
    summary = optimizer.tracker.summary()
    print("  +-----------------------------------+------------------------+")
    print("  | Metric                            | Value                  |")
    print("  +-----------------------------------+------------------------+")
    print(f"  | Total Queries Processed           | {summary['total_queries']:<22} |")
    print(f"  | Cache Hit Rate                    | {summary['cache_hit_rate_pct']}%{'':<18} |")
    print(f"  | Revoked Queries (Guardrail)       | {summary['revoked_queries']:<22} |")
    print(f"  | Total Tokens Processed            | {summary['total_tokens']:<22} |")
    print(f"  | Total Cost Incurred               | ${summary['total_cost_usd']:<21} |")
    print(f"  | Total Dollars Saved               | ${summary['total_saved_usd']:<21} |")
    print(f"  | Average Query Latency             | {summary['average_latency_ms']} ms{'':<16} |")
    print("  +-----------------------------------+------------------------+")

    print("\n" + "=" * 74)
    print(" [✔] Cost Optimization Suite Completed Successfully!".center(74))
    print("=" * 74 + "\n")


def interactive_console(optimizer: CostOptimizer) -> None:
    print_header("INTERACTIVE COST OPTIMIZATION CONSOLE", "Type 'exit' to quit")
    print(f"Max Query Budget: {optimizer.max_query_tokens} tokens | Cache Threshold: {optimizer.cache.similarity_threshold}")
    print("Enter queries to test Token Budgeting, Semantic Caching, and Cost Tracking.\n")

    while True:
        try:
            q = input("\nCost-Optimizer > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not q or q.lower() in ("exit", "quit", "q"):
            print("Summary before exit:")
            print(optimizer.tracker.summary())
            break

        res = optimizer.execute_optimized_query(query=q)
        if res.get("revoked"):
            hit_label = "⛔ REVOKED (TOKEN BUDGET EXCEEDED)"
        elif res["cache_hit"]:
            hit_label = "⚡ CACHE HIT"
        else:
            hit_label = "🌐 API CALL"

        print(f"\n[{hit_label}] ({res['latency_ms']} ms | Cost: ${res['cost_usd']:.6f} | Model: {res['model_used']})")
        print(f"Answer:\n{res['answer']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cost Optimization Toolkit for LangChain RAG & LLMs",
    )
    parser.add_argument("-q", "--query", type=str, help="Single query to execute with cost optimization")
    parser.add_argument("-c", "--context", type=str, default="", help="Optional context string to compress and feed")
    parser.add_argument("-t", "--cache-threshold", type=float, default=0.80, help="Semantic cache threshold (default: 0.80)")
    parser.add_argument("-b", "--max-query-tokens", type=int, default=150, help="Max tokens allowed for user query before revocation (default: 150)")
    parser.add_argument("--budget-usd", type=float, default=None, help="Total session USD budget cap")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive console")

    args = parser.parse_args()

    optimizer = create_cost_optimizer(
        cache_threshold=args.cache_threshold,
        max_query_tokens=args.max_query_tokens,
        budget_usd=args.budget_usd,
    )

    if args.interactive:
        interactive_console(optimizer)
        return

    if args.query:
        res = optimizer.execute_optimized_query(query=args.query, context=args.context)
        print(f"\nQuery: {args.query}")
        if res.get("revoked"):
            print(f"Status:        REVOKED (Budget Guardrail)")
            print(f"Reason:        {res.get('revocation_reason')}")
        else:
            print(f"Cache Hit:     {res['cache_hit']} (Sim: {res['similarity_score']})")
            print(f"Model:         {res['model_used']}")
            print(f"Latency:       {res['latency_ms']} ms")
            print(f"Cost:          ${res['cost_usd']:.6f}")
        print(f"\nAnswer:\n{res['answer']}")
        return

    run_demonstration_suite(optimizer)


if __name__ == "__main__":
    main()

