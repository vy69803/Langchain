#!/usr/bin/env python3
"""Monitoring Demonstration & Telemetry Harness for LangChain RAG Pipelines.

Demonstrates:
  1. MetricsCollector: Aggregating requests, errors, input/output tokens, latency, and cache hit rates.
  2. Metrics Summary Display: Exact format matching production dashboard requirements.
  3. LangChain MetricsCallbackHandler: Automated telemetry on LLM invocations.
  4. RAGMetricsTracker: Tracking end-to-end RAG latency and errors.
  5. Prometheus Export: Generating prometheus-compliant exposition text.

Usage:
  # Run the full monitoring demonstration:
  python monitoring.py

  # Test single query with live metrics tracking:
  python monitoring.py -q "What is ChromaDB used for?"

  # Run interactive monitoring console:
  python monitoring.py --interactive
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src directory is in sys.path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
from langchain_rag.cost_optimization import SemanticCache, estimate_tokens
from langchain_rag.llm import get_llm
from langchain_rag.monitoring import (
    MetricsCallbackHandler,
    MetricsCollector,
    RAGMetricsTracker,
)

load_dotenv()


class MonitoringDemo:
    """Demonstration class to showcase and test all telemetry & metrics collection features."""

    def __init__(self) -> None:
        self.collector = MetricsCollector()
        self.callback_handler = MetricsCallbackHandler(collector=self.collector)
        self.rag_tracker = RAGMetricsTracker(collector=self.collector)
        self.cache = SemanticCache()

    def run_direct_simulation(self) -> None:
        """Simulate direct request recording matching sample workflow."""
        print("=" * 65)
        print(" SCENARIO 1: Direct MetricsCollector Tracking Simulation".center(65))
        print("=" * 65)
        print("Simulating 3 requests (2 LLM API calls + 1 Cache Hit)...")

        # Request 1: Normal LLM Call
        self.collector.record_request(
            latency_ms=1240.50,
            input_tokens=5,
            output_tokens=320,
            error=False,
            cache_hit=False,
        )

        # Request 2: Normal LLM Call
        self.collector.record_request(
            latency_ms=1450.20,
            input_tokens=9,
            output_tokens=556,
            error=False,
            cache_hit=False,
        )

        # Request 3: Cache Hit (sub-millisecond latency, zero tokens)
        self.collector.record_request(
            latency_ms=3.10,
            input_tokens=0,
            output_tokens=0,
            error=False,
            cache_hit=True,
        )

        print("\nDisplaying Output:")
        self.collector.print_summary()

    def run_live_llm_chain_demo(self) -> None:
        """Run live LLM queries with automated Callback and Cache tracking."""
        print("\n" + "=" * 65)
        print(" SCENARIO 2: Live LLM Invocation with Automated Callback".center(65))
        print("=" * 65)

        # Reset collector for a clean live test
        self.collector.reset()

        queries = [
            "Explain vector embeddings in one sentence.",
            "What is ChromaDB used for?",
            "Explain vector embeddings in one sentence.",  # Repeat -> Cache Hit
        ]

        print("Executing queries through monitored pipeline...\n")
        for i, q in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] Query: \"{q}\"")

            # Check cache
            cached_resp, score = self.cache.get(q)
            if cached_resp is not None:
                print(f"   ⚡ Cache Hit (Sim: {score:.4f}) -> 0 tokens, instant response.")
                self.collector.record_request(
                    latency_ms=1.85,
                    input_tokens=0,
                    output_tokens=0,
                    error=False,
                    cache_hit=True,
                )
                print(f"   Answer: {cached_resp}\n")
                continue

            # Execute LLM call with LangChain callback handler
            start_t = time.perf_counter()
            try:
                llm = get_llm(callbacks=[self.callback_handler])
                resp = llm.invoke(q)
                answer = resp.content if hasattr(resp, "content") else str(resp)
                lat = (time.perf_counter() - start_t) * 1000.0
                
                # Seed cache
                self.cache.set(q, answer)
                print(f"   🌐 API Call ({lat:.2f} ms)")
                print(f"   Answer: {answer.strip()}\n")
            except Exception as err:
                lat = (time.perf_counter() - start_t) * 1000.0
                print(f"   ✖ Error: {err}")
                self.collector.record_request(
                    latency_ms=lat,
                    input_tokens=estimate_tokens(q),
                    output_tokens=0,
                    error=True,
                    cache_hit=False,
                )

        print("-" * 65)
        self.collector.print_summary()

    def show_prometheus_metrics(self) -> None:
        """Display Prometheus formatted metrics."""
        print("\n" + "=" * 65)
        print(" SCENARIO 3: Prometheus Metrics Exposition".center(65))
        print("=" * 65)
        print(self.collector.to_prometheus_format())


def interactive_console(demo: MonitoringDemo) -> None:
    """Interactive console for real-time metrics telemetry."""
    print("=" * 65)
    print(" INTERACTIVE RAG MONITORING CONSOLE ".center(65))
    print("=" * 65)
    print("Enter queries to invoke LLM, observe telemetry, and track metrics.")
    print("Type 'summary' to view metrics, 'reset' to clear, 'exit' to quit.\n")

    while True:
        try:
            q = input("\nMonitor-Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not q or q.lower() in ("exit", "quit", "q"):
            print("\nFinal Session Metrics:")
            demo.collector.print_summary()
            break

        if q.lower() == "summary":
            demo.collector.print_summary()
            continue

        if q.lower() == "reset":
            demo.collector.reset()
            print("✔ Metrics reset to zero.")
            continue

        # Check cache
        cached_resp, score = demo.cache.get(q)
        if cached_resp:
            print(f"⚡ [CACHE HIT] Similarity: {score:.4f} | Latency: ~2.0 ms | Tokens: 0")
            demo.collector.record_request(
                latency_ms=2.0,
                input_tokens=0,
                output_tokens=0,
                error=False,
                cache_hit=True,
            )
            print(f"Answer:\n{cached_resp}")
            continue

        # LLM Call
        start_t = time.perf_counter()
        try:
            llm = get_llm(callbacks=[demo.callback_handler])
            resp = llm.invoke(q)
            ans = resp.content if hasattr(resp, "content") else str(resp)
            lat = (time.perf_counter() - start_t) * 1000.0
            demo.cache.set(q, ans)
            print(f"🌐 [API CALL] Latency: {lat:.2f} ms")
            print(f"Answer:\n{ans}")
        except Exception as e:
            lat = (time.perf_counter() - start_t) * 1000.0
            print(f"✖ [ERROR] {e}")
            demo.collector.record_request(
                latency_ms=lat,
                input_tokens=estimate_tokens(q),
                output_tokens=0,
                error=True,
                cache_hit=False,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitoring & Telemetry Harness for LangChain RAG")
    parser.add_argument("-q", "--query", type=str, help="Single query to test and monitor")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive telemetry console")

    args = parser.parse_args()
    demo = MonitoringDemo()

    if args.interactive:
        interactive_console(demo)
        return

    if args.query:
        demo.collector.reset()
        start_t = time.perf_counter()
        try:
            llm = get_llm(callbacks=[demo.callback_handler])
            resp = llm.invoke(args.query)
            ans = resp.content if hasattr(resp, "content") else str(resp)
            lat = (time.perf_counter() - start_t) * 1000.0
            print(f"\nQuery: {args.query}")
            print(f"Latency: {lat:.2f} ms")
            print(f"Answer:\n{ans}\n")
        except Exception as e:
            print(f"Error: {e}")
        demo.collector.print_summary()
        return

    # Run full demonstration
    demo.run_direct_simulation()
    demo.run_live_llm_chain_demo()
    demo.show_prometheus_metrics()


if __name__ == "__main__":
    main()
