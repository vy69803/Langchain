"""Monitoring and Metrics Collection Module for LangChain RAG Pipelines.

Provides:
  1. MetricsCollector: Core aggregation engine tracking requests, latencies, tokens, errors, and cache hits.
  2. MetricsCallbackHandler: LangChain BaseCallbackHandler for automated telemetry on LLM/chain execution.
  3. RAGMetricsTracker: Context manager and wrapper for measuring retrieval, generation, and cache performance.
"""

from __future__ import annotations

import time
from typing import Any, Sequence
from dataclasses import dataclass, field

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class MetricsCollector:
    """Production metrics collector for LLM requests, latency, tokens, errors, and cache hits.

    Tracks cumulative counters and raw sample distributions for computing averages,
    rates, and percentile latencies (p50, p95, p99).
    """

    def __init__(self) -> None:
        self.metrics: dict[str, Any] = {
            "requests_total": 0,
            "latency_sum": 0.0,
            "latency_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "errors_total": 0,
            "cache_hits_total": 0,
        }
        self._latencies: list[float] = []

    def record_request(
        self,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        error: bool = False,
        cache_hit: bool = False,
    ) -> None:
        """Record a completed or failed request."""
        self.metrics["requests_total"] += 1
        self.metrics["latency_sum"] += latency_ms
        self.metrics["latency_count"] += 1
        self.metrics["tokens_input"] += input_tokens
        self.metrics["tokens_output"] += output_tokens
        self._latencies.append(latency_ms)

        if error:
            self.metrics["errors_total"] += 1

        if cache_hit:
            self.metrics["cache_hits_total"] += 1

    def reset(self) -> None:
        """Reset all metrics back to zero."""
        for key in self.metrics:
            if isinstance(self.metrics[key], float):
                self.metrics[key] = 0.0
            else:
                self.metrics[key] = 0
        self._latencies.clear()

    def get_percentile_latency(self, percentile: float) -> float:
        """Calculate percentile latency (e.g. 50, 95, 99) in milliseconds."""
        if not self._latencies:
            return 0.0
        sorted_latencies = sorted(self._latencies)
        idx = int((percentile / 100.0) * len(sorted_latencies))
        idx = min(idx, len(sorted_latencies) - 1)
        return round(sorted_latencies[idx], 2)

    def get_summary(self) -> dict[str, Any]:
        """Compute structured metrics summary."""
        total_requests = self.metrics["requests_total"]
        total_errors = self.metrics["errors_total"]
        latency_count = self.metrics["latency_count"]
        latency_sum = self.metrics["latency_sum"]
        cache_hits = self.metrics["cache_hits_total"]

        avg_latency = (latency_sum / latency_count) if latency_count > 0 else 0.0
        error_rate = (total_errors / total_requests * 100.0) if total_requests > 0 else 0.0
        cache_hit_rate = (cache_hits / total_requests * 100.0) if total_requests > 0 else 0.0

        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate_pct": round(error_rate, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "p50_latency_ms": self.get_percentile_latency(50),
            "p95_latency_ms": self.get_percentile_latency(95),
            "p99_latency_ms": self.get_percentile_latency(99),
            "total_input_tokens": self.metrics["tokens_input"],
            "total_output_tokens": self.metrics["tokens_output"],
            "total_tokens": self.metrics["tokens_input"] + self.metrics["tokens_output"],
            "total_cache_hits": cache_hits,
            "cache_hit_rate_pct": round(cache_hit_rate, 2),
        }

    def print_summary(self) -> None:
        """Print formatted metrics summary to stdout."""
        summary = self.get_summary()
        print("Metrics Summary:")
        print(f"  total_requests: {summary['total_requests']}")
        print(f"  total_errors: {summary['total_errors']}")
        print(f"  error_rate: {summary['error_rate_pct']:.2f}%")
        print(f"  avg_latency_ms: {summary['avg_latency_ms']:.2f}")
        print(f"  total_input_tokens: {summary['total_input_tokens']}")
        print(f"  total_output_tokens: {summary['total_output_tokens']}")
        print(f"  cache_hit_rate: {summary['cache_hit_rate_pct']:.2f}%")

    def to_prometheus_format(self) -> str:
        """Export metrics in standard Prometheus exposition format."""
        s = self.get_summary()
        return (
            f"# HELP llm_requests_total Total number of LLM requests processed\n"
            f"# TYPE llm_requests_total counter\n"
            f"llm_requests_total {s['total_requests']}\n\n"
            f"# HELP llm_errors_total Total number of LLM errors\n"
            f"# TYPE llm_errors_total counter\n"
            f"llm_errors_total {s['total_errors']}\n\n"
            f"# HELP llm_tokens_input_total Total input prompt tokens\n"
            f"# TYPE llm_tokens_input_total counter\n"
            f"llm_tokens_input_total {s['total_input_tokens']}\n\n"
            f"# HELP llm_tokens_output_total Total generated completion tokens\n"
            f"# TYPE llm_tokens_output_total counter\n"
            f"llm_tokens_output_total {s['total_output_tokens']}\n\n"
            f"# HELP llm_latency_average_ms Average request latency in milliseconds\n"
            f"# TYPE llm_latency_average_ms gauge\n"
            f"llm_latency_average_ms {s['avg_latency_ms']}\n\n"
            f"# HELP llm_cache_hits_total Total semantic cache hits\n"
            f"# TYPE llm_cache_hits_total counter\n"
            f"llm_cache_hits_total {s['total_cache_hits']}\n"
        )


class MetricsCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that automatically feeds metrics to a MetricsCollector."""

    def __init__(self, collector: MetricsCollector | None = None) -> None:
        super().__init__()
        self.collector = collector or MetricsCollector()
        self._start_times: dict[str, float] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Record start time when LLM invocation begins."""
        key = str(run_id) if run_id else "default"
        self._start_times[key] = time.perf_counter()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Calculate latency and token usage when LLM completes."""
        key = str(run_id) if run_id else "default"
        start_t = self._start_times.pop(key, time.perf_counter())
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        input_tokens = 0
        output_tokens = 0

        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
        else:
            # Fallback heuristic: calculate tokens from generation text
            for gens in response.generations:
                for gen in gens:
                    output_tokens += max(1, len(gen.text.split()))

        self.collector.record_request(
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=False,
            cache_hit=False,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Record error when LLM invocation fails."""
        key = str(run_id) if run_id else "default"
        start_t = self._start_times.pop(key, time.perf_counter())
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        self.collector.record_request(
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            error=True,
            cache_hit=False,
        )


class RAGMetricsTracker:
    """Context manager and tracker for end-to-end RAG pipelines."""

    def __init__(self, collector: MetricsCollector | None = None) -> None:
        self.collector = collector or MetricsCollector()

    def track(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_hit: bool = False,
        error: bool = False,
    ):
        """Context manager to measure latency of a block."""
        return _TrackerContext(
            collector=self.collector,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit=cache_hit,
            error=error,
        )


class _TrackerContext:
    def __init__(
        self,
        collector: MetricsCollector,
        input_tokens: int,
        output_tokens: int,
        cache_hit: bool,
        error: bool,
    ) -> None:
        self.collector = collector
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_hit = cache_hit
        self.error = error
        self.start_time: float = 0.0

    def __enter__(self) -> _TrackerContext:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        latency_ms = (time.perf_counter() - self.start_time) * 1000.0
        has_error = self.error or (exc_type is not None)
        self.collector.record_request(
            latency_ms=latency_ms,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            error=has_error,
            cache_hit=self.cache_hit,
        )


if __name__ == "__main__":
    print("=" * 60)
    print(" Testing MetricsCollector directly from module")
    print("=" * 60)
    collector = MetricsCollector()

    # Record sample requests
    collector.record_request(latency_ms=1240.50, input_tokens=5, output_tokens=320, error=False, cache_hit=False)
    collector.record_request(latency_ms=1450.20, input_tokens=9, output_tokens=556, error=False, cache_hit=False)
    collector.record_request(latency_ms=3.10, input_tokens=0, output_tokens=0, error=False, cache_hit=True)

    print()
    collector.print_summary()
    print("=" * 60)

