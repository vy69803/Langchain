"""Monitoring, telemetry, JSON structured logging, and metrics collection for Production API."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# ============================================================================
# JSON Structured Logger
# ============================================================================

class JSONFormatter(logging.Formatter):
    """Custom logging formatter that outputs structured JSON log entries."""

    # Standard LogRecord attributes to ignore when parsing extra context
    RESERVED_ATTRS = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record as a JSON string."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": record.process,
            "thread_name": record.threadName,
        }

        # Include exception trace if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            log_entry["exception"] = record.exc_text

        if record.stack_info:
            log_entry["stack_info"] = self.formatStack(record.stack_info)

        # Include extra attributes supplied via logger.info("...", extra={...})
        extra_fields: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                extra_fields[key] = value

        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str)


def setup_json_logger(
    name: str = "production_api",
    level: str | int = "INFO",
    stream: Any = sys.stdout,
) -> logging.Logger:
    """Configure and return a structured JSON logger.

    Args:
        name: Logger name.
        level: Log level (e.g. "INFO", "DEBUG", "ERROR").
        stream: Output stream (default: stdout).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if isinstance(level, str):
        log_level = getattr(logging, level.upper(), logging.INFO)
    else:
        log_level = level
    logger.setLevel(log_level)

    # Avoid duplicate handlers if setup is invoked multiple times
    if not any(isinstance(h.formatter, JSONFormatter) for h in logger.handlers):
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    # Avoid propagating to root logger to prevent duplicate logs if root has handlers
    logger.propagate = False
    return logger


def get_logger(name: str = "production_api") -> logging.Logger:
    """Get or create a structured JSON logger."""
    return setup_json_logger(name)


logger = get_logger("production_api")


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """Thread-safe production metrics collector for API requests, latency, errors, cache, and LLM telemetry."""

    def __init__(self, max_latency_samples: int = 10000) -> None:
        self._lock = threading.Lock()
        self.max_latency_samples = max_latency_samples
        self.reset()

    def reset(self) -> None:
        """Reset all metrics back to initial state."""
        with self._lock:
            self._requests_total: int = 0
            self._requests_by_endpoint: dict[str, int] = {}
            self._requests_by_method: dict[str, int] = {}
            self._requests_by_status: dict[int, int] = {}

            self._latency_sum_ms: float = 0.0
            self._latency_count: int = 0
            self._latency_samples: list[float] = []

            self._errors_total: int = 0
            self._errors_by_type: dict[str, int] = {}

            self._cache_hits: int = 0
            self._cache_misses: int = 0

            self._llm_calls_total: int = 0
            self._llm_tokens_input: int = 0
            self._llm_tokens_output: int = 0
            self._llm_errors_total: int = 0
            self._llm_latency_sum_ms: float = 0.0

            self._custom_counters: dict[str, int] = {}
            self._custom_gauges: dict[str, float] = {}

    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        """Record an API request lifecycle event.

        Args:
            method: HTTP method (e.g., 'GET', 'POST').
            endpoint: URL path or route template.
            status_code: HTTP response status code.
            duration_seconds: Duration of request processing in seconds.
        """
        duration_ms = duration_seconds * 1000.0
        method_upper = method.upper()

        with self._lock:
            self._requests_total += 1
            self._requests_by_method[method_upper] = self._requests_by_method.get(method_upper, 0) + 1
            self._requests_by_endpoint[endpoint] = self._requests_by_endpoint.get(endpoint, 0) + 1
            self._requests_by_status[status_code] = self._requests_by_status.get(status_code, 0) + 1

            self._latency_sum_ms += duration_ms
            self._latency_count += 1

            if len(self._latency_samples) < self.max_latency_samples:
                self._latency_samples.append(duration_ms)
            else:
                # Replace a slot to keep representative distribution
                self._latency_samples[self._requests_total % self.max_latency_samples] = duration_ms

            if status_code >= 400:
                self._errors_total += 1
                err_category = f"HTTP_{status_code}"
                self._errors_by_type[err_category] = self._errors_by_type.get(err_category, 0) + 1

    def record_error(self, error_type: str, status_code: int = 500) -> None:
        """Record an application or system error."""
        with self._lock:
            self._errors_total += 1
            self._errors_by_type[error_type] = self._errors_by_type.get(error_type, 0) + 1
            self._requests_by_status[status_code] = self._requests_by_status.get(status_code, 0) + 1

    def record_cache(self, hit: bool) -> None:
        """Record a cache hit or miss event."""
        with self._lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        duration_seconds: float = 0.0,
        success: bool = True,
    ) -> None:
        """Record LLM inference telemetry."""
        with self._lock:
            self._llm_calls_total += 1
            self._llm_tokens_input += prompt_tokens
            self._llm_tokens_output += completion_tokens
            self._llm_latency_sum_ms += duration_seconds * 1000.0

            if not success:
                self._llm_errors_total += 1

    def increment(self, metric_name: str, value: int = 1) -> None:
        """Increment a custom counter."""
        with self._lock:
            self._custom_counters[metric_name] = self._custom_counters.get(metric_name, 0) + value

    def gauge(self, metric_name: str, value: float) -> None:
        """Set a custom gauge value."""
        with self._lock:
            self._custom_gauges[metric_name] = value

    def get_percentile_latency(self, percentile: float) -> float:
        """Calculate percentile request latency in milliseconds."""
        with self._lock:
            if not self._latency_samples:
                return 0.0
            sorted_samples = sorted(self._latency_samples)
            idx = int((percentile / 100.0) * len(sorted_samples))
            idx = min(idx, len(sorted_samples) - 1)
            return round(sorted_samples[idx], 2)

    def get_summary(self) -> dict[str, Any]:
        """Compute structured metrics summary."""
        with self._lock:
            total_reqs = self._requests_total
            total_errors = self._errors_total
            lat_count = self._latency_count
            lat_sum = self._latency_sum_ms
            cache_hits = self._cache_hits
            cache_misses = self._cache_misses
            total_cache_lookups = cache_hits + cache_misses

            avg_latency = (lat_sum / lat_count) if lat_count > 0 else 0.0
            error_rate = (total_errors / total_reqs * 100.0) if total_reqs > 0 else 0.0
            cache_hit_rate = (cache_hits / total_cache_lookups * 100.0) if total_cache_lookups > 0 else 0.0

            llm_calls = self._llm_calls_total
            avg_llm_latency = (self._llm_latency_sum_ms / llm_calls) if llm_calls > 0 else 0.0

            min_latency = min(self._latency_samples) if self._latency_samples else 0.0
            max_latency = max(self._latency_samples) if self._latency_samples else 0.0

            sorted_samples = sorted(self._latency_samples) if self._latency_samples else []

            def calc_p(p: float) -> float:
                if not sorted_samples:
                    return 0.0
                i = min(int((p / 100.0) * len(sorted_samples)), len(sorted_samples) - 1)
                return round(sorted_samples[i], 2)

            return {
                "requests": {
                    "total": total_reqs,
                    "by_method": dict(self._requests_by_method),
                    "by_endpoint": dict(self._requests_by_endpoint),
                    "by_status": dict(self._requests_by_status),
                },
                "latency_ms": {
                    "avg": round(avg_latency, 2),
                    "min": round(min_latency, 2),
                    "max": round(max_latency, 2),
                    "p50": calc_p(50),
                    "p90": calc_p(90),
                    "p95": calc_p(95),
                    "p99": calc_p(99),
                },
                "errors": {
                    "total": total_errors,
                    "rate_pct": round(error_rate, 2),
                    "by_type": dict(self._errors_by_type),
                },
                "cache": {
                    "hits": cache_hits,
                    "misses": cache_misses,
                    "total_lookups": total_cache_lookups,
                    "hit_rate_pct": round(cache_hit_rate, 2),
                },
                "llm": {
                    "calls_total": llm_calls,
                    "errors_total": self._llm_errors_total,
                    "tokens_input": self._llm_tokens_input,
                    "tokens_output": self._llm_tokens_output,
                    "tokens_total": self._llm_tokens_input + self._llm_tokens_output,
                    "avg_latency_ms": round(avg_llm_latency, 2),
                },
                "custom_counters": dict(self._custom_counters),
                "custom_gauges": dict(self._custom_gauges),
            }

    def to_prometheus_format(self) -> str:
        """Export metrics in standard Prometheus text format."""
        summary = self.get_summary()
        reqs = summary["requests"]
        lat = summary["latency_ms"]
        errs = summary["errors"]
        cache = summary["cache"]
        llm = summary["llm"]

        lines = [
            "# HELP http_requests_total Total HTTP requests received",
            "# TYPE http_requests_total counter",
            f"http_requests_total {reqs['total']}",
            "",
            "# HELP http_request_duration_ms_avg Average HTTP request duration in ms",
            "# TYPE http_request_duration_ms_avg gauge",
            f"http_request_duration_ms_avg {lat['avg']}",
            "",
            "# HELP http_request_duration_ms_p95 95th percentile HTTP request duration in ms",
            "# TYPE http_request_duration_ms_p95 gauge",
            f"http_request_duration_ms_p95 {lat['p95']}",
            "",
            "# HELP http_errors_total Total HTTP error responses",
            "# TYPE http_errors_total counter",
            f"http_errors_total {errs['total']}",
            "",
            "# HELP cache_hits_total Total semantic cache hits",
            "# TYPE cache_hits_total counter",
            f"cache_hits_total {cache['hits']}",
            "",
            "# HELP cache_misses_total Total semantic cache misses",
            "# TYPE cache_misses_total counter",
            f"cache_misses_total {cache['misses']}",
            "",
            "# HELP llm_calls_total Total LLM calls initiated",
            "# TYPE llm_calls_total counter",
            f"llm_calls_total {llm['calls_total']}",
            "",
            "# HELP llm_tokens_total Total LLM tokens processed",
            "# TYPE llm_tokens_total counter",
            f"llm_tokens_total {llm['tokens_total']}",
        ]
        return "\n".join(lines) + "\n"


# Global singleton metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global MetricsCollector instance."""
    return metrics_collector


# ============================================================================
# FastAPI Middlewares
# ============================================================================

class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to measure latency, add X-Process-Time header, and record metrics."""

    def __init__(
        self,
        app: Any,
        collector: MetricsCollector | None = None,
        json_logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(app)
        self.collector = collector or metrics_collector
        self.logger = json_logger or logger

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        start_time = time.perf_counter()
        endpoint = request.url.path
        method = request.method

        try:
            response = await call_next(request)
            process_time = time.perf_counter() - start_time
            response.headers["X-Process-Time"] = f"{process_time:.4f}s"

            self.collector.record_request(
                method=method,
                endpoint=endpoint,
                status_code=response.status_code,
                duration_seconds=process_time,
            )
            return response
        except Exception as exc:
            process_time = time.perf_counter() - start_time
            self.collector.record_request(
                method=method,
                endpoint=endpoint,
                status_code=500,
                duration_seconds=process_time,
            )
            self.collector.record_error(error_type=type(exc).__name__, status_code=500)
            self.logger.error(
                f"Unhandled exception during request processing: {exc}",
                exc_info=True,
                extra={
                    "method": method,
                    "endpoint": endpoint,
                    "duration_ms": round(process_time * 1000, 2),
                },
            )
            raise


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log request/response cycles in structured JSON format."""

    def __init__(
        self,
        app: Any,
        json_logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(app)
        self.logger = json_logger or logger

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        start_time = time.perf_counter()

        client_host = request.client.host if request.client else "unknown"
        endpoint = request.url.path
        method = request.method

        self.logger.info(
            f"Incoming request: {method} {endpoint}",
            extra={
                "request_id": request_id,
                "method": method,
                "endpoint": endpoint,
                "client_ip": client_host,
            },
        )

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            response.headers["X-Request-ID"] = request_id

            self.logger.info(
                f"Completed request: {method} {endpoint} -> {response.status_code}",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            self.logger.error(
                f"Request failed: {method} {endpoint} - {exc}",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "method": method,
                    "endpoint": endpoint,
                    "duration_ms": round(duration_ms, 2),
                    "error": str(exc),
                },
            )
            raise
