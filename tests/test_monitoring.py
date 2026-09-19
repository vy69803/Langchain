"""Tests for JSON logger and MetricsCollector in production_api.monitoring."""

import io
import json
import logging
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from production_api.monitoring import (
    JSONFormatter,
    MetricsCollector,
    StructuredLoggingMiddleware,
    TimingMiddleware,
    get_logger,
    get_metrics_collector,
    metrics_collector,
    setup_json_logger,
)


def test_json_formatter_standard_fields():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Test message with %s",
        args=("param",),
        exc_info=None,
    )
    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["logger"] == "test_logger"
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Test message with param"
    assert parsed["line"] == 42
    assert "timestamp" in parsed


def test_json_formatter_extra_and_exception():
    formatter = JSONFormatter()
    try:
        raise ValueError("Sample error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="error_logger",
        level=logging.ERROR,
        pathname="test.py",
        lineno=10,
        msg="Failed operation",
        args=(),
        exc_info=exc_info,
    )
    record.request_id = "req-12345"
    record.user_id = "user-99"

    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["level"] == "ERROR"
    assert "exception" in parsed
    assert "ValueError: Sample error" in parsed["exception"]
    assert "extra" in parsed
    assert parsed["extra"]["request_id"] == "req-12345"
    assert parsed["extra"]["user_id"] == "user-99"


def test_setup_json_logger():
    stream = io.StringIO()
    test_logger = setup_json_logger(name="test_stream_logger", level="DEBUG", stream=stream)
    test_logger.info("Hello structured log", extra={"key": "val"})

    output = stream.getvalue().strip()
    assert output != ""
    data = json.loads(output)
    assert data["message"] == "Hello structured log"
    assert data["extra"]["key"] == "val"


def test_metrics_collector_basic():
    collector = MetricsCollector()
    collector.record_request(method="GET", endpoint="/health", status_code=200, duration_seconds=0.015)
    collector.record_request(method="POST", endpoint="/query", status_code=200, duration_seconds=0.120)
    collector.record_request(method="POST", endpoint="/query", status_code=500, duration_seconds=0.050)
    collector.record_cache(hit=True)
    collector.record_cache(hit=False)
    collector.record_llm_call(model="claude-3-5-sonnet", prompt_tokens=100, completion_tokens=50, duration_seconds=1.2)
    collector.increment("custom_event", 3)
    collector.gauge("active_sessions", 12.0)

    summary = collector.get_summary()

    assert summary["requests"]["total"] == 3
    assert summary["requests"]["by_method"]["GET"] == 1
    assert summary["requests"]["by_method"]["POST"] == 2
    assert summary["requests"]["by_status"][200] == 2
    assert summary["requests"]["by_status"][500] == 1
    assert summary["errors"]["total"] == 1
    assert summary["cache"]["hits"] == 1
    assert summary["cache"]["misses"] == 1
    assert summary["cache"]["hit_rate_pct"] == 50.0
    assert summary["llm"]["calls_total"] == 1
    assert summary["llm"]["tokens_input"] == 100
    assert summary["llm"]["tokens_output"] == 50
    assert summary["llm"]["tokens_total"] == 150
    assert summary["custom_counters"]["custom_event"] == 3
    assert summary["custom_gauges"]["active_sessions"] == 12.0

    prom = collector.to_prometheus_format()
    assert "http_requests_total 3" in prom
    assert "cache_hits_total 1" in prom
    assert "llm_tokens_total 150" in prom


def test_metrics_collector_percentiles():
    collector = MetricsCollector()
    for lat in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        collector.record_request(method="GET", endpoint="/api", status_code=200, duration_seconds=lat / 1000.0)

    p50 = collector.get_percentile_latency(50)
    p90 = collector.get_percentile_latency(90)
    assert p50 >= 40.0
    assert p90 >= 80.0

    collector.reset()
    assert collector.get_summary()["requests"]["total"] == 0


@pytest.mark.asyncio
async def test_timing_and_logging_middleware():
    test_app = FastAPI()
    custom_collector = MetricsCollector()
    log_stream = io.StringIO()
    test_logger = setup_json_logger(name="test_mw_logger", level="INFO", stream=log_stream)

    test_app.add_middleware(TimingMiddleware, collector=custom_collector, json_logger=test_logger)
    test_app.add_middleware(StructuredLoggingMiddleware, json_logger=test_logger)

    @test_app.get("/test")
    async def sample_endpoint():
        return {"status": "ok"}

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/test")
        assert response.status_code == 200
        assert "X-Process-Time" in response.headers
        assert "X-Request-ID" in response.headers

    # Check that metric was collected
    assert custom_collector.get_summary()["requests"]["total"] == 1
    assert custom_collector.get_summary()["requests"]["by_endpoint"]["/test"] == 1

    # Check log output
    log_content = log_stream.getvalue().strip()
    assert "Incoming request: GET /test" in log_content
    assert "Completed request: GET /test -> 200" in log_content
