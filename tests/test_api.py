"""Tests for FastAPI endpoints and rate limiting."""

import pytest
from httpx import ASGITransport, AsyncClient
from production_api.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
        assert "X-Process-Time" in response.headers
        assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "components" in data


@pytest.mark.asyncio
async def test_metrics_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests" in data
        assert "latency_ms" in data
        assert "errors" in data
        assert "cache" in data
        assert "llm" in data


@pytest.mark.asyncio
async def test_chat_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "message": "Hello, how are you?",
            "session_id": "test-session-1",
            "use_cache": True,
        }
        response = await client.post("/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["session_id"] == "test-session-1"


@pytest.mark.asyncio
async def test_chat_stream_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "message": "Hello",
            "session_id": "stream-session-1",
            "use_cache": False,
        }
        response = await client.post("/chat/stream", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "data:" in response.text


@pytest.mark.asyncio
async def test_feedback_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "score": 1.0,
            "session_id": "test-session-1",
            "comment": "Great response!",
        }
        response = await client.post("/feedback", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"
        assert "feedback_id" in data


@pytest.mark.asyncio
async def test_cache_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Get cache stats
        stats_resp = await client.get("/cache/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert "total_entries" in stats
        assert "hit_count" in stats

        # Clear cache
        clear_resp = await client.post("/cache/clear")
        assert clear_resp.status_code == 200
        assert clear_resp.json()["status"] == "success"
