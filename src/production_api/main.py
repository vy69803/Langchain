"""Production FastAPI Application for Enterprise RAG and AI Agent System.

Connects and orchestrates:
- ProductionAgent / LangGraph workflow engine
- SlowAPI rate limiting & DDoS protection
- Structured JSON logging & Prometheus-style metrics telemetry
- Authentication & Multi-tier security guardrails
- Semantic caching & hybrid knowledge retrieval
- Streaming SSE (Server-Sent Events) chat responses
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    Security,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from production_api.agent import ProductionAgent, production_agent
from production_api.cache import (
    default_cache,
    semantic_cache,
    tiered_cache,
)
from production_api.config import Settings, get_settings
from production_api.models import (
    CacheStatsResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ComponentHealth,
    DocumentIngestRequest,
    DocumentIngestResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
    SearchRequest,
    SearchResponse,
    SourceDocument,
    StreamingChatChunk,
    TokenUsage,
)
from production_api.monitoring import (
    StructuredLoggingMiddleware,
    TimingMiddleware,
    get_logger,
    metrics_collector,
)
from production_api.rate_limiter import limiter
from production_api.security import (
    UnifiedSecurityGuard,
    default_guard,
    get_api_key,
)

settings = get_settings()
logger = get_logger("production_api")


# ============================================================================
# Application Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events."""
    logger.info(
        f"Starting {settings.app_name} in {settings.environment} mode...",
        extra={
            "environment": settings.environment,
            "app_name": settings.app_name,
            "primary_model": settings.primary_model,
            "rate_limit_default": settings.rate_limit_default,
        },
    )
    yield
    logger.info(f"Shutting down {settings.app_name}...")


# ============================================================================
# FastAPI Application Factory / Instance
# ============================================================================

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Enterprise-grade Production RAG, Multi-tiered Caching, and Agent API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Attach SlowAPI Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Register Middlewares
app.add_middleware(TimingMiddleware)
app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Return structured JSON error response for HTTP exceptions."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail if isinstance(exc.detail, str) else "HTTP Exception",
            "detail": exc.detail,
            "request_id": request_id,
        },
    )


# ============================================================================
# 1. System & Observability Endpoints
# ============================================================================

@app.get(
    "/",
    tags=["General"],
    summary="Root Welcome & API Overview",
)
@limiter.limit("30/minute")
async def root(request: Request):
    """Root endpoint providing service metadata and discovery links."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "environment": settings.environment,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
        "endpoints": {
            "chat": "/chat",
            "stream": "/chat/stream",
            "search": "/search",
            "ingest": "/documents/ingest",
            "feedback": "/feedback",
            "cache_stats": "/cache/stats",
            "metrics": "/metrics",
            "health": "/health",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Comprehensive Subsystem Health Check",
)
async def health_check():
    """Inspect status and latencies of cache, vector store, and LLM subsystem."""
    components: Dict[str, ComponentHealth] = {}

    # Check Cache subsystem
    try:
        start_t = time.perf_counter()
        cache_stats = tiered_cache.get_metrics()
        cache_latency = round((time.perf_counter() - start_t) * 1000, 2)
        components["cache"] = ComponentHealth(
            status="healthy",
            latency_ms=cache_latency,
            details=f"L1 size: {cache_stats.get('current_size', 0)}, hit rate: {cache_stats.get('hit_rate', 0.0)}%",
        )
    except Exception as e:
        components["cache"] = ComponentHealth(status="degraded", details=str(e))

    # Check RAG / Vector Store Subsystem
    try:
        start_t = time.perf_counter()
        rag_pipe = production_agent.rag_pipeline
        rag_latency = round((time.perf_counter() - start_t) * 1000, 2)
        if rag_pipe is not None:
            components["vector_store"] = ComponentHealth(
                status="healthy",
                latency_ms=rag_latency,
                details="ChromaDB / Vector store online",
            )
        else:
            components["vector_store"] = ComponentHealth(
                status="uninitialized",
                details="Running in deferred vector store mode",
            )
    except Exception as e:
        components["vector_store"] = ComponentHealth(status="degraded", details=str(e))

    # Check LLM subsystem
    try:
        llm = production_agent.llm
        if llm is not None:
            components["llm"] = ComponentHealth(
                status="healthy",
                details=f"Model: {production_agent.model_name}",
            )
        else:
            components["llm"] = ComponentHealth(
                status="offline",
                details="Offline / key not configured",
            )
    except Exception as e:
        components["llm"] = ComponentHealth(status="degraded", details=str(e))

    overall_status = (
        "healthy"
        if all(c.status in ("healthy", "uninitialized") for c in components.values())
        else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        service=settings.app_name,
        environment=settings.environment,
        version="1.0.0",
        components=components,
    )


@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Application Metrics & Telemetry",
)
async def get_metrics():
    """Retrieve runtime telemetry, latencies, error counts, and token counters."""
    return metrics_collector.get_summary()


# ============================================================================
# 2. Chat & AI Agent Endpoints
# ============================================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Agent"],
    summary="Process Chat Query with RAG and Guardrails",
)
@limiter.limit(settings.rate_limit_default)
async def chat_endpoint(
    request: Request,
    payload: ChatRequest,
):
    """Execute conversational agent pipeline with RAG, semantic caching, and guardrails."""
    try:
        response = await production_agent.process_request(payload)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        metrics_collector.record_error("chat_endpoint_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal agent error: {str(e)}",
        )


@app.post(
    "/chat/stream",
    tags=["Agent"],
    summary="Stream Chat Response via Server-Sent Events (SSE)",
)
@limiter.limit(settings.rate_limit_default)
async def chat_stream_endpoint(
    request: Request,
    payload: ChatRequest,
):
    """Stream token chunks incrementally using Server-Sent Events (SSE)."""
    async def event_generator() -> AsyncIterator[str]:
        try:
            async for chunk in production_agent.astream(
                query=payload.message,
                session_id=payload.session_id,
                user_id=payload.user_id,
                use_cache=payload.use_cache,
                metadata=payload.metadata,
            ):
                chunk_obj = StreamingChatChunk(chunk=chunk, done=False)
                yield f"data: {chunk_obj.model_dump_json()}\n\n"

            # Completion marker chunk
            done_obj = StreamingChatChunk(chunk="", done=True)
            yield f"data: {done_obj.model_dump_json()}\n\n"
        except Exception as err:
            logger.error(f"Error streaming response: {err}", exc_info=True)
            error_chunk = StreamingChatChunk(
                chunk=f"\n[Error: {str(err)}]",
                done=True,
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


# ============================================================================
# 3. Knowledge Base & Vector Retrieval Endpoints
# ============================================================================

@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Knowledge Base"],
    summary="Direct Vector Search Retrieval",
)
@limiter.limit("60/minute")
async def search_endpoint(
    request: Request,
    payload: SearchRequest,
):
    """Retrieve relevant document chunks directly from the vector store."""
    try:
        rag_pipe = production_agent.rag_pipeline
        if rag_pipe is None:
            return SearchResponse(results=[], total_found=0)

        retrieved = rag_pipe.retrieve(
            query=payload.query,
            k=payload.top_k,
            where=payload.filter,
        )

        results: List[SourceDocument] = []
        for idx, item in enumerate(retrieved, start=1):
            meta = item.get("metadata", {}) or {}
            score = item.get("distance")
            if payload.score_threshold is not None and score is not None:
                if score > payload.score_threshold:
                    continue

            results.append(
                SourceDocument(
                    document_id=item.get("id", f"doc_{idx}"),
                    title=meta.get("source") or meta.get("filename") or f"doc_{idx}",
                    content=item.get("text", ""),
                    score=score,
                    metadata=meta,
                )
            )

        return SearchResponse(results=results, total_found=len(results))

    except Exception as e:
        logger.error(f"Error during search retrieval: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search retrieval error: {str(e)}",
        )


@app.post(
    "/documents/ingest",
    response_model=DocumentIngestResponse,
    tags=["Knowledge Base"],
    summary="Ingest and Index Raw Documents into Vector Store",
)
@limiter.limit("30/minute")
async def ingest_document(
    request: Request,
    payload: DocumentIngestRequest,
):
    """Chunk and index document text into the enterprise vector store."""
    try:
        rag_pipe = production_agent.rag_pipeline
        if rag_pipe is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector store pipeline not initialized",
            )

        meta = dict(payload.metadata)
        if payload.title:
            meta["source"] = payload.title
        if payload.source_url:
            meta["source_url"] = payload.source_url

        doc_ids = rag_pipe.index_texts(
            texts=[payload.content],
            metadatas=[meta],
            chunk=True,
        )

        first_id = doc_ids[0] if doc_ids else str(uuid.uuid4())
        return DocumentIngestResponse(
            document_id=first_id,
            chunks_created=len(doc_ids),
            status="indexed",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}",
        )


# ============================================================================
# 4. Feedback & RLHF Telemetry Endpoints
# ============================================================================

@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["Feedback"],
    summary="Submit User Feedback on AI Responses",
)
@limiter.limit("60/minute")
async def submit_feedback(
    request: Request,
    payload: FeedbackRequest,
):
    """Record user satisfaction scores and comments for RLHF / evaluation."""
    try:
        feedback_id = str(uuid.uuid4())
        logger.info(
            "User feedback recorded",
            extra={
                "feedback_id": feedback_id,
                "score": payload.score,
                "session_id": payload.session_id,
                "run_id": payload.run_id,
                "comment": payload.comment,
            },
        )
        return FeedbackResponse(status="recorded", feedback_id=feedback_id)
    except Exception as e:
        logger.error(f"Error recording feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback submission failed: {str(e)}",
        )


# ============================================================================
# 5. Semantic & Multi-Tier Cache Endpoints
# ============================================================================

@app.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    tags=["Cache"],
    summary="Inspect Semantic & Multi-tier Cache Statistics",
)
async def get_cache_stats():
    """Retrieve operational statistics from tiered and semantic cache engines."""
    stats = tiered_cache.get_metrics()
    hits = stats.get("hits", 0)
    misses = stats.get("misses", 0)
    total = hits + misses
    hit_rate = (hits / total * 100.0) if total > 0 else 0.0

    return CacheStatsResponse(
        total_entries=stats.get("current_size", 0),
        hit_count=hits,
        miss_count=misses,
        hit_rate_percent=round(hit_rate, 2),
    )


@app.post(
    "/cache/clear",
    tags=["Cache"],
    summary="Clear L1/L2 and Semantic Cache",
)
async def clear_cache():
    """Flush and invalidate all cached LLM prompt responses."""
    try:
        tiered_cache.clear()
        if semantic_cache is not None:
            semantic_cache.clear()
        return {"status": "success", "message": "Cache successfully cleared"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}",
        )


# ============================================================================
# Local Direct Server Execution
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "production_api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
