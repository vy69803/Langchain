"""FastAPI Application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.security import get_api_key
from app.cache import cache
from app.monitoring import TimingMiddleware
from app.agent import agent_service

settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit_default])


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 Starting {settings.app_name} in {settings.environment} mode...")
    yield
    print(f"🛑 Shutting down {settings.app_name}...")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware
app.add_middleware(TimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
@limiter.limit("30/minute")
async def root(request: Request):
    return {
        "message": f"Welcome to {settings.app_name}",
        "environment": settings.environment,
        "docs_url": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    return HealthResponse(
        status="healthy",
        service=settings.app_name,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("20/minute")
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(get_api_key),
):
    # Check cache
    cached_val = cache.get(chat_request.message)
    if cached_val:
        return ChatResponse(
            response=cached_val,
            cached=True,
            metadata={"source": "cache"},
        )

    # Process via agent
    result = await agent_service.run(chat_request.message)
    answer = result.get("answer", "")
    cache.set(chat_request.message, answer)

    return ChatResponse(
        response=answer,
        cached=False,
        metadata={"sources": result.get("sources", [])},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
