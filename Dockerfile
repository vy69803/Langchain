# ==============================================================================
# Multi-stage Dockerfile for Production LangChain RAG & FastAPI Agent System
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Builder Stage: Install dependencies using uv
# ------------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM python:3.14-slim AS builder

WORKDIR /app

# Copy uv binaries from astral-sh image
COPY --from=uv_bin /uv /uvx /bin/

# Environment variables for uv & Python build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install build dependencies if needed for native C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition files first (leverage Docker layer caching)
COPY pyproject.toml uv.lock* ./

# Install production dependencies only into a standalone virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# ------------------------------------------------------------------------------
# 2. Production Runtime Stage
# ------------------------------------------------------------------------------
FROM python:3.14-slim AS runner

WORKDIR /app

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:/app" \
    PORT=8000 \
    HOST="0.0.0.0"

# Install runtime dependencies (e.g., curl for container health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root system user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source code and configuration
COPY src/ /app/src/
COPY main.py /app/main.py
COPY pyproject.toml /app/pyproject.toml

# Set permissions for non-root execution
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose FastAPI application port
EXPOSE 8000

# Health check against the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the production ASGI server
CMD ["uvicorn", "production_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
