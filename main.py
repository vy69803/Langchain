"""Enterprise AI RAG & Agent Production Server Entrypoint.

Connects and serves the full FastAPI application with SlowAPI rate limiting,
LangGraph agent reasoning workflows, multi-tiered semantic caching, structured
telemetry, and knowledge retrieval.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src directory is in sys.path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv

load_dotenv()

# Expose the FastAPI application instance for ASGI servers (e.g. `uvicorn main:app`)
from production_api.main import app
from production_api.config import get_settings
from production_api.agent import production_agent


def main() -> None:
    """Entrypoint for running the Production API server or CLI test queries."""
    parser = argparse.ArgumentParser(
        description="Production RAG API & Agent Server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the API server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the API server (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=True,
        help="Enable auto-reload on code changes (default: True)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a direct one-off query through the Production Agent CLI instead of starting the server",
    )

    args = parser.parse_args()
    settings = get_settings()

    if args.query:
        print(f"\n[Agent CLI Test] Query: {args.query}")
        print("-" * 60)
        result = production_agent.run(args.query)
        print(f"Answer:\n{result.get('answer')}\n")
        print(f"Cached: {result.get('is_cached')}")
        print(f"Latency: {result.get('latency_ms')} ms")
        print(f"Security: {result.get('security_status')}")
        print("-" * 60)
        return

    import uvicorn

    print("=" * 65)
    print(f" Starting {settings.app_name} ({settings.environment})")
    print(f" Host: http://{args.host}:{args.port}")
    print(f" Swagger Docs: http://{args.host}:{args.port}/docs")
    print(f" ReDoc Docs:   http://{args.host}:{args.port}/redoc")
    print(f" Rate Limiting: Enabled via SlowAPI ({settings.rate_limit_default})")
    print("=" * 65)

    uvicorn.run(
        "production_api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
