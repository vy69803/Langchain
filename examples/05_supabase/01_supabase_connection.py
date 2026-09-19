#!/usr/bin/env python3
"""01_supabase_connection.py - Direct Supabase PostgreSQL Connection & Health Check.

Demonstrates:
  1. Secure environment variable loading (.env)
  2. Direct PostgreSQL connection via psycopg2 / psycopg
  3. Server information & pgvector extension diagnostics
  4. Robust error handling with actionable troubleshooting tips

Usage:
  uv run python 6-supabase/01_supabase_connection.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

# Load environment variables
load_dotenv(dotenv_path=project_root / ".env")


DATABASE_URL = os.getenv("DATABASE_URL")


def connect_to_supabase():
    """Connect to Supabase pgvector"""

    # Use OpenRouter or OpenAI API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        openai_api_base=base_url if os.getenv("OPENROUTER_API_KEY") else None,
    )

    # Note: Supabase has pgvector pre-installed
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="production_docs",
        connection=DATABASE_URL or get_database_url(),
        use_jsonb=True,
    )
    return vectorstore



def verify_connection(vectorstore):
    """Verify the connection works"""
    # Add a test document
    test_doc = Document(
        page_content="This is a test document to verify Supabase connection",
        metadata={"test": True},
    )

    print("\n[*] Adding test document to Supabase vectorstore...")
    vectorstore.add_documents([test_doc])

    print("[*] Running test similarity search query...")
    results = vectorstore.similarity_search("test document", k=1)
    if results:
        print(f"[✔] Successfully retrieved {len(results)} document(s) from Supabase!")
        print(f"    Sample content: \"{results[0].page_content}\"")
        print(f"    Metadata: {results[0].metadata}")
    return results


def get_database_url() -> str:


    """Retrieve and validate the Supabase Database URL from environment variables."""
    db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not db_url or "[YOUR-PASSWORD]" in db_url:
        # Fallback to individual components if available
        user = os.getenv("SUPABASE_DB_USER", "postgres")
        pwd = os.getenv("SUPABASE_DB_PASSWORD", "")
        host = os.getenv("SUPABASE_DB_HOST", "db.owziqwaxwbfzvcdlgjte.supabase.co")
        port = os.getenv("SUPABASE_DB_PORT", "5432")
        dbname = os.getenv("SUPABASE_DB_NAME", "postgres")

        if pwd and pwd != "[YOUR-PASSWORD]":
            db_url = f"postgresql://{user}:{pwd}@{host}:{port}/{dbname}"
        else:
            raise ValueError(
                "DATABASE_URL or SUPABASE_DB_PASSWORD is missing or contains placeholder '[YOUR-PASSWORD]'. "
                "Please configure your .env file."
            )
    return db_url


def test_psycopg_connection(db_url: str) -> Dict[str, Any]:
    """Test connection using psycopg (v3) or psycopg2 with diagnostic queries."""
    results: Dict[str, Any] = {}
    start_time = time.perf_counter()

    try:
        # Try psycopg (v3) first
        import psycopg

        print("[*] Connecting using psycopg (v3)...")
        with psycopg.connect(db_url, autocommit=True, connect_timeout=10) as conn:
            latency_ms = (time.perf_counter() - start_time) * 1000
            results["latency_ms"] = latency_ms

            with conn.cursor() as cur:
                # 1. Database & User info
                cur.execute("SELECT current_database(), current_user, version();")
                row = cur.fetchone()
                if row:
                    results["database"] = row[0]
                    results["user"] = row[1]
                    results["version"] = row[2].split(",")[0]

                # 2. Check pgvector extension
                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');"
                )
                ext_row = cur.fetchone()
                results["pgvector_installed"] = ext_row[0] if ext_row else False

                # 3. Connection count & active transactions
                cur.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database();"
                )
                conn_count = cur.fetchone()
                results["active_connections"] = conn_count[0] if conn_count else 1

            results["driver"] = "psycopg3"
            return results

    except ImportError:
        # Fallback to psycopg2
        import psycopg2

        print("[*] Connecting using psycopg2...")
        conn = psycopg2.connect(db_url, connect_timeout=10)
        try:
            latency_ms = (time.perf_counter() - start_time) * 1000
            results["latency_ms"] = latency_ms

            with conn.cursor() as cur:
                cur.execute("SELECT current_database(), current_user, version();")
                row = cur.fetchone()
                if row:
                    results["database"] = row[0]
                    results["user"] = row[1]
                    results["version"] = row[2].split(",")[0]

                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');"
                )
                ext_row = cur.fetchone()
                results["pgvector_installed"] = ext_row[0] if ext_row else False

            results["driver"] = "psycopg2"
            return results
        finally:
            conn.close()


def enable_pgvector_extension(db_url: str) -> bool:
    """Attempt to enable the pgvector extension for AI vector search."""
    try:
        import psycopg

        with psycopg.connect(db_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        return True
    except Exception as e:
        print(f"[!] Note on enabling pgvector: {e}")
        return False


def main():
    print("=" * 65)
    print("  Supabase Database Connection & Diagnostics (Step 01)")
    print("=" * 65)

    try:
        db_url = get_database_url()
        # Mask password in URL for display
        masked_url = db_url
        if "@" in db_url and "://" in db_url:
            prefix, rest = db_url.split("://", 1)
            user_pass, host_part = rest.split("@", 1)
            user = user_pass.split(":")[0] if ":" in user_pass else user_pass
            masked_url = f"{prefix}://{user}:******@{host_part}"

        print(f"\n[+] Configured Endpoint: {masked_url}")
        print("[+] Initiating TCP/SSL handshake...")

        diag = test_psycopg_connection(db_url)

        print("\n" + "-" * 65)
        print("  Connection Successful! Diagnostic Report:")
        print("-" * 65)
        print(f"  • Driver           : {diag.get('driver')}")
        print(f"  • Database Name    : {diag.get('database')}")
        print(f"  • Connected User   : {diag.get('user')}")
        print(f"  • Latency (Ping)   : {diag.get('latency_ms', 0):.2f} ms")
        print(f"  • Postgres Version : {diag.get('version')}")
        print(
            f"  • pgvector Support : {'[ACTIVE]' if diag.get('pgvector_installed') else '[NOT INSTALLED]'}"
        )

        if not diag.get("pgvector_installed"):
            print("\n[*] Enabling 'vector' extension for RAG vector stores...")
            if enable_pgvector_extension(db_url):
                print("[✔] pgvector extension activated successfully.")

        # Initialize LangChain PGVector and verify by writing & reading a document
        print("\n" + "=" * 65)
        print("  Initializing LangChain PGVector Store & Creating Tables...")
        print("=" * 65)
        vectorstore = connect_to_supabase()
        verify_connection(vectorstore)

        print("\n[✔] Supabase tables created and verified successfully!")


    except Exception as exc:
        print(f"\n[X] Connection Failed: {exc}")
        print("\nTroubleshooting Tips:")
        print("  1. Verify the password in .env matches your Supabase Project Settings.")
        print("  2. Ensure your IP is not blocked by network firewalls.")
        print("  3. If connecting directly (port 5432), ensure IPv6 or pooler is reachable.")
        print("  4. For serverless/pooler mode, use port 6543 (PgBouncer/Supavisor).")
        sys.exit(1)


if __name__ == "__main__":
    main()
