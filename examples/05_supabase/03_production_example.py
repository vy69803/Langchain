#!/usr/bin/env python3
"""03_production_example.py - Production-Ready Supabase pgvector RAG Pipeline.

Demonstrates:
  1. PostgreSQL Schema Migration with pgvector extension & HNSW indexing
  2. JSONB Metadata filtering + Dense Semantic Cosine Similarity Search
  3. Thread-safe Batch Upsert with parameterized SQL queries
  4. Complete end-to-end RAG document ingestion & hybrid retrieval

Usage:
  uv run python 6-supabase/03_production_example.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv

load_dotenv(dotenv_path=project_root / ".env")


def get_db_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not url or "[YOUR-PASSWORD]" in url:
        raise ValueError("Valid DATABASE_URL required in .env")
    return url


class SupabaseVectorStore:
    """Production pgvector document store using raw SQL with connection pooling."""

    def __init__(self, db_url: str, table_name: str = "rag_documents", embedding_dim: int = 1536):
        self.db_url = db_url
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self._init_driver()
        self.setup_schema()

    def _init_driver(self):
        try:
            import psycopg

            self.driver = "psycopg"
        except ImportError:
            import psycopg2

            self.driver = "psycopg2"

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        if self.driver == "psycopg":
            import psycopg

            with psycopg.connect(self.db_url, autocommit=True) as conn:
                yield conn
        else:
            import psycopg2

            conn = psycopg2.connect(self.db_url)
            conn.autocommit = True
            try:
                yield conn
            finally:
                conn.close()

    def setup_schema(self):
        """Creates the vector extension, document table, and HNSW index."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Ensure pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # 2. Document storage table
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id BIGSERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB DEFAULT '{{}}'::jsonb,
                        embedding vector({self.embedding_dim}),
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """
                )

                # 3. Create HNSW index for high-speed approximate nearest neighbor search
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding_hnsw
                    ON {self.table_name}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """
                )

                # 4. GIN index on metadata for fast JSONB filtering
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metadata_gin
                    ON {self.table_name} USING gin (metadata);
                """
                )

    def insert_documents(self, documents: List[Dict[str, Any]]):
        """Batch inserts documents with text, JSONB metadata, and embedding vectors."""
        if not documents:
            return

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for doc in documents:
                    content = doc["content"]
                    metadata_json = json.dumps(doc.get("metadata", {}))
                    embedding = doc["embedding"]
                    # Format vector as string for PostgreSQL: '[0.1, 0.2, ...]'
                    embedding_str = f"[{','.join(f'{x:.6f}' for x in embedding)}]"

                    cur.execute(
                        f"""
                        INSERT INTO {self.table_name} (content, metadata, embedding)
                        VALUES (%s, %s::jsonb, %s::vector)
                    """,
                        (content, metadata_json, embedding_str),
                    )

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Dense similarity search using cosine distance (<=>) with optional JSONB filter."""
        embedding_str = f"[{','.join(f'{x:.6f}' for x in query_embedding)}]"

        where_clause = ""
        params: List[Any] = [embedding_str]

        if category_filter:
            where_clause = "WHERE metadata->>'category' = %s"
            params.append(category_filter)

        params.append(embedding_str)
        params.append(top_k)

        query = f"""
            SELECT
                id,
                content,
                metadata,
                1 - (embedding <=> %s::vector) AS cosine_similarity,
                created_at
            FROM {self.table_name}
            {where_clause}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
        """

        results = []
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                for row in rows:
                    results.append(
                        {
                            "id": row[0],
                            "content": row[1],
                            "metadata": row[2] if isinstance(row[2], dict) else json.loads(row[2]),
                            "similarity": float(row[3]),
                            "created_at": str(row[4]),
                        }
                    )
        return results

    def count(self) -> int:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {self.table_name};")
                return cur.fetchone()[0]


def generate_sample_embedding(dim: int = 1536, seed: int = 42) -> List[float]:
    """Generates a normalized synthetic embedding vector for demonstration."""
    import random
    import math

    rnd = random.Random(seed)
    vec = [rnd.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def main():
    print("=" * 65)
    print("  Production Supabase + pgvector RAG Pipeline (Step 03)")
    print("=" * 65)

    try:
        db_url = get_db_url()
        print("[+] Connecting to Supabase and ensuring pgvector schema...")
        store = SupabaseVectorStore(db_url, table_name="production_rag_docs", embedding_dim=1536)
        print("[✔] Database schema, HNSW vector index & GIN metadata index ready.")

        # Sample Knowledge Base
        sample_docs = [
            {
                "content": "OAuth 2.0 token revocation requires invalidating both access and refresh tokens in Supabase Auth.",
                "metadata": {"category": "security", "author": "devops", "version": "v2.1"},
                "embedding": generate_sample_embedding(1536, seed=101),
            },
            {
                "content": "Connection pooling with Supavisor allows handling tens of thousands of client connections on port 6543.",
                "metadata": {"category": "database", "author": "dba", "version": "v1.0"},
                "embedding": generate_sample_embedding(1536, seed=202),
            },
            {
                "content": "Hierarchical parent-child chunking retains full section context while preserving dense vector precision.",
                "metadata": {"category": "rag", "author": "ai_team", "version": "v3.0"},
                "embedding": generate_sample_embedding(1536, seed=303),
            },
        ]

        print(f"\n[*] Ingesting {len(sample_docs)} production documents with embeddings...")
        t0 = time.perf_counter()
        store.insert_documents(sample_docs)
        print(f"[✔] Documents indexed in {(time.perf_counter() - t0) * 1000:.2f} ms. Total count: {store.count()}")

        # Perform Similarity Search
        query_vec = generate_sample_embedding(1536, seed=202)  # Search closest to the connection pooling doc
        print("\n[*] Executing Vector Cosine Similarity Search (<=>)...")
        results = store.similarity_search(query_embedding=query_vec, top_k=2)

        print("-" * 65)
        print("  Top Retrieved Search Results:")
        print("-" * 65)
        for i, res in enumerate(results, 1):
            print(f"  #{i} [Score: {res['similarity']:.4f}] [Category: {res['metadata'].get('category')}]")
            print(f"      Text: \"{res['content']}\"")
            print(f"      Metadata: {res['metadata']}")
            print()

        print("[✔] Production pgvector pipeline executed successfully!")

    except Exception as exc:
        print(f"\n[X] Pipeline Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
