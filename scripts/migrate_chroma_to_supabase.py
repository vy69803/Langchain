#!/usr/bin/env python3
"""scripts/migrate_chroma_to_supabase.py

High-performance, zero-cost vector migration from local ChromaDB to Supabase pgvector.

Optimizations Implemented:
  1. halfvec(384) FP16 quantization: cuts vector storage in half (~103 MB for 134k rows)
  2. Extracted metadata columns (company, filing_year, doc_type) with B-Tree indexes
  3. IVFFlat indexing (lists = 360): built AFTER bulk insert, occupying only ~18 MB
  4. psycopg2.extras.execute_values: bulk streaming for fast batch ingestion
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

load_dotenv(dotenv_path=project_root / ".env")


def get_db_url() -> str:
    url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError(
            "SUPABASE_DB_URL or DATABASE_URL not found in .env file. "
            "Please configure your Supabase connection string."
        )
    return url


def init_supabase_schema(conn, recreate: bool = False) -> None:
    """Initialize table schema and B-tree indexes."""
    with conn.cursor() as cur:
        # 1. Enable extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        if recreate:
            print("[!] Dropping existing 'financebench_docs' table...")
            cur.execute("DROP TABLE IF EXISTS financebench_docs CASCADE;")

        # 2. Create table with halfvec(384)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS financebench_docs (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                company VARCHAR(64),
                filing_year SMALLINT,
                doc_type VARCHAR(16),
                metadata JSONB DEFAULT '{}'::jsonb,
                embedding halfvec(384),
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 3. Enable extended compression
        cur.execute("ALTER TABLE financebench_docs ALTER COLUMN content SET STORAGE EXTENDED;")

        # 4. Create B-tree indexes on filtered columns
        cur.execute("CREATE INDEX IF NOT EXISTS idx_financebench_company ON financebench_docs (company);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_financebench_year ON financebench_docs (filing_year);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_financebench_doc_type ON financebench_docs (doc_type);")

    conn.commit()
    print("[✔] Supabase schema and B-tree indexes initialized.")


def build_ivfflat_index(conn, lists: int = 360) -> None:
    """Build the compact IVFFlat index after all data has been ingested."""
    print(f"\n[*] Building IVFFlat index on 'embedding' with lists={lists}...")
    print("    (This trains Voronoi centroid clusters across the full 134k dataset)")
    start_t = time.perf_counter()
    with conn.cursor() as cur:
        # Increase maintenance_work_mem for faster index creation if permitted
        try:
            cur.execute("SET maintenance_work_mem = '64MB';")
        except Exception:
            pass

        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_financebench_ivfflat 
            ON financebench_docs USING ivfflat (embedding halfvec_cosine_ops)
            WITH (lists = {lists});
        """)
    conn.commit()
    duration = time.perf_counter() - start_t
    print(f"[✔] IVFFlat index built successfully in {duration:.2f}s!")


def print_storage_diagnostics(conn) -> None:
    """Print exact table and index size on disk in Supabase."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                pg_size_pretty(pg_relation_size('financebench_docs')) AS table_size,
                pg_size_pretty(pg_indexes_size('financebench_docs')) AS index_size,
                pg_size_pretty(pg_total_relation_size('financebench_docs')) AS total_size,
                COUNT(*) AS total_rows
            FROM financebench_docs;
        """)
        row = cur.fetchone()
        if row:
            table_sz, idx_sz, total_sz, row_cnt = row
            print("\n" + "=" * 60)
            print("         SUPABASE STORAGE USAGE REPORT")
            print("=" * 60)
            print(f" Total Rows Migrated : {row_cnt:,}")
            print(f" Table Data Size     : {table_sz}")
            print(f" Indexes Size        : {idx_sz}")
            print(f" Total Disk Footprint: {total_sz} (Quota: 500 MB Free Tier)")
            print("=" * 60 + "\n")


def clean_str(val: Optional[str]) -> Optional[str]:
    """Remove NUL (0x00) characters prohibited by PostgreSQL."""
    if val is None:
        return None
    return str(val).replace("\x00", "")


def clean_dict(d: Any) -> Any:
    """Recursively clean NUL bytes from metadata dictionary keys and string values."""
    if isinstance(d, str):
        return d.replace("\x00", "")
    elif isinstance(d, dict):
        return {clean_str(k): clean_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_dict(x) for x in d]
    return d


def migrate(
    chroma_dir: str = "./data/chroma_db",
    collection_name: str = "financebench",
    batch_size: int = 1000,
    limit: Optional[int] = None,
    recreate: bool = False,
    skip_index: bool = False,
) -> None:
    db_url = get_db_url()

    print("=" * 65)
    print("  Zero-Cost ChromaDB -> Supabase pgvector (halfvec) Migration")
    print("=" * 65)
    print(f" Source ChromaDB Dir : {chroma_dir}")
    print(f" Source Collection   : {collection_name}")
    print(f" Batch Size          : {batch_size}")
    if limit:
        print(f" Limit               : {limit} chunks (Test Run)")
    print("=" * 65)

    # 1. Connect to ChromaDB
    print("\n[*] Opening local ChromaDB...")
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(collection_name)
    total_docs = collection.count()
    print(f"[✔] Found {total_docs:,} total documents in ChromaDB '{collection_name}'.")

    target_count = min(total_docs, limit) if limit else total_docs

    # 2. Connect to Supabase
    print("[*] Connecting to Supabase PostgreSQL...")
    conn = psycopg2.connect(db_url)
    init_supabase_schema(conn, recreate=recreate)

    # 3. Stream batches from ChromaDB and bulk insert
    offset = 0
    start_time = time.perf_counter()
    migrated_count = 0

    insert_sql = """
        INSERT INTO financebench_docs (id, content, company, filing_year, doc_type, metadata, embedding)
        VALUES %s
        ON CONFLICT (id) DO UPDATE 
        SET content = EXCLUDED.content,
            company = EXCLUDED.company,
            filing_year = EXCLUDED.filing_year,
            doc_type = EXCLUDED.doc_type,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding;
    """

    print(f"[*] Starting migration of {target_count:,} records...")

    while offset < target_count:
        cur_batch_size = min(batch_size, target_count - offset)
        batch = collection.get(
            limit=cur_batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )

        ids = batch.get("ids", [])
        if not ids:
            break

        docs = batch.get("documents", [])
        metas = batch.get("metadatas", [])
        embeddings = batch.get("embeddings", [])

        records = []
        for doc_id, text, meta, emb in zip(ids, docs, metas, embeddings):
            meta_dict = clean_dict(meta or {})
            company = clean_str(meta_dict.get("company"))
            
            # Extract filing_year
            raw_period = meta_dict.get("doc_period", "")
            filing_year = int(raw_period) if str(raw_period).isdigit() else None
            
            doc_type = clean_str(meta_dict.get("doc_type"))
            
            # Format 384-d vector string for halfvec
            # Rounded to 5 decimal places to reduce network bandwidth while preserving precision
            emb_str = f"[{','.join(f'{x:.5f}' for x in emb)}]"
            
            clean_text = clean_str(text) or ""
            clean_id = clean_str(doc_id) or ""
            
            records.append((
                clean_id,
                clean_text,
                company,
                filing_year,
                doc_type,
                json.dumps(meta_dict),
                emb_str,
            ))

        with conn.cursor() as cur:
            execute_values(
                cur,
                insert_sql,
                records,
                template="(%s, %s, %s, %s, %s, %s::jsonb, %s::halfvec)",
                page_size=batch_size,
            )
        conn.commit()

        offset += len(ids)
        migrated_count += len(ids)
        pct = (offset / target_count) * 100
        elapsed = time.perf_counter() - start_time
        rate = migrated_count / elapsed if elapsed > 0 else 0
        eta = (target_count - offset) / rate if rate > 0 else 0

        print(
            f"\r[+] Migrated: {offset:,}/{target_count:,} ({pct:.1f}%) "
            f"| Rate: {rate:.0f} rows/s | ETA: {eta:.0f}s",
            end="",
            flush=True,
        )

    total_time = time.perf_counter() - start_time
    print(f"\n[✔] Insert completed! Migrated {migrated_count:,} records in {total_time:.1f}s.")

    # 4. Build IVFFlat index on populated data
    if not skip_index:
        build_ivfflat_index(conn, lists=360)

    # 5. Diagnostic report
    print_storage_diagnostics(conn)

    conn.close()
    print("[✔] Migration process finished cleanly.")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate ChromaDB vectors to Supabase pgvector using halfvec(384) & IVFFlat"
    )
    parser.add_argument("--chroma-dir", type=str, default="./data/chroma_db", help="Path to ChromaDB directory")
    parser.add_argument("--collection", type=str, default="financebench", help="Chroma collection name")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for migration")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to migrate (for testing)")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate target table before migration")
    parser.add_argument("--skip-index", action="store_true", help="Skip building the IVFFlat index")

    args = parser.parse_args()
    migrate(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        batch_size=args.batch_size,
        limit=args.limit,
        recreate=args.recreate,
        skip_index=args.skip_index,
    )


if __name__ == "__main__":
    main()
