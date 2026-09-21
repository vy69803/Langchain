#!/usr/bin/env python
"""CLI Tool for Ingesting the GitLab Handbook Dataset into LangChain RAG.

Usage:
    # Quick dry-run test on a 10-file sample
    python scripts/ingest_handbook.py --dry-run --sample 10

    # Ingest the values department
    python scripts/ingest_handbook.py --department values

    # Ingest a 50-file sample and query it
    python scripts/ingest_handbook.py --sample 50 --query "What are GitLab's core values?"

    # Full ingestion
    python scripts/ingest_handbook.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_rag.handbook_pipeline import HandbookIngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest GitLab Handbook dataset into ChromaDB and BM25 index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        type=str,
        default="handbook",
        help="Path to the GitLab handbook repository root.",
    )
    parser.add_argument(
        "--department",
        type=str,
        default=None,
        help="Filter ingestion to a specific department directory (e.g. 'values', 'engineering').",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit number of documents processed (useful for rapid testing).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing of all documents, ignoring the content-hash manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk files without writing to ChromaDB or BM25.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./data/chroma_db",
        help="Directory to persist ChromaDB vector embeddings.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="gitlab_handbook",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="./data/handbook_manifest.json",
        help="Path to JSON file tracking indexed file hashes.",
    )
    parser.add_argument(
        "--bm25-path",
        type=str,
        default="./data/bm25_handbook.json",
        help="Path to BM25 index state file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of chunks per database write batch.",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1000,
        help="Maximum character length per chunk.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional test query to run after ingestion.",
    )
    parser.add_argument(
        "--query-type",
        type=str,
        choices=["hybrid", "dense", "sparse"],
        default="hybrid",
        help="Search mode for the test query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to retrieve for test query.",
    )
    return parser.parse_args()


def print_banner(args: argparse.Namespace) -> None:
    print("=" * 70)
    print("  GitLab Handbook Ingestion Pipeline")
    print("=" * 70)
    print(f"  Handbook Path : {args.path}")
    print(f"  Department    : {args.department or '[All Departments]'}")
    print(f"  Sample Limit  : {args.sample or '[None - All Files]'}")
    print(f"  Mode          : {'DRY RUN (No Writes)' if args.dry_run else 'PERSISTENT INGESTION'}")
    print(f"  Force Rebuild : {args.force}")
    print(f"  Chroma DB Dir : {args.persist_dir}")
    print(f"  Collection    : {args.collection}")
    print(f"  BM25 Index    : {args.bm25_path}")
    print("=" * 70)


def print_summary(summary: dict) -> None:
    print("\n" + "-" * 70)
    print("  Ingestion Execution Summary")
    print("-" * 70)
    print(f"  Total Files Scanned      : {summary['total_scanned']}")
    print(f"  Skipped (Unchanged)      : {summary['skipped_unchanged']}")
    print(f"  Processed Files          : {summary['processed_files']}")
    print(f"  Total Chunks Generated   : {summary['total_chunks_generated']}")
    print(f"  Duration                 : {summary['duration_seconds']:.2f} seconds")
    if not summary.get("dry_run"):
        print(f"  ChromaDB Total Documents : {summary.get('chroma_collection_count')}")
        print(f"  BM25 Total Chunks        : {summary.get('bm25_chunk_count')}")
    else:
        print("  ChromaDB / BM25          : Skipped (Dry Run)")

    if summary.get("errors"):
        print(f"\n  [!] Warnings/Errors encountered: {len(summary['errors'])}")
        for err in summary["errors"][:5]:
            print(f"      - {err['file']}: {err['error']}")
    print("-" * 70 + "\n")


def run_test_query(pipeline: HandbookIngestionPipeline, query: str, search_type: str, top_k: int) -> None:
    print(f"\n🔎 Executing Test Retrieval: \"{query}\" (Type: {search_type}, Top: {top_k})")
    print("=" * 70)
    results = pipeline.query(query, top_k=top_k, search_type=search_type)

    if not results:
        print("  No documents found.")
        return

    for rank, item in enumerate(results, 1):
        meta = item.get("metadata") or {}
        title = meta.get("title", "Untitled")
        url = item.get("source") or meta.get("url") or "N/A"
        breadcrumbs = meta.get("breadcrumbs") or "N/A"
        score_info = (
            f"Hybrid RRF: {item.get('hybrid_score')}"
            if "hybrid_score" in item
            else f"Distance: {item.get('distance')}"
            if "distance" in item
            else f"BM25: {item.get('bm25_score')}"
        )

        print(f"\n[Result #{rank}] {title}")
        print(f"  Breadcrumbs : {breadcrumbs}")
        print(f"  Source URL  : {url}")
        print(f"  Score       : {score_info}")
        preview = item.get("text", "").strip()
        # Truncate preview if too long
        if len(preview) > 300:
            preview = preview[:300] + "..."
        print(f"  Content Preview:\n    {preview}\n")


def main() -> None:
    args = parse_args()
    print_banner(args)

    pipeline = HandbookIngestionPipeline(
        handbook_dir=args.path,
        persist_directory=args.persist_dir,
        collection_name=args.collection,
        manifest_path=args.manifest_path,
        bm25_path=args.bm25_path,
        max_chunk_chars=args.max_chunk_chars,
        batch_size=args.batch_size,
    )

    summary = pipeline.run(
        department=args.department,
        sample=args.sample,
        force=args.force,
        dry_run=args.dry_run,
    )

    print_summary(summary)

    # Optional query execution
    if args.query and not args.dry_run:
        run_test_query(pipeline, args.query, args.query_type, args.top_k)


if __name__ == "__main__":
    main()
