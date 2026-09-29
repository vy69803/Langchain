#!/usr/bin/env python
"""CLI Tool for Ingesting FinanceBench SEC Filings into LangChain RAG.

Usage Examples:
    # Dry-run test on a 3-file sample
    python scripts/ingest_financebench.py --dry-run --sample 3

    # Ingest 3M filings and run an ad-hoc query
    python scripts/ingest_financebench.py --company 3M --query "What is 3M's capital expenditure in 2018?"

    # Ingest a sample of 10 filings for quick benchmark setup
    python scripts/ingest_financebench.py --sample 10

    # Full ingestion of all 368 filings
    python scripts/ingest_financebench.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_rag.finance_pipeline import FinanceBenchIngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest FinanceBench SEC filings (PDF) into ChromaDB and BM25 index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="data/financebench/pdfs",
        help="Path to FinanceBench PDF filings directory.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/financebench/financebench_document_information.jsonl",
        help="Path to FinanceBench document metadata JSONL file.",
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Filter ingestion to a specific company (e.g. '3M', 'Apple', 'Amazon').",
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        default=None,
        help="Filter ingestion to a specific document type (e.g. '10k', '10q', '8k').",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit number of documents processed (useful for quick testing).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit maximum pages to parse per document.",
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
        default="financebench",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="./data/financebench_manifest.json",
        help="Path to JSON file tracking indexed file hashes.",
    )
    parser.add_argument(
        "--bm25-path",
        type=str,
        default="./data/bm25_financebench.json",
        help="Path to BM25 index state file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of chunks per database write batch.",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1200,
        help="Target maximum character length per chunk.",
    )
    parser.add_argument(
        "--disable-bm25",
        action="store_true",
        help="Disable BM25 sparse index creation.",
    )
    parser.add_argument(
        "--disable-graph",
        action="store_true",
        help="Disable Neo4j Knowledge Graph ingestion.",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=None,
        help="Neo4j connection URI (e.g. 'neo4j+s://xxx.databases.neo4j.io').",
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default=None,
        help="Neo4j username (defaults to 'neo4j').",
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=None,
        help="Neo4j password.",
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Skip ingestion phase and directly execute test query against existing index.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional test query to run immediately after ingestion.",
    )
    parser.add_argument(
        "--use-agent",
        action="store_true",
        help="Use full LangGraph Financial Intelligence Agent for query answering.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return for test query.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable FlashRank reranking for test query.",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        choices=["hybrid", "dense", "sparse"],
        default="hybrid",
        help="Search mode for test query.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("ingest_financebench")

    if args.neo4j_uri:
        os.environ["NEO4J_URI"] = args.neo4j_uri
    if args.neo4j_user:
        os.environ["NEO4J_USERNAME"] = args.neo4j_user
    if args.neo4j_password:
        os.environ["NEO4J_PASSWORD"] = args.neo4j_password

    pipeline = FinanceBenchIngestionPipeline(
        pdf_dir=args.pdf_dir,
        metadata_path=args.metadata_path,
        persist_directory=args.persist_dir,
        collection_name=args.collection,
        manifest_path=args.manifest_path,
        bm25_path=args.bm25_path,
        max_chunk_chars=args.max_chunk_chars,
        batch_size=args.batch_size,
        enable_bm25=not args.disable_bm25,
        enable_graph=not args.disable_graph,
    )

    if not args.query_only:
        print("\n" + "=" * 70)
        print("   FINANCEBENCH SEC FILINGS INGESTION PIPELINE")
        print("=" * 70)
        print(f" PDF Directory     : {Path(args.pdf_dir).resolve()}")
        print(f" Metadata Path     : {Path(args.metadata_path).resolve()}")
        print(f" ChromaDB Path     : {Path(args.persist_dir).resolve()} (collection: {args.collection})")
        print(f" BM25 Index Path   : {Path(args.bm25_path).resolve()} (enabled: {not args.disable_bm25})")
        print(f" Manifest Path     : {Path(args.manifest_path).resolve()}")
        print(f" Max Chunk Chars   : {args.max_chunk_chars}")
        print(f" Batch Size        : {args.batch_size}")
        if args.company:
            print(f" Company Filter    : {args.company}")
        if args.doc_type:
            print(f" Doc Type Filter   : {args.doc_type}")
        if args.sample:
            print(f" Sample Limit      : {args.sample} documents")
        if args.max_pages:
            print(f" Max Pages / Doc   : {args.max_pages}")
        if args.dry_run:
            print(" MODE              : DRY-RUN (no writes)")
        if args.force:
            print(" MODE              : FORCE RE-INDEX (ignoring manifest)")
        print("=" * 70 + "\n")

        def on_progress(info: dict) -> None:
            curr = info["current"]
            total = info["total"]
            pct = (curr / total) * 100.0 if total else 0.0
            bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
            sys.stdout.write(
                f"\r[{bar}] {pct:5.1f}% ({curr}/{total}) | {info['file'][:30]:<30} (+{info['chunks']} chunks)"
            )
            sys.stdout.flush()

        stats = pipeline.ingest(
            force=args.force,
            dry_run=args.dry_run,
            sample=args.sample,
            company=args.company,
            doc_type=args.doc_type,
            max_pages_per_doc=args.max_pages,
            progress_callback=on_progress,
        )
        print("\n")

        print("=" * 70)
        print("   INGESTION RESULTS SUMMARY")
        print("=" * 70)
        print(f" Total Files Discovered : {stats['total_files_discovered']}")
        print(f" Files Indexed          : {stats['files_to_index']}")
        print(f" Files Skipped (Unchg)  : {stats['files_unchanged']}")
        print(f" Chunks Generated       : {stats['chunks_created']}")
        print(f" Chunks Upserted        : {stats['chunks_upserted']}")
        print(f" Graph Entities Ingested: {stats.get('graph_entities_ingested', 0)}")
        print(f" Execution Duration     : {stats['duration_seconds']}s")
        if stats["errors"]:
            print(f" Errors Encountered     : {len(stats['errors'])}")
            for err in stats["errors"][:5]:
                print(f"   - {err['file']}: {err['error']}")
        print("=" * 70 + "\n")
    if args.query:
        if args.use_agent:
            import asyncio
            from langchain_rag.agent.graph_agent import FinancialIntelligenceAgent
            from langchain_rag.adapters.storage_adapters import ChromaVectorAdapter

            print(f"\n--- Running Financial Intelligence Agent: '{args.query}' ---")
            vector_adapter = ChromaVectorAdapter(
                collection_name=args.collection,
                persist_directory=args.persist_dir,
            )
            agent = FinancialIntelligenceAgent(vector_adapter=vector_adapter)
            res = asyncio.run(agent.ainvoke(args.query))
            print("\n" + "=" * 70)
            print(res.get("final_output", "No response generated."))
            print("=" * 70)
        else:
            print(f"\n--- Testing Query: '{args.query}' ---")
            results = pipeline.query(
                query_text=args.query,
                top_k=args.top_k,
                search_type=args.search_type,
                rerank=not args.no_rerank,
                company=args.company,
            )
            print(f"Retrieved {len(results)} results:")
            for idx, res in enumerate(results, start=1):
                meta = res.get("metadata", {})
                score_str = f"Rerank Score: {res['rerank_score']:.4f}" if "rerank_score" in res else f"RRF/Score"
                print(f"\n[{idx}] {meta.get('company', '')} | {meta.get('doc_name', '')} | Page {meta.get('page', '')} | Section: {meta.get('section', '')} ({score_str})")
                snippet = res.get("text", "")[:250].replace("\n", " ")
                print(f"    Snippet: {snippet}...")


if __name__ == "__main__":
    main()
