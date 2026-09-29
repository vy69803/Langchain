#!/usr/bin/env python
"""Evaluation Benchmark Runner for FinanceBench on LangChain RAG.

Evaluates retrieval accuracy (Doc Hit@K, Page Hit@K, MRR) against the 150 gold-standard
questions in FinanceBench.

Usage Examples:
    # Quick retrieval benchmark on first 10 questions
    python scripts/evaluate_financebench.py --sample 10

    # Evaluate 3M questions
    python scripts/evaluate_financebench.py --company 3M

    # Compare search modes: dense vs sparse vs hybrid with reranking
    python scripts/evaluate_financebench.py --search-type hybrid --rerank --sample 25

    # Run full 150-question benchmark
    python scripts/evaluate_financebench.py --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_rag.finance_parser import load_financebench_qa
from langchain_rag.finance_pipeline import FinanceBenchIngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LangChain RAG retrieval on FinanceBench dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qa-path",
        type=str,
        default="data/financebench/financebench_open_source.jsonl",
        help="Path to FinanceBench open_source QA JSONL file.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./data/chroma_db",
        help="Path to ChromaDB storage directory.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="financebench",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--bm25-path",
        type=str,
        default="./data/bm25_financebench.json",
        help="Path to BM25 index state file.",
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Filter evaluation to questions about a specific company.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit number of evaluation questions.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks to evaluate per question.",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        choices=["hybrid", "dense", "sparse"],
        default="hybrid",
        help="Retrieval search strategy.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=True,
        help="Apply FlashRank reranking.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_false",
        dest="rerank",
        help="Disable reranking.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./data/financebench_eval_results.json",
        help="Path to save evaluation results JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)

    print("\n" + "=" * 70)
    print("   FINANCEBENCH RETRIEVAL BENCHMARK EVALUATOR")
    print("=" * 70)
    print(f" QA Dataset Path   : {Path(args.qa_path).resolve()}")
    print(f" Collection Name   : {args.collection}")
    print(f" Search Strategy   : {args.search_type} (Rerank: {args.rerank})")
    print(f" Top-K Retrieval   : {args.top_k}")
    if args.company:
        print(f" Company Filter    : {args.company}")
    if args.sample:
        print(f" Sample Size       : {args.sample} questions")
    print("=" * 70 + "\n")

    qa_items = load_financebench_qa(args.qa_path)

    if args.company:
        qa_items = [q for q in qa_items if q.get("company", "").lower() == args.company.lower()]

    if args.sample and args.sample > 0:
        qa_items = qa_items[:args.sample]

    if not qa_items:
        print("No evaluation questions found matching filters.")
        return

    print(f"Loaded {len(qa_items)} questions for evaluation.\n")

    pipeline = FinanceBenchIngestionPipeline(
        persist_directory=args.persist_dir,
        collection_name=args.collection,
        bm25_path=args.bm25_path,
        enable_bm25=args.search_type in ("sparse", "hybrid"),
    )

    doc_hits = {1: 0, 3: 0, 5: 0, args.top_k: 0}
    page_hits = {1: 0, 3: 0, 5: 0, args.top_k: 0}
    mrr_page_sum = 0.0
    question_results: list[dict[str, Any]] = []

    start_time = time.perf_counter()

    for idx, item in enumerate(qa_items, start=1):
        qid = item.get("financebench_id", f"q_{idx}")
        question = item["question"]
        target_doc = item.get("doc_name", "")
        evidence_list = item.get("evidence", [])
        target_pages = {e["evidence_page_num"] for e in evidence_list if "evidence_page_num" in e}

        retrieved = pipeline.query(
            query_text=question,
            top_k=args.top_k,
            search_type=args.search_type,
            rerank=args.rerank,
        )

        retrieved_docs = [r.get("metadata", {}).get("doc_name", "") for r in retrieved]
        retrieved_pages = [r.get("metadata", {}).get("page") for r in retrieved]

        # Calculate Doc Hit@K
        doc_hit_k = {k: any(target_doc == d for d in retrieved_docs[:k]) for k in doc_hits}
        for k, hit in doc_hit_k.items():
            if hit:
                doc_hits[k] += 1

        # Calculate Page Hit@K & Page MRR
        page_hit_k = {k: any(p in target_pages for p in retrieved_pages[:k]) for k in page_hits}
        for k, hit in page_hit_k.items():
            if hit:
                page_hits[k] += 1

        # Reciprocal rank of first matching page
        first_page_rank = 0
        for rank, p in enumerate(retrieved_pages, start=1):
            if p in target_pages:
                first_page_rank = rank
                break
        reciprocal_rank = (1.0 / first_page_rank) if first_page_rank > 0 else 0.0
        mrr_page_sum += reciprocal_rank

        question_results.append({
            "id": qid,
            "company": item.get("company", ""),
            "target_doc": target_doc,
            "target_pages": list(target_pages),
            "doc_hit": doc_hit_k.get(args.top_k, False),
            "page_hit": page_hit_k.get(args.top_k, False),
            "page_reciprocal_rank": reciprocal_rank,
            "retrieved_pages": retrieved_pages,
            "retrieved_docs": retrieved_docs,
        })

        sys.stdout.write(
            f"\r[{idx}/{len(qa_items)}] Page Hit@{args.top_k}: {page_hits[args.top_k]}/{idx} "
            f"({(page_hits[args.top_k]/idx)*100.0:.1f}%) | MRR: {mrr_page_sum/idx:.3f}"
        )
        sys.stdout.flush()

    total_q = len(qa_items)
    duration = time.perf_counter() - start_time
    print("\n")

    print("=" * 70)
    print("   FINANCEBENCH EVALUATION RESULTS")
    print("=" * 70)
    print(f" Questions Evaluated      : {total_q}")
    print(f" Evaluation Duration      : {duration:.2f}s ({duration/total_q:.2f}s/query)")
    print("-" * 70)
    print(" DOCUMENT-LEVEL RETRIEVAL ACCURACY:")
    for k in sorted(doc_hits.keys()):
        if k <= args.top_k:
            pct = (doc_hits[k] / total_q) * 100.0
            print(f"   Hit@{k:<2}                 : {doc_hits[k]:>3}/{total_q} ({pct:5.1f}%)")
    print("-" * 70)
    print(" PAGE-LEVEL EVIDENCE RETRIEVAL ACCURACY:")
    for k in sorted(page_hits.keys()):
        if k <= args.top_k:
            pct = (page_hits[k] / total_q) * 100.0
            print(f"   Page Hit@{k:<2}            : {page_hits[k]:>3}/{total_q} ({pct:5.1f}%)")
    print(f"   Page Evidence MRR     : {mrr_page_sum / total_q:.4f}")
    print("=" * 70 + "\n")

    # Save results to JSON
    out_path = Path(args.output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_questions": total_q,
        "search_type": args.search_type,
        "rerank": args.rerank,
        "top_k": args.top_k,
        "doc_hits": {f"hit_{k}": doc_hits[k] / total_q for k in doc_hits if k <= args.top_k},
        "page_hits": {f"page_hit_{k}": page_hits[k] / total_q for k in page_hits if k <= args.top_k},
        "page_mrr": round(mrr_page_sum / total_q, 4),
        "duration_seconds": round(duration, 2),
        "details": question_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"Detailed evaluation report saved to: {out_path}")


if __name__ == "__main__":
    main()
