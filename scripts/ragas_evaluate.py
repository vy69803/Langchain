"""RAGAS Evaluation Script with Langfuse Tracing.

Runs automated evaluation of the local RAG pipeline using RAGAS metrics
(Faithfulness, Answer Relevance, Context Recall) and logs every trace to Langfuse
for observability, debugging, and longitudinal quality tracking.

Usage:
    python scripts/ragas_evaluate.py                     # Run with built-in sample dataset
    python scripts/ragas_evaluate.py --dataset eval.json # Run with custom JSON dataset
    python scripts/ragas_evaluate.py --top-k 5           # Change retrieval depth
    python scripts/ragas_evaluate.py --verbose            # Print per-sample details
    python scripts/ragas_evaluate.py --no-langfuse        # Dry run without tracing

Dataset JSON format (array of objects):
    [
      {
        "question": "What is ...?",
        "ground_truth": "The answer is ..."    // optional, for reference
      },
      ...
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Ensure src/ on sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ragas_eval")

# ── RAGAS imports (v0.4.x) ───────────────────────────────────────────────
try:
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_recall import LLMContextRecall
except ImportError as exc:
    logger.error(
        "ragas is not installed or has an incompatible version. "
        "Run:  uv add ragas\n"
        f"Import error: {exc}"
    )
    sys.exit(1)

# ── Langfuse imports (v4.x) ─────────────────────────────────────────────
try:
    from langfuse import Langfuse
except ImportError as exc:
    logger.error(
        "langfuse is not installed. Run:  uv add langfuse\n"
        f"Import error: {exc}"
    )
    sys.exit(1)

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper

# ── Local pipeline imports ───────────────────────────────────────────────
from langchain_rag.llm import get_llm
from langchain_rag.rag_pipeline import RAGPipeline, create_rag_pipeline

# ═════════════════════════════════════════════════════════════════════════
# Sample evaluation dataset (FinanceBench financial questions)
# ═════════════════════════════════════════════════════════════════════════
SAMPLE_EVAL_DATASET: list[dict[str, str]] = [
    {
        "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
        "ground_truth": (
            "In FY2018, 3M's capital expenditures (purchases of property, plant and equipment) "
            "were $1,577 million as reported in the Consolidated Statement of Cash Flows."
        ),
    },
    {
        "question": "What are the primary business segments reported by 3M?",
        "ground_truth": (
            "3M manages its operations in four operating business segments: "
            "Safety and Industrial, Transportation and Electronics, Health Care, and Consumer."
        ),
    },
    {
        "question": "What was Amazon's total net sales in FY2019?",
        "ground_truth": (
            "Amazon reported total net sales of $280,522 million in FY2019, "
            "compared to $232,887 million in FY2018."
        ),
    },
    {
        "question": "What is Apple's primary source of revenue?",
        "ground_truth": (
            "Apple's primary revenue source is iPhone sales, complemented by Services, "
            "Wearables, Home and Accessories, Mac, and iPad."
        ),
    },
]


# ═════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═════════════════════════════════════════════════════════════════════════

def _get_langfuse_client() -> Langfuse | None:
    """Initialize the Langfuse client from environment variables.

    Required env vars:
        LANGFUSE_PUBLIC_KEY
        LANGFUSE_SECRET_KEY
        LANGFUSE_HOST  (defaults to https://cloud.langfuse.com)

    Returns None and logs a warning if credentials are missing.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or public_key.startswith("pk-lf-your") or not secret_key or secret_key.startswith("sk-lf-your"):
        logger.error(
            "Langfuse credentials missing or still set to placeholders.\n"
            "  Add real keys to your .env file:\n"
            "    LANGFUSE_PUBLIC_KEY=pk-lf-...\n"
            "    LANGFUSE_SECRET_KEY=sk-lf-...\n"
            "    LANGFUSE_HOST=https://cloud.langfuse.com\n"
            "  Get your keys at: https://cloud.langfuse.com → Settings → API Keys"
        )
        return None

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )


class LocalChromaEmbeddings(Embeddings):
    """Local embeddings provider using ChromaDB's ONNX MiniLM model without external API keys."""

    def __init__(self) -> None:
        import chromadb.utils.embedding_functions as ef
        self._ef = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, vec)) for vec in self._ef(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(map(float, self._ef([text])[0]))


def _build_evaluator_embeddings() -> LangchainEmbeddingsWrapper:
    """Build a local embedding model wrapper for RAGAS metrics."""
    return LangchainEmbeddingsWrapper(LocalChromaEmbeddings())


def _build_evaluator_llm() -> ChatOpenAI:
    """Build the LLM instance that RAGAS uses as its evaluator judge.

    Uses the same OpenRouter setup and agentic harness headers as the pipeline.
    """
    return get_llm(temperature=0)


def _load_dataset(path: str | Path | None) -> list[dict[str, str]]:
    """Load evaluation dataset from a JSON file or fall back to built-in samples."""
    if path:
        p = Path(path)
        if not p.is_file():
            logger.error(f"Dataset file not found: {p}")
            sys.exit(1)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} evaluation samples from {p}")
        return data

    logger.info(f"Using built-in sample dataset ({len(SAMPLE_EVAL_DATASET)} samples)")
    return SAMPLE_EVAL_DATASET


# ═════════════════════════════════════════════════════════════════════════
# Core evaluation pipeline
# ═════════════════════════════════════════════════════════════════════════

def run_rag_and_collect(
    pipeline: RAGPipeline,
    eval_data: list[dict[str, str]],
    top_k: int = 3,
    langfuse: Langfuse | None = None,
    verbose: bool = False,
) -> list[SingleTurnSample]:
    """Run each question through the RAG pipeline and collect RAGAS samples.

    For each question:
      1. Query the pipeline (retrieval + generation).
      2. Capture the answer, retrieved contexts, and optional ground truth.
      3. Log the individual query trace to Langfuse.

    Returns:
        List of RAGAS SingleTurnSample objects ready for evaluation.
    """
    samples: list[SingleTurnSample] = []
    session_id = f"ragas-eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    for idx, item in enumerate(eval_data, start=1):
        question = item["question"]
        ground_truth = (
            item.get("ground_truth")
            or item.get("expected_answer")
            or item.get("reference")
        )

        logger.info(f"[{idx}/{len(eval_data)}] Querying: {question[:80]}...")

        # ── Run the RAG pipeline ─────────────────────────────────────
        start = time.perf_counter()
        try:
            result = pipeline.query(question=question, k=top_k)
        except Exception as exc:
            logger.error(f"  Pipeline error on sample {idx}: {exc}")
            continue
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        answer = result.get("answer", "")
        raw_results = result.get("raw_results", [])

        # Extract context strings from retrieved chunks
        contexts = []
        for r in raw_results:
            text = r.get("text", "").strip()
            if text:
                contexts.append(text)

        # ── Log pipeline output to Langfuse ──────────────────────────
        if langfuse:
            try:
                if hasattr(langfuse, "start_observation"):
                    # Langfuse v3 / v4 API
                    trace = langfuse.start_observation(
                        name=f"rag-eval-query-{idx}",
                        input={"question": question},
                        output={
                            "answer": answer[:500],
                            "num_contexts": len(contexts),
                            "latency_ms": latency_ms,
                        },
                        metadata={
                            "session_id": session_id,
                            "eval_index": idx,
                            "has_ground_truth": ground_truth is not None,
                            "top_k": top_k,
                            "tags": ["ragas", "eval-query"],
                        },
                    )

                    # Log the retrieval step as a child span
                    retrieval_span = trace.start_observation(
                        name="retrieval",
                        as_type="span",
                        input={"query": question, "top_k": top_k},
                        output={
                            "num_contexts": len(contexts),
                            "context_lengths": [len(c) for c in contexts],
                            "context_previews": [c[:150] + "..." for c in contexts],
                        },
                        metadata={"latency_ms": latency_ms},
                    )
                    retrieval_span.end()

                    # Log the generation step
                    gen_span = trace.start_observation(
                        name="llm-generation",
                        as_type="generation",
                        model="thinkingmachines/inkling:free",
                        input=result.get("context", "")[:500],
                        output=answer[:1000],
                        metadata={"latency_ms": latency_ms},
                    )
                    gen_span.end()

                    trace.end()
                elif hasattr(langfuse, "trace"):
                    # Legacy Langfuse v2 API
                    trace = langfuse.trace(
                        name=f"rag-eval-query-{idx}",
                        session_id=session_id,
                        input={"question": question},
                        output={
                            "answer": answer[:500],
                            "num_contexts": len(contexts),
                            "latency_ms": latency_ms,
                        },
                        metadata={
                            "eval_index": idx,
                            "has_ground_truth": ground_truth is not None,
                            "top_k": top_k,
                        },
                        tags=["ragas", "eval-query"],
                    )

                    retrieval_span = trace.span(
                        name="retrieval",
                        input={"query": question, "top_k": top_k},
                        output={
                            "num_contexts": len(contexts),
                            "context_lengths": [len(c) for c in contexts],
                            "context_previews": [c[:150] + "..." for c in contexts],
                        },
                        metadata={"latency_ms": latency_ms},
                    )
                    retrieval_span.end()

                    trace.generation(
                        name="llm-generation",
                        model="thinkingmachines/inkling:free",
                        input=result.get("context", "")[:500],
                        output=answer[:1000],
                        metadata={"latency_ms": latency_ms},
                    )
            except Exception as exc:
                logger.warning(f"  Langfuse logging error on sample {idx}: {exc}")

        # ── Build RAGAS sample ───────────────────────────────────────
        sample_kwargs: dict[str, Any] = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts if contexts else ["No context retrieved."],
        }
        if ground_truth:
            sample_kwargs["reference"] = ground_truth

        samples.append(SingleTurnSample(**sample_kwargs))

        if verbose:
            print(f"\n{'─'*70}")
            print(f"  Q{idx}: {question}")
            print(f"  Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print(f"  Contexts: {len(contexts)} chunks retrieved")
            print(f"  Latency: {latency_ms} ms")
            print(f"{'─'*70}")

    # Flush all Langfuse events
    if langfuse:
        try:
            langfuse.flush()
        except Exception as exc:
            logger.warning(f"Langfuse flush warning: {exc}")

    return samples


def run_evaluation(
    samples: list[SingleTurnSample],
    include_context_recall: bool = True,
) -> Any:
    """Run RAGAS evaluation on collected samples.

    Metrics:
        - Faithfulness: Is the answer grounded in the retrieved contexts?
        - Answer Relevancy: Does the answer actually address the question?
        - Context Recall: Did the retrieved context contain the ground-truth information?

    Returns:
        RAGAS EvaluationResult object with metric scores.
    """
    if not samples:
        logger.warning("No samples to evaluate!")
        return {"error": "No samples collected"}

    evaluator_llm = _build_evaluator_llm()
    evaluator_embeddings = _build_evaluator_embeddings()

    # Instantiate generation quality metrics with local embeddings
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    # Context Recall requires ground-truth references on all evaluated samples
    if include_context_recall:
        missing_ref_count = sum(1 for s in samples if not getattr(s, "reference", None))
        if missing_ref_count == 0:
            metrics.append(LLMContextRecall(llm=evaluator_llm))
            logger.info("Context Recall metric included (all samples have ground-truth references)")
        else:
            logger.warning(
                f"Skipping Context Recall: {missing_ref_count}/{len(samples)} samples lack ground-truth reference."
            )

    logger.info(
        f"Running RAGAS evaluation on {len(samples)} samples "
        f"with metrics: {[m.name for m in metrics]}"
    )

    eval_dataset = EvaluationDataset(samples=samples)

    # Run RAGAS evaluation with explicit local embeddings
    ragas_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return ragas_result


def _extract_scores(ragas_result: Any) -> tuple[dict[str, float], list[dict]]:
    """Extract aggregate and per-sample scores from a RAGAS result object.

    Works with multiple RAGAS result formats (dict, DataFrame, .scores attr).
    """
    scores: dict[str, float] = {}
    per_sample: list[dict] = []

    # RAGAS 0.4.x returns an EvaluationResult with .scores (list of dicts)
    if hasattr(ragas_result, "scores") and ragas_result.scores:
        per_sample = ragas_result.scores
        all_keys = per_sample[0].keys()
        for key in all_keys:
            vals = [s[key] for s in per_sample if key in s and s[key] is not None]
            if vals:
                scores[key] = sum(vals) / len(vals)
    # Some RAGAS versions return a DataFrame via .to_pandas()
    elif hasattr(ragas_result, "to_pandas"):
        try:
            df = ragas_result.to_pandas()
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            for col in numeric_cols:
                scores[col] = float(df[col].mean())
            per_sample = df[numeric_cols].to_dict("records")
        except Exception:
            pass
    # Fallback: treat as dict
    elif isinstance(ragas_result, dict):
        scores = {k: v for k, v in ragas_result.items() if isinstance(v, (int, float))}

    return scores, per_sample


def log_results_to_langfuse(
    ragas_result: Any,
    langfuse: Langfuse,
    num_samples: int,
) -> None:
    """Log the aggregate RAGAS scores to Langfuse as a dedicated trace."""
    try:
        scores, _ = _extract_scores(ragas_result)

        trace_id = None
        if hasattr(langfuse, "start_observation"):
            summary_obs = langfuse.start_observation(
                name="ragas-evaluation-summary",
                input={"num_samples": num_samples, "metrics": list(scores.keys())},
                output=scores,
                metadata={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "evaluation_type": "ragas_automated",
                    "tags": ["ragas", "summary", "evaluation"],
                },
            )
            trace_id = summary_obs.trace_id
            summary_obs.end()
        elif hasattr(langfuse, "trace"):
            trace = langfuse.trace(
                name="ragas-evaluation-summary",
                input={"num_samples": num_samples, "metrics": list(scores.keys())},
                output=scores,
                metadata={
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "evaluation_type": "ragas_automated",
                },
                tags=["ragas", "summary", "evaluation"],
            )
            trace_id = trace.id

        # Log each metric as a Langfuse score on the summary trace
        if trace_id:
            for metric_name, metric_value in scores.items():
                langfuse.create_score(
                    trace_id=trace_id,
                    name=metric_name,
                    value=round(metric_value, 4),
                    comment=f"RAGAS {metric_name} (mean over {num_samples} samples)",
                )

        langfuse.flush()
        logger.info(f"Logged RAGAS scores to Langfuse trace: {trace_id}")
    except Exception as exc:
        logger.warning(f"Failed to log summary scores to Langfuse: {exc}")


def print_report(ragas_result: Any, duration: float) -> None:
    """Print a formatted evaluation report to stdout."""
    print("\n" + "═" * 70)
    print("  RAGAS EVALUATION REPORT")
    print("═" * 70)
    print(f"  Timestamp : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Duration  : {duration:.1f}s")

    scores, per_sample = _extract_scores(ragas_result)

    print(f"  Samples   : {len(per_sample) if per_sample else 'N/A'}")
    print("─" * 70)
    print("  AGGREGATE SCORES:")
    for metric, value in scores.items():
        bar_len = max(0, min(20, int(value * 20)))
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {metric:<25s}  {bar}  {value:.4f}")

    if per_sample:
        print("\n" + "─" * 70)
        print("  PER-SAMPLE BREAKDOWN:")
        header_keys = list(per_sample[0].keys())
        print(f"  {'#':<4s} ", end="")
        for key in header_keys:
            print(f"{key:<20s} ", end="")
        print()
        print("  " + "─" * 66)
        for idx, sample_scores in enumerate(per_sample, start=1):
            print(f"  {idx:<4d} ", end="")
            for val in sample_scores.values():
                if isinstance(val, float):
                    print(f"{val:<20.4f} ", end="")
                else:
                    print(f"{str(val):<20s} ", end="")
            print()

    print("═" * 70)

    # Quality assessment
    faith = scores.get("faithfulness", 0)
    relevancy = scores.get("answer_relevancy", 0)
    recall = scores.get("context_recall")

    eval_metrics = [("Faithfulness", faith), ("Answer Relevancy", relevancy)]
    if recall is not None:
        eval_metrics.append(("Context Recall", recall))

    all_values = [v for _, v in eval_metrics]
    if all(v >= 0.80 for v in all_values):
        print(f"  ✅ Pipeline quality: EXCELLENT — All {len(eval_metrics)} metrics above 0.80 threshold")
    elif all(v >= 0.60 for v in all_values):
        print(f"  ⚠️  Pipeline quality: FAIR — All metrics above 0.60, but some below 0.80")
    else:
        below_60 = [name for name, v in eval_metrics if v < 0.60]
        print(f"  ❌ Pipeline quality: NEEDS IMPROVEMENT — Below 0.60 on: {', '.join(below_60)}")
    print("═" * 70 + "\n")


# ═════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the local RAG pipeline with Langfuse tracing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a JSON file with evaluation questions (default: built-in samples)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of context chunks to retrieve per question (default: 3)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="financebench",
        help="ChromaDB collection name to query (default: financebench)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./data/chroma_db",
        help="ChromaDB persist directory (default: ./data/chroma_db)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query details during execution",
    )
    parser.add_argument(
        "--no-langfuse",
        action="store_true",
        help="Skip Langfuse tracing (useful for dry-run testing)",
    )
    parser.add_argument(
        "--no-context-recall",
        action="store_true",
        help="Skip Context Recall metric calculation (useful if dataset lacks ground truth references)",
    )

    args = parser.parse_args()

    # ── 1. Load evaluation dataset ───────────────────────────────────
    eval_data = _load_dataset(args.dataset)

    # ── 2. Initialize the RAG pipeline ───────────────────────────────
    logger.info(
        f"Initializing RAG pipeline "
        f"(collection={args.collection}, persist_dir={args.persist_dir})"
    )
    pipeline = create_rag_pipeline(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
    )
    logger.info(f"Vector store document count: {pipeline.vector_store.count()}")

    # ── 3. Initialize Langfuse ───────────────────────────────────────
    langfuse_client: Langfuse | None = None
    if not args.no_langfuse:
        langfuse_client = _get_langfuse_client()
        if langfuse_client:
            logger.info("Langfuse tracing enabled — traces will be logged")
        else:
            logger.warning("Langfuse tracing DISABLED (credentials missing)")
    else:
        logger.info("Langfuse tracing DISABLED (--no-langfuse flag)")

    # ── 4. Run RAG queries and collect samples ───────────────────────
    start_time = time.perf_counter()

    samples = run_rag_and_collect(
        pipeline=pipeline,
        eval_data=eval_data,
        top_k=args.top_k,
        langfuse=langfuse_client,
        verbose=args.verbose,
    )
    logger.info(f"Collected {len(samples)} RAGAS samples from pipeline")

    # ── 5. Run RAGAS evaluation ──────────────────────────────────────
    ragas_result = run_evaluation(
        samples=samples,
        include_context_recall=not args.no_context_recall,
    )

    total_duration = time.perf_counter() - start_time

    # ── 6. Log results to Langfuse ───────────────────────────────────
    if langfuse_client:
        log_results_to_langfuse(
            ragas_result=ragas_result,
            langfuse=langfuse_client,
            num_samples=len(samples),
        )

    # ── 7. Print report ──────────────────────────────────────────────
    print_report(ragas_result, total_duration)

    # ── 8. Save results to JSON ──────────────────────────────────────
    results_dir = ROOT / "data"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "ragas_eval_results.json"

    scores, per_sample = _extract_scores(ragas_result)

    export_data: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(samples),
        "duration_seconds": round(total_duration, 2),
        "collection": args.collection,
        "top_k": args.top_k,
        "aggregate_scores": {k: round(v, 4) for k, v in scores.items()},
    }
    if per_sample:
        export_data["per_sample_scores"] = per_sample

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Results saved to {results_path}")

    # ── 9. Cleanup ───────────────────────────────────────────────────
    if langfuse_client:
        try:
            langfuse_client.flush()
            langfuse_client.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
