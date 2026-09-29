"""Information Retrieval (IR) and RAG Retrieval Quality Metrics.

Implements core retrieval quality metrics for RAG and search systems:
- Precision@K: Fraction of retrieved documents in top-K that are relevant.
- Recall@K: Fraction of all known relevant documents retrieved in top-K.
- Mean Reciprocal Rank (MRR / RR): Reciprocal rank of the first relevant document.
- Normalized Discounted Cumulative Gain (NDCG@K): Graded relevance ranking metric
  penalizing systems that push highly relevant documents down the list.
- HitRate@K (Hit@K): Whether at least one relevant document appears in top-K.
- Average Precision (AP@K) / MAP: Mean average precision across retrieval depths.
"""

from __future__ import annotations

import math
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

RelevantDocs = Union[Container[str], Mapping[str, float], Sequence[str]]


def _get_relevance_score(doc_id: str, relevant: RelevantDocs) -> float:
    """Extract numeric relevance score for a document.
    
    Supports:
    - Mapping[str, float]: Graded relevance, e.g. {"chunk_1": 3.0, "chunk_2": 1.0}
    - Set[str] or Sequence[str]: Binary relevance, returning 1.0 if present else 0.0
    """
    if isinstance(relevant, Mapping):
        return float(relevant.get(doc_id, 0.0))
    return 1.0 if doc_id in relevant else 0.0


def _is_relevant(doc_id: str, relevant: RelevantDocs, threshold: float = 0.0) -> bool:
    """Check if document meets the relevance threshold (score > threshold)."""
    return _get_relevance_score(doc_id, relevant) > threshold


def _get_total_relevant_count(relevant: RelevantDocs, threshold: float = 0.0) -> int:
    """Return the total number of relevant documents available in ground truth."""
    if isinstance(relevant, Mapping):
        return sum(1 for score in relevant.values() if float(score) > threshold)
    if isinstance(relevant, (set, list, tuple)):
        return len(set(relevant))
    # General container fallback
    return len(relevant)  # type: ignore[arg-type]


def precision_at_k(
    retrieved: Sequence[str],
    relevant: RelevantDocs,
    k: Optional[int] = None,
) -> float:
    """Compute Precision@K.

    Precision@K measures the percentage of retrieved results in top-K that are useful.
    Formula:
        Precision@K = |Retrieved_K ∩ Relevant| / K

    Args:
        retrieved: Ordered list of retrieved document/chunk IDs.
        relevant: Ground-truth relevant document IDs (as set/list or dict of scores).
        k: Retrieval depth. If None, uses len(retrieved).

    Returns:
        Float in [0.0, 1.0].
    """
    if k is None:
        k = len(retrieved)
    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    if not top_k:
        return 0.0

    relevant_hits = sum(1 for doc_id in top_k if _is_relevant(doc_id, relevant))
    return relevant_hits / float(k)


def recall_at_k(
    retrieved: Sequence[str],
    relevant: RelevantDocs,
    k: Optional[int] = None,
) -> float:
    """Compute Recall@K.

    Recall@K measures the fraction of all available good documents successfully found in top-K.
    Formula:
        Recall@K = |Retrieved_K ∩ Relevant| / |Total Relevant|

    Args:
        retrieved: Ordered list of retrieved document/chunk IDs.
        relevant: Ground-truth relevant document IDs (as set/list or dict of scores).
        k: Retrieval depth. If None, uses len(retrieved).

    Returns:
        Float in [0.0, 1.0]. If no relevant documents exist, returns 0.0.
    """
    total_relevant = _get_total_relevant_count(relevant)
    if total_relevant == 0:
        return 0.0

    if k is None:
        top_k = retrieved
    elif k <= 0:
        return 0.0
    else:
        top_k = retrieved[:k]

    hits = sum(1 for doc_id in top_k if _is_relevant(doc_id, relevant))
    return min(1.0, hits / float(total_relevant))


def hit_rate_at_k(
    retrieved: Sequence[str],
    relevant: RelevantDocs,
    k: Optional[int] = None,
) -> float:
    """Compute HitRate@K (Hit@K).

    HitRate@K is 1.0 if at least one relevant document is in top-K, otherwise 0.0.

    Args:
        retrieved: Ordered list of retrieved document/chunk IDs.
        relevant: Ground-truth relevant document IDs.
        k: Retrieval depth.

    Returns:
        1.0 if a hit occurred in top-K, else 0.0.
    """
    if k is None:
        top_k = retrieved
    elif k <= 0:
        return 0.0
    else:
        top_k = retrieved[:k]

    for doc_id in top_k:
        if _is_relevant(doc_id, relevant):
            return 1.0
    return 0.0


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: RelevantDocs,
    k: Optional[int] = None,
) -> float:
    """Compute Reciprocal Rank (RR).

    Score checking how high the very first good result appears.
    Rank 1 gives 1.0, rank 2 gives 0.5, rank 3 gives 0.333, etc.
    Formula:
        RR = 1.0 / rank_first (where rank_first is 1-indexed)

    Args:
        retrieved: Ordered list of retrieved document/chunk IDs.
        relevant: Ground-truth relevant document IDs.
        k: Maximum cutoff rank to consider. If None, considers all retrieved items.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if no relevant result is found within top-K.
    """
    if k is not None:
        if k <= 0:
            return 0.0
        retrieved = retrieved[:k]

    for rank, doc_id in enumerate(retrieved, start=1):
        if _is_relevant(doc_id, relevant):
            return 1.0 / float(rank)
    return 0.0


def mean_reciprocal_rank(
    retrieved_list: Sequence[Sequence[str]],
    relevant_list: Sequence[RelevantDocs],
    k: Optional[int] = None,
) -> float:
    """Compute Mean Reciprocal Rank (MRR) across multiple queries.

    Formula:
        MRR = (1 / Q) * sum(RR_q for q in Q)

    Args:
        retrieved_list: List of retrieved ID lists (one per query).
        relevant_list: List of ground-truth relevant sets/mappings (one per query).
        k: Cutoff rank.

    Returns:
        Mean Reciprocal Rank score in [0.0, 1.0].
    """
    if len(retrieved_list) != len(relevant_list):
        raise ValueError(
            f"Length mismatch: {len(retrieved_list)} retrieved queries vs "
            f"{len(relevant_list)} relevant queries"
        )
    if not retrieved_list:
        return 0.0

    rrs = [
        reciprocal_rank(ret, rel, k=k)
        for ret, rel in zip(retrieved_list, relevant_list)
    ]
    return sum(rrs) / float(len(rrs))


def dcg_at_k(
    retrieved: Sequence[str],
    relevance_scores: RelevantDocs,
    k: Optional[int] = None,
    method: str = "exponential",
) -> float:
    """Compute Discounted Cumulative Gain (DCG@K).

    Formula (exponential gain, standard in TREC / MS MARCO):
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)

    Formula (linear gain):
        DCG@K = sum_{i=1}^K rel_i / log2(i + 1)

    Args:
        retrieved: Ordered list of retrieved document/chunk IDs.
        relevance_scores: Graded relevance mapping or set of binary relevant IDs.
        k: Retrieval depth.
        method: "exponential" (default, 2^rel - 1) or "linear" (rel).

    Returns:
        DCG score (>= 0.0).
    """
    if k is not None:
        if k <= 0:
            return 0.0
        retrieved = retrieved[:k]

    dcg = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        rel = _get_relevance_score(doc_id, relevance_scores)
        if rel <= 0.0:
            continue
        if method == "exponential":
            gain = math.pow(2.0, rel) - 1.0
        elif method == "linear":
            gain = rel
        else:
            raise ValueError(f"Unknown DCG method '{method}'. Choose 'exponential' or 'linear'.")
        discount = math.log2(rank + 1.0)
        dcg += gain / discount

    return dcg


def idcg_at_k(
    relevance_scores: RelevantDocs,
    k: Optional[int] = None,
    method: str = "exponential",
) -> float:
    """Compute Ideal Discounted Cumulative Gain (IDCG@K).

    IDCG is the DCG of the perfectly sorted ground-truth relevant documents in descending order.

    Args:
        relevance_scores: Graded relevance mapping or set of binary relevant IDs.
        k: Retrieval depth.
        method: "exponential" or "linear".

    Returns:
        Ideal DCG score (>= 0.0).
    """
    if isinstance(relevance_scores, Mapping):
        scores = [float(s) for s in relevance_scores.values() if float(s) > 0.0]
    else:
        scores = [1.0] * len(set(relevance_scores))

    scores.sort(reverse=True)
    if k is not None:
        scores = scores[:k]

    idcg = 0.0
    for rank, rel in enumerate(scores, start=1):
        if method == "exponential":
            gain = math.pow(2.0, rel) - 1.0
        elif method == "linear":
            gain = rel
        else:
            raise ValueError(f"Unknown DCG method '{method}'.")
        discount = math.log2(rank + 1.0)
        idcg += gain / discount

    return idcg


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance_scores: RelevantDocs,
    k: Optional[int] = None,
    method: str = "exponential",
) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@K).

    Measures ranking quality with graded relevance. Perfectly ranked results
    yield 1.0. Systems that push relevant documents down are heavily penalized.
    Formula:
        NDCG@K = DCG@K / IDCG@K

    Args:
        retrieved: Ordered list of retrieved document/chunk IDs.
        relevance_scores: Graded relevance mapping or set/sequence of relevant IDs.
        k: Retrieval depth.
        method: "exponential" (default) or "linear".

    Returns:
        Float in [0.0, 1.0]. If ideal DCG is 0.0 (no relevant items), returns 0.0.
    """
    ideal_dcg = idcg_at_k(relevance_scores, k=k, method=method)
    if ideal_dcg == 0.0:
        return 0.0

    actual_dcg = dcg_at_k(retrieved, relevance_scores, k=k, method=method)
    ndcg = actual_dcg / ideal_dcg
    return min(1.0, max(0.0, ndcg))


def average_precision_at_k(
    retrieved: Sequence[str],
    relevant: RelevantDocs,
    k: Optional[int] = None,
) -> float:
    """Compute Average Precision at K (AP@K).

    Area under the precision-recall curve up to rank K.
    Formula:
        AP@K = sum_{i=1}^K (Precision@i * is_relevant(doc_i)) / min(K, total_relevant)
    """
    total_relevant = _get_total_relevant_count(relevant)
    if total_relevant == 0:
        return 0.0

    if k is not None:
        if k <= 0:
            return 0.0
        retrieved = retrieved[:k]

    score = 0.0
    num_hits = 0

    for i, doc_id in enumerate(retrieved, start=1):
        if _is_relevant(doc_id, relevant):
            num_hits += 1
            precision_at_i = num_hits / float(i)
            score += precision_at_i

    denominator = min(len(retrieved), total_relevant) if k is None else min(k, total_relevant)
    if denominator == 0:
        return 0.0
    return score / float(denominator)


@dataclass
class QueryRetrievalResult:
    """Single query evaluation result across multiple metrics."""
    query_id: str
    precision: Dict[int, float] = field(default_factory=dict)
    recall: Dict[int, float] = field(default_factory=dict)
    hit_rate: Dict[int, float] = field(default_factory=dict)
    ndcg: Dict[int, float] = field(default_factory=dict)
    reciprocal_rank: float = 0.0


def evaluate_query_retrieval(
    retrieved: Sequence[str],
    relevant: RelevantDocs,
    query_id: str = "query_1",
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> QueryRetrievalResult:
    """Evaluate retrieval quality for a single query across multiple K values."""
    res = QueryRetrievalResult(
        query_id=query_id,
        reciprocal_rank=reciprocal_rank(retrieved, relevant),
    )
    for k in k_values:
        res.precision[k] = precision_at_k(retrieved, relevant, k=k)
        res.recall[k] = recall_at_k(retrieved, relevant, k=k)
        res.hit_rate[k] = hit_rate_at_k(retrieved, relevant, k=k)
        res.ndcg[k] = ndcg_at_k(retrieved, relevant, k=k)
    return res


class RetrievalBenchmarkEvaluator:
    """Evaluates batches of queries against ground truth, aggregating all core retrieval metrics."""

    def __init__(self, k_values: Sequence[int] = (1, 3, 5, 10)):
        self.k_values = list(k_values)

    def evaluate_batch(
        self,
        retrieved_batch: Sequence[Sequence[str]],
        relevant_batch: Sequence[RelevantDocs],
        query_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Compute aggregated metrics over a batch of queries."""
        if len(retrieved_batch) != len(relevant_batch):
            raise ValueError("Size mismatch between retrieved and relevant query batches")

        n = len(retrieved_batch)
        if n == 0:
            return {"count": 0}

        if query_ids is None:
            query_ids = [f"q_{i+1}" for i in range(n)]

        results: List[QueryRetrievalResult] = []
        for qid, ret, rel in zip(query_ids, retrieved_batch, relevant_batch):
            results.append(evaluate_query_retrieval(ret, rel, query_id=qid, k_values=self.k_values))

        summary: Dict[str, Any] = {
            "total_queries": n,
            "mrr": sum(r.reciprocal_rank for r in results) / float(n),
        }

        for k in self.k_values:
            summary[f"precision@{k}"] = sum(r.precision[k] for r in results) / float(n)
            summary[f"recall@{k}"] = sum(r.recall[k] for r in results) / float(n)
            summary[f"hit_rate@{k}"] = sum(r.hit_rate[k] for r in results) / float(n)
            summary[f"ndcg@{k}"] = sum(r.ndcg[k] for r in results) / float(n)

        return summary
