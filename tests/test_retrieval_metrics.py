"""Comprehensive Unit Tests for RAG Retrieval Quality Metrics.

Tests the core retrieval quality metrics:
1. Precision@K: Checks percentage of retrieved results that are actually useful.
2. Recall@K: Checks fraction of available good documents successfully found.
3. Mean Reciprocal Rank (MRR / RR): Checks how high the first good result appears (rank 1 -> 1.0, rank 2 -> 0.5, etc.).
4. Normalized Discounted Cumulative Gain (NDCG@K): Checks graded relevance and penalization of degraded rankings.
5. HitRate@K and Batch Benchmark Evaluator.
"""

from __future__ import annotations

import math
import unittest
from typing import Dict, List, Set

from langchain_rag.retrieval_metrics import (
    RetrievalBenchmarkEvaluator,
    average_precision_at_k,
    dcg_at_k,
    evaluate_query_retrieval,
    hit_rate_at_k,
    idcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestPrecisionAtK(unittest.TestCase):
    """Test suite for Precision@K calculation."""

    def test_user_scenario_7_out_of_10(self):
        """User prompt example: 'If you pull 10 text blocks and 7 help, precision is 0.7 (or 70%).'"""
        retrieved = [f"doc_{i}" for i in range(1, 11)]  # doc_1 ... doc_10
        # 7 of the 10 help
        relevant = {f"doc_{i}" for i in range(1, 8)}    # doc_1 ... doc_7

        p10 = precision_at_k(retrieved, relevant, k=10)
        self.assertAlmostEqual(p10, 0.7, places=5)

    def test_precision_at_varying_k(self):
        """Precision at cutoff k=1, 2, 3, 5 with changing relevance."""
        retrieved = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
        relevant = {"doc_A", "doc_C"}

        # k=1: ["doc_A"] -> 1/1 = 1.0
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=1), 1.0)
        # k=2: ["doc_A", "doc_B"] -> 1/2 = 0.5
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=2), 0.5)
        # k=3: ["doc_A", "doc_B", "doc_C"] -> 2/3 = ~0.66667
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=3), 2.0 / 3.0)
        # k=4: ["doc_A", "doc_B", "doc_C", "doc_D"] -> 2/4 = 0.5
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=4), 0.5)
        # k=5: ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"] -> 2/5 = 0.4
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=5), 0.4)

    def test_precision_with_graded_relevance_dict(self):
        """Relevance passed as a mapping with positive scores."""
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1": 3.0, "d2": 0.0, "d3": 1.5}  # d1 and d3 are relevant
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, k=3), 2.0 / 3.0)

    def test_precision_edge_cases(self):
        """Edge cases: empty inputs, zero k, k greater than retrieved length."""
        relevant = {"d1", "d2"}
        # Empty retrieved
        self.assertEqual(precision_at_k([], relevant, k=5), 0.0)
        # Zero or negative k
        self.assertEqual(precision_at_k(["d1"], relevant, k=0), 0.0)
        self.assertEqual(precision_at_k(["d1"], relevant, k=-1), 0.0)
        # Empty relevant set
        self.assertEqual(precision_at_k(["d1", "d2"], set(), k=2), 0.0)
        # None k uses entire retrieved list length
        self.assertEqual(precision_at_k(["d1", "d2", "d3"], {"d1"}, k=None), 1.0 / 3.0)
        # k > len(retrieved) divides by k
        self.assertEqual(precision_at_k(["d1", "d2"], {"d1"}, k=5), 1.0 / 5.0)


class TestRecallAtK(unittest.TestCase):
    """Test suite for Recall@K calculation."""

    def test_user_scenario_7_out_of_10_available(self):
        """User prompt example: 'If 10 total good blocks exist and you retrieved 7 of them, recall is 0.7.'"""
        all_good_blocks = {f"good_{i}" for i in range(1, 11)}  # 10 good blocks exist
        retrieved = [f"good_{i}" for i in range(1, 8)] + ["irrelevant_1", "irrelevant_2", "irrelevant_3"]

        r10 = recall_at_k(retrieved, all_good_blocks, k=10)
        self.assertAlmostEqual(r10, 0.7, places=5)

    def test_recall_at_varying_k(self):
        """Recall increases as cutoff k reaches more relevant documents."""
        all_relevant = {"doc_1", "doc_2", "doc_3", "doc_4"}  # 4 total
        retrieved = ["doc_1", "other_1", "doc_2", "other_2", "doc_3", "doc_4"]

        # k=1: retrieves doc_1 -> 1/4 = 0.25
        self.assertAlmostEqual(recall_at_k(retrieved, all_relevant, k=1), 0.25)
        # k=2: retrieves doc_1, other_1 -> 1/4 = 0.25
        self.assertAlmostEqual(recall_at_k(retrieved, all_relevant, k=2), 0.25)
        # k=3: retrieves doc_1, other_1, doc_2 -> 2/4 = 0.5
        self.assertAlmostEqual(recall_at_k(retrieved, all_relevant, k=3), 0.50)
        # k=5: retrieves doc_1, doc_2, doc_3 -> 3/4 = 0.75
        self.assertAlmostEqual(recall_at_k(retrieved, all_relevant, k=5), 0.75)
        # k=6: retrieves all 4 -> 4/4 = 1.0
        self.assertAlmostEqual(recall_at_k(retrieved, all_relevant, k=6), 1.0)

    def test_recall_edge_cases(self):
        """Edge cases: empty inputs, zero relevant docs, zero hits."""
        self.assertEqual(recall_at_k([], {"d1"}, k=5), 0.0)
        self.assertEqual(recall_at_k(["d1"], set(), k=5), 0.0)  # no relevant docs exist
        self.assertEqual(recall_at_k(["d1", "d2"], {"d3"}, k=2), 0.0)  # zero hits
        self.assertEqual(recall_at_k(["d1", "d2"], {"d1"}, k=0), 0.0)


class TestReciprocalRankAndMRR(unittest.TestCase):
    """Test suite for Reciprocal Rank (RR) and Mean Reciprocal Rank (MRR)."""

    def test_user_scenario_ranks(self):
        """User prompt example: 'Rank 1 gives a score of 1.0, rank 2 gives 0.5, and lower ranks score much less.'"""
        relevant = {"target"}

        # Rank 1 -> 1.0
        self.assertAlmostEqual(reciprocal_rank(["target", "other1", "other2"], relevant), 1.0)
        # Rank 2 -> 0.5
        self.assertAlmostEqual(reciprocal_rank(["other1", "target", "other2"], relevant), 0.5)
        # Rank 3 -> 1/3 ~ 0.33333
        self.assertAlmostEqual(reciprocal_rank(["other1", "other2", "target"], relevant), 1.0 / 3.0)
        # Rank 4 -> 0.25
        self.assertAlmostEqual(reciprocal_rank(["o1", "o2", "o3", "target"], relevant), 0.25)
        # Rank 5 -> 0.20
        self.assertAlmostEqual(reciprocal_rank(["o1", "o2", "o3", "o4", "target"], relevant), 0.2)

    def test_reciprocal_rank_with_k_cutoff(self):
        """If first relevant doc appears after rank K, score is 0.0."""
        relevant = {"target"}
        retrieved = ["o1", "o2", "target", "o4"]

        # k=2: target is at rank 3, which is > 2, so RR@2 is 0.0
        self.assertEqual(reciprocal_rank(retrieved, relevant, k=2), 0.0)
        # k=3: target is at rank 3, so RR@3 is 1/3
        self.assertAlmostEqual(reciprocal_rank(retrieved, relevant, k=3), 1.0 / 3.0)

    def test_reciprocal_rank_no_hit(self):
        """When no relevant doc is retrieved, RR is 0.0."""
        self.assertEqual(reciprocal_rank(["o1", "o2", "o3"], {"target"}), 0.0)

    def test_mean_reciprocal_rank_batch(self):
        """Compute MRR over multiple queries."""
        batch_retrieved = [
            ["doc_hit", "other"],           # Query 1: hit at rank 1 -> RR = 1.0
            ["other", "doc_hit"],           # Query 2: hit at rank 2 -> RR = 0.5
            ["other1", "other2"],           # Query 3: no hit -> RR = 0.0
            ["other1", "other2", "doc_hit"],# Query 4: hit at rank 3 -> RR = 1/3
        ]
        batch_relevant = [
            {"doc_hit"},
            {"doc_hit"},
            {"doc_hit"},
            {"doc_hit"},
        ]

        expected_mrr = (1.0 + 0.5 + 0.0 + (1.0 / 3.0)) / 4.0
        actual_mrr = mean_reciprocal_rank(batch_retrieved, batch_relevant)
        self.assertAlmostEqual(actual_mrr, expected_mrr, places=5)

    def test_mrr_validation_and_empty(self):
        """MRR handles empty batch and validates matching lengths."""
        self.assertEqual(mean_reciprocal_rank([], []), 0.0)
        with self.assertRaises(ValueError):
            mean_reciprocal_rank([["d1"]], [])


class TestNDCG(unittest.TestCase):
    """Test suite for Normalized Discounted Cumulative Gain (NDCG@K)."""

    def test_perfect_ranking_yields_one(self):
        """If retrieved documents are perfectly ordered by relevance, NDCG is exactly 1.0."""
        relevance_scores = {
            "doc_high": 3.0,
            "doc_med": 2.0,
            "doc_low": 1.0,
            "doc_none": 0.0,
        }
        perfect_retrieval = ["doc_high", "doc_med", "doc_low"]

        ndcg = ndcg_at_k(perfect_retrieval, relevance_scores, k=3)
        self.assertAlmostEqual(ndcg, 1.0, places=5)

    def test_penalizes_pushing_good_data_down(self):
        """User prompt specification:
        'It checks if the most useful documents rank near the top and heavily
         penalizes systems that push good data down the list.'
        """
        relevance_scores = {
            "doc_critical": 3.0,
            "doc_helpful": 2.0,
            "doc_minor": 1.0,
        }

        # System A ranks the most critical first
        system_a = ["doc_critical", "doc_helpful", "doc_minor"]
        # System B pushes the most critical to the bottom
        system_b = ["doc_minor", "doc_helpful", "doc_critical"]

        ndcg_a = ndcg_at_k(system_a, relevance_scores, k=3)
        ndcg_b = ndcg_at_k(system_b, relevance_scores, k=3)

        self.assertAlmostEqual(ndcg_a, 1.0, places=5)
        # System B should be heavily penalized
        self.assertLess(ndcg_b, 0.75)
        self.assertGreater(ndcg_b, 0.0)

    def test_exact_ndcg_formula_calculation(self):
        """Manually verify exponential gain formula:
        DCG = sum (2^rel - 1) / log2(rank + 1)
        """
        scores = {"A": 3.0, "B": 2.0, "C": 1.0}

        # Ideal ranking: [A, B, C]
        # Rank 1 (A, rel=3): (2^3 - 1)/log2(2) = 7 / 1.0 = 7.0
        # Rank 2 (B, rel=2): (2^2 - 1)/log2(3) = 3 / 1.5849625 ≈ 1.892789
        # Rank 3 (C, rel=1): (2^1 - 1)/log2(4) = 1 / 2.0 = 0.5
        expected_idcg = 7.0 + (3.0 / math.log2(3)) + 0.5
        calculated_idcg = idcg_at_k(scores, k=3)
        self.assertAlmostEqual(calculated_idcg, expected_idcg, places=5)

        # Actual ranking: [C, B, A]
        # Rank 1 (C): 1 / 1.0 = 1.0
        # Rank 2 (B): 3 / log2(3) ≈ 1.892789
        # Rank 3 (A): 7 / 2.0 = 3.5
        expected_dcg = 1.0 + (3.0 / math.log2(3)) + 3.5
        calculated_dcg = dcg_at_k(["C", "B", "A"], scores, k=3)
        self.assertAlmostEqual(calculated_dcg, expected_dcg, places=5)

        expected_ndcg = expected_dcg / expected_idcg
        actual_ndcg = ndcg_at_k(["C", "B", "A"], scores, k=3)
        self.assertAlmostEqual(actual_ndcg, expected_ndcg, places=5)

    def test_binary_relevance_ndcg(self):
        """NDCG works seamlessly with binary relevance sets."""
        relevant_set = {"doc_1", "doc_2"}

        # Perfect ranking: [doc_1, doc_2, irrelevant]
        ndcg_ideal = ndcg_at_k(["doc_1", "doc_2", "doc_irrelevant"], relevant_set, k=3)
        self.assertAlmostEqual(ndcg_ideal, 1.0)

        # Delayed ranking: [irrelevant, doc_1, doc_2]
        ndcg_delayed = ndcg_at_k(["doc_irrelevant", "doc_1", "doc_2"], relevant_set, k=3)
        self.assertLess(ndcg_delayed, ndcg_ideal)
        self.assertGreater(ndcg_delayed, 0.5)

    def test_ndcg_edge_cases(self):
        """Edge cases: completely irrelevant results, empty lists, zero k."""
        scores = {"A": 3.0}
        # Zero relevant retrieved
        self.assertEqual(ndcg_at_k(["X", "Y", "Z"], scores, k=3), 0.0)
        # Empty relevant set
        self.assertEqual(ndcg_at_k(["A", "B"], set(), k=3), 0.0)
        # Empty retrieved list
        self.assertEqual(ndcg_at_k([], scores, k=3), 0.0)
        # k <= 0
        self.assertEqual(ndcg_at_k(["A"], scores, k=0), 0.0)


class TestHitRateAndAveragePrecision(unittest.TestCase):
    """Test suite for HitRate@K and Average Precision."""

    def test_hit_rate(self):
        relevant = {"doc_target"}
        retrieved = ["o1", "o2", "doc_target", "o4"]

        self.assertEqual(hit_rate_at_k(retrieved, relevant, k=2), 0.0)
        self.assertEqual(hit_rate_at_k(retrieved, relevant, k=3), 1.0)
        self.assertEqual(hit_rate_at_k(retrieved, relevant, k=5), 1.0)

    def test_average_precision(self):
        relevant = {"doc_1", "doc_3"}
        retrieved = ["doc_1", "other", "doc_3"]

        # Rank 1: doc_1 -> precision=1/1
        # Rank 2: other -> not relevant
        # Rank 3: doc_3 -> precision=2/3
        # Total relevant = 2
        # AP = (1.0 + 2/3) / 2 = 5/6 ≈ 0.8333
        ap = average_precision_at_k(retrieved, relevant, k=3)
        self.assertAlmostEqual(ap, (1.0 + 2.0 / 3.0) / 2.0, places=5)


class TestRetrievalBenchmarkEvaluator(unittest.TestCase):
    """Integration test suite for the batch RetrievalBenchmarkEvaluator."""

    def test_evaluate_query_retrieval(self):
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        relevant = {"doc_1", "doc_3"}

        res = evaluate_query_retrieval(retrieved, relevant, query_id="q1", k_values=(1, 3, 5))
        self.assertEqual(res.query_id, "q1")
        self.assertAlmostEqual(res.reciprocal_rank, 1.0)
        self.assertAlmostEqual(res.precision[1], 1.0)
        self.assertAlmostEqual(res.precision[3], 2.0 / 3.0)
        self.assertAlmostEqual(res.precision[5], 2.0 / 5.0)
        self.assertAlmostEqual(res.recall[1], 0.5)
        self.assertAlmostEqual(res.recall[3], 1.0)
        self.assertAlmostEqual(res.hit_rate[1], 1.0)
        # doc_1 is at rank 1, doc_3 is at rank 3 (doc_2 at rank 2 is irrelevant).
        # DCG@3 = 1.0 + 0 + 1/log2(4) = 1.5; IDCG@3 = 1.0 + 1/log2(3) ≈ 1.630929
        # NDCG@3 = 1.5 / 1.630929 ≈ 0.91972
        self.assertAlmostEqual(res.ndcg[3], 1.5 / (1.0 + 1.0 / math.log2(3)), places=5)

    def test_batch_benchmark_evaluator(self):
        evaluator = RetrievalBenchmarkEvaluator(k_values=[1, 3, 5])

        # 2 queries
        retrieved_batch = [
            ["doc_A", "doc_B", "doc_C"],  # q1: doc_A is relevant
            ["other_1", "doc_B", "other_2"], # q2: doc_B is relevant at rank 2
        ]
        relevant_batch = [
            {"doc_A"},
            {"doc_B"},
        ]

        summary = evaluator.evaluate_batch(retrieved_batch, relevant_batch)

        self.assertEqual(summary["total_queries"], 2)
        # MRR: q1 is 1.0, q2 is 0.5 -> average = 0.75
        self.assertAlmostEqual(summary["mrr"], 0.75)
        # Precision@1: q1=1.0, q2=0.0 -> average = 0.5
        self.assertAlmostEqual(summary["precision@1"], 0.5)
        # Precision@3: q1=1/3, q2=1/3 -> average = 1/3
        self.assertAlmostEqual(summary["precision@3"], 1.0 / 3.0)
        # Recall@1: q1=1.0, q2=0.0 -> average = 0.5
        self.assertAlmostEqual(summary["recall@1"], 0.5)
        # Recall@3: q1=1.0, q2=1.0 -> average = 1.0
        self.assertAlmostEqual(summary["recall@3"], 1.0)
        # HitRate@1: q1=1.0, q2=0.0 -> average = 0.5
        self.assertAlmostEqual(summary["hit_rate@1"], 0.5)
        # HitRate@3: q1=1.0, q2=1.0 -> average = 1.0
        self.assertAlmostEqual(summary["hit_rate@3"], 1.0)


if __name__ == "__main__":
    unittest.main()
