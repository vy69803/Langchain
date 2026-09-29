"""Unit tests for RAGAS evaluation with Context Recall."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from ragas import SingleTurnSample
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_recall import LLMContextRecall

from scripts.ragas_evaluate import (
    _extract_scores,
    run_evaluation,
    run_rag_and_collect,
    SAMPLE_EVAL_DATASET,
)


class TestRagasEvaluateContextRecall(unittest.TestCase):
    """Test suite validating Context Recall integration in RAGAS evaluation."""

    def test_llm_context_recall_metric_attributes(self):
        """Verify LLMContextRecall has expected name and required columns."""
        metric = LLMContextRecall()
        self.assertEqual(metric.name, "context_recall")
        # Check required columns for single-turn samples
        from ragas.metrics.base import MetricType
        required = metric._required_columns[MetricType.SINGLE_TURN]
        self.assertIn("user_input", required)
        self.assertIn("retrieved_contexts", required)
        self.assertIn("reference", required)

    def test_run_rag_and_collect_populates_reference(self):
        """Verify run_rag_and_collect builds SingleTurnSample with reference."""
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "answer": "GitLab values CREDIT.",
            "raw_results": [{"text": "GitLab core values: Collaboration, Results, Efficiency..."}],
        }

        test_data = [
            {"question": "Q1", "ground_truth": "GT 1"},
            {"question": "Q2", "expected_answer": "EA 2"},
            {"question": "Q3", "reference": "REF 3"},
        ]

        samples = run_rag_and_collect(mock_pipeline, test_data, top_k=1, langfuse=None)
        self.assertEqual(len(samples), 3)

        self.assertEqual(samples[0].user_input, "Q1")
        self.assertEqual(samples[0].reference, "GT 1")

        self.assertEqual(samples[1].user_input, "Q2")
        self.assertEqual(samples[1].reference, "EA 2")

        self.assertEqual(samples[2].user_input, "Q3")
        self.assertEqual(samples[2].reference, "REF 3")

    @patch("scripts.ragas_evaluate._build_evaluator_llm")
    @patch("scripts.ragas_evaluate._build_evaluator_embeddings")
    @patch("scripts.ragas_evaluate.evaluate")
    def test_run_evaluation_includes_context_recall_when_reference_present(
        self, mock_evaluate, mock_embeddings, mock_llm
    ):
        """Verify LLMContextRecall is included when all samples have reference."""
        mock_llm.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()
        mock_evaluate.return_value = MagicMock()

        samples = [
            SingleTurnSample(
                user_input="What is CREDIT?",
                response="It stands for GitLab values.",
                retrieved_contexts=["GitLab CREDIT values..."],
                reference="Collaboration, Results, Efficiency...",
            )
        ]

        run_evaluation(samples, include_context_recall=True)

        call_kwargs = mock_evaluate.call_args[1]
        metrics = call_kwargs["metrics"]
        metric_names = [m.name for m in metrics]

        self.assertIn("faithfulness", metric_names)
        self.assertIn("answer_relevancy", metric_names)
        self.assertIn("context_recall", metric_names)

    @patch("scripts.ragas_evaluate._build_evaluator_llm")
    @patch("scripts.ragas_evaluate._build_evaluator_embeddings")
    @patch("scripts.ragas_evaluate.evaluate")
    def test_run_evaluation_skips_context_recall_when_flag_false(
        self, mock_evaluate, mock_embeddings, mock_llm
    ):
        """Verify LLMContextRecall is skipped when include_context_recall=False."""
        mock_llm.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()
        mock_evaluate.return_value = MagicMock()

        samples = [
            SingleTurnSample(
                user_input="What is CREDIT?",
                response="It stands for GitLab values.",
                retrieved_contexts=["GitLab CREDIT values..."],
                reference="Collaboration, Results, Efficiency...",
            )
        ]

        run_evaluation(samples, include_context_recall=False)

        call_kwargs = mock_evaluate.call_args[1]
        metrics = call_kwargs["metrics"]
        metric_names = [m.name for m in metrics]

        self.assertIn("faithfulness", metric_names)
        self.assertIn("answer_relevancy", metric_names)
        self.assertNotIn("context_recall", metric_names)

    @patch("scripts.ragas_evaluate._build_evaluator_llm")
    @patch("scripts.ragas_evaluate._build_evaluator_embeddings")
    @patch("scripts.ragas_evaluate.evaluate")
    def test_run_evaluation_skips_context_recall_when_sample_lacks_reference(
        self, mock_evaluate, mock_embeddings, mock_llm
    ):
        """Verify LLMContextRecall is skipped if a sample lacks reference."""
        mock_llm.return_value = MagicMock()
        mock_embeddings.return_value = MagicMock()
        mock_evaluate.return_value = MagicMock()

        samples = [
            SingleTurnSample(
                user_input="What is CREDIT?",
                response="It stands for GitLab values.",
                retrieved_contexts=["GitLab CREDIT values..."],
                reference=None,
            )
        ]

        run_evaluation(samples, include_context_recall=True)

        call_kwargs = mock_evaluate.call_args[1]
        metrics = call_kwargs["metrics"]
        metric_names = [m.name for m in metrics]

        self.assertNotIn("context_recall", metric_names)

    def test_extract_scores_with_context_recall(self):
        """Verify _extract_scores correctly parses context_recall."""
        mock_result = MagicMock()
        mock_result.scores = [
            {"faithfulness": 0.9, "answer_relevancy": 0.85, "context_recall": 1.0},
            {"faithfulness": 0.8, "answer_relevancy": 0.75, "context_recall": 0.6},
        ]

        scores, per_sample = _extract_scores(mock_result)
        self.assertAlmostEqual(scores["faithfulness"], 0.85)
        self.assertAlmostEqual(scores["answer_relevancy"], 0.80)
        self.assertAlmostEqual(scores["context_recall"], 0.80)
        self.assertEqual(len(per_sample), 2)


if __name__ == "__main__":
    unittest.main()
