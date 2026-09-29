"""Unit tests for the FlashRank and Cross-Encoder reranking module."""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from langchain_core.documents import Document
from langchain_rag.reranker import BaseReranker, FlashRankReranker, get_reranker
from langchain_rag.rag_pipeline import RAGPipeline
from langchain_rag.hybrid_search import HybridSearchEngine


class TestReranker(unittest.TestCase):
    def setUp(self):
        self.reranker = get_reranker(reranker_type="flashrank")

    def test_imports_and_types(self):
        self.assertIsInstance(self.reranker, BaseReranker)
        self.assertIsInstance(self.reranker, FlashRankReranker)

    def test_reranker_elevates_relevant_chunk_from_rank_6_to_1(self):
        """Verify that a highly relevant chunk at rank 6 out of 10 is pulled to rank 1."""
        query = "What CI/CD and deployment features does GitLab support?"

        # 10 chunks where chunk #6 (index 5) is the true answer
        chunks = [
            {"id": "doc_1", "text": "Tomatoes grow best in full sunlight with well-drained soil.", "metadata": {"source": "gardening"}},
            {"id": "doc_2", "text": "The solar system has eight planets orbiting the Sun.", "metadata": {"source": "astronomy"}},
            {"id": "doc_3", "text": "Baking bread requires flour, water, yeast, and salt.", "metadata": {"source": "cooking"}},
            {"id": "doc_4", "text": "Electric cars utilize lithium-ion battery packs.", "metadata": {"source": "automotive"}},
            {"id": "doc_5", "text": "The French Revolution began in 1789 with the storming of the Bastille.", "metadata": {"source": "history"}},
            {
                "id": "target_doc_6",
                "text": "GitLab CI/CD automates software builds, testing pipelines, Auto DevOps, and container deployments to Kubernetes.",
                "metadata": {"source": "gitlab_docs"},
            },
            {"id": "doc_7", "text": "Photosynthesis converts light energy into chemical energy in plants.", "metadata": {"source": "biology"}},
            {"id": "doc_8", "text": "Mount Everest is the highest mountain peak above sea level.", "metadata": {"source": "geography"}},
            {"id": "doc_9", "text": "Quantum computers use qubits that can exist in multiple states simultaneously.", "metadata": {"source": "physics"}},
            {"id": "doc_10", "text": "Shakespeare wrote numerous plays including Hamlet and Macbeth.", "metadata": {"source": "literature"}},
        ]

        reranked = self.reranker.rerank(query=query, documents=chunks, top_k=3)

        self.assertEqual(len(reranked), 3)
        # Verify chunk #6 was pulled to rank #1
        self.assertEqual(reranked[0]["id"], "target_doc_6")
        self.assertEqual(reranked[0]["rerank_rank"], 1)
        self.assertGreater(reranked[0]["score"], 0.8)
        self.assertIn("rerank_score", reranked[0])
        # Verify metadata was preserved
        self.assertEqual(reranked[0]["metadata"]["source"], "gitlab_docs")

    def test_reranker_handles_documents_objects(self):
        """Verify that LangChain Document objects are supported."""
        query = "Who developed Python?"
        docs = [
            Document(page_content="JavaScript was created by Brendan Eich in 1995.", metadata={"id": "d1"}),
            Document(page_content="Python was created by Guido van Rossum and released in 1991.", metadata={"id": "d2"}),
        ]
        reranked = self.reranker.rerank(query=query, documents=docs, top_k=1)
        self.assertEqual(len(reranked), 1)
        self.assertEqual(reranked[0]["id"], "d2")
        self.assertIn("Guido van Rossum", reranked[0]["text"])

    def test_reranker_edge_cases(self):
        # Empty documents list
        empty_res = self.reranker.rerank("any query", [])
        self.assertEqual(empty_res, [])

        # Empty query string returns original docs
        sample_doc = [{"id": "d1", "text": "Hello world"}]
        fallback = self.reranker.rerank("", sample_doc)
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0]["id"], "d1")


class TestRAGPipelineRerankIntegration(unittest.TestCase):
    def test_rag_pipeline_retrieve_reranked(self):
        # Mock vector store returning 5 chunks
        mock_vs = MagicMock()
        mock_vs.query.return_value = [
            {"id": "chunk_1", "text": "Irrelevant text about gardening", "metadata": {}, "distance": 0.2},
            {"id": "chunk_2", "text": "Irrelevant text about cooking", "metadata": {}, "distance": 0.3},
            {"id": "chunk_3", "text": "ChromaDB stores high-dimensional embeddings for fast nearest neighbor search.", "metadata": {"source": "chroma"}, "distance": 0.4},
        ]

        pipeline = RAGPipeline(vector_store=mock_vs, use_reranker=True)
        results = pipeline.retrieve_reranked(query="What is ChromaDB used for?", candidate_k=3, top_k=1)

        # ChromaDB chunk should be pulled to rank 1
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "chunk_3")
        self.assertIn("rerank_score", results[0])

    def test_hybrid_search_with_rerank(self):
        dense_mock = MagicMock()
        dense_mock.query.return_value = [
            {"id": "h1", "text": "Irrelevant astronomy note", "metadata": {}, "distance": 0.1},
            {"id": "h2", "text": "PostgreSQL is a powerful relational database.", "metadata": {}, "distance": 0.2},
        ]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = []

        engine = HybridSearchEngine(vector_store=dense_mock, bm25_index=bm25_mock)
        results = engine.search("PostgreSQL relational database", k=1, rerank=True)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "h2")
        self.assertIn("rerank_score", results[0])

    def test_finance_pipeline_two_stage_rerank(self):
        from langchain_rag.finance_pipeline import FinanceBenchIngestionPipeline

        pipeline = FinanceBenchIngestionPipeline.__new__(FinanceBenchIngestionPipeline)
        mock_vs = MagicMock()
        mock_vs.query.return_value = [
            {"id": "doc_noise", "text": "Noise content about apples", "metadata": {}},
            {"id": "doc_target", "text": "3M capital expenditure in FY2018 was $1,577 million.", "metadata": {"company": "3M", "source": "3M_2018_10K.pdf"}},
        ]
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = []

        pipeline._vector_store = mock_vs
        pipeline._bm25_index = mock_bm25
        pipeline._reranker = get_reranker(reranker_type="flashrank")
        pipeline.reranker_model = "ms-marco-TinyBERT-L-2-v2"
        pipeline.enable_bm25 = True

        results = pipeline.query(
            "What is 3M capital expenditure in 2018?",
            top_k=1,
            search_type="hybrid",
            rerank=True,
            candidate_k=5,
            score_threshold=0.01,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "doc_target")
        self.assertEqual(results[0]["rerank_rank"], 1)
        self.assertIn("rerank_score", results[0])


if __name__ == "__main__":
    unittest.main()
