"""Unit tests for Query Expansion, Query Rewriting, and HyDE module."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_rag.query_expansion import (
    HyDERetriever,
    MultiQueryRetriever,
    QueryExpander,
    create_hyde_retriever,
    create_multi_query_retriever,
    create_query_expander,
    fuse_multiquery_results,
    _clean_query_lines,
)
from langchain_rag.rag_pipeline import RAGPipeline
from langchain_rag.hybrid_search import HybridSearchEngine, BM25Index
from langchain_rag.vector_stores import VectorStore


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class TestQueryExpander(unittest.TestCase):
    def test_clean_query_lines(self):
        raw = "1. First query\n2) Second query\n- Third query\n* Fourth query\n\"Fifth query\""
        cleaned = _clean_query_lines(raw)
        self.assertEqual(
            cleaned,
            ["First query", "Second query", "Third query", "Fourth query", "Fifth query"],
        )

    def test_expand_query_with_mock_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse(
            "1. alternative query one\n2. alternative query two\n3. alternative query three"
        )
        expander = QueryExpander(llm=mock_llm)
        queries = expander.expand_query("original query", num_queries=3)

        self.assertEqual(len(queries), 4)  # original + 3 variations
        self.assertEqual(queries[0], "original query")
        self.assertIn("alternative query one", queries)
        self.assertIn("alternative query two", queries)
        self.assertIn("alternative query three", queries)

    def test_rewrite_query_with_history(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse("GitLab Duo Enterprise pricing and subscription cost")
        expander = QueryExpander(llm=mock_llm)

        history = [
            {"role": "user", "content": "What is GitLab Duo Enterprise?"},
            {"role": "assistant", "content": "GitLab Duo Enterprise is an AI-powered DevSecOps add-on."},
        ]
        rewritten = expander.rewrite_query("How much does it cost?", chat_history=history)

        self.assertEqual(rewritten, "GitLab Duo Enterprise pricing and subscription cost")
        mock_llm.invoke.assert_called_once()

    def test_generate_hyde_doc(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse(
            "GitLab Duo Enterprise is priced at $39 per user per month billed annually."
        )
        expander = QueryExpander(llm=mock_llm)
        hyde_doc = expander.generate_hyde_doc("How much does GitLab Duo Enterprise cost?")

        self.assertIn("$39 per user per month", hyde_doc)

    def test_decompose_query(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse(
            "What is ChromaDB?\nWhat is BM25?\nHow does Reciprocal Rank Fusion work?"
        )
        expander = QueryExpander(llm=mock_llm)
        sub_queries = expander.decompose_query("Compare ChromaDB and BM25 using RRF")

        self.assertEqual(len(sub_queries), 3)
        self.assertEqual(sub_queries[0], "What is ChromaDB?")

    def test_step_back_query(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse(
            "What is the architecture of GitLab CI runner execution environments?"
        )
        expander = QueryExpander(llm=mock_llm)
        step_back = expander.step_back_query("How do I configure concurrent jobs in a Kubernetes runner?")

        self.assertEqual(step_back, "What is the architecture of GitLab CI runner execution environments?")

    def test_fallback_when_llm_fails(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("OpenRouter API connection timeout")
        expander = QueryExpander(llm=mock_llm)

        # Should never crash, must return graceful fallbacks
        self.assertEqual(expander.expand_query("test query"), ["test query"])
        self.assertEqual(expander.rewrite_query("test query"), "test query")
        self.assertEqual(expander.generate_hyde_doc("test query"), "test query")
        self.assertEqual(expander.decompose_query("test query"), ["test query"])
        self.assertEqual(expander.step_back_query("test query"), "test query")


class TestRankFusion(unittest.TestCase):
    def test_fuse_multiquery_results(self):
        results_q1 = [
            {"id": "doc1", "text": "Doc 1 content", "metadata": {}},
            {"id": "doc2", "text": "Doc 2 content", "metadata": {}},
        ]
        results_q2 = [
            {"id": "doc2", "text": "Doc 2 content", "metadata": {}},
            {"id": "doc3", "text": "Doc 3 content", "metadata": {}},
        ]

        all_results = [
            ("query 1", results_q1),
            ("query 2", results_q2),
        ]

        fused = fuse_multiquery_results(all_results, rrf_k=60, top_k=3)

        # doc2 appears in both results, so it should rank #1 due to higher RRF score
        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0]["id"], "doc2")
        self.assertEqual(fused[0]["fused_rank"], 1)
        self.assertEqual(fused[0]["query_count"], 2)
        self.assertIn("query 1", fused[0]["matched_queries"])
        self.assertIn("query 2", fused[0]["matched_queries"])


class TestMultiQueryRetriever(unittest.TestCase):
    def test_multi_query_retriever_with_custom_callable(self):
        # Database mock
        corpus = {
            "token revocation": [
                {"id": "c1", "text": "ERR-9021 Token revoked by admin", "metadata": {}},
            ],
            "oauth expired": [
                {"id": "c2", "text": "OAuth session expired after 24 hours", "metadata": {}},
                {"id": "c1", "text": "ERR-9021 Token revoked by admin", "metadata": {}},
            ],
        }

        def mock_search(query: str, k: int):
            for k_term, docs in corpus.items():
                if k_term in query.lower():
                    return docs[:k]
            return []

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse("oauth expired token")

        expander = QueryExpander(llm=mock_llm)
        mqr = MultiQueryRetriever(retriever=mock_search, expander=expander, num_queries=1)

        results = mqr.retrieve("token revocation", k=5)
        self.assertTrue(any(r["id"] == "c1" for r in results))

    def test_as_langchain_retriever(self):
        docs = [Document(page_content="Test document", metadata={"id": "doc_lc"})]
        mock_target = MagicMock(spec=BaseRetriever)
        mock_target.invoke.return_value = docs


        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse("expanded query")

        mqr = create_multi_query_retriever(retriever=mock_target, llm=mock_llm)
        lc_retriever = mqr.as_retriever(k=2)

        results = lc_retriever.invoke("original query")
        self.assertGreaterEqual(len(results), 1)
        self.assertIsInstance(results[0], Document)
        self.assertIn("fused_rank", results[0].metadata)


class TestHyDERetriever(unittest.TestCase):
    def test_hyde_retriever(self):
        mock_vector_store = MagicMock()
        mock_vector_store.query.return_value = [
            {"id": "doc_h1", "text": "Actual document passage", "metadata": {}}
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse("Hypothetical technical passage")

        hyde = create_hyde_retriever(vector_retriever=mock_vector_store, llm=mock_llm)
        results, passage = hyde.retrieve("How does X work?", k=2, return_hypothetical_doc=True)

        self.assertEqual(passage, "Hypothetical technical passage")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["retrieval_mode"], "hyde")
        mock_vector_store.query.assert_called_once_with(
            query_text="Hypothetical technical passage",
            n_results=2,
        )


class TestRAGPipelineIntegration(unittest.TestCase):
    def test_pipeline_expanded_query(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse(
            "1. alternative search term A\n2. alternative search term B"
        )

        pipeline = RAGPipeline(
            collection_name="test_expansion_pipeline",
            persist_directory=None,
            llm=mock_llm,
        )
        pipeline.index_texts([
            "PostgreSQL connection pooling is managed via PgBouncer.",
            "GitLab Duo provides code suggestions and code review in the IDE.",
        ])

        # Test retrieve with multi_query
        results = pipeline.retrieve(
            query="Tell me about PgBouncer",
            k=2,
            query_transform="multi_query",
        )
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(any("PgBouncer" in r["text"] for r in results))

    def test_pipeline_rewritten_query(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse("PgBouncer database connection pooling settings")

        pipeline = RAGPipeline(
            collection_name="test_rewrite_pipeline",
            persist_directory=None,
            llm=mock_llm,
        )
        pipeline.index_texts(["PostgreSQL connection pooling is managed via PgBouncer."])

        results = pipeline.retrieve_rewritten(
            query="Can you tell me how it pools connections?",
            k=1,
            chat_history=[{"role": "user", "content": "What is PgBouncer?"}],
        )
        self.assertEqual(len(results), 1)
        self.assertIn("rewritten_query", results[0])


class TestHybridSearchExpanded(unittest.TestCase):
    def test_hybrid_search_expanded(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MockLLMResponse("gitlab code suggestions ai")
        expander = QueryExpander(llm=mock_llm)

        engine = HybridSearchEngine(collection_name="test_hybrid_expanded")
        engine.add_texts([
            "GitLab Duo Code Suggestions uses generative AI to recommend code in real time.",
            "Database backups run automatically at midnight via CronJob.",
        ])

        results = engine.search_expanded(
            query="How does Code Suggestions work?",
            num_queries=1,
            k=2,
            expander=expander,
        )
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("Code Suggestions", results[0]["text"])


if __name__ == "__main__":
    unittest.main()
