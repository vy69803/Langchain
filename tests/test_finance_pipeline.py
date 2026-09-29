"""Unit tests for FinanceBench Parser and Ingestion Pipeline."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from langchain_rag.finance_parser import (
    FinanceBenchParser,
    compute_file_hash,
    detect_has_table,
    detect_section_header,
    load_financebench_metadata,
    load_financebench_qa,
    sanitize_metadata_for_chroma,
)
from langchain_rag.finance_pipeline import FinanceBenchIngestionPipeline
from langchain_core.documents import Document


class TestFinanceBenchParser(unittest.TestCase):
    def test_detect_section_header(self):
        text1 = "CONSOLIDATED STATEMENTS OF INCOME\nFor the Year Ended December 31, 2018"
        self.assertEqual(detect_section_header(text1), "Income Statement")

        text2 = "CONSOLIDATED BALANCE SHEETS\nAssets and Liabilities"
        self.assertEqual(detect_section_header(text2), "Balance Sheet")

        text3 = "CONSOLIDATED STATEMENTS OF CASH FLOWS\nOperating Activities"
        self.assertEqual(detect_section_header(text3), "Cash Flow Statement")

        text4 = "Item 1A. Risk Factors\nRisks relating to our business"
        self.assertEqual(detect_section_header(text4), "Item 1A. Risk Factors")

        text5 = "Item 7. Management's Discussion and Analysis of Financial Condition"
        self.assertEqual(detect_section_header(text5), "Item 7. MD&A")

        text6 = "General corporate information and trademarks."
        self.assertEqual(detect_section_header(text6), "General")

    def test_detect_has_table(self):
        table_text = """
        Revenue: $5,363 million
        Operating Income: $1,488 million
        Net Income: $1,577 million
        Diluted EPS: $2.50
        """
        self.assertTrue(detect_has_table(table_text))

        prose_text = "This company was founded in Delaware in 1902 and produces materials."
        self.assertFalse(detect_has_table(prose_text))

    def test_sanitize_metadata_for_chroma(self):
        meta = {
            "str_key": "val",
            "int_key": 42,
            "float_key": 3.14,
            "bool_key": True,
            "none_key": None,
            "list_key": ["a", "b"],
            "dict_key": {"sub": "obj"},
        }
        sanitized = sanitize_metadata_for_chroma(meta)
        self.assertEqual(sanitized["str_key"], "val")
        self.assertEqual(sanitized["int_key"], 42)
        self.assertEqual(sanitized["float_key"], 3.14)
        self.assertEqual(sanitized["bool_key"], True)
        self.assertEqual(sanitized["none_key"], "")
        self.assertIsInstance(sanitized["list_key"], str)
        self.assertIsInstance(sanitized["dict_key"], str)

    def test_chunk_document_preserves_breadcrumbs(self):
        parser = FinanceBenchParser()
        content = "[Company: 3M | Filing: 3M_2018_10K | Page: 59/160 | Section: Cash Flow]\n\n" + (
            "Paragraph one with some text.\n\n" * 15
        )
        doc = Document(page_content=content, metadata={"company": "3M", "page": 59})
        chunks = parser.chunk_document(doc, max_chunk_chars=300)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk.page_content.startswith("[Company: 3M"))
            self.assertEqual(chunk.metadata["company"], "3M")
            self.assertEqual(chunk.metadata["page"], 59)


class TestFinanceBenchPipeline(unittest.TestCase):
    def test_pipeline_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = FinanceBenchIngestionPipeline(
                pdf_dir=tmpdir,
                persist_directory=os.path.join(tmpdir, "chroma"),
                manifest_path=os.path.join(tmpdir, "manifest.json"),
                bm25_path=os.path.join(tmpdir, "bm25.json"),
            )
            self.assertEqual(pipeline.collection_name, "financebench")
            self.assertTrue(pipeline.enable_bm25)

    def test_pipeline_two_stage_rerank(self):
        pipeline = FinanceBenchIngestionPipeline.__new__(FinanceBenchIngestionPipeline)
        mock_vs = MagicMock()
        mock_vs.query.return_value = [
            {"id": "doc_noise", "text": "Noise content about consumer goods", "metadata": {"company": "Target"}},
            {
                "id": "doc_target",
                "text": "3M FY2018 capital expenditure (purchases of PP&E) was $1,577 million.",
                "metadata": {"company": "3M", "page": 59, "doc_name": "3M_2018_10K"},
            },
        ]
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = []

        pipeline._vector_store = mock_vs
        pipeline._bm25_index = mock_bm25
        from langchain_rag.reranker import get_reranker
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
        self.assertIn("rerank_score", results[0])


if __name__ == "__main__":
    unittest.main()
