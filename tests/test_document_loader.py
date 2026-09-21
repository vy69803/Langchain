"""Unit tests for DocumentLoader and Docling integration."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from langchain_rag.document_loader import (
    DOCLING_SUPPORTED_EXTENSIONS,
    DocumentLoader,
    is_docling_available,
    load_directory,
    load_document,
    load_text,
    load_with_docling,
)


class TestDocumentLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_text_file(self):
        file_path = self.dir_path / "sample.txt"
        file_path.write_text("Hello, world! This is a test.", encoding="utf-8")

        docs = load_document(file_path)
        self.assertEqual(len(docs), 1)
        self.assertIn("Hello, world!", docs[0].page_content)
        self.assertEqual(docs[0].metadata["file_type"], ".txt")
        self.assertEqual(docs[0].metadata["filename"], "sample.txt")

    def test_load_csv_file(self):
        file_path = self.dir_path / "data.csv"
        file_path.write_text("name,role\nAlice,Engineer\nBob,Designer", encoding="utf-8")

        docs = load_document(file_path)
        self.assertEqual(len(docs), 1)
        self.assertIn("Alice", docs[0].page_content)
        self.assertEqual(docs[0].metadata["row_count"], 2)

    def test_load_json_file(self):
        file_path = self.dir_path / "data.json"
        file_path.write_text('{"project": "rag", "version": 1}', encoding="utf-8")

        docs = load_document(file_path)
        self.assertEqual(len(docs), 1)
        self.assertIn('"project": "rag"', docs[0].page_content)

    def test_load_text_helper(self):
        doc = load_text("Raw text content", {"source": "memory", "custom": 123})
        self.assertEqual(doc.page_content, "Raw text content")
        self.assertEqual(doc.metadata["custom"], 123)

    def test_docling_supported_extensions(self):
        self.assertIn(".pdf", DOCLING_SUPPORTED_EXTENSIONS)
        self.assertIn(".docx", DOCLING_SUPPORTED_EXTENSIONS)
        self.assertIn(".pptx", DOCLING_SUPPORTED_EXTENSIONS)
        self.assertIn(".xlsx", DOCLING_SUPPORTED_EXTENSIONS)
        self.assertIn(".html", DOCLING_SUPPORTED_EXTENSIONS)

    def test_is_docling_available_type(self):
        self.assertIsInstance(is_docling_available(), bool)

    def test_docling_not_installed_raises_importerror(self):
        file_path = self.dir_path / "report.pdf"
        file_path.write_bytes(b"%PDF-1.4 mock pdf content")

        # Simulate docling being unavailable
        with patch("langchain_rag.document_loader.is_docling_available", return_value=False):
            loader = DocumentLoader(file_path, use_docling=True)
            with self.assertRaises(ImportError) as ctx:
                loader.load()
            self.assertIn("Docling is required", str(ctx.exception))

    def test_docling_mocked_conversion(self):
        file_path = self.dir_path / "table_doc.pdf"
        file_path.write_bytes(b"%PDF-1.4 mock pdf with table")

        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "# Header\n\n| Item | Price |\n| --- | --- |\n| Apple | $1.00 |\n"
        mock_doc.pages = [MagicMock(), MagicMock()]
        mock_doc.tables = [MagicMock()]
        mock_doc.pictures = []

        mock_conv_res = MagicMock()
        mock_conv_res.document = mock_doc

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_conv_res

        mock_docling_module = MagicMock()
        mock_docling_module.DocumentConverter.return_value = mock_converter

        with patch("langchain_rag.document_loader.is_docling_available", return_value=True):
            with patch.dict("sys.modules", {"docling.document_converter": mock_docling_module}):
                docs = load_with_docling(file_path)

                self.assertEqual(len(docs), 1)
                self.assertIn("| Apple | $1.00 |", docs[0].page_content)
                self.assertEqual(docs[0].metadata["parser"], "docling")
                self.assertEqual(docs[0].metadata["total_pages"], 2)
                self.assertEqual(docs[0].metadata["table_count"], 1)

    def test_load_directory_filtering(self):
        (self.dir_path / "a.txt").write_text("file a", encoding="utf-8")
        (self.dir_path / "b.py").write_text("# python", encoding="utf-8")
        (self.dir_path / "c.md").write_text("# markdown", encoding="utf-8")

        txt_docs = load_directory(self.dir_path, extensions=[".txt"])
        self.assertEqual(len(txt_docs), 1)
        self.assertEqual(txt_docs[0].metadata["filename"], "a.txt")


if __name__ == "__main__":
    unittest.main()
