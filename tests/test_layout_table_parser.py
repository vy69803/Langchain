"""Unit and integration tests for LayoutTableParser, footnote attachment, and Graph Ingestion."""

import json
from pathlib import Path
import pytest

from langchain_rag.parsers.layout_table_parser import (
    LayoutTableParser,
    parse_numeric_value,
)
from langchain_rag.graph.ingest_graph import FinancialGraphIngestor
from langchain_rag.adapters.storage_adapters import BaseGraphStoreAdapter, Neo4jGraphAdapter


SAMPLE_INCOME_STATEMENT_TEXT = """
Apple Inc.
Consolidated Statements of Operations
(In millions, except number of shares which are reflected in thousands and per share amounts)

Years ended September 30, 2023
Total net sales (1)                             $ 383,285
Cost of sales                                     214,137
Gross margin                                      169,148

Operating expenses:
Research and development (2)                       29,915
Selling, general and administrative                24,932
Total operating expenses                           54,847

Operating income                                  114,301
Total other income/(expense), net                    (382)
Income before provision for income taxes          113,919
Provision for income taxes                         16,741
Net income                                      $  96,995

(1) Includes net sales from Products and Services.
(2) Research and development includes software capitalization amortization of $1,250 million.
"""


def test_parse_numeric_value():
    """Verify numeric value parsing for various financial accounting formats."""
    assert parse_numeric_value("$ 383,285") == 383285.0
    assert parse_numeric_value("(382)") == -382.0
    assert parse_numeric_value("$  96,995") == 96995.0
    assert parse_numeric_value("-") == 0.0
    assert parse_numeric_value("—") == 0.0
    assert parse_numeric_value("Invalid Text") is None


def test_table_parsing_and_footnote_resolution(tmp_path: Path):
    """Verify table parsing extracts line items and resolves footnote disclosures."""
    dlq_file = tmp_path / "test_dlq.jsonl"
    parser = LayoutTableParser(dlq_path=dlq_file)

    table = parser.parse_table_from_text(
        page_text=SAMPLE_INCOME_STATEMENT_TEXT,
        page_num=48,
        statement_name="Income Statement",
        file_path="AAPL_2023_10K.pdf",
    )

    assert table is not None
    assert len(table.line_items) >= 7

    # Check Footnote definitions were parsed
    assert "1" in table.footnote_definitions or "(1)" in str(table.footnote_definitions)
    assert len(table.footnote_definitions) >= 2

    # Check specific line items
    items_by_name = {item.raw_name: item for item in table.line_items}

    # Verify Net Sales mapped to REVENUE
    sales_item = next((item for item in table.line_items if "sales" in item.raw_name.lower()), None)
    assert sales_item is not None
    assert sales_item.canonical_metric == "REVENUE"
    assert sales_item.value == 383285.0
    assert len(sales_item.footnote_refs) > 0
    assert any("Products and Services" in fn for fn in sales_item.footnote_texts)

    # Verify R&D mapped to RD_EXPENSE
    rd_item = next((item for item in table.line_items if "research" in item.raw_name.lower()), None)
    assert rd_item is not None
    assert rd_item.canonical_metric == "RD_EXPENSE"
    assert rd_item.value == 29915.0
    assert any("software capitalization" in fn for fn in rd_item.footnote_texts)


def test_parent_child_chunking(tmp_path: Path):
    """Verify parent-child chunk hierarchy is constructed with linked metadata."""
    parser = LayoutTableParser(dlq_path=tmp_path / "dlq.jsonl")

    table = parser.parse_table_from_text(
        page_text=SAMPLE_INCOME_STATEMENT_TEXT,
        page_num=48,
        statement_name="Income Statement",
    )
    assert table is not None

    meta = {
        "doc_name": "AAPL_2023_10K",
        "company": "Apple",
        "doc_period": "2023",
        "doc_type": "10k",
    }

    parent_doc, child_docs = parser.create_parent_child_chunks(table, meta)

    # Validate Parent Chunk
    assert parent_doc.metadata["chunk_type"] == "parent_financial_statement"
    assert "AAPL_2023_10K_p48_Income_Statement" in parent_doc.metadata["chunk_id"]
    assert "### Income Statement (Page 48)" in parent_doc.page_content

    # Validate Child Chunks
    assert len(child_docs) == len(table.line_items)
    for child in child_docs:
        assert child.metadata["chunk_type"] == "child_line_item"
        assert child.metadata["parent_id"] == parent_doc.metadata["chunk_id"]
        assert "neo4j_lineitem_id" in child.metadata


class MockGraphStoreAdapter(BaseGraphStoreAdapter):
    def __init__(self):
        self.executed_queries = []

    def health_check(self) -> bool:
        return False

    def execute_cypher(self, query: str, params=None, read_only=True):
        self.executed_queries.append((query, params))
        return []


def test_graph_ingestion_offline_mode():
    """Verify FinancialGraphIngestor executes without crashing when Neo4j is offline."""
    parser = LayoutTableParser()
    table = parser.parse_table_from_text(
        page_text=SAMPLE_INCOME_STATEMENT_TEXT,
        page_num=48,
        statement_name="Income Statement",
    )
    assert table is not None

    meta = {
        "doc_name": "AAPL_2023_10K",
        "company": "AAPL",
        "doc_period": "2023",
        "doc_type": "10k",
    }

    # Offline adapter: health_check is False -> fast bypass, no queries executed
    mock_adapter = MockGraphStoreAdapter()
    ingestor = FinancialGraphIngestor(graph_adapter=mock_adapter)
    committed = ingestor.ingest_table(table, meta)
    assert committed == len(table.line_items)
    assert len(mock_adapter.executed_queries) == 0

    # Online adapter: health_check is True -> line items are executed
    class MockOnlineGraphStoreAdapter(BaseGraphStoreAdapter):
        def __init__(self):
            self.executed_queries = []

        def health_check(self) -> bool:
            return True

        def execute_cypher(self, query: str, params=None, read_only=True):
            self.executed_queries.append((query, params))
            return []

    online_adapter = MockOnlineGraphStoreAdapter()
    online_ingestor = FinancialGraphIngestor(graph_adapter=online_adapter)
    online_committed = online_ingestor.ingest_table(table, meta)
    assert online_committed == len(table.line_items)
    # Includes constraints DDL + line item queries
    assert len(online_adapter.executed_queries) >= len(table.line_items)
