"""Unit tests for Graph schema, DecisionAdapter, and storage adapters."""

import pytest
from pydantic import BaseModel, Field

from langchain_rag.adapters.decision_adapter import (
    DecisionAdapter,
    PydanticLLMAdapter,
    VerificationResult,
)
from langchain_rag.adapters.storage_adapters import (
    BaseGraphStoreAdapter,
    BaseVectorStoreAdapter,
    Neo4jGraphAdapter,
)
from langchain_rag.graph.schema import (
    CANONICAL_METRICS,
    normalize_line_item_name,
    resolve_canonical_metric,
)


class SampleQueryPlan(BaseModel):
    ticker: str = Field(default="AAPL")
    metric: str = Field(default="REVENUE")
    fiscal_year: int = Field(default=2023)


def test_canonical_metric_resolution():
    """Verify raw 10-K variations resolve to canonical XBRL-mapped metrics."""
    # Test Revenue aliases
    assert resolve_canonical_metric("Net Sales") is not None
    assert resolve_canonical_metric("Net Sales").canonical_name == "REVENUE"
    assert resolve_canonical_metric("Total net sales (1)") is not None
    assert resolve_canonical_metric("Total net sales (1)").canonical_name == "REVENUE"

    # Test R&D aliases
    rd = resolve_canonical_metric("Research, development and engineering")
    assert rd is not None
    assert rd.canonical_name == "RD_EXPENSE"

    # Test CapEx aliases
    capex = resolve_canonical_metric("Payments for property, plant and equipment")
    assert capex is not None
    assert capex.canonical_name == "CAPEX"

    # Test unmapped metric
    assert resolve_canonical_metric("Random executive flight allowance") is None


def test_cypher_read_only_injection_guard():
    """Verify Neo4jGraphAdapter blocks write/delete operations in read-only mode."""
    adapter = Neo4jGraphAdapter(uri="bolt://localhost:7687", user="mock", password="mock")

    # Forbidden Cypher statements
    forbidden_queries = [
        "MATCH (n) DETACH DELETE n",
        "MATCH (c:Company) DROP CONSTRAINT c",
        "CREATE (c:Company {ticker: 'BAD'})",
        "MATCH (c:Company) SET c.name = 'HACKED'",
        "MERGE (c:Company {ticker: 'INJECT'})",
    ]

    for query in forbidden_queries:
        with pytest.raises(PermissionError):
            adapter.execute_cypher(query, read_only=True)


def test_decision_adapter_offline_classification():
    """Verify PydanticLLMAdapter constructs fallback schema when offline."""
    adapter = PydanticLLMAdapter(mock=True)
    import asyncio

    result = asyncio.run(adapter.classify("Compare Apple revenue", SampleQueryPlan))
    assert isinstance(result, SampleQueryPlan)
    assert result.ticker == "AAPL"


def test_decision_adapter_offline_verification():
    """Verify offline fallback for factual verification."""
    adapter = PydanticLLMAdapter(mock=True)
    import asyncio

    res = asyncio.run(
        adapter.verify(
            claim="Apple 2023 R&D was $29.9B",
            evidence=[{"table": "Operations", "R&D": 29915000000}],
        )
    )
    assert isinstance(res, VerificationResult)
    assert res.is_faithful is True
