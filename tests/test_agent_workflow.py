"""End-to-End Workflow Tests for FinancialIntelligenceAgent and LangGraph State Machine."""

import pytest
import asyncio
from typing import Any, Dict, List

from langchain_rag.adapters.decision_adapter import PydanticLLMAdapter, VerificationResult
from langchain_rag.adapters.storage_adapters import BaseGraphStoreAdapter
from langchain_rag.agent.decomposer import QueryDecomposer, rule_based_query_decomposition
from langchain_rag.agent.financial_math import FinancialCalculator
from langchain_rag.agent.graph_agent import FinancialIntelligenceAgent


class MockFinancialGraphStore(BaseGraphStoreAdapter):
    """Mock Graph Store providing deterministic financial metrics for test filings."""

    def health_check(self) -> bool:
        return True

    def execute_cypher(
        self,
        query: str,
        params: Dict[str, Any] | None = None,
        read_only: bool = True,
    ) -> List[Dict[str, Any]]:
        # Return mock 3-year financial history for AAPL & MSFT
        return [
            {
                "ticker": "AAPL",
                "year": 2021,
                "statement": "Income Statement",
                "metric": "REVENUE",
                "raw_name": "Total net sales",
                "value": 365817.0,
                "unit": "USD",
                "footnotes": ["Includes Products and Services"],
            },
            {
                "ticker": "AAPL",
                "year": 2021,
                "statement": "Income Statement",
                "metric": "RD_EXPENSE",
                "raw_name": "Research and development",
                "value": 21914.0,
                "unit": "USD",
                "footnotes": [],
            },
            {
                "ticker": "AAPL",
                "year": 2022,
                "statement": "Income Statement",
                "metric": "REVENUE",
                "raw_name": "Total net sales",
                "value": 394328.0,
                "unit": "USD",
                "footnotes": [],
            },
            {
                "ticker": "AAPL",
                "year": 2022,
                "statement": "Income Statement",
                "metric": "RD_EXPENSE",
                "raw_name": "Research and development",
                "value": 26251.0,
                "unit": "USD",
                "footnotes": [],
            },
            {
                "ticker": "AAPL",
                "year": 2023,
                "statement": "Income Statement",
                "metric": "REVENUE",
                "raw_name": "Total net sales",
                "value": 383285.0,
                "unit": "USD",
                "footnotes": [],
            },
            {
                "ticker": "AAPL",
                "year": 2023,
                "statement": "Income Statement",
                "metric": "RD_EXPENSE",
                "raw_name": "Research and development",
                "value": 29915.0,
                "unit": "USD",
                "footnotes": ["Software capitalization amortization included"],
            },
        ]


def test_query_decomposer_rule_based():
    """Verify rule-based decomposition on the complex prompt."""
    query = "Compare the R&D spending efficiency of Company A and Company B over the last 3 years"
    plan = rule_based_query_decomposition(query)

    assert plan.query_type == "cross_company_comparison"
    assert "COMP_A" in plan.target_companies
    assert "COMP_B" in plan.target_companies
    assert 2021 in plan.fiscal_years and 2023 in plan.fiscal_years
    assert "RD_EXPENSE" in plan.required_metrics
    assert "REVENUE" in plan.required_metrics
    assert plan.requires_ratio_math is True


def test_financial_calculator():
    """Verify financial math accurately calculates intensities, margins, and growth."""
    metrics_by_year = {
        2022: {"REVENUE": 394328.0, "RD_EXPENSE": 26251.0, "GROSS_PROFIT": 170782.0},
        2023: {"REVENUE": 383285.0, "RD_EXPENSE": 29915.0, "GROSS_PROFIT": 169148.0},
    }

    ratios = FinancialCalculator.compute_all_ratios(metrics_by_year)

    # Check 2023 R&D Intensity = 29915 / 383285 = 7.80%
    assert ratios["rd_intensity_by_year"][2023] == 7.8
    # Check 2022 R&D Intensity = 26251 / 394328 = 6.66%
    assert ratios["rd_intensity_by_year"][2022] == 6.66

    # Check YoY Revenue Growth (2022 to 2023): (383285 - 394328) / 394328 = -2.8%
    growth = ratios["yoy_growth"]["2022_to_2023"]
    assert growth["REVENUE_growth_pct"] == -2.8
    # Check YoY R&D Growth: (29915 - 26251) / 26251 = +13.96%
    assert growth["RD_EXPENSE_growth_pct"] == 13.96


@pytest.mark.asyncio
async def test_full_agent_execution_end_to_end():
    """Test full LangGraph agent execution on golden cross-document analytical query."""
    mock_graph = MockFinancialGraphStore()
    decision_adapter = PydanticLLMAdapter(mock=True)

    agent = FinancialIntelligenceAgent(
        decision_adapter=decision_adapter,
        graph_adapter=mock_graph,
        vector_adapter=None,
    )

    query = "Compare Apple R&D spending efficiency from 2021 to 2023"
    result = await agent.ainvoke(query)

    # Validate output
    assert result is not None
    assert "final_output" in result and result["final_output"] is not None
    output_text = result["final_output"]

    # Assert Markdown table is present
    assert "| Company | Fiscal Year | Metric | Value (USD Millions) | Disclosures / Footnotes |" in output_text
    assert "AAPL" in output_text
    assert "$29,915.0" in output_text or "29,915" in output_text

    # Assert Computed Ratios are present
    assert "Calculated Efficiency & Performance Ratios" in output_text
    assert "R&D Intensity" in output_text

    # Assert Audit Trail
    assert "Audit Trail & Citations" in output_text
    assert result["iteration_count"] <= 2
    assert result["is_verified"] is True
