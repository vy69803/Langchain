"""Query Decomposition and Intent Analysis for the Financial Intelligence Agent."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from langchain_rag.adapters.decision_adapter import DecisionAdapter, PydanticLLMAdapter
from langchain_rag.graph.schema import CANONICAL_METRICS, resolve_canonical_metric

logger = logging.getLogger("langchain_rag.agent.decomposer")


class FinancialQueryPlan(BaseModel):
    """Structured decomposition plan for an analytical financial question."""

    query_type: Literal[
        "point_metric",
        "longitudinal_trend",
        "cross_company_comparison",
        "qualitative_risk",
    ] = Field(
        default="point_metric",
        description="Categorical type of the financial query.",
    )
    target_companies: List[str] = Field(
        default_factory=list,
        description="List of company tickers or names identified in the prompt (e.g. ['AAPL', 'MSFT']).",
    )
    fiscal_years: List[int] = Field(
        default_factory=list,
        description="Target fiscal years (e.g. [2021, 2022, 2023]).",
    )
    required_metrics: List[str] = Field(
        default_factory=list,
        description="Canonical financial metrics needed (e.g. ['REVENUE', 'RD_EXPENSE']).",
    )
    requires_cypher: bool = Field(
        default=True,
        description="Whether structured Knowledge Graph Cypher retrieval is needed for exact figures.",
    )
    requires_vector: bool = Field(
        default=True,
        description="Whether unstructured vector search is needed for MD&A, risk factors, or footnotes.",
    )
    requires_ratio_math: bool = Field(
        default=False,
        description="Whether derived mathematical ratios (e.g. efficiency, intensity, margins) are required.",
    )
    sub_questions: List[str] = Field(
        default_factory=list,
        description="Decomposed sub-queries to execute against specialized retrievers.",
    )


def rule_based_query_decomposition(query: str) -> FinancialQueryPlan:
    """Deterministic fallback decomposer extracting companies, years, and metrics via regex."""
    query_lower = query.lower()

    # 1. Detect Years (e.g. 2018, FY2018, 2022, 2023)
    years = [int(y) for y in re.findall(r"(?:fy)?(20\d{2}|19\d{2})\b", query, re.IGNORECASE)]
    years = sorted(list(set(years)))

    # If phrases like "last 3 years" or "over 3 years" are present
    if not years and ("3 years" in query_lower or "three years" in query_lower):
        years = [2021, 2022, 2023]

    # 2. Detect Common Company Names / Tickers
    companies: List[str] = []
    company_aliases = {
        "apple": "AAPL",
        "aapl": "AAPL",
        "microsoft": "MSFT",
        "msft": "MSFT",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "3m": "MMM",
        "amazon": "AMZN",
        "meta": "META",
        "nvidia": "NVDA",
        "company a": "COMP_A",
        "company b": "COMP_B",
    }
    for alias, ticker in company_aliases.items():
        if re.search(rf"\b{alias}\b", query_lower):
            if ticker not in companies:
                companies.append(ticker)

    # 3. Detect Metrics
    metrics: List[str] = []
    for metric_name, metric_def in CANONICAL_METRICS.items():
        for alias in metric_def.aliases:
            if alias in query_lower:
                if metric_name not in metrics:
                    metrics.append(metric_name)
                break

    # If "efficiency" or "intensity" or "compare" is mentioned, add revenue
    if "efficiency" in query_lower or "intensity" in query_lower:
        if "REVENUE" not in metrics:
            metrics.append("REVENUE")
        if "RD_EXPENSE" not in metrics and "r&d" in query_lower:
            metrics.append("RD_EXPENSE")

    # 4. Classify Query Type
    if len(companies) > 1 or "compare" in query_lower or "versus" in query_lower or " vs " in query_lower:
        q_type = "cross_company_comparison"
        math_needed = True
    elif len(years) > 1 or "trend" in query_lower or "growth" in query_lower:
        q_type = "longitudinal_trend"
        math_needed = True
    elif any(term in query_lower for term in ["risk", "mda", "strategy", "why", "explain"]):
        q_type = "qualitative_risk"
        math_needed = False
    else:
        q_type = "point_metric"
        math_needed = False

    sub_qs: List[str] = []
    for c in companies:
        for y in (years or [2023]):
            sub_qs.append(f"Retrieve financial statements for {c} for fiscal year {y}")

    return FinancialQueryPlan(
        query_type=q_type,
        target_companies=companies or ["UNKNOWN"],
        fiscal_years=years or [2023],
        required_metrics=metrics,
        requires_cypher=len(metrics) > 0,
        requires_vector=True,
        requires_ratio_math=math_needed,
        sub_questions=sub_qs or [query],
    )


class QueryDecomposer:
    """Orchestrates structured intent decomposition using DecisionAdapter with rule fallback."""

    def __init__(self, adapter: Optional[DecisionAdapter] = None) -> None:
        self.adapter = adapter or PydanticLLMAdapter()

    async def decompose(self, query: str) -> FinancialQueryPlan:
        """Decompose a user question into a structured FinancialQueryPlan."""
        try:
            plan = await self.adapter.classify(
                text=query,
                schema=FinancialQueryPlan,
                context={"domain": "SEC 10-K Financial Analysis"},
            )
            # Ensure minimal valid fields
            if not plan.target_companies or plan.target_companies == ["UNKNOWN"]:
                fallback = rule_based_query_decomposition(query)
                return fallback
            if not plan.requires_cypher and not plan.requires_vector:
                plan.requires_vector = True
                plan.requires_cypher = len(plan.required_metrics) > 0
            return plan
        except Exception as e:
            logger.info(f"Using rule-based decomposition fallback: {e}")
            return rule_based_query_decomposition(query)
