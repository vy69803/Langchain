"""State definitions for the Financial Intelligence LangGraph Agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class FinancialAgentState(TypedDict):
    """Execution state tracking throughout the LangGraph workflow."""

    # Raw User Query & Analytical Intent
    raw_query: str
    query_type: str  # "point_metric" | "longitudinal_trend" | "cross_company_comparison" | "qualitative_risk"
    target_companies: List[str]  # e.g., ["AAPL", "MSFT"]
    fiscal_years: List[int]  # e.g., [2021, 2022, 2023]
    required_metrics: List[str]  # e.g., ["RD_EXPENSE", "REVENUE"]

    # Sub-Agent Decomposition Plan
    sub_tasks: List[Dict[str, Any]]

    # Multi-Source Evidence Stores
    cypher_results: List[Dict[str, Any]]
    vector_chunks: List[Dict[str, Any]]
    computed_ratios: Dict[str, Any]

    # Circuit Breakers & Resilience
    cypher_errors: int
    retrieval_fallback_triggered: bool

    # Generation, Reflection & Cycle Limits
    draft_response: str
    verification_feedback: Optional[str]
    is_verified: bool
    iteration_count: int  # Strict safety cap <= 2 to prevent infinite self-correction
    final_output: Optional[str]
