"""Canonical Financial Knowledge Graph Schema and XBRL Mapping.

Defines the ~35 core canonical financial metrics, their standard SEC EDGAR XBRL tags,
and common 10-K textual aliases for robust entity linking and normalization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CanonicalMetricDef:
    """Definition of a canonical financial metric."""

    canonical_name: str
    category: str  # "IncomeStatement" | "BalanceSheet" | "CashFlow" | "PerShare"
    xbrl_tag: str
    description: str
    aliases: List[str] = field(default_factory=list)


CANONICAL_METRICS: Dict[str, CanonicalMetricDef] = {
    # --- Income Statement ---
    "REVENUE": CanonicalMetricDef(
        canonical_name="REVENUE",
        category="IncomeStatement",
        xbrl_tag="us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax",
        description="Total revenue, net sales, or turnover from contracts with customers.",
        aliases=[
            "net sales",
            "total net sales",
            "revenue",
            "total revenue",
            "total revenues",
            "operating revenues",
            "sales",
        ],
    ),
    "COST_OF_REVENUE": CanonicalMetricDef(
        canonical_name="COST_OF_REVENUE",
        category="IncomeStatement",
        xbrl_tag="us-gaap/CostOfGoodsAndServicesSold",
        description="Total direct cost of sales, goods, and services sold.",
        aliases=[
            "cost of sales",
            "cost of goods sold",
            "cogs",
            "cost of revenues",
            "total cost of sales",
        ],
    ),
    "GROSS_PROFIT": CanonicalMetricDef(
        canonical_name="GROSS_PROFIT",
        category="IncomeStatement",
        xbrl_tag="us-gaap/GrossProfit",
        description="Gross margin or gross profit (Revenue minus Cost of Revenue).",
        aliases=["gross profit", "gross margin", "total gross profit"],
    ),
    "RD_EXPENSE": CanonicalMetricDef(
        canonical_name="RD_EXPENSE",
        category="IncomeStatement",
        xbrl_tag="us-gaap/ResearchAndDevelopmentExpense",
        description="Research, development, and engineering expenses incurred.",
        aliases=[
            "research and development",
            "r&d",
            "research, development and engineering",
            "research and development expense",
            "total research and development",
        ],
    ),
    "SGA_EXPENSE": CanonicalMetricDef(
        canonical_name="SGA_EXPENSE",
        category="IncomeStatement",
        xbrl_tag="us-gaap/SellingGeneralAndAdministrativeExpense",
        description="Selling, general, and administrative operating overhead.",
        aliases=[
            "selling, general and administrative",
            "sg&a",
            "selling, general, and administrative expenses",
            "general and administrative",
        ],
    ),
    "OPERATING_EXPENSES": CanonicalMetricDef(
        canonical_name="OPERATING_EXPENSES",
        category="IncomeStatement",
        xbrl_tag="us-gaap/OperatingExpenses",
        description="Total operating expenses including R&D and SG&A.",
        aliases=["total operating expenses", "operating costs and expenses"],
    ),
    "OPERATING_INCOME": CanonicalMetricDef(
        canonical_name="OPERATING_INCOME",
        category="IncomeStatement",
        xbrl_tag="us-gaap/OperatingIncomeLoss",
        description="Operating profit before interest and tax (EBIT equivalent).",
        aliases=[
            "operating income",
            "operating profit",
            "income from operations",
            "operating earnings",
        ],
    ),
    "NET_INCOME": CanonicalMetricDef(
        canonical_name="NET_INCOME",
        category="IncomeStatement",
        xbrl_tag="us-gaap/NetIncomeLoss",
        description="Net income, net earnings, or bottom-line profit after tax.",
        aliases=[
            "net income",
            "net earnings",
            "net profit",
            "net income attributable to shareholders",
        ],
    ),
    "EPS_DILUTED": CanonicalMetricDef(
        canonical_name="EPS_DILUTED",
        category="PerShare",
        xbrl_tag="us-gaap/EarningsPerShareDiluted",
        description="Diluted earnings per share.",
        aliases=["diluted earnings per share", "diluted eps", "earnings per common share - diluted"],
    ),
    # --- Balance Sheet ---
    "TOTAL_ASSETS": CanonicalMetricDef(
        canonical_name="TOTAL_ASSETS",
        category="BalanceSheet",
        xbrl_tag="us-gaap/Assets",
        description="Total current and non-current economic resources.",
        aliases=["total assets", "assets, total"],
    ),
    "CASH_AND_EQUIVALENTS": CanonicalMetricDef(
        canonical_name="CASH_AND_EQUIVALENTS",
        category="BalanceSheet",
        xbrl_tag="us-gaap/CashAndCashEquivalentsAtCarryingValue",
        description="Cash, commercial paper, and short-term liquid investments.",
        aliases=[
            "cash and cash equivalents",
            "cash, cash equivalents and marketable securities",
            "cash and short-term investments",
        ],
    ),
    "TOTAL_LIABILITIES": CanonicalMetricDef(
        canonical_name="TOTAL_LIABILITIES",
        category="BalanceSheet",
        xbrl_tag="us-gaap/Liabilities",
        description="Total current and long-term financial obligations.",
        aliases=["total liabilities", "liabilities, total"],
    ),
    "TOTAL_DEBT": CanonicalMetricDef(
        canonical_name="TOTAL_DEBT",
        category="BalanceSheet",
        xbrl_tag="us-gaap/DebtInstrumentCarryingAmount",
        description="Short-term borrowings plus long-term debt.",
        aliases=["total debt", "long-term debt", "commercial paper and total debt"],
    ),
    "STOCKHOLDERS_EQUITY": CanonicalMetricDef(
        canonical_name="STOCKHOLDERS_EQUITY",
        category="BalanceSheet",
        xbrl_tag="us-gaap/StockholdersEquity",
        description="Total shareholders equity or book value.",
        aliases=[
            "total shareholders' equity",
            "total stockholders' equity",
            "shareholders' equity",
        ],
    ),
    # --- Cash Flow Statement ---
    "OPERATING_CASH_FLOW": CanonicalMetricDef(
        canonical_name="OPERATING_CASH_FLOW",
        category="CashFlow",
        xbrl_tag="us-gaap/NetCashProvidedByUsedInOperatingActivities",
        description="Net cash flow provided by operating activities.",
        aliases=[
            "cash provided by operating activities",
            "net cash provided by operating activities",
            "operating cash flow",
            "cash from operations",
        ],
    ),
    "CAPEX": CanonicalMetricDef(
        canonical_name="CAPEX",
        category="CashFlow",
        xbrl_tag="us-gaap/PaymentsToAcquirePropertyPlantAndEquipment",
        description="Capital expenditures for property, plant, and equipment additions.",
        aliases=[
            "capital expenditures",
            "capital expenditure",
            "purchases of property, plant and equipment",
            "payments for property, plant and equipment",
            "additions to property, plant and equipment",
            "capex",
        ],
    ),
    "FREE_CASH_FLOW": CanonicalMetricDef(
        canonical_name="FREE_CASH_FLOW",
        category="CashFlow",
        xbrl_tag="custom/FreeCashFlow",
        description="Operating Cash Flow minus Capital Expenditures.",
        aliases=["free cash flow", "fcf"],
    ),
}


def normalize_line_item_name(raw_name: str) -> str:
    """Normalize string for fuzzy alias comparison."""
    cleaned = re.sub(r"[\(\[\{].*?[\)\]\}]", "", raw_name)  # Remove bracketed footnote numbers
    cleaned = re.sub(r"[^\w\s]", " ", cleaned).lower()
    return " ".join(cleaned.split())


def resolve_canonical_metric(raw_name: str) -> Optional[CanonicalMetricDef]:
    """Map raw 10-K line item text to a canonical financial metric definition.

    Args:
        raw_name: Extracted row header from a financial table.

    Returns:
        CanonicalMetricDef if a match is found, else None.
    """
    normalized = normalize_line_item_name(raw_name)

    for metric_def in CANONICAL_METRICS.values():
        for alias in metric_def.aliases:
            alias_norm = normalize_line_item_name(alias)
            if normalized == alias_norm or normalized.startswith(alias_norm):
                return metric_def

    return None
