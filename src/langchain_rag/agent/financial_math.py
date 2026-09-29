"""Financial Math Sandbox for deterministic calculation of accounting ratios and metrics."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("langchain_rag.agent.financial_math")


class FinancialCalculator:
    """Calculates standardized financial ratios, growth rates, and margins."""

    @staticmethod
    def calculate_growth_rate(prior_value: float, current_value: float) -> Optional[float]:
        """Calculate Year-over-Year (YoY) percentage growth rate.

        Formula: (Current - Prior) / abs(Prior) * 100
        """
        if prior_value == 0:
            return None
        return round(((current_value - prior_value) / abs(prior_value)) * 100.0, 2)

    @staticmethod
    def calculate_ratio(numerator: float, denominator: float, as_percent: bool = True) -> Optional[float]:
        """Calculate standardized financial ratio or percentage."""
        if denominator == 0:
            return None
        val = numerator / denominator
        return round(val * 100.0 if as_percent else val, 2)

    @classmethod
    def compute_all_ratios(
        cls,
        metrics_by_year: Dict[int, Dict[str, float]],
    ) -> Dict[str, Any]:
        """Compute standard corporate performance ratios across fiscal periods.

        Args:
            metrics_by_year: Mapping of year -> {canonical_metric_name: value}
                e.g. {2022: {"REVENUE": 394328, "RD_EXPENSE": 26251}, 2023: {...}}

        Returns:
            Dictionary containing computed margins, intensities, and YoY growth series.
        """
        results: Dict[str, Any] = {
            "margins_by_year": {},
            "rd_intensity_by_year": {},
            "free_cash_flow_by_year": {},
            "yoy_growth": {},
        }

        years = sorted(metrics_by_year.keys())

        # 1. Per-Year Ratios
        for year in years:
            data = metrics_by_year[year]
            rev = data.get("REVENUE")
            rd = data.get("RD_EXPENSE")
            gp = data.get("GROSS_PROFIT")
            op_inc = data.get("OPERATING_INCOME")
            ocf = data.get("OPERATING_CASH_FLOW")
            capex = data.get("CAPEX")

            year_margins: Dict[str, Any] = {}
            if rev and rev > 0:
                if rd is not None:
                    rd_pct = cls.calculate_ratio(rd, rev, as_percent=True)
                    results["rd_intensity_by_year"][year] = rd_pct
                if gp is not None:
                    year_margins["gross_margin_pct"] = cls.calculate_ratio(gp, rev, as_percent=True)
                if op_inc is not None:
                    year_margins["operating_margin_pct"] = cls.calculate_ratio(op_inc, rev, as_percent=True)

            if year_margins:
                results["margins_by_year"][year] = year_margins

            if ocf is not None and capex is not None:
                # FCF = Operating Cash Flow - CapEx
                results["free_cash_flow_by_year"][year] = round(ocf - capex, 2)

        # 2. Multi-Year YoY Growth
        for i in range(1, len(years)):
            prev_yr = years[i - 1]
            curr_yr = years[i]
            prev_data = metrics_by_year[prev_yr]
            curr_data = metrics_by_year[curr_yr]

            yoy_pair: Dict[str, Any] = {}
            for metric in ["REVENUE", "RD_EXPENSE", "NET_INCOME", "OPERATING_INCOME"]:
                v_prev = prev_data.get(metric)
                v_curr = curr_data.get(metric)
                if v_prev is not None and v_curr is not None:
                    yoy_pair[f"{metric}_growth_pct"] = cls.calculate_growth_rate(v_prev, v_curr)

            if yoy_pair:
                results["yoy_growth"][f"{prev_yr}_to_{curr_yr}"] = yoy_pair

        return results
