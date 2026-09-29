"""LangGraph Orchestrator for Multi-Modal GraphRAG Financial Intelligence.

Implements the stateful workflow combining Cypher Knowledge Graph traversal,
hybrid vector search, deterministic financial ratio math, and fact-checking reflection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import END, StateGraph

from langchain_rag.adapters.decision_adapter import DecisionAdapter, PydanticLLMAdapter, VerificationResult
from langchain_rag.adapters.storage_adapters import BaseGraphStoreAdapter, BaseVectorStoreAdapter, Neo4jGraphAdapter
from langchain_rag.agent.decomposer import FinancialQueryPlan, QueryDecomposer
from langchain_rag.agent.financial_math import FinancialCalculator
from langchain_rag.agent.state import FinancialAgentState

logger = logging.getLogger("langchain_rag.agent.graph_agent")


class FinancialIntelligenceAgent:
    """Stateful Multi-Modal GraphRAG Financial Agent."""

    def __init__(
        self,
        decision_adapter: Optional[DecisionAdapter] = None,
        graph_adapter: Optional[BaseGraphStoreAdapter] = None,
        vector_adapter: Optional[BaseVectorStoreAdapter] = None,
    ) -> None:
        self.decision_adapter = decision_adapter or PydanticLLMAdapter()
        self.graph_adapter = graph_adapter or Neo4jGraphAdapter()
        self.vector_adapter = vector_adapter
        self.decomposer = QueryDecomposer(adapter=self.decision_adapter)
        self.graph = self._build_graph()

    # --- Node 1: Query Decomposition ---
    async def node_decompose(self, state: FinancialAgentState) -> Dict[str, Any]:
        """Decompose user query into structured retrieval sub-tasks."""
        plan: FinancialQueryPlan = await self.decomposer.decompose(state["raw_query"])
        return {
            "query_type": plan.query_type,
            "target_companies": plan.target_companies,
            "fiscal_years": plan.fiscal_years,
            "required_metrics": plan.required_metrics,
            "sub_tasks": [{"desc": q} for q in plan.sub_questions],
            "cypher_results": state.get("cypher_results", []),
            "vector_chunks": state.get("vector_chunks", []),
            "computed_ratios": state.get("computed_ratios", {}),
            "cypher_errors": 0,
            "retrieval_fallback_triggered": False,
            "iteration_count": 0,
            "is_verified": False,
        }

    # --- Node 2: Cypher Graph Retrieval ---
    async def node_cypher_retrieval(self, state: FinancialAgentState) -> Dict[str, Any]:
        """Query Neo4j Knowledge Graph for exact historical metrics."""
        companies = state.get("target_companies", [])
        years = state.get("fiscal_years", [])
        metrics = state.get("required_metrics", [])

        cypher_query = """
        MATCH (c:Company)-[:FILED]->(f:Filing)-[:CONTAINS]->(s:FinancialStatement)-[:HAS_LINE_ITEM]->(l:LineItem)
        WHERE c.ticker IN $companies AND f.year IN $years
        OPTIONAL MATCH (l)-[:QUALIFIED_BY]->(fn:Footnote)
        RETURN 
            c.ticker AS ticker,
            f.year AS year,
            s.type AS statement,
            l.canonical_name AS metric,
            l.raw_name AS raw_name,
            l.value AS value,
            l.unit AS unit,
            collect(fn.text) AS footnotes
        ORDER BY c.ticker, f.year, l.canonical_name
        """

        results: List[Dict[str, Any]] = []
        cypher_error = 0
        fallback_triggered = False

        try:
            results = self.graph_adapter.execute_cypher(
                cypher_query,
                params={"companies": companies, "years": years},
                read_only=True,
            )
            # Filter to valid canonical metrics and exclude date/table noise
            valid_canonical = {"REVENUE", "CAPEX", "NET_INCOME", "OPERATING_INCOME", "TOTAL_ASSETS", "TOTAL_DEBT", "RD_EXPENSE", "GROSS_PROFIT", "CASH_AND_EQUIVALENTS"}
            noise_terms = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
            cleaned = [
                r for r in results 
                if (r.get("metric") in valid_canonical or any(rm.lower() in (r.get("metric") or "").lower() for rm in metrics))
                and not any((r.get("raw_name") or "").lower().startswith(nt) for nt in noise_terms)
            ]
            results = cleaned
            if not results:
                logger.info("Cypher returned 0 valid financial records. Triggering vector retrieval fallback.")
                fallback_triggered = True

        except Exception as e:
            logger.warning(f"Cypher execution failed: {e}. Diverting to vector retrieval.")
            cypher_error = 1
            fallback_triggered = True

        return {
            "cypher_results": results,
            "cypher_errors": cypher_error,
            "retrieval_fallback_triggered": fallback_triggered,
        }

    # --- Node 3: Vector Search Retrieval ---
    async def node_vector_retrieval(self, state: FinancialAgentState) -> Dict[str, Any]:
        """Retrieve unstructured narrative, MD&A, and table chunks from vector store."""
        chunks: List[Dict[str, Any]] = []

        if self.vector_adapter:
            for company in state.get("target_companies", []):
                query = f"{company} " + " ".join(state.get("required_metrics", [])) + " " + state["raw_query"]
                filter_meta = {"company": company} if company and company != "UNKNOWN" else None
                try:
                    results = self.vector_adapter.similarity_search(query=query, k=6, filter_metadata=filter_meta)
                except Exception:
                    results = self.vector_adapter.similarity_search(query=query, k=6)
                chunks.extend(results)
        else:
            # Deterministic mock chunk when vector adapter not passed
            chunks.append(
                {
                    "id": "mock_chunk_1",
                    "text": f"SEC 10-K Qualitative Disclosures for {state.get('target_companies')}: "
                    f"R&D investments and revenue performance over {state.get('fiscal_years')}.",
                    "metadata": {"company": state.get("target_companies")},
                }
            )

        return {"vector_chunks": chunks}

    # --- Node 4: Financial Math Sandbox ---
    async def node_financial_math(self, state: FinancialAgentState) -> Dict[str, Any]:
        """Compute financial ratios, margins, and multi-year growth rates."""
        cypher_data = state.get("cypher_results", [])
        metrics_by_year: Dict[int, Dict[str, float]] = {}

        # Aggregate cypher metrics by year
        for row in cypher_data:
            yr = row.get("year")
            metric = row.get("metric")
            val = row.get("value")
            if yr and metric and val is not None:
                if yr not in metrics_by_year:
                    metrics_by_year[yr] = {}
                metrics_by_year[yr][metric] = float(val)

        ratios = FinancialCalculator.compute_all_ratios(metrics_by_year)
        return {"computed_ratios": ratios}

    # --- Node 5: Report Synthesis ---
    async def node_synthesis(self, state: FinancialAgentState) -> Dict[str, Any]:
        """Synthesize comparative markdown report with tables and inline citations."""
        companies = ", ".join(state.get("target_companies", []))
        years = ", ".join(str(y) for y in state.get("fiscal_years", []))
        metrics = state.get("cypher_results", [])
        ratios = state.get("computed_ratios", {})
        feedback = state.get("verification_feedback")

        vector_chunks = state.get("vector_chunks", [])
        
        # 1. Filter Cypher metrics to remove noise/month fragments
        noise_terms = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "note", "item"}
        req_metrics = [rm.lower() for rm in state.get("required_metrics", [])]
        filtered_metrics = []
        for m in metrics:
            m_name = (m.get("metric") or "").strip()
            r_name = (m.get("raw_name") or "").strip().lower()
            is_noise = any(r_name == nt or r_name.startswith(nt) for nt in noise_terms)
            is_req = any(rm in m_name.lower() or rm in r_name for rm in req_metrics)
            is_canonical = m_name in ["REVENUE", "CAPEX", "NET_INCOME", "OPERATING_INCOME", "TOTAL_ASSETS", "TOTAL_DEBT", "RD_EXPENSE", "GROSS_PROFIT", "CASH_AND_EQUIVALENTS"]
            if (is_req or is_canonical) and not is_noise:
                filtered_metrics.append(m)

        # 2. Synthesize direct executive summary with LLM if available
        exec_summary = f"Cross-document comparative analysis addressing: *{state['raw_query']}*"
        if hasattr(self.decision_adapter, "llm") and self.decision_adapter.llm is not None:
            try:
                ev_str = "\n".join([f"[{i}] {c.get('metadata', {}).get('doc_name', '')} (p.{c.get('metadata', {}).get('page', '')}): {c.get('text', '')[:250]}" for i, c in enumerate(vector_chunks[:3], 1)])
                cyp_str = "\n".join([f"- {m.get('ticker')} {m.get('year')} {m.get('metric')}: ${m.get('value'):,.1f}M" for m in filtered_metrics[:5]])
                prompt = (
                    f"You are a Senior Financial Analyst. Answer this question in 2-3 sentences based strictly on the retrieved SEC filing evidence.\n"
                    f"State the exact dollar figures/ratios, financial statement name, and filing citations.\n"
                    f"Question: {state['raw_query']}\n"
                    f"Metrics: {cyp_str or 'None'}\n"
                    f"Evidence Chunks:\n{ev_str}\n"
                    f"Verification Feedback: {feedback or 'None'}\n\n"
                    f"Executive Summary:"
                )
                resp = await self.decision_adapter.llm.ainvoke(prompt)
                if resp and hasattr(resp, "content") and resp.content:
                    exec_summary = resp.content.strip()
            except Exception as e:
                logger.debug(f"LLM summary generation skipped: {e}")

        # Generate structured analytical response
        lines = [
            f"# Financial Analysis: {companies} ({years})",
            "",
            "## Executive Summary",
            exec_summary,
            "",
            "## Consolidated Financial Metrics",
            "| Company | Fiscal Year | Metric | Value (USD Millions) | Disclosures / Footnotes |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        if filtered_metrics:
            for m in filtered_metrics[:10]:
                fn_str = " | ".join(m.get("footnotes", [])) or "Standard Disclosure"
                lines.append(
                    f"| {m.get('ticker')} | {m.get('year')} | {m.get('metric')} | ${m.get('value'):,.1f} | {fn_str} |"
                )
        else:
            lines.append(f"| {companies} | {years} | Extracted via Vector Search | See Evidence Disclosures below | - |")

        if vector_chunks:
            lines.extend(["", "## Retrieved Evidence & Document Disclosures"])
            for idx, chunk in enumerate(vector_chunks[:3], start=1):
                meta = chunk.get("metadata", {})
                doc_name = meta.get("doc_name", "SEC Filing")
                page = meta.get("page", "N/A")
                txt = chunk.get("text", "").strip()[:350].replace("\n", " ")
                lines.append(f"- **[{idx}] {doc_name} (Page {page}):** {txt}...")

        lines.extend(["", "## Calculated Efficiency & Performance Ratios"])
        rd_intensity = ratios.get("rd_intensity_by_year", {})
        if rd_intensity:
            lines.append("### R&D Intensity (R&D Expense / Total Revenue):")
            for yr, val in rd_intensity.items():
                lines.append(f"- **FY{yr}:** {val}%")

        yoy = ratios.get("yoy_growth", {})
        if yoy:
            lines.append("### Multi-Year YoY Growth Trends:")
            for period, metrics_dict in yoy.items():
                for k, v in metrics_dict.items():
                    lines.append(f"- **{period} {k.replace('_', ' ').title()}:** {v}%")

        lines.extend([
            "",
            "## Audit Trail & Citations",
            f"- Source Filings: SEC Form 10-K for {companies} (Fiscal Periods {years}).",
            "- Bounding Box & Page Reference: Item 8 Consolidated Financial Statements.",
        ])

        if feedback:
            lines.append(f"\n> **Verification Revision Note:** {feedback}")

        return {"draft_response": "\n".join(lines)}

    # --- Node 6: Verification Judge (Self-Correction) ---
    async def node_verification(self, state: FinancialAgentState) -> Dict[str, Any]:
        """Cross-check numerical claims against retrieved evidence."""
        draft = state.get("draft_response", "")
        evidence = state.get("cypher_results", []) + state.get("vector_chunks", [])
        current_iter = state.get("iteration_count", 0) + 1

        result: VerificationResult = await self.decision_adapter.verify(
            claim=draft,
            evidence=evidence,
        )

        return {
            "is_verified": result.is_faithful,
            "verification_feedback": result.discrepancy_details,
            "iteration_count": current_iter,
        }

    # --- Terminal Nodes ---
    async def node_final_output(self, state: FinancialAgentState) -> Dict[str, Any]:
        """Format final verified report."""
        return {"final_output": state["draft_response"]}

    async def node_fallback_output(self, state: FinancialAgentState) -> Dict[str, Any]:
        """Deliver report with uncertainty warning when verification cap is reached."""
        warning = (
            "> **Disclaimer:** Maximum verification iterations reached. "
            "Please cross-reference figures directly with referenced SEC footnotes.\n\n"
        )
        return {"final_output": warning + state["draft_response"]}

    # --- Edge Conditionals ---
    def check_verification(self, state: FinancialAgentState) -> Literal["final_output", "self_correction", "fallback"]:
        """Determine whether to accept response, retry synthesis, or terminate with fallback."""
        if state.get("is_verified", False):
            return "final_output"
        if state.get("iteration_count", 0) >= 2:
            return "fallback"
        return "self_correction"

    def _build_graph(self) -> Any:
        """Construct the LangGraph state machine."""
        workflow = StateGraph(FinancialAgentState)

        # Register Nodes
        workflow.add_node("decompose", self.node_decompose)
        workflow.add_node("cypher_retrieval", self.node_cypher_retrieval)
        workflow.add_node("vector_retrieval", self.node_vector_retrieval)
        workflow.add_node("financial_math", self.node_financial_math)
        workflow.add_node("synthesis", self.node_synthesis)
        workflow.add_node("verify", self.node_verification)
        workflow.add_node("final_output", self.node_final_output)
        workflow.add_node("fallback_output", self.node_fallback_output)

        # Set Entry Point
        workflow.set_entry_point("decompose")

        # Edges
        workflow.add_edge("decompose", "cypher_retrieval")
        workflow.add_edge("cypher_retrieval", "vector_retrieval")
        workflow.add_edge("vector_retrieval", "financial_math")
        workflow.add_edge("financial_math", "synthesis")
        workflow.add_edge("synthesis", "verify")

        # Conditional Verification Routing
        workflow.add_conditional_edges(
            "verify",
            self.check_verification,
            {
                "final_output": "final_output",
                "self_correction": "synthesis",
                "fallback": "fallback_output",
            },
        )

        workflow.add_edge("final_output", END)
        workflow.add_edge("fallback_output", END)

        return workflow.compile()

    async def ainvoke(self, query: str) -> Dict[str, Any]:
        """Execute the agent graph asynchronously for a given user query."""
        initial_state: FinancialAgentState = {
            "raw_query": query,
            "query_type": "point_metric",
            "target_companies": [],
            "fiscal_years": [],
            "required_metrics": [],
            "sub_tasks": [],
            "cypher_results": [],
            "vector_chunks": [],
            "computed_ratios": {},
            "cypher_errors": 0,
            "retrieval_fallback_triggered": False,
            "draft_response": "",
            "verification_feedback": None,
            "is_verified": False,
            "iteration_count": 0,
            "final_output": None,
        }
        return await self.graph.ainvoke(initial_state)
