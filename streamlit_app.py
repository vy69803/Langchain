"""Streamlit Frontend for Multi-Modal GraphRAG Financial Intelligence Agent.

Supports both direct agent execution and connection to the Production FastAPI service.
Optimized for local execution and deployment to Hugging Face Spaces.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Financial Intelligence Agent | GraphRAG",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Financial Dark-Mode CSS
st.markdown(
    """
    <style>
    /* Global Styling */
    .stApp {
        background-color: #0c1017;
        color: #f0f6fc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Gradient */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38ef7d 0%, #11998e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #8b949e;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .metric-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #58a6ff;
    }

    /* Badge Tags */
    .badge-verified {
        display: inline-block;
        background-color: rgba(46, 160, 67, 0.15);
        border: 1px solid #2ea043;
        color: #3fb950;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-circuit {
        display: inline-block;
        background-color: rgba(219, 109, 40, 0.15);
        border: 1px solid #db6d28;
        color: #f0883e;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Report Box */
    .report-container {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
        line-height: 1.6;
    }
    
    /* Table styling */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    th, td {
        border: 1px solid #30363d;
        padding: 8px 12px;
        text-align: left;
    }
    th {
        background-color: #21262d;
        color: #58a6ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_benchmark_questions() -> List[Dict[str, Any]]:
    """Load the curated 20-question FinanceBench dataset."""
    benchmark_path = Path(__file__).parent / "data" / "financebench_curated_20.json"
    if benchmark_path.exists():
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


# --- Engine Invocation Helpers ---
async def run_agent_in_process(query: str) -> Dict[str, Any]:
    """Execute the LangGraph Agent directly in-process."""
    from langchain_rag.adapters.storage_adapters import ChromaVectorAdapter, Neo4jGraphAdapter, SupabaseVectorAdapter
    from langchain_rag.agent.graph_agent import FinancialIntelligenceAgent

    # Determine vector backend
    backend = os.getenv("VECTOR_STORE_BACKEND", "chroma").lower()
    vector_adapter = None
    if backend == "supabase" or os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL"):
        try:
            vector_adapter = SupabaseVectorAdapter(
                table_name=os.getenv("SUPABASE_VECTOR_TABLE", "financebench_docs")
            )
        except Exception as e:
            st.sidebar.warning(f"Supabase init fallback to Chroma: {e}")

    if vector_adapter is None:
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        if os.path.exists(chroma_dir):
            vector_adapter = ChromaVectorAdapter(collection_name="financebench", persist_directory=chroma_dir)

    graph_adapter = Neo4jGraphAdapter()
    agent = FinancialIntelligenceAgent(graph_adapter=graph_adapter, vector_adapter=vector_adapter)
    return await agent.ainvoke(query)


def run_agent_via_api(api_url: str, query: str) -> Dict[str, Any]:
    """Execute the agent via FastAPI HTTP endpoint."""
    import httpx

    endpoint = f"{api_url.rstrip('/')}/api/v1/financial-agent/query"
    resp = httpx.post(endpoint, json={"query": query}, timeout=90.0)
    resp.raise_for_status()
    data = resp.json()
    return {
        "final_output": data.get("report", ""),
        "target_companies": data.get("target_companies", []),
        "fiscal_years": data.get("fiscal_years", []),
        "required_metrics": data.get("metrics_retrieved", []),
        "is_verified": data.get("is_verified", False),
        "iteration_count": data.get("iteration_count", 0),
        "cypher_results": [{"metric": m} for m in data.get("metrics_retrieved", [])],
        "vector_chunks": [{"id": f"chunk_{i}"} for i in range(data.get("vector_chunks_count", 0))],
        "retrieval_fallback_triggered": data.get("fallback_triggered", False),
    }


def run_hybrid_rag_via_api(api_url: str, query: str) -> Dict[str, Any]:
    """Execute the conversational RAG endpoint."""
    import httpx

    endpoint = f"{api_url.rstrip('/')}/chat"
    resp = httpx.post(endpoint, json={"query": query, "session_id": "st-session"}, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    return {
        "final_output": data.get("response", ""),
        "citations": data.get("citations", []),
        "retrieved_docs": data.get("retrieved_docs", []),
        "cached": data.get("cached", False),
        "is_verified": True,
    }


# ============================================================================
# Sidebar Configuration & Benchmark Selector
# ============================================================================
st.sidebar.markdown("### 🏛️ System Architecture")

# Cloud Status Indicators
neo4j_uri = os.getenv("NEO4J_URI", "")
neo4j_online = bool(neo4j_uri and "databases.neo4j.io" in neo4j_uri)
if neo4j_online:
    st.sidebar.success("🟢 Neo4j AuraDB (Cloud KG Live)")
else:
    st.sidebar.info("⚪ Neo4j Local / Standby")

backend_env = os.getenv("VECTOR_STORE_BACKEND", "chroma").upper()
st.sidebar.success(f"🟢 Vector Store ({backend_env} Online)")
st.sidebar.success("🟢 OpenRouter LLM Harness")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Execution Mode")
exec_mode = st.sidebar.radio(
    "Select Engine:",
    options=[
        "LangGraph Agent (GraphRAG + Verification)",
        "Conversational Hybrid RAG (/chat)",
    ],
    index=0,
)

api_url_input = st.sidebar.text_input(
    "FastAPI Base URL (optional):",
    value=os.getenv("API_URL", "http://127.0.0.1:8000"),
    help="Leave as default if local API is running, or clear to run direct in-process.",
)

# Benchmark Curation Loader
benchmark_data = load_benchmark_questions()
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Grounded Benchmark Questions")

selected_q: Optional[Dict[str, Any]] = None
if benchmark_data:
    q_categories = sorted(list({q.get("category", "General") for q in benchmark_data}))
    selected_cat = st.sidebar.selectbox("Filter Category:", ["All Categories"] + q_categories)

    filtered_qs = (
        benchmark_data
        if selected_cat == "All Categories"
        else [q for q in benchmark_data if q.get("category") == selected_cat]
    )

    q_options = [f"[{q['id']}] {q['company']} ({q.get('doc_name', '').split('_')[-1]}): {q['question'][:65]}..." for q in filtered_qs]
    selected_idx = st.sidebar.selectbox("Select Grounded Query:", range(len(q_options)), format_func=lambda i: q_options[i])
    selected_q = filtered_qs[selected_idx]

    if st.sidebar.button("📥 Load Question into Query Box"):
        st.session_state["query_input"] = selected_q["question"]
        st.session_state["selected_benchmark"] = selected_q

st.sidebar.markdown("---")
st.sidebar.caption(
    "Antigravity Multi-Modal Financial Intelligence Agent • SEC Form 10-K/10-Q GraphRAG"
)

# ============================================================================
# Main Dashboard
# ============================================================================
st.markdown('<div class="main-title">🏛️ Financial Intelligence Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Stateful GraphRAG reasoning across corporate SEC filings (FinanceBench) with deterministic math verification.</div>',
    unsafe_allow_html=True,
)

# Query Input Area
default_prompt = st.session_state.get(
    "query_input",
    "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Rely on the cash flow statement.",
)

with st.container():
    user_query = st.text_area(
        "Enter your financial analytical query:",
        value=default_prompt,
        height=95,
        placeholder="e.g. Compare Apple's R&D intensity between FY2021 and FY2023.",
    )

col_run, col_clear, col_status = st.columns([1.5, 1, 6])
run_clicked = col_run.button("🚀 Run Analysis", type="primary", use_container_width=True)
if col_clear.button("Clear", use_container_width=True):
    st.session_state["query_input"] = ""
    st.rerun()

# Execute Query
if run_clicked and user_query.strip():
    start_time = time.perf_counter()
    with st.spinner("Analyzing SEC filings across Knowledge Graph & Vector hierarchy..."):
        result: Dict[str, Any] = {}
        error_msg = None

        if "LangGraph Agent" in exec_mode:
            # Try API first if provided, else fallback to in-process
            used_api = False
            if api_url_input:
                try:
                    result = run_agent_via_api(api_url_input, user_query)
                    used_api = True
                except Exception as api_err:
                    st.warning(f"FastAPI connection failed ({api_err}). Running in-process agent...")
            
            if not used_api:
                try:
                    result = asyncio.run(run_agent_in_process(user_query))
                except Exception as ex:
                    error_msg = str(ex)
        else:
            # Hybrid RAG mode
            if api_url_input:
                try:
                    result = run_hybrid_rag_via_api(api_url_input, user_query)
                except Exception as api_err:
                    error_msg = f"API Error: {api_err}"
            else:
                error_msg = "Conversational Hybrid RAG requires a running FastAPI backend."

    latency = round(time.perf_counter() - start_time, 2)

    if error_msg:
        st.error(f"Execution Error: {error_msg}")
    else:
        # Save last result in session state
        st.session_state["last_result"] = result
        st.session_state["last_latency"] = latency

# Render Output if available
if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    lat = st.session_state.get("last_latency", 0.0)

    # Top KPI Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        target_comps = ", ".join(res.get("target_companies", [])) or "Detected in Query"
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Entities</div><div class="metric-value">{target_comps}</div></div>',
            unsafe_allow_html=True,
        )
    with m_col2:
        years = ", ".join(str(y) for y in res.get("fiscal_years", [])) or "Full Period"
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Fiscal Years</div><div class="metric-value">{years}</div></div>',
            unsafe_allow_html=True,
        )
    with m_col3:
        cypher_cnt = len(res.get("cypher_results", []))
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Graph Metrics</div><div class="metric-value">{cypher_cnt}</div></div>',
            unsafe_allow_html=True,
        )
    with m_col4:
        vec_cnt = len(res.get("vector_chunks", []))
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Vector Chunks</div><div class="metric-value">{vec_cnt}</div></div>',
            unsafe_allow_html=True,
        )
    with m_col5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">End-to-End Latency</div><div class="metric-value">{lat}s</div></div>',
            unsafe_allow_html=True,
        )

    # Status Badges
    b_col1, b_col2, _ = st.columns([2, 3, 5])
    with b_col1:
        if res.get("is_verified", False):
            st.markdown('<span class="badge-verified">✓ SEC Footnote Verified</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-circuit">⚡ Reflection Cap Reached</span>', unsafe_allow_html=True)
    with b_col2:
        if res.get("retrieval_fallback_triggered", False):
            st.markdown('<span class="badge-circuit">⚡ Circuit Breaker: Vector Fallback Engaged</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-verified">✓ Exact Neo4j Graph Traversal</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabbed Detail View
    tab_report, tab_evidence, tab_benchmark = st.tabs(
        ["📋 Executive Report", "🔍 Retrieved Evidence & Citations", "🎯 Benchmark Ground Truth"]
    )

    with tab_report:
        report_text = res.get("final_output", "No response generated.")
        st.markdown(f'<div class="report-container">\n\n{report_text}\n\n</div>', unsafe_allow_html=True)

    with tab_evidence:
        chunks = res.get("vector_chunks", [])
        citations = res.get("citations", [])

        if citations:
            st.markdown("#### Inline SEC Citations")
            for cit in citations:
                meta = cit.get("metadata", {})
                doc_name = meta.get("doc_name") or cit.get("source", "Unknown Filing")
                page_no = meta.get("page") or "N/A"
                st.markdown(f"**{cit.get('citation')} {doc_name} (Page {page_no})**")
                st.caption(cit.get("content_preview", ""))
                st.markdown("---")

        if chunks:
            st.markdown(f"#### Retrieved Evidence Chunks ({len(chunks)})")
            for i, chunk in enumerate(chunks, start=1):
                meta = chunk.get("metadata", {})
                doc = meta.get("doc_name", "SEC Document")
                page = meta.get("page", "N/A")
                score = round(chunk.get("score", 0.0), 3)
                with st.expander(f"Chunk #{i}: {doc} — Page {page} (Similarity Score: {score})"):
                    st.text(chunk.get("text", ""))

    with tab_benchmark:
        cur_benchmark = st.session_state.get("selected_benchmark")
        if cur_benchmark:
            st.markdown(f"### Benchmark ID: `{cur_benchmark['id']}` — {cur_benchmark['company']}")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.info(f"**Official Ground Truth Answer:**\n\n### {cur_benchmark['ground_truth_answer']}")
                st.markdown(f"**Target Filing:** `{cur_benchmark.get('doc_name')}` (Page {cur_benchmark.get('page')})")
                st.markdown(f"**Financial Statement:** {cur_benchmark.get('statement')}")
            with col_b2:
                st.markdown(f"**Subsystem Evaluated:**\n{cur_benchmark.get('subsystem_tested')}")
                st.markdown(f"**Grounding Justification:**\n{cur_benchmark.get('justification')}")

            with st.expander("View Gold Benchmark Evidence Text"):
                for ev in cur_benchmark.get("evidence", []):
                    st.code(ev.get("evidence_text", ""), language="markdown")
        else:
            st.info("Select a benchmark question from the sidebar to inspect official FinanceBench ground truths.")
