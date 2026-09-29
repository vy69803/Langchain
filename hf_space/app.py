"""Hugging Face Spaces Streamlit App for Financial Intelligence Agent."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Configure Page
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
    .stApp {
        background-color: #0c1017;
        color: #f0f6fc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
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
    .report-container {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.5rem;
        line-height: 1.6;
    }
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
    if not benchmark_path.exists():
        # Fallback to parent data directory if running from repo root
        benchmark_path = Path(__file__).parent.parent / "data" / "financebench_curated_20.json"

    if benchmark_path.exists():
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def query_backend_api(api_url: str, endpoint: str, payload: Dict[str, Any], timeout: float = 90.0) -> Dict[str, Any]:
    """Execute an analytical query against the FastAPI backend."""
    import httpx

    full_url = f"{api_url.rstrip('/')}{endpoint}"
    resp = httpx.post(full_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ============================================================================
# Sidebar Configuration & Grounded Benchmark Loader
# ============================================================================
st.sidebar.markdown("### 🏛️ System Architecture")

# Detect Cloud Services from Space Secrets
neo4j_uri = os.getenv("NEO4J_URI", "")
supabase_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL", "")
openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

if neo4j_uri:
    st.sidebar.success("🟢 Neo4j AuraDB (Cloud KG Connected)")
else:
    st.sidebar.info("⚪ Neo4j Secret Standby")

if supabase_url:
    st.sidebar.success("🟢 Supabase pgvector (Cloud)")
else:
    st.sidebar.info("⚪ Supabase Secret Standby")

if openrouter_key:
    st.sidebar.success("🟢 OpenRouter LLM Active")
else:
    st.sidebar.info("⚪ OpenRouter Key Standby")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Mode")
exec_mode = st.sidebar.radio(
    "Select Engine:",
    options=[
        "LangGraph Agent (GraphRAG + Verification)",
        "Conversational Hybrid RAG (/chat)",
    ],
    index=0,
)

api_url_default = os.getenv("API_URL", "http://127.0.0.1:8000")
api_url_input = st.sidebar.text_input(
    "Backend API URL:",
    value=api_url_default,
    help="URL of your running FastAPI service (e.g. Railway, GCP, or Local Tunnel).",
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

    q_options = [f"[{q['id']}] {q['company']}: {q['question'][:65]}..." for q in filtered_qs]
    selected_idx = st.sidebar.selectbox("Select Grounded Query:", range(len(q_options)), format_func=lambda i: q_options[i])
    selected_q = filtered_qs[selected_idx]

    if st.sidebar.button("📥 Load Question into Query Box"):
        st.session_state["query_input"] = selected_q["question"]
        st.session_state["selected_benchmark"] = selected_q

st.sidebar.markdown("---")
st.sidebar.caption("FinanceBench Multi-Modal GraphRAG • Hugging Face Space Edition")


# ============================================================================
# Main Dashboard
# ============================================================================
st.markdown('<div class="main-title">🏛️ Financial Intelligence Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Stateful GraphRAG reasoning across corporate SEC filings (FinanceBench) with deterministic math verification.</div>',
    unsafe_allow_html=True,
)

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

col_run, col_clear, _ = st.columns([1.5, 1, 6])
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

        try:
            if "LangGraph Agent" in exec_mode:
                raw_res = query_backend_api(
                    api_url_input,
                    "/api/v1/financial-agent/query",
                    {"query": user_query},
                    timeout=100.0,
                )
                result = {
                    "final_output": raw_res.get("report", ""),
                    "target_companies": raw_res.get("target_companies", []),
                    "fiscal_years": raw_res.get("fiscal_years", []),
                    "required_metrics": raw_res.get("metrics_retrieved", []),
                    "is_verified": raw_res.get("is_verified", False),
                    "iteration_count": raw_res.get("iteration_count", 0),
                    "cypher_results": [{"metric": m} for m in raw_res.get("metrics_retrieved", [])],
                    "vector_chunks": [{"id": f"chunk_{i}"} for i in range(raw_res.get("vector_chunks_count", 0))],
                    "retrieval_fallback_triggered": raw_res.get("fallback_triggered", False),
                }
            else:
                raw_res = query_backend_api(
                    api_url_input,
                    "/chat",
                    {"query": user_query, "session_id": "hf-space-user"},
                    timeout=60.0,
                )
                result = {
                    "final_output": raw_res.get("response", ""),
                    "citations": raw_res.get("citations", []),
                    "retrieved_docs": raw_res.get("retrieved_docs", []),
                    "cached": raw_res.get("cached", False),
                    "is_verified": True,
                }
        except Exception as ex:
            error_msg = f"Backend Connection Error: {str(ex)}. Please ensure your FastAPI backend is running and accessible."

    latency = round(time.perf_counter() - start_time, 2)

    if error_msg:
        st.error(error_msg)
    else:
        st.session_state["last_result"] = result
        st.session_state["last_latency"] = latency

# Render Output View
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
            f'<div class="metric-card"><div class="metric-title">Latency</div><div class="metric-value">{lat}s</div></div>',
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

    # Detail Tabs
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
                with st.expander(f"Chunk #{i}: {doc} — Page {page}"):
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
