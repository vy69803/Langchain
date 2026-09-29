---
title: Financial Intelligence Agent
emoji: 🏛️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# 🏛️ Multi-Modal GraphRAG Financial Intelligence Agent

A production-grade, stateful AI agent combining **Neo4j Knowledge Graph**, **Supabase / pgvector**, **Python Financial Math Sandbox**, and **Fact-Checking Reflection** on corporate SEC filings (FinanceBench dataset).

## 🚀 Features
- **Knowledge Graph Traversal**: Queries canonical SEC XBRL financial statements on **Neo4j AuraDB**.
- **Vector Search & Footnote Resolution**: Resolves 2D table disclosures and superscript footnote qualifications (`(1)`, `Note 3`) in Supabase pgvector / ChromaDB.
- **Circuit Breaker Fallback**: Automatically degrades gracefully from Cypher Knowledge Graph to dense vector search if exact statement tags are absent.
- **Deterministic Math Sandbox**: Computes R&D intensity, YoY growth, EBITDA margins, and efficiency ratios directly in Python.
- **Grounded Benchmark Explorer**: Interactive inspector for 20 curated gold questions from FinanceBench.

## 🛠️ Required Space Secrets
To run in cloud standalone mode, configure the following secrets in **Space Settings $\rightarrow$ Repository Secrets**:

| Secret Key | Description |
| :--- | :--- |
| `NEO4J_URI` | Neo4j Aura connection URI (e.g. `neo4j+ssc://...`) |
| `NEO4J_USERNAME` | Neo4j database username |
| `NEO4J_PASSWORD` | Neo4j database password |
| `SUPABASE_DB_URL` | Supabase connection string for pgvector (`postgresql://...`) |
| `OPENROUTER_API_KEY` | OpenRouter API Key for structured LLM reasoning |
| `API_URL` *(Optional)* | Public URL of your deployed FastAPI backend (if running external API) |

## 📦 Deployment to Hugging Face
```bash
# Clone your Hugging Face Space repository
git clone https://huggingface.co/spaces/<your-username>/financial-intelligence-agent

# Copy files from hf_space/ into the clone
cp -r hf_space/* financial-intelligence-agent/

# Push to Hugging Face
cd financial-intelligence-agent
git add .
git commit -m "Deploy Financial Intelligence Agent Streamlit Space"
git push
```
