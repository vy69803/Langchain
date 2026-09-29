# 📊 FinanceBench Curated 20 Grounded Benchmark Questions

This document details the **20 representative benchmark questions** specifically selected to evaluate the **Multi-Modal, Graph-Enhanced Financial Intelligence Agent**.

The corresponding machine-readable JSON dataset is located at [`data/financebench_curated_20.json`](file:///c:/Users/Asus/Langchain/data/financebench_curated_20.json).

---

## 🏛️ Dataset Lineage & Grounding
Every question in this suite is grounded directly in the SEC filings indexed within this repository:
- **Master QA Source:** [`data/financebench/financebench_open_source.jsonl`](file:///c:/Users/Asus/Langchain/data/financebench/financebench_open_source.jsonl)
- **Vector Index:** 134,663 chunks across 366 filings in [`data/chroma_db`](file:///c:/Users/Asus/Langchain/data/chroma_db)
- **Sparse Index:** BM25 inverted index in [`data/bm25_financebench.json`](file:///c:/Users/Asus/Langchain/data/bm25_financebench.json)
- **Knowledge Graph:** 41,569 Line Items & 50,738 relationships in **Neo4j AuraDB Free**

---

## 📋 Benchmark Questions Summary Table

| ID | Company | Filing | Question Summary | Ground Truth Target | Category |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FB-01** | 3M (`MMM`) | `3M_2018_10K` (p.59) | FY2018 Capital Expenditure (Cash Flow) | `$1,577.00 million` | Point-in-Time Metric |
| **FB-02** | Boeing (`BA`) | `BOEING_2018_10K` (p.51) | FY2018 Net PP&E (Balance Sheet) | `$12,645.00 million` | Point-in-Time Metric |
| **FB-03** | Microsoft (`MSFT`) | `MICROSOFT_2016_10K` (p.51) | FY2016 Cost of Revenues (COGS) | `$32,780.00 million` | Point-in-Time Metric |
| **FB-04** | Best Buy (`BBY`) | `BESTBUY_2019_10K` (p.51) | FY2019 Total Inventories | `$5,409.00 million` | Point-in-Time Metric |
| **FB-05** | PepsiCo (`PEP`) | `PEPSICO_2021_10K` (p.62) | FY2021 CapEx in USD Billions | `$4.60 billion` | Point-in-Time Metric |
| **FB-06** | Amazon (`AMZN`) | `AMAZON_2017_10K` (p.37) | YoY Revenue Growth (FY16 to FY17) | `30.8%` | Longitudinal Trend |
| **FB-07** | Adobe (`ADBE`) | `ADOBE_2016_10K` (p.61) | Operating Income Growth (FY15 to FY16) | `65.4%` | Longitudinal Trend |
| **FB-08** | Microsoft (`MSFT`) | `MICROSOFT_2023_10K` (p.59) | Balance Sheet Debt Change (FY22 vs FY23) | Decreased by `$2.5B` | Longitudinal Trend |
| **FB-09** | Walmart (`WMT`) | `WALMART_2019_10K` (p.47) | Operating Margin Delta (FY18 to FY19) | `+0.2%` (20 bps) | Longitudinal Trend |
| **FB-10** | AMD (`AMD`) | `AMD_2022_10K` (p.55) | FY2022 Quick Ratio & Liquidity Profile | `1.57` (Healthy buffer) | Financial Ratio Math |
| **FB-11** | Adobe (`ADBE`) | `ADOBE_2015_10K` (p.58) | FY2015 Operating Cash Flow Ratio | `0.66` | Financial Ratio Math |
| **FB-12** | Amazon (`AMZN`) | `AMAZON_2017_10K` (p.37) | FY2017 Days Payable Outstanding (DPO) | `93.86 days` | Financial Ratio Math |
| **FB-13** | Johnson & Johnson (`JNJ`) | `JOHNSON_JOHNSON_2022_10K` (p.45) | FY2022 Inventory Turnover Ratio | `2.7 times` | Financial Ratio Math |
| **FB-14** | General Mills (`GIS`) | `GENERALMILLS_2019_10K` (p.52) | FY2019 Cash Conversion Cycle (CCC) | `-3.7 days` | Financial Ratio Math |
| **FB-15** | Boeing (`BA`) | `BOEING_2022_10K` (p.61) | Revenue Concentration by Segment >20% | Commercial (39%), Defense (35%), Services (26%) | Footnote & Segment |
| **FB-16** | Johnson & Johnson (`JNJ`) | `JOHNSON_JOHNSON_2022_10K` (p.33) | FY2022 Gross Margin Contraction Drivers | Vaccine exit, FX drag, inflation | Footnote & MD&A |
| **FB-17** | Boeing (`BA`) | `BOEING_2022_10K` (p.112) | FY2022 Material Legal Contingencies | 737 MAX Crash Lawsuits | Footnote & Disclosures |
| **FB-18** | 3M (`MMM`) | `3M_2022_10K` (p.22) | FY2022 Capital Intensity Assessment | CapEx/Rev 5.1%, Fixed Assets/Assets 20% | Domain Reasoning |
| **FB-19** | PepsiCo vs Coca-Cola | `PEPSICO_2021_10K` / `COCACOLA_2021_10K` | FY2021 CapEx & Operating Margin | PEP: $4,625M / 14.1% vs KO: $1,367M / 26.7% | Peer Comparison |
| **FB-20** | Apple vs Microsoft | `APPLE_2022_10K` / `MICROSOFT_2022_10K` | FY2022 R&D Spending & Intensity | AAPL: $26.25B (6.66%) vs MSFT: $24.51B (12.36%) | Peer Comparison |

---

## 🚀 How to Execute Queries from this Suite

### 1. Through CLI Pipeline:
```powershell
# Run Question FB-01 (3M CapEx)
.venv\Scripts\python.exe scripts/ingest_financebench.py --query-only --query "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement." --use-agent

# Run Question FB-10 (AMD Quick Ratio)
.venv\Scripts\python.exe scripts/ingest_financebench.py --query-only --query "Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22?" --use-agent
```

### 2. Through FastAPI REST Endpoint:
Send a `POST` request to `http://127.0.0.1:8000/api/v1/financial-agent/query`:
```json
{
  "query": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement."
}
```
