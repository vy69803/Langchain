# FinanceBench Multimodal & Financial RAG Integration

## Overview

FinanceBench is an open-source financial benchmark dataset curated by Patronus AI, specifically designed to test the financial and layout understanding capabilities of LLMs and RAG systems with real-world enterprise financial filings packed with:
- **Complex Financial Statements**: Balance Sheets, Income Statements, Consolidated Statements of Cash Flows, and Stockholders' Equity.
- **Multicolumn Tables & Nested Rows**: Capital expenditures, segment breakdowns, credit risk disclosures, and footnote adjustments.
- **Rich Document Metadata**: GICS industry sectors, filing dates, SEC EDGAR links, and company tickers across 40 leading corporations.

---

## Dataset Layout

The dataset files are stored in `data/financebench/`:

```
data/financebench/
├── pdfs/                                    # 368 full SEC filing PDFs (~672 MB)
│   ├── 3M_2015_10K.pdf
│   ├── 3M_2018_10K.pdf
│   ├── AMAZON_2019_10K.pdf
│   ├── APPLE_2022_10K.pdf
│   ├── MICROSOFT_2023_10K.pdf
│   └── ...
├── financebench_document_information.jsonl  # Metadata for 361 SEC filings
├── financebench_open_source.jsonl           # 150 gold-standard QA evaluation pairs
└── evaluation_playground.ipynb              # Patronus AI evaluation notebook
```

---

## Pipeline Architecture

```
  ┌────────────────────────────────────────────────────────┐
  │                 FinanceBench SEC PDFs                  │
  │            (10-K, 10-Q, 8-K, Earnings)                 │
  └───────────────────────────┬────────────────────────────┘
                              │
                    pypdf / Layout Parser
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │                 FinanceBenchParser                     │
  │  - Statement section detection (Balance Sheet, CF)     │
  │  - Numerical table detection ($ amounts, %, commas)    │
  │  - Contextual breadcrumb injection:                    │
  │    "[Company: 3M | Filing: 3M_2018_10K | Page 59]"    │
  └───────────────────────────┬────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
  ┌───────────────────────┐       ┌───────────────────────┐
  │ Dense Vector Index    │       │ Sparse Lexical Index  │
  │ (ChromaDB)            │       │ (Okapi BM25)          │
  └───────────┬───────────┘       └───────────┬───────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │             Reciprocal Rank Fusion (RRF)               │
  └───────────────────────────┬────────────────────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │        Two-Stage Reranking (FlashRank Cross-Encoder)   │
  └───────────────────────────┬────────────────────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │            Top-K Financially Grounded Context          │
  └────────────────────────────────────────────────────────┘
```

---

## CLI Usage

### 1. Ingestion (`scripts/ingest_financebench.py`)

```bash
# Dry-run test on a 3-file sample
python scripts/ingest_financebench.py --dry-run --sample 3

# Ingest filings for a specific company (e.g. 3M)
python scripts/ingest_financebench.py --company 3M

# Ingest 10 documents and test query retrieval immediately
python scripts/ingest_financebench.py --sample 10 --query "What is 3M's capital expenditure in 2018?"

# Full production ingestion across all 368 filings
python scripts/ingest_financebench.py
```

### 2. Evaluation Benchmark (`scripts/evaluate_financebench.py`)

Run retrieval benchmark against the 150 gold-standard questions:

```bash
# Quick test on first 10 questions
python scripts/evaluate_financebench.py --sample 10

# Full benchmark evaluation
python scripts/evaluate_financebench.py --top-k 5 --search-type hybrid --rerank
```

Metrics tracked:
- **Doc Hit@K**: Target SEC filing document retrieved in Top-K.
- **Page Hit@K**: Exact evidence page (`evidence_page_num`) retrieved in Top-K.
- **Page Evidence MRR**: Mean Reciprocal Rank of the gold-standard evidence page.
