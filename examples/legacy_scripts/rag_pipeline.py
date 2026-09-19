#!/usr/bin/env python3
"""LangChain RAG Pipeline with Source Citations & Document Attribution.

Features:
  1. Full RAG pipeline with automatic source tracking and citation grounding.
  2. Numbered inline citations (e.g., [1], [2]) pointing to exact source documents.
  3. Structured output format returning:
     - 'question': Original user question
     - 'answer': Generated answer with inline bracketed citations
     - 'sources': Structured list of cited files, line/page references, and previews
     - 'context': Clean formatted multi-document numbered context string
  4. LangChain Expression Language (LCEL) chain integration via create_rag_with_sources_chain.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src directory is in sys.path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
from langchain_rag.rag_pipeline import (
    RAGPipeline,
    create_rag_pipeline,
    create_rag_with_sources_chain,
    format_docs_with_sources,
    rag_with_sources,
)

load_dotenv()

SAMPLE_KNOWLEDGE_BASE = [
    {
        "text": "Thinking Machines: Inkling is the default high-speed language model used in this LangChain RAG architecture.",
        "metadata": {"source": "architecture_guide.md", "topic": "models", "section": "1.1"},
    },
    {
        "text": "DocumentLoader automatically handles ingestion for Markdown (.md), Plain Text (.txt), PDF (.pdf), and Web URLs.",
        "metadata": {"source": "loader_specs.md", "topic": "loaders", "section": "2.0"},
    },
    {
        "text": "VectorStore leverages ChromaDB powered by local ONNX all-MiniLM-L6-v2 embeddings for fast, free similarity search.",
        "metadata": {"source": "vectorstore_specs.md", "topic": "storage", "section": "3.2"},
    },
    {
        "text": "ParentDocumentRetriever splits large documents into small child vectors for search accuracy while returning full parent context to the LLM.",
        "metadata": {"source": "retrieval_strategies.md", "topic": "advanced_rag", "section": "4.1"},
    },
    {
        "text": "ContextualCompressionRetriever uses LLMChainExtractor to strip out irrelevant noise from retrieved candidate documents.",
        "metadata": {"source": "retrieval_strategies.md", "topic": "advanced_rag", "section": "4.2"},
    },
]


def demo_rag_with_sources(question: str = "What models and vector store does this project use?") -> None:
    """Demonstrate RAG with sources, inline citations, and metadata attribution."""
    print("=" * 72)
    print("      LANGCHAIN RAG PIPELINE - RAG WITH SOURCE CITATIONS      ".center(72))
    print("=" * 72)

    # 1. Initialize RAG pipeline
    pipeline = create_rag_pipeline(collection_name="demo_rag_sources_kb")

    # 2. Ingest sample documents with metadata
    print("\n[Step 1] Ingesting knowledge documents with rich source metadata...")
    texts = [item["text"] for item in SAMPLE_KNOWLEDGE_BASE]
    metadatas = [item["metadata"] for item in SAMPLE_KNOWLEDGE_BASE]
    pipeline.index_texts(texts, metadatas=metadatas)
    print(f"✔ Indexed {len(texts)} documents into ChromaDB.")

    # 3. Query with Sources
    print(f"\n[Step 2] Executing query_with_sources(question='{question}', k=3)...")
    result = pipeline.query_with_sources(question, k=3)

    # 4. Display Formatted Context
    print("\n" + "-" * 72)
    print(" [Formatted Numbered Context Fed to LLM]")
    print("-" * 72)
    print(result.get("context", "N/A"))

    # 5. Display Grounded Answer
    print("\n" + "-" * 72)
    print(" [Grounded LLM Answer with Citations]")
    print("-" * 72)
    print(result.get("answer", "N/A"))

    # 6. Display Structured Citations Table
    print("\n" + "-" * 72)
    print(" [Source Citations Breakdown]")
    print("-" * 72)
    sources = result.get("sources") or []
    for s in sources:
        citation_tag = s.get("citation", "[?]")
        source_name = s.get("source", "unknown")
        meta = s.get("metadata") or {}
        topic = meta.get("topic", "general")
        preview = s.get("content_preview", "")
        print(f"  {citation_tag:<5} | File: {source_name:<24} | Topic: {topic:<12}")
        print(f"        Snippet: \"{preview}\"\n")

    print("=" * 72)
    print(" [✔] RAG with Sources Executed Successfully!".center(72))
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain RAG Pipeline with Sources & Inline Citations")
    parser.add_argument("-q", "--query", type=str, help="Search query to run against RAG pipeline")
    parser.add_argument("-k", type=int, default=3, help="Number of context sources to retrieve (default: 3)")
    parser.add_argument("-f", "--file", type=str, help="Optional document file to index before querying")

    args = parser.parse_args()

    if args.query or args.file:
        pipeline = create_rag_pipeline(collection_name="cli_rag_sources_kb")
        if args.file:
            path = Path(args.file)
            if path.exists():
                print(f"Indexing '{path.name}'...")
                pipeline.index_file(path)
            else:
                print(f"Error: File '{path}' not found.")
                sys.exit(1)
        else:
            texts = [item["text"] for item in SAMPLE_KNOWLEDGE_BASE]
            metadatas = [item["metadata"] for item in SAMPLE_KNOWLEDGE_BASE]
            pipeline.index_texts(texts, metadatas=metadatas)

        q = args.query or "What are the supported document loaders and storage backends?"
        demo_rag_with_sources(question=q)
    else:
        demo_rag_with_sources()


if __name__ == "__main__":
    main()
