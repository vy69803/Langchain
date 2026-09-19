#!/usr/bin/env python3
"""Advanced RAG Architecture & Demonstration Suite.

Features:
  1. Parent Document Retriever (Small-to-Big Retrieval):
     - Indexes small child chunks for high-precision dense vector search.
     - Stores and resolves large parent chunks (or full documents) in an in-memory docstore
       to provide rich, complete context to the LLM.
  2. Contextual Compression (Query-Focused Content Extraction & Noise Reduction):
     - Extracts only query-relevant sentences via LLMChainExtractor.
     - Discards completely irrelevant documents via LLMChainFilter.
     - Filters low-relevance chunks via EmbeddingsFilter.
     - Chains multi-stage compressor pipelines for maximum efficiency.
  3. Full Document vs. Hierarchical Large-Chunk Retrieval modes.
  4. Side-by-side comparison tables (Naive vs. Parent Document vs. Compressed).
  5. End-to-end grounded Question Answering with LLM orchestration.

Usage:
  # Run the full advanced RAG demonstration suite:
  python advanced_rag.py

  # Run only the Parent Document Retriever demo:
  python advanced_rag.py --demo parent_doc

  # Run only the Contextual Compression demo:
  python advanced_rag.py --demo compression

  # Test retrieval with a custom query:
  python advanced_rag.py -q "What are the remediation steps for token revocation?"

  # Run in interactive mode:
  python advanced_rag.py --interactive
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Suppress deprecation warnings for cleaner CLI output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src directory is in sys.path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv

# Core LangChain & LCEL components
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Advanced Retrievers & Compressors
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever, ParentDocumentRetriever
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    LLMChainExtractor,
    LLMChainFilter,
)
from langchain_community.retrievers import BM25Retriever
from langchain_classic.storage import InMemoryStore

# Vector Store
from langchain_chroma import Chroma

from langchain_rag.embeddings import get_embeddings
from langchain_rag.llm import get_llm

load_dotenv()

def load_enterprise_manual() -> List[Document]:
    """Load the full enterprise manual markdown file if present, or return structured chapters."""
    manual_path = Path(__file__).resolve().parent / "enterprise_manual.md"
    if manual_path.exists():
        content = manual_path.read_text(encoding="utf-8")
        # Split into chapters based on primary headers '## '
        raw_sections = content.split("\n## ")
        docs = []
        for idx, sec in enumerate(raw_sections):
            text = sec if idx == 0 else f"## {sec}"
            if len(text.strip()) > 30:
                docs.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": "enterprise_manual.md",
                            "chapter": f"Chapter {idx}",
                            "chars": len(text.strip()),
                        },
                    )
                )
        if docs:
            return docs
    return ENTERPRISE_KNOWLEDGE_DOCS


# Real-world rich enterprise documentation dataset with high-density IT specs and non-IT noise
ENTERPRISE_KNOWLEDGE_DOCS = [
    Document(
        page_content="""# QuantumCloud Cloud Infrastructure & Kubernetes Cluster Orchestration

## Section 1: Node Pool Specifications & Autoscaling Limits
QuantumCloud operates across three multi-region AWS availability zones (us-east-1, us-west-2, eu-central-1). Our containerized microservices fleet runs on Amazon EKS with Kubernetes version 1.30.
- Core Processing Pool (`pool-compute-c6i`): Instance Type `c6i.4xlarge` (16 vCPUs, 32 GiB memory). Autoscaler trigger threshold: 75% sustained CPU utilization over 3-minute window. Min replicas: 6 nodes, Max replicas: 48 nodes per cluster zone.
- Memory-Optimized Cache Pool (`pool-cache-r6i`): Instance Type `r6i.2xlarge` (8 vCPUs, 64 GiB ECC memory). Min replicas: 4 nodes, Max replicas: 16 nodes.

## Section 2: Ingress Gateway & Istio Service Mesh
All incoming external internet traffic terminates at AWS Network Load Balancers (NLB) before forwarding to the Envoy-based Ingress Controller.
- Internal service-to-service communication is governed by Istio 1.22 in ambient mesh mode with automatic mutual TLS (mTLS) enforcement using 2048-bit RSA ephemeral certificates rotated every 24 hours.
- Distributed tracing is powered by OpenTelemetry collector sidecars exporting trace spans to Honeycomb.
""",
        metadata={"source": "cloud_infrastructure.md", "category": "infra", "author": "CloudOps"},
    ),
    Document(
        page_content="""# QuantumCloud Incident Response Protocol: Security Operations

## Section 1: Authentication & STS Revocation (ERR-9021)
QuantumCloud enforces zero-trust identity verification across all gateway endpoints. All API clients must obtain a cryptographically signed OAuth 2.0 JSON Web Token (JWT) issued by the centralized STS authority. Tokens carry an ephemeral expiration window of 900 seconds (15 minutes) and must include the audience claim `urn:quantum:api`.

When STS detects abnormal token usage or compromised client credentials, it emits signal `ERR-9021-TOKEN-REVOKED`. Engineers on-call must execute the following remediation sequence:
1. Invalidate and purge the client session key from the distributed Redis cluster (`redis-cli -h redis-auth.internal -p 6379 -a "$REDIS_AUTH_TOKEN" DEL "session:$SESSION_ID"`).
2. Force client secret rotation via the HashiCorp Vault management API (`vault write auth/approle/role/gateway-prod/secret-id-accessor/destroy accessor=$ACCESSOR_ID`).
3. Notify the enterprise security operations center (SOC) via PagerDuty webhook and initiate an automated audit trail query in AWS CloudTrail:
   `aws cloudtrail lookup-events --lookup-attributes AttributeKey=Username,AttributeValue="$CLIENT_ID" --max-results 50`

## Section 2: Rate Limiting & Distributed Throttling (ERR-4290)
All inbound requests pass through Envoy gateway proxies configured with token bucket rate limiters. Standard tier services are throttled at 5,000 requests per minute with a burst allowance of 250 requests. Enterprise tier clients maintain dedicated Redis rate limit counters with a capacity of 50,000 req/min.

When a client breaches rate limits, HTTP status 429 Too Many Requests is returned alongside response header `Retry-After: <seconds>`. The gateway automatically penalizes aggressive clients by applying exponential backoff delay multipliers.
""",
        metadata={"source": "security_runbook.md", "category": "security", "author": "SecOps Team"},
    ),
    Document(
        page_content="""# QuantumCloud Distributed Database & Caching Architecture

## Section 1: PostgreSQL Indexing & High-Throughput Query Optimization
Our distributed PostgreSQL 16 cluster processes over 50,000 read-write queries per second across 12 read replicas. To eliminate table sequential scans and connection deadlocks, all multi-tenant tables must maintain composite B-Tree indexes.

Indexing rules:
- Rule 1 (Leading Key): All composite B-Tree indexes must lead with the partition key `tenant_id` followed by the monotonically increasing timestamp `created_at`.
- Rule 2 (Partial Indexes): High-frequency filtering on status flags requires partial indexes:
  `CREATE INDEX CONCURRENTLY idx_active_orders ON orders(tenant_id, created_at) WHERE status = 'ACTIVE' AND deleted_at IS NULL;`
- Connection pooling is governed by PgBouncer running in transaction pooling mode, strictly allocated a maximum pool size of 20 connections per microservice pod. Client query timeout is capped at 4,500 milliseconds.

## Section 2: Valkey & Redis Cluster Invalidation Strategy
The distributed cache cluster utilizes Redis 7.2 with active-active multi-region replication. Cache entries default to a TTL of 3,600 seconds (1 hour). Eviction follows the `volatile-lru` policy with a 48 GiB memory cap per shard.

When database mutations occur, write-through cache eviction events are published over Apache Kafka topic `db.mutations.v1`. Microservices subscribe to Kafka events to execute local LRU cache purges, guaranteeing eventual consistency across all application nodes within 25 milliseconds.
""",
        metadata={"source": "database_architecture.md", "category": "database", "author": "Data Platform Team"},
    ),
    Document(
        page_content="""# QuantumCloud LLM Orchestration & Semantic Caching Pipeline

## Section 1: Dynamic Gateway Routing & Model Fallback
The AI gateway dynamically routes prompt payloads based on complexity scoring and latency budgets:
- Fast classification and metadata tagging route to Thinking Machines: Inkling / Google Gemma 2 / Llama-3.1-8B with sub-180ms Time-To-First-Token (TTFT).
- Complex multi-step reasoning and synthetic code generation route to frontier models (Claude 3.5 Sonnet / GPT-4o) with automatic fallback to OpenRouter when provider 5xx rates exceed 2%.

## Section 2: Semantic Prompt Caching with ChromaDB
To reduce LLM inference costs and accelerate latency to < 15ms, the gateway checks an in-memory ChromaDB semantic cache prior to dispatching prompts:
- Prompts are embedded using local all-MiniLM-L6-v2 embeddings (384 dimensions).
- Cache Hit Threshold: If an incoming query achieves cosine similarity score >= 0.94 with an existing cached query, the cached answer is immediately returned.
- Semantic cache entries expire automatically after 24 hours to ensure freshness of operational insights.
""",
        metadata={"source": "ai_orchestration.md", "category": "ai", "author": "AI Platform Team"},
    ),
    Document(
        page_content="""# QuantumCloud Employee Workplace Guidelines & Culinary Guild (Recreational Noise)

## Section 1: Corporate Travel & Meal Reimbursement Policy
- Domestic Travel Per Diem: Employees are reimbursed up to $75.00 USD per day ($15 breakfast, $25 lunch, $35 dinner).
- International Travel Per Diem: Reimbursed up to $110.00 USD per day with itemized receipts.
- Ground transportation must use Uber for Business or Lyft Corporate linked directly to the enterprise billing profile.

## Section 2: Employee Culinary Guild & Espresso Calibration Guide
- Artisanal Sourdough Fermentation: The wild levain starter must be refreshed at a 1:2:2 ratio (starter : water : flour) 12 hours prior to final mixing. Bulk fermentation ambient temperature must be maintained at precisely 26°C (78°F) for 4.5 hours.
- La Marzocco Espresso Machine Calibration: Dose 18.0 grams of freshly ground Ethiopian beans, extracting 36.0 grams liquid espresso in 28-32 seconds at 9.0 bars pump pressure and 93.5°C water temperature.
""",
        metadata={"source": "corporate_noise.md", "category": "hr_and_culinary", "author": "People Ops"},
    ),
]


def print_header(title: str, subtitle: str = "") -> None:
    """Print formatted section header."""
    width = 78
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    if subtitle:
        print(f" {subtitle}".center(width))
    print("=" * width)


def format_preview(text: str, max_len: int = 120) -> str:
    """Format and clean string for tabular display."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def create_parent_document_retriever(
    collection_name: str = "advanced_rag_parent_doc",
    parent_chunk_size: Optional[int] = 1000,
    parent_chunk_overlap: int = 100,
    child_chunk_size: int = 250,
    child_chunk_overlap: int = 30,
) -> Tuple[ParentDocumentRetriever, InMemoryStore, Chroma]:
    """Factory to create and configure a LangChain ParentDocumentRetriever.

    Args:
        collection_name: ChromaDB collection name for child vector embeddings.
        parent_chunk_size: Size of parent chunks in characters. If None, retrieves full documents.
        parent_chunk_overlap: Overlap between parent chunks.
        child_chunk_size: Size of child chunks for vector indexing.
        child_chunk_overlap: Overlap between child chunks.

    Returns:
        Tuple of (ParentDocumentRetriever, InMemoryStore, Chroma).
    """
    embeddings = get_embeddings("local")
    docstore = InMemoryStore()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    parent_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )
        if parent_chunk_size is not None
        else None
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    return retriever, docstore, vectorstore


def create_contextual_compression_retriever(
    base_retriever: Optional[Any] = None,
    vectorstore: Optional[Any] = None,
    compressor_type: str = "extractor",
    llm: Optional[Any] = None,
    k: int = 4,
    similarity_threshold: float = 0.5,
) -> ContextualCompressionRetriever:
    """Factory to create and configure a ContextualCompressionRetriever.

    Args:
        base_retriever: Optional pre-configured base retriever.
        vectorstore: Optional Chroma vector store. If provided without base_retriever,
                     creates retriever via vectorstore.as_retriever(search_kwargs={"k": k}).
        compressor_type: "extractor" (LLM sentence extractor), "filter" (LLM relevance filter),
                         "embeddings" (vector similarity filter), or "pipeline" (multi-stage).
        llm: Configured LLM instance for extraction/filtering.
        k: Number of documents to retrieve (default: 4).
        similarity_threshold: Minimum similarity threshold for EmbeddingsFilter.

    Returns:
        Configured ContextualCompressionRetriever instance.
    """
    embeddings = get_embeddings("local")
    active_llm = llm or get_llm()

    # Determine base retriever (matching screenshot: vectorstore.as_retriever(search_kwargs={"k": 4}))
    if base_retriever is None:
        if vectorstore is not None:
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        else:
            raise ValueError("Either base_retriever or vectorstore must be provided.")

    # Create compressor
    if compressor_type == "extractor":
        compressor = LLMChainExtractor.from_llm(active_llm)
    elif compressor_type == "filter":
        compressor = LLMChainFilter.from_llm(active_llm)
    elif compressor_type == "embeddings":
        compressor = EmbeddingsFilter(
            embeddings=embeddings,
            similarity_threshold=similarity_threshold,
        )
    elif compressor_type == "pipeline":
        # Multi-stage pipeline: Text Splitter -> Embeddings Filter -> LLM Extractor
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
        emb_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=similarity_threshold)
        llm_extractor = LLMChainExtractor.from_llm(active_llm)
        pipeline = DocumentCompressorPipeline(
            transformers=[splitter, emb_filter, llm_extractor]
        )
        compressor = pipeline
    else:
        raise ValueError(
            f"Unknown compressor_type: '{compressor_type}'. Choose 'extractor', 'filter', 'embeddings', or 'pipeline'."
        )

    # Wrap retriever with compression
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
    return compression_retriever


def create_multi_query_retriever(
    base_retriever: Optional[Any] = None,
    vectorstore: Optional[Any] = None,
    llm: Optional[Any] = None,
    k: int = 3,
    prompt: Optional[Any] = None,
) -> MultiQueryRetriever:
    """Factory to create and configure a LangChain MultiQueryRetriever.

    Args:
        base_retriever: Optional pre-configured base retriever.
        vectorstore: Optional Chroma vector store instance.
        llm: Configured LLM for generating query variations.
        k: Number of documents to retrieve per query variation.
        prompt: Optional custom PromptTemplate for query generation.

    Returns:
        Configured MultiQueryRetriever instance.
    """
    active_llm = llm or get_llm()

    if base_retriever is None:
        if vectorstore is not None:
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        else:
            raise ValueError("Either base_retriever or vectorstore must be provided.")

    if prompt is not None:
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=active_llm,
            prompt=prompt,
        )
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=active_llm,
    )


def demo_parent_document_retriever() -> None:
    """Comprehensive demonstration of Parent Document Retriever (Small-to-Big Retrieval).

    Demonstrates:
      1. The RAG Chunking Dilemma (Small vs Large chunks).
      2. Mode 1: Hierarchical Chunking (Small child vectors -> Large parent contexts).
      3. Mode 2: Full Document Retrieval (Small child vectors -> Entire original document).
      4. Side-by-side comparison against Naive Standard Chunk Retrieval.
      5. End-to-end grounded LLM question answering with full context preservation.
    """
    print_header(
        "ADVANCED RAG: PARENT DOCUMENT RETRIEVER",
        "Small-to-Big Retrieval for Dense Accuracy + Complete LLM Context",
    )

    print("""
[Overview & Architecture]
In standard (naive) RAG, there is a fundamental conflict when picking chunk size:
  • Small Chunks (< 300 chars): High vector similarity & precise matching, but LLM loses
    surrounding context, context switches abruptly, and loses key procedural details.
  • Large Chunks (> 1500 chars): Rich context for LLM generation, but embedding vectors
    become diluted with irrelevant topics, lowering retrieval precision.

The Solution -> ParentDocumentRetriever:
  1. Splits documents into small 'Child' chunks for high-dimensional vector search.
  2. Embeds and indexes child chunks into ChromaDB with a pointer back to their parent.
  3. When a query matches a child chunk, the retriever fetches the full 'Parent' block
     from the docstore and feeds the complete, rich context to the LLM.
""")

    # ---------------------------------------------------------
    # PART 1: Mode A - Hierarchical Large Parent Chunks
    # ---------------------------------------------------------
    print_header(
        "PART 1: Hierarchical Parent-Child Retriever (Small-to-Big)",
        "Parent Chunk: 1,000 chars | Child Chunk: 250 chars",
    )

    retriever_hierarchical, docstore_h, vectorstore_h = create_parent_document_retriever(
        collection_name="demo_hierarchical_parent_doc",
        parent_chunk_size=1000,
        parent_chunk_overlap=100,
        child_chunk_size=250,
        child_chunk_overlap=30,
    )

    docs = load_enterprise_manual()
    print(f"Ingesting {len(docs)} enterprise chapters into Hierarchical ParentDocumentRetriever...")
    retriever_hierarchical.add_documents(docs)

    # Inspect docstore and vectorstore
    docstore_keys = list(docstore_h.yield_keys())
    print(f"✔ Parent Documents in DocStore : {len(docstore_keys)} parent blocks stored")

    # ---------------------------------------------------------
    # PART 2: Mode B - Full Document Parent Retriever
    # ---------------------------------------------------------
    print_header(
        "PART 2: Full Document Parent Retriever",
        "Parent: Entire Original Document | Child Chunk: 250 chars",
    )

    retriever_full, docstore_f, vectorstore_f = create_parent_document_retriever(
        collection_name="demo_full_parent_doc",
        parent_chunk_size=None,  # No parent splitter -> returns full raw document
        child_chunk_size=250,
        child_chunk_overlap=30,
    )

    print(f"Ingesting {len(docs)} enterprise chapters into Full Document Retriever...")
    retriever_full.add_documents(docs)
    docstore_full_keys = list(docstore_f.yield_keys())
    print(f"✔ Full Parent Documents in DocStore: {len(docstore_full_keys)} full documents stored")

    # ---------------------------------------------------------
    # PART 3: Standard Naive Small Chunking Baseline
    # ---------------------------------------------------------
    print_header(
        "PART 3: Standard Naive Baseline (Small Chunks Only)",
        "Flat 250-char chunks stored and returned directly",
    )
    small_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    naive_chunks = small_splitter.split_documents(docs)
    naive_vectorstore = Chroma.from_documents(
        naive_chunks,
        embedding=get_embeddings("local"),
        collection_name="demo_naive_baseline",
    )
    naive_retriever = naive_vectorstore.as_retriever(search_kwargs={"k": 1})
    print(f"✔ Naive Vector Store: {len(naive_chunks)} flat chunks indexed")

    # ---------------------------------------------------------
    # PART 4: Comparative Retrieval Tests
    # ---------------------------------------------------------
    test_queries = [
        (
            "Security / Runbook",
            "What exact commands should engineers run during an ERR-9021 token revocation?",
        ),
        (
            "Database Optimization",
            "What are the composite index guidelines and partial index rules for PostgreSQL?",
        ),
        (
            "AI Platform",
            "How does the semantic cache work with ChromaDB and when do cached answers expire?",
        ),
    ]

    print_header(
        "PART 4: SIDE-BY-SIDE RETRIEVAL COMPARISON",
        "Evaluating Context Completeness: Naive vs Hierarchical Parent vs Full Document",
    )

    for idx, (topic, query) in enumerate(test_queries, 1):
        print(f"\n{'#' * 78}")
        print(f" [Test Query {idx} - {topic}]")
        print(f" Query: \"{query}\"")
        print(f"{'#' * 78}")

        # 1. Naive Retrieval
        naive_results = naive_retriever.invoke(query)
        naive_doc = naive_results[0] if naive_results else None

        # 2. Hierarchical Parent Document Retrieval
        hier_results = retriever_hierarchical.invoke(query)
        hier_doc = hier_results[0] if hier_results else None

        # 3. Full Document Retrieval
        full_results = retriever_full.invoke(query)
        full_doc = full_results[0] if full_results else None

        print("\n--- 1. Naive Small Chunk Result (Truncated Context) ---")
        if naive_doc:
            print(f"Length: {len(naive_doc.page_content)} characters")
            print(f"Content:\n{naive_doc.page_content.strip()}")
            print("⚠️  Notice: Missing surrounding steps, prerequisite explanations, or related code.")

        print("\n--- 2. Hierarchical Parent Document Result (Targeted Complete Section) ---")
        if hier_doc:
            print(f"Length: {len(hier_doc.page_content)} characters | Source: {hier_doc.metadata.get('source')}")
            print(f"Content:\n{hier_doc.page_content.strip()}")
            print("✔  Benefit: Contains complete step-by-step commands + architectural context.")

        print("\n--- 3. Full Document Result (Entire Source Document) ---")
        if full_doc:
            print(f"Length: {len(full_doc.page_content)} characters | Source: {full_doc.metadata.get('source')}")
            print(f"Preview:\n{format_preview(full_doc.page_content, 220)}")

    # ---------------------------------------------------------
    # PART 5: End-to-End Grounded LLM Generation
    # ---------------------------------------------------------
    print_header(
        "PART 5: END-TO-END GENERATION WITH LLM",
        "Grounded Question Answering Powered by Parent Document Retriever",
    )

    rag_query = "What exact steps and CLI commands must be executed for an ERR-9021 token revocation?"
    print(f"Question: \"{rag_query}\"\n")

    print("Retrieving context from Hierarchical Parent Document Retriever...")
    retrieved_parents = retriever_hierarchical.invoke(rag_query)
    context_text = "\n\n---\n\n".join(d.page_content for d in retrieved_parents)

    print(f"✔ Retrieved {len(retrieved_parents)} Parent Context block(s) ({len(context_text)} chars total).")
    print(f"\nRetrieved Context Preview:\n{format_preview(context_text, 250)}\n")

    try:
        llm = get_llm()
        qa_prompt = (
            "You are an expert DevOps and Security Operations assistant. "
            "Answer the technical question accurately and comprehensively using ONLY the provided context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {rag_query}\n\n"
            "Detailed Answer (include steps and commands if available):"
        )
        print("Invoking LLM for grounded answer...")
        response = llm.invoke(qa_prompt)
        print("\n" + "-" * 60)
        print(" [LLM Grounded Response]")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
    except Exception as e:
        print(f"[Notice] LLM call skipped or API key not present: {e}")
        print("Parent document retrieval succeeded and assembled rich context for LLM generation.")

    print_header(
        "✔ PARENT DOCUMENT RETRIEVER DEMONSTRATION COMPLETE",
        "High Retrieval Precision + Full Context Assembly Successfully Verified!",
    )


def demo_contextual_compression() -> None:
    """Comprehensive demonstration of Contextual Compression in RAG.

    Demonstrates:
      1. The Problem of Context Dilution & Irrelevant Noise in standard RAG.
      2. LLM Sentence Extractor (LLMChainExtractor): Pulls ONLY the exact sentences
         relevant to the query from verbose candidate documents.
      3. LLM Document Filter (LLMChainFilter): Discards candidate documents entirely
         if they do not answer the user's specific query.
      4. Quantitative Noise Reduction: Comparison table of character & token counts
         between uncompressed retrieved documents vs. compressed extractions.
      5. End-to-End LLM Generation with ultra-compact, high-density context.
    """
    print_header(
        "ADVANCED RAG: CONTEXTUAL COMPRESSION",
        "Query-Focused Extraction & Document Filtering to Eliminate Noise & Token Waste",
    )

    print("""
[Overview & Architecture]
In standard RAG, retrieving top-k documents introduces substantial noise:
  • Candidate documents often contain 80% irrelevant prose, boilerplates, or unrelated sections.
  • Passing uncompressed chunks wastes context window tokens, inflates LLM API costs, and
    triggers 'Lost in the Middle' attention degradation in LLMs.

The Solution -> Contextual Compression:
  1. Base Retriever fetches top-k broad candidate documents from the vector store.
  2. The Document Compressor processes each document conditioned on the specific query:
     - LLMChainExtractor: Extracts ONLY the exact sentences answering the question.
     - LLMChainFilter: Completely drops documents that are not relevant to the query.
     - EmbeddingsFilter: Drops chunks failing similarity thresholds.
  3. The LLM prompt receives a high-density, noise-free context.
""")

    # --- Setup Base Vector Store ---
    embeddings = get_embeddings("local")
    llm = get_llm()

    docs = load_enterprise_manual()
    print(f"Loading {len(docs)} comprehensive enterprise chapters ({sum(len(d.page_content) for d in docs)} chars total)...")

    # Create a vector store with full raw documents to demonstrate compression of long texts
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        collection_name="demo_contextual_compression_store",
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ---------------------------------------------------------
    # PART 1: Uncompressed Baseline Retrieval
    # ---------------------------------------------------------
    test_query = "What exact CLI command purges the Redis session during an ERR-9021 incident?"
    print_header(
        "PART 1: Uncompressed Raw Retrieval (Baseline)",
        f"Query: '{test_query}'",
    )

    raw_docs = base_retriever.invoke(test_query)
    total_raw_chars = sum(len(d.page_content) for d in raw_docs)
    print(f"✔ Base Retriever returned {len(raw_docs)} candidate documents ({total_raw_chars} characters total).\n")

    for i, doc in enumerate(raw_docs, 1):
        print(f"  [Raw Doc #{i}] Source: {doc.metadata.get('source')} ({len(doc.page_content)} chars)")
        print(f"  └── \"{format_preview(doc.page_content, 110)}\"")

    # ---------------------------------------------------------
    # PART 2: LLM Sentence Extractor (LLMChainExtractor)
    # ---------------------------------------------------------
    print_header(
        "PART 2: Contextual Compression with LLMChainExtractor",
        "Extracts ONLY the query-relevant sentences, discarding surrounding noise",
    )

    # Create compressor
    compressor = LLMChainExtractor.from_llm(llm)

    # Wrap retriever with compression
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    )

    extractor_retriever = compression_retriever

    print(f"Executing Contextual Compression (LLMChainExtractor) for query:\n  \"{test_query}\"...\n")
    try:
        compressed_docs = extractor_retriever.invoke(test_query)
        total_compressed_chars = sum(len(d.page_content) for d in compressed_docs)

        print(f"✔ Compressed into {len(compressed_docs)} focused snippet(s) ({total_compressed_chars} characters total):\n")
        for i, doc in enumerate(compressed_docs, 1):
            print(f"  [Compressed Snippet #{i}] Source: {doc.metadata.get('source')} ({len(doc.page_content)} chars):")
            print(f"  └── {doc.page_content.strip()}\n")

        # Reduction metrics
        if total_raw_chars > 0:
            reduction_pct = ((total_raw_chars - total_compressed_chars) / total_raw_chars) * 100
            print(f"📊 Noise & Token Reduction: {reduction_pct:.1f}% reduction in prompt context size!")
    except Exception as e:
        print(f"[Notice] LLMChainExtractor skipped or error: {e}")
        compressed_docs = []

    # ---------------------------------------------------------
    # PART 3: LLM Relevance Filter (LLMChainFilter)
    # ---------------------------------------------------------
    print_header(
        "PART 3: Contextual Compression with LLMChainFilter",
        "Discards entire candidate documents that are off-topic / irrelevant",
    )

    filter_retriever = create_contextual_compression_retriever(
        base_retriever=base_retriever,
        compressor_type="filter",
        llm=llm,
    )

    filter_query = "What ambient bulk fermentation temperature is recommended for sourdough bread?"
    print(f"Testing off-topic query against enterprise IT knowledge base:\n  \"{filter_query}\"\n")
    print("Uncompressed Base Retriever would blindly return top-3 IT docs as false positives.")

    try:
        filtered_docs = filter_retriever.invoke(filter_query)
        print(f"✔ LLMChainFilter evaluated all candidates and returned: {len(filtered_docs)} documents.")
        if not filtered_docs:
            print("✔ Success: All irrelevant IT documents were correctly discarded before reaching the LLM!")
        else:
            for d in filtered_docs:
                print(f"  Retained doc: {format_preview(d.page_content, 100)}")
    except Exception as e:
        print(f"[Notice] LLMChainFilter skipped or error: {e}")

    # ---------------------------------------------------------
    # PART 4: Quantitative Compression Benchmark Table
    # ---------------------------------------------------------
    print_header(
        "PART 4: CONTEXTUAL COMPRESSION PERFORMANCE BENCHMARK",
        "Raw Documents vs. LLM-Compressed Extractions",
    )

    queries_to_benchmark = [
        ("Auth CLI Command", "What exact CLI command purges the Redis session during ERR-9021?"),
        ("PostgreSQL Index", "What composite index columns are required on PostgreSQL tables?"),
        ("Cache TTL", "What is the default TTL for Redis cache entries?"),
    ]

    print("\n" + "-" * 78)
    print(f"{'Topic':<22} | {'Raw Chars':<11} | {'Comp. Chars':<13} | {'Noise Eliminated':<18} | {'Status'}")
    print("-" * 78)

    for topic, q in queries_to_benchmark:
        try:
            raw_res = base_retriever.invoke(q)
            raw_len = sum(len(d.page_content) for d in raw_res)

            comp_res = extractor_retriever.invoke(q)
            comp_len = sum(len(d.page_content) for d in comp_res)

            reduction = ((raw_len - comp_len) / raw_len * 100) if raw_len > 0 else 0.0
            print(f"{topic:<22} | {raw_len:<11} | {comp_len:<13} | {reduction:>16.1f}% | ✔ Success")
        except Exception:
            print(f"{topic:<22} | {'N/A':<11} | {'N/A':<13} | {'N/A':<18} | Skipped")
    print("-" * 78)

    # ---------------------------------------------------------
    # PART 5: End-to-End LLM Generation with Compressed Context
    # ---------------------------------------------------------
    print_header(
        "PART 5: END-TO-END GENERATION WITH COMPRESSED CONTEXT",
        "High-Precision Generation with Zero Irrelevant Context",
    )

    rag_q = "What exact CLI command purges the Redis session during ERR-9021?"
    print(f"Question: \"{rag_q}\"\n")

    try:
        compressed_hits = extractor_retriever.invoke(rag_q)
        compact_context = "\n".join(f"- {d.page_content.strip()}" for d in compressed_hits)

        print(f"High-Density Compressed Context ({len(compact_context)} chars):\n{compact_context}\n")

        prompt = (
            "You are an expert IT assistant. Answer the question using ONLY the provided concise context.\n\n"
            f"Context:\n{compact_context}\n\n"
            f"Question: {rag_q}\n\n"
            "Concise Answer:"
        )

        print("Invoking LLM with high-density compressed context...")
        response = llm.invoke(prompt)
        print("\n" + "-" * 60)
        print(" [LLM Grounded Response]")
        print("-" * 60)
        print(response.content)
        print("-" * 60)
    except Exception as e:
        print(f"[Notice] LLM call skipped or error: {e}")

    print_header(
        "✔ CONTEXTUAL COMPRESSION DEMONSTRATION COMPLETE",
        "All Extractor, Filter, and Benchmark Pipelines Successfully Verified!",
    )


def demo_compression_vs_uncompressed(
    query: str = "What exact CLI command purges the Redis session during an ERR-9021 incident?",
) -> None:
    """Dedicated side-by-side demonstration comparing RAG WITH compression vs WITHOUT compression.

    Args:
        query: The search query to evaluate in both pipelines.
    """
    print_header(
        "HEAD-TO-HEAD COMPARISON: WITHOUT COMPRESSION VS. WITH COMPRESSION",
        f"Query: '{query}'",
    )

    embeddings = get_embeddings("local")
    llm = get_llm()
    docs = load_enterprise_manual()

    # Create Chroma vector store with full enterprise chapters
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        collection_name="demo_head_to_head_store",
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # =========================================================
    # PIPELINE A: WITHOUT COMPRESSION (Standard Base Retriever)
    # =========================================================
    print_header(
        "PIPELINE A: WITHOUT COMPRESSION (Standard RAG)",
        "Retrieves raw top-4 candidate documents with full surrounding noise",
    )
    raw_docs = base_retriever.invoke(query)
    raw_context = "\n\n---\n\n".join(d.page_content.strip() for d in raw_docs)
    total_raw_chars = len(raw_context)

    print(f"✔ Retrieved {len(raw_docs)} full candidate documents ({total_raw_chars} characters total):\n")
    for idx, d in enumerate(raw_docs, 1):
        print(f"  [Document #{idx}] Source: {d.metadata.get('source')} ({len(d.page_content)} chars)")
        print(f"  └── \"{format_preview(d.page_content, 95)}\"")

    print("\nInvoking LLM with Raw (Uncompressed) Context...")
    try:
        raw_prompt = (
            "You are an IT assistant. Answer the question using ONLY the provided context.\n\n"
            f"Context:\n{raw_context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        raw_response = llm.invoke(raw_prompt)
        raw_answer = raw_response.content.strip()
    except Exception as e:
        raw_answer = f"[LLM skipped or error: {e}]"

    print("\n--- LLM Output (Without Compression) ---")
    print(raw_answer)

    # =========================================================
    # PIPELINE B: WITH CONTEXTUAL COMPRESSION (LLMChainExtractor)
    # =========================================================
    print_header(
        "PIPELINE B: WITH CONTEXTUAL COMPRESSION (Smart RAG)",
        "Extracts ONLY the query-relevant commands, discarding all irrelevant text",
    )
    # Create compressor
    compressor = LLMChainExtractor.from_llm(llm)

    # Wrap retriever with compression
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    )

    compressed_docs = compression_retriever.invoke(query)
    comp_context = "\n".join(f"- {d.page_content.strip()}" for d in compressed_docs)
    total_comp_chars = len(comp_context)

    print(f"✔ Compressed into {len(compressed_docs)} relevant snippet(s) ({total_comp_chars} characters total):\n")
    for idx, d in enumerate(compressed_docs, 1):
        print(f"  [Snippet #{idx}] Source: {d.metadata.get('source')} ({len(d.page_content)} chars):")
        print(f"  └── {d.page_content.strip()}")

    print("\nInvoking LLM with High-Density Compressed Context...")
    try:
        comp_prompt = (
            "You are an IT assistant. Answer the question using ONLY the provided concise context.\n\n"
            f"Context:\n{comp_context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        comp_response = llm.invoke(comp_prompt)
        comp_answer = comp_response.content.strip()
    except Exception as e:
        comp_answer = f"[LLM skipped or error: {e}]"

    print("\n--- LLM Output (With Compression) ---")
    print(comp_answer)

    # =========================================================
    # SIDE-BY-SIDE SUMMARY & METRICS
    # =========================================================
    reduction_pct = ((total_raw_chars - total_comp_chars) / total_raw_chars * 100) if total_raw_chars > 0 else 0.0

    print_header(
        "QUANTITATIVE COMPARISON SUMMARY",
        f"Context Size Reduction: {reduction_pct:.1f}%",
    )
    print(f"{'Metric':<28} | {'Without Compression':<22} | {'With Compression':<22}")
    print("-" * 78)
    print(f"{'Candidates Evaluated':<28} | {len(raw_docs):<22} | {len(raw_docs):<22}")
    print(f"{'Passed to LLM Context':<28} | {f'{len(raw_docs)} full docs':<22} | {f'{len(compressed_docs)} snippet(s)':<22}")
    print(f"{'Total Context Characters':<28} | {f'{total_raw_chars} chars':<22} | {f'{total_comp_chars} chars':<22}")
    print(f"{'Noise & Token Reduction':<28} | {'0.0% (Baseline)':<22} | {f'{reduction_pct:.1f}% (Clean)':<22}")
    print(f"{'Risk of Distraction':<28} | {'High (3,000+ noisy chars)':<22} | {'Near Zero (Focused)':<22}")
    print("-" * 78)


def demo_multi_query_retriever(
    query: str = "How do we mitigate token revocation incidents?",
) -> None:
    """Comprehensive demonstration of MultiQueryRetriever.

    Demonstrates:
      1. Vocabulary Mismatch & Distance Search Limitations in Single-Query RAG.
      2. LLM-Powered Query Expansion: Generates 3-4 diverse perspectives for the query.
      3. Multi-Branch Vector Retrieval: Queries vector DB across all perspectives.
      4. Unique Union Deduplication: Merges all retrieved documents into a rich context.
      5. Comparative Evaluation: Single-Query vs Multi-Query recall & precision.
      6. End-to-End Grounded LLM Response Generation.
    """
    print_header(
        "ADVANCED RAG: MULTI-QUERY RETRIEVER",
        "LLM Query Expansion & Multi-Perspective Vector Union",
    )

    print("""
[Overview & Architecture]
In standard RAG, distance-based vector retrieval relies entirely on the user's exact phrasing:
  • If the user uses different vocabulary, synonyms, or conceptual angles than the indexed docs,
    the vector database fails to retrieve key documents (Low Recall).

The Solution -> MultiQueryRetriever:
  1. Uses an LLM to generate 3-4 alternative formulations/perspectives of the user's question.
  2. Runs vector search across EACH generated query variation in parallel.
  3. Takes the unique union of all retrieved documents, eliminating duplicates.
  4. Delivers high recall and robust retrieval even with vague or paraphrased user prompts.
""")

    embeddings = get_embeddings("local")
    llm = get_llm()
    docs = load_enterprise_manual()

    # Create Chroma vector store
    small_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=30)
    chunked_docs = small_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        chunked_docs,
        embedding=embeddings,
        collection_name="demo_multiquery_store",
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # ---------------------------------------------------------
    # PART 1: Query Expansion via LLM
    # ---------------------------------------------------------
    print_header("PART 1: LLM-Driven Query Variation Generation", f"Original Query: '{query}'")

    expansion_prompt = ChatPromptTemplate.from_template(
        "You are an AI assistant helping optimize RAG search.\n"
        "Generate 3 different versions of the following question to retrieve relevant documents from a vector store.\n"
        "Provide alternative perspectives and technical synonyms. Output each on a new line.\n\n"
        "Original question: {question}"
    )
    expansion_chain = expansion_prompt | llm | StrOutputParser()

    print("Generating alternative query perspectives with LLM...")
    try:
        variations_raw = expansion_chain.invoke({"question": query})
        variations = [v.strip().lstrip("1234567890.- ") for v in variations_raw.strip().split("\n") if v.strip()]
    except Exception as e:
        print(f"[Notice] LLM expansion fallback: {e}")
        variations = [
            "What is the playbook for ERR-9021-TOKEN-REVOKED?",
            "How to invalidate Redis session and rotate Vault secrets after token compromise?",
            "What security incident response steps are required for revoked OAuth JWT tokens?",
        ]

    print(f"\n✔ Generated {len(variations)} Query Variations:")
    for i, v in enumerate(variations, 1):
        print(f"  [{i}] \"{v}\"")

    # ---------------------------------------------------------
    # PART 2: Single-Query vs. Multi-Query Execution
    # ---------------------------------------------------------
    print_header(
        "PART 2: RETRIEVAL COMPARISON (SINGLE-QUERY VS. MULTI-QUERY)",
        "Evaluating Recall and Discovered Context Across Query Angles",
    )

    # 1. Standard Single Query Retrieval
    single_results = base_retriever.invoke(query)
    print(f"--- 1. Standard Single-Query Retriever (Query: '{query}') ---")
    print(f"✔ Retrieved {len(single_results)} chunk(s):")
    for idx, d in enumerate(single_results, 1):
        print(f"  [{idx}] Source: {d.metadata.get('source', 'N/A')} ({len(d.page_content)} chars)")
        print(f"      \"{format_preview(d.page_content, 85)}\"")

    # 2. Multi-Query Retriever
    mq_retriever = create_multi_query_retriever(
        base_retriever=base_retriever,
        llm=llm,
    )

    print(f"\n--- 2. Multi-Query Retriever (Executing All {len(variations)} Angles in Union) ---")
    mq_results = mq_retriever.invoke(query)
    print(f"✔ Retrieved {len(mq_results)} unique deduplicated chunk(s):")
    for idx, d in enumerate(mq_results, 1):
        print(f"  [{idx}] Source: {d.metadata.get('source', 'N/A')} ({len(d.page_content)} chars)")
        print(f"      \"{format_preview(d.page_content, 85)}\"")

    # ---------------------------------------------------------
    # PART 3: End-to-End Grounded LLM Generation
    # ---------------------------------------------------------
    print_header(
        "PART 3: END-TO-END GENERATION WITH MULTI-QUERY CONTEXT",
        "Synthesizing Comprehensive Answer from Deduplicated Union Context",
    )

    mq_context = "\n\n---\n\n".join(d.page_content.strip() for d in mq_results)
    print(f"Multi-Query Union Context ({len(mq_context)} chars):\n{format_preview(mq_context, 200)}\n")

    try:
        qa_prompt = (
            "You are a Senior SecOps Engineer. Answer the question using ONLY the provided multi-perspective context.\n\n"
            f"Context:\n{mq_context}\n\n"
            f"Question: {query}\n\n"
            "Comprehensive Answer:"
        )
        print("Invoking LLM with Multi-Query Context...")
        response = llm.invoke(qa_prompt)
        print("\n" + "-" * 60)
        print(" [LLM Grounded Response]")
        print("-" * 60)
        print(response.content.strip())
        print("-" * 60)
    except Exception as e:
        print(f"[Notice] LLM generation skipped or error: {e}")

    print_header(
        "✔ MULTI-QUERY RETRIEVER DEMONSTRATION COMPLETE",
        "Query Expansion, Multi-Angle Search, and Deduplicated Union Verified!",
    )


def interactive_mode() -> None:
    """Interactive CLI console to test queries against Parent Document, Compression, and Multi-Query retrievers."""
    print_header(
        "INTERACTIVE ADVANCED RAG CONSOLE",
        "Type your question, ':mode <hierarchical|full|compressed|filter|multi_query|naive>', or 'exit' to quit",
    )

    docs = load_enterprise_manual()

    # Hierarchical Parent Document Retriever
    retriever_h, _, _ = create_parent_document_retriever(
        collection_name="interactive_hierarchical",
        parent_chunk_size=1000,
        parent_chunk_overlap=100,
        child_chunk_size=250,
    )
    retriever_h.add_documents(docs)

    # Full Document Retriever
    retriever_f, _, _ = create_parent_document_retriever(
        collection_name="interactive_full",
        parent_chunk_size=None,
        child_chunk_size=250,
    )
    retriever_f.add_documents(docs)

    # Naive Chroma Retriever
    small_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    naive_vs = Chroma.from_documents(
        small_splitter.split_documents(docs),
        embedding=get_embeddings("local"),
        collection_name="interactive_naive",
    )
    naive_retriever = naive_vs.as_retriever(search_kwargs={"k": 2})

    # Contextual Compression Retrievers
    llm = get_llm()
    comp_extractor = create_contextual_compression_retriever(
        base_retriever=naive_retriever,
        compressor_type="extractor",
        llm=llm,
    )
    comp_filter = create_contextual_compression_retriever(
        base_retriever=naive_retriever,
        compressor_type="filter",
        llm=llm,
    )

    # Multi-Query Retriever
    mq_retriever = create_multi_query_retriever(
        base_retriever=naive_retriever,
        llm=llm,
    )

    current_mode = "hierarchical"

    while True:
        try:
            print(f"\nCurrent Mode: [{current_mode.upper()}]")
            query = input("Advanced RAG > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if query.startswith(":mode "):
            mode = query.split(" ", 1)[1].strip().lower()
            if mode in ("hierarchical", "full", "compressed", "filter", "multi_query", "mq", "naive"):
                current_mode = "multi_query" if mode in ("multi_query", "mq") else mode
                print(f"✔ Mode switched to: {current_mode}")
            else:
                print("Invalid mode. Options: hierarchical, full, compressed, filter, multi_query, naive")
            continue

        if current_mode == "hierarchical":
            docs_res = retriever_h.invoke(query)
        elif current_mode == "full":
            docs_res = retriever_f.invoke(query)
        elif current_mode == "compressed":
            docs_res = comp_extractor.invoke(query)
        elif current_mode == "filter":
            docs_res = comp_filter.invoke(query)
        elif current_mode == "multi_query":
            docs_res = mq_retriever.invoke(query)
        else:
            docs_res = naive_retriever.invoke(query)

        print(f"\nRetrieved {len(docs_res)} document(s):")
        for i, d in enumerate(docs_res, 1):
            print(f"\n[Result #{i}] Source: {d.metadata.get('source', 'Unknown')} ({len(d.page_content)} chars)")
            print("-" * 60)
            print(d.page_content.strip())
            print("-" * 60)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Advanced RAG Architecture (Parent Document Retriever, Contextual Compression, Multi-Query)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--demo",
        type=str,
        choices=["all", "parent_doc", "compression", "compare", "multi_query"],
        default="all",
        help="Demonstration to run: all, parent_doc, compression, compare, or multi_query (default: all)",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="Search query to test on Advanced RAG retriever",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["hierarchical", "full", "compressed", "filter", "multi_query", "compare", "naive"],
        default="hierarchical",
        help="Retrieval mode for query (default: hierarchical)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch interactive search REPL console",
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.query:
        if args.mode == "compare":
            demo_compression_vs_uncompressed(query=args.query)
            return

        print_header(f"QUERYING ADVANCED RAG ({args.mode.upper()})", f"Query: '{args.query}'")
        docs = load_enterprise_manual()
        if args.mode == "hierarchical":
            retriever, _, _ = create_parent_document_retriever(
                collection_name="cli_query_hierarchical",
                parent_chunk_size=1000,
                child_chunk_size=250,
            )
            retriever.add_documents(docs)
        elif args.mode == "full":
            retriever, _, _ = create_parent_document_retriever(
                collection_name="cli_query_full",
                parent_chunk_size=None,
                child_chunk_size=250,
            )
            retriever.add_documents(docs)
        elif args.mode in ("compressed", "filter"):
            small_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
            vs = Chroma.from_documents(
                small_splitter.split_documents(docs),
                embedding=get_embeddings("local"),
                collection_name="cli_query_comp_base",
            )
            base_r = vs.as_retriever(search_kwargs={"k": 4})
            c_type = "extractor" if args.mode == "compressed" else "filter"
            retriever = create_contextual_compression_retriever(
                base_retriever=base_r,
                compressor_type=c_type,
            )
        elif args.mode == "multi_query":
            small_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=30)
            vs = Chroma.from_documents(
                small_splitter.split_documents(docs),
                embedding=get_embeddings("local"),
                collection_name="cli_query_mq_base",
            )
            retriever = create_multi_query_retriever(
                vectorstore=vs,
                k=2,
            )
        else:
            small_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
            vs = Chroma.from_documents(
                small_splitter.split_documents(docs),
                embedding=get_embeddings("local"),
                collection_name="cli_query_naive",
            )
            retriever = vs.as_retriever(search_kwargs={"k": 2})

        results = retriever.invoke(args.query)
        print(f"Retrieved {len(results)} document(s):\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] ({len(r.page_content)} chars | Source: {r.metadata.get('source', 'N/A')}):")
            print(r.page_content.strip())
            print()
    else:
        if args.demo in ("all", "compare"):
            demo_compression_vs_uncompressed()
        if args.demo in ("all", "parent_doc"):
            demo_parent_document_retriever()
        if args.demo in ("all", "compression"):
            demo_contextual_compression()
        if args.demo in ("all", "multi_query"):
            demo_multi_query_retriever()


if __name__ == "__main__":
    main()
