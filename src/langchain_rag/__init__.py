import os
import sys
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_rag.llm import get_llm
from langchain_rag.document_loader import (
    DOCLING_SUPPORTED_EXTENSIONS,
    DocumentLoader,
    is_docling_available,
    load_directory,
    load_document,
    load_text,
    load_url,
    load_with_docling,
)
from langchain_rag.vector_stores import VectorStore, create_vector_store
from langchain_rag.rag_pipeline import (
    RAGPipeline,
    create_rag_pipeline,
    create_rag_with_sources_chain,
    format_docs_with_sources,
    rag_with_sources,
)
from langchain_rag.text_splitter import (
    TextSplitter,
    split_documents,
    split_text,
)
from langchain_rag.embeddings import (
    APIEmbeddings,
    LocalEmbeddings,
    calculate_similarity,
    cosine_similarity,
    get_embeddings,
)

from langchain_rag.hybrid_search import (
    BM25Index,
    HybridSearchEngine,
    create_hybrid_search_engine,
)
from langchain_rag.reranker import (
    BaseReranker,
    FlashRankReranker,
    get_reranker,
)
from langchain_rag.cost_optimization import (
    CostTracker,
    ModelTier,
    PromptCompressor,
    SemanticCache,
    TieredRouter,
    create_cost_optimizer,
)
from langchain_rag.semantic_chunking import (
    ChunkMetrics,
    ContextualMarkdownChunker,
    EmbeddingCache,
    ParentChildChunker,
    ProductionChunker,
    ProductionSemanticChunker,
    SentenceSplitter,
    create_production_chunker,
    smart_chunker,
)
from langchain_rag.cached_embeddings import (
    CacheBackedEmbeddings,
    LocalFileByteStore,
    SQLiteByteStore,
    create_cached_embeddings,
)
from langchain_rag.monitoring import (
    MetricsCallbackHandler,
    MetricsCollector,
    RAGMetricsTracker,
)
from langchain_rag.finance_parser import (
    FinanceBenchParser,
    detect_has_table,
    detect_section_header,
    load_financebench_metadata,
    load_financebench_qa,
)
from langchain_rag.finance_pipeline import (
    FinanceBenchIngestionPipeline,
)
from langchain_rag.query_expansion import (
    HyDERetriever,
    MultiQueryRetriever,
    QueryExpander,
    create_hyde_retriever,
    create_multi_query_retriever,
    create_query_expander,
    fuse_multiquery_results,
)
from langchain_rag.retrieval_metrics import (
    RetrievalBenchmarkEvaluator,
    average_precision_at_k,
    dcg_at_k,
    evaluate_query_retrieval,
    hit_rate_at_k,
    idcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


load_dotenv()


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        print(
            "Please set your OPENROUTER_API_KEY in the .env file before running.\n"
            "Get your key at: https://openrouter.ai/settings/keys"
        )
        return

    print("Initializing Thinking Machines: Inkling (free) via OpenRouter...")
    llm = get_llm()

    question = "What makes an AI agent architecture 'stateful'?"
    print(f"Prompt: {question}\n")
    print("Response:")
    response = llm.invoke(question)
    print(response.content)


__all__ = [
    "get_llm",
    "main",
    "DocumentLoader",
    "DOCLING_SUPPORTED_EXTENSIONS",
    "is_docling_available",
    "load_document",
    "load_directory",
    "load_text",
    "load_url",
    "load_with_docling",
    "VectorStore",
    "create_vector_store",
    "RAGPipeline",
    "create_rag_pipeline",
    "create_rag_with_sources_chain",
    "format_docs_with_sources",
    "rag_with_sources",
    "TextSplitter",
    "split_documents",
    "split_text",
    "LocalEmbeddings",
    "APIEmbeddings",
    "get_embeddings",
    "cosine_similarity",
    "calculate_similarity",
    "BM25Index",
    "HybridSearchEngine",
    "create_hybrid_search_engine",
    "BaseReranker",
    "FlashRankReranker",
    "get_reranker",
    "CostTracker",
    "ModelTier",
    "PromptCompressor",
    "SemanticCache",
    "TieredRouter",
    "create_cost_optimizer",
    "ChunkMetrics",
    "ContextualMarkdownChunker",
    "EmbeddingCache",
    "ParentChildChunker",
    "ProductionChunker",
    "ProductionSemanticChunker",
    "SentenceSplitter",
    "create_production_chunker",
    "smart_chunker",
    "CacheBackedEmbeddings",
    "LocalFileByteStore",
    "SQLiteByteStore",
    "create_cached_embeddings",
    "MetricsCollector",
    "MetricsCallbackHandler",
    "FinanceBenchParser",
    "FinanceBenchIngestionPipeline",
    "load_financebench_metadata",
    "load_financebench_qa",
    "detect_has_table",
    "detect_section_header",
    "QueryExpander",
    "MultiQueryRetriever",
    "HyDERetriever",
    "create_query_expander",
    "create_multi_query_retriever",
    "create_hyde_retriever",
    "fuse_multiquery_results",
    "precision_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "dcg_at_k",
    "idcg_at_k",
    "ndcg_at_k",
    "average_precision_at_k",
    "evaluate_query_retrieval",
    "RetrievalBenchmarkEvaluator",
]

