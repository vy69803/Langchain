import os
import sys
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_rag.llm import get_llm
from langchain_rag.document_loader import (
    DocumentLoader,
    load_directory,
    load_document,
    load_text,
    load_url,
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
    "load_document",
    "load_directory",
    "load_text",
    "load_url",
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
    "RAGMetricsTracker",
]
