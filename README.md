# LangChain RAG with Thinking Machines: Inkling

This project uses the **Thinking Machines: Inkling (free)** model via [OpenRouter](https://openrouter.ai/thinkingmachines/inkling:free) using LangChain's `langchain-openai` integration.

## Setup

1. **Clone or navigate to the repository**:
   ```bash
   cd "Langchain"
   ```

2. **Configure your API Key**:
   Copy `.env.example` to `.env` (or edit the existing `.env`):
   ```bash
   cp .env.example .env
   ```
   Add your OpenRouter key:
   ```env
   OPENROUTER_API_KEY=sk-or-v1-...
   ```
   Get a free OpenRouter key from [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys).

3. **Install Dependencies**:
   ```bash
   uv sync
   ```

## Project Structure

```
Langchain/
├── src/                      # Source Code Layer
│   ├── langchain_rag/        # Core Production RAG Engine
│   └── production_api/       # FastAPI Backend Application
├── examples/                 # Tutorials, Deep Dives & Demonstrations
│   ├── 01_advanced_rag/      # End-to-end advanced RAG showcase
│   ├── 02_chunking/          # Chunking experiments & comparisons
│   ├── 03_embeddings/        # Deep-dive embedding benchmarks
│   ├── 04_hybrid_search/     # Hybrid search benchmarks
│   ├── 05_supabase/          # Supabase connection & pooling examples
│   └── legacy_scripts/       # Preserved standalone scripts
├── scripts/                  # DevOps, Setup, & Tooling (LangSmith verification)
├── docs/                     # Architecture Manuals & Documentation
├── tests/                    # Automated Test Suites
├── main.py                   # Root execution entrypoint
└── pyproject.toml            # Project & dependency configuration
```

## Usage

### Run the Project
You can run the script using either:
```bash
uv run main.py
```
or via the CLI script:
```bash
uv run langchain-rag
```

### Run FastAPI Production Server
```bash
uv run uvicorn production_api.main:app --reload --port 8000
```

### Run DevOps & Verification Scripts
```bash
uv run python scripts/langsmith_setup.py
```

### Use in Python Code

#### 1. Calling LLM
```python
import os
from dotenv import load_dotenv
from langchain_rag.llm import get_llm

load_dotenv()

# Get the configured Thinking Machines model instance
llm = get_llm()

# Invoke the model
response = llm.invoke("What makes an AI agent architecture 'stateful'?")
print(response.content)
```

#### 2. Loading Documents
```python
from langchain_rag.document_loader import (
    DocumentLoader,
    load_document,
    load_directory,
    load_text,
)

# Load a single file (markdown, text, code, json, csv, etc.)
docs = load_document("README.md")
print(f"Loaded {len(docs)} document(s):", docs[0].metadata)

# Load all markdown or python files in a directory
dir_docs = load_directory("src", extensions=[".py", ".md"])
print(f"Found {len(dir_docs)} document(s) in directory")

# In-memory document creation
doc = load_text("Knowledge base context...", metadata={"topic": "ai"})
```

#### 3. Text Splitting & Chunking
```python
from langchain_rag.text_splitter import TextSplitter, split_documents, split_text

# Initialize splitter with chunk size and overlap
splitter = TextSplitter(chunk_size=500, chunk_overlap=50)

# Chunk plain string
chunks = splitter.split_text("Long text content...")

# Chunk LangChain Document objects (preserves & enriches metadata)
doc_chunks = splitter.split_documents(docs)
print(f"Split {len(docs)} document into {len(doc_chunks)} chunks")
```

#### 4. Embeddings & Semantic Similarity
```python
from langchain_rag.embeddings import get_embeddings, calculate_similarity

# Local ONNX model (offline, fast, free)
embeddings = get_embeddings("local")

# Generate embedding vector
vector = embeddings.embed_query("LangChain RAG pipeline")
print(f"Vector dimensions: {len(vector)}")

# Compute similarity between two texts
score = calculate_similarity(
    "AI helps doctors diagnose illness",
    "Machine learning assists physicians in medicine",
    embeddings=embeddings,
)
print(f"Similarity score: {score:.4f}")
```

#### 4.1 Cached Embeddings (`CacheBackedEmbeddings`)
Eliminate redundant vector calculations, slash API embedding costs, and achieve sub-millisecond warm lookups across SQLite, local files, or memory.

Run the deep dive benchmark and CLI:
```bash
# Run the complete embeddings deep dive suite:
uv run python examples/03_embeddings/embeddings_deep.py

# Benchmark with specific backend (sqlite, file, memory):
uv run python examples/03_embeddings/embeddings_deep.py --backend sqlite

# Interactive embeddings & latency lab:
uv run python examples/03_embeddings/embeddings_deep.py --interactive
```

Python usage:
```python
from langchain_rag.cached_embeddings import create_cached_embeddings

# Wrap local or API embeddings with an ACID SQLite byte store
cached_embeddings = create_cached_embeddings(
    store_type="sqlite",
    cache_dir=".cache/embeddings",
    namespace="production_minilm",
)

# 1. Cold embedding run (computes vectors and saves to cache store)
docs = ["Document chunk 1...", "Document chunk 2..."]
vectors = cached_embeddings.embed_documents(docs)

# 2. Warm embedding run (retrieves from cache in ~0.3ms - 500x speedup!)
vectors_warm = cached_embeddings.embed_documents(docs)
print(cached_embeddings.get_stats())
```

#### 5. Vector Store (ChromaDB)
```python
from langchain_rag.vector_stores import create_vector_store

store = create_vector_store("my_collection")
store.add_texts(["AI is transforming healthcare.", "ChromaDB is a vector database."])

results = store.query("healthcare", n_results=1)
print("Top match:", results[0]["text"])
```

#### 6. End-to-End RAG Pipeline
```python
from langchain_rag.rag_pipeline import create_rag_pipeline

pipeline = create_rag_pipeline()

# Index knowledge documents or files (automatically chunked)
pipeline.index_file("README.md")
pipeline.index_texts(["LangChain simplifies AI application workflows."])

# Ask questions grounded in your documents
result = pipeline.query("What does this project do?")
print("Answer:", result["answer"])
print("Sources:", result["sources"])
```

#### 7. Production Hybrid Search (Dense + Sparse BM25 Fusion)
Run the standalone hybrid search CLI and demonstration:
```bash
# Run comparison suite across Dense vs Sparse vs RRF:
uv run python examples/04_hybrid_search/prod_hybrid_search.py

# Query with Reciprocal Rank Fusion (RRF):
uv run python examples/04_hybrid_search/prod_hybrid_search.py -q "ERR-9021-TOKEN-REVOKED" --mode compare

# Launch interactive search REPL:
uv run python examples/04_hybrid_search/prod_hybrid_search.py --interactive
```

Python usage:
```python
from langchain_rag.hybrid_search import create_hybrid_search_engine

engine = create_hybrid_search_engine()
engine.add_texts([
    "Error Code ERR-9021-TOKEN-REVOKED: The OAuth bearer token was invalidated.",
    "SSO portal handles expired user authentication timeouts.",
])

# Reciprocal Rank Fusion (RRF)
results = engine.search("ERR-9021-TOKEN-REVOKED", k=2, fusion_mode="rrf")
print("Top match:", results[0]["text"])

# LangChain Retriever adapter
retriever = engine.as_retriever(k=2)
docs = retriever.invoke("token revoked")
```

#### 8. Cost Optimization (Semantic Caching, Pruning, Routing & Token Budgeting)
Run the cost optimization demonstration and interactive console:
```bash
# Run cost optimization demonstration suite:
uv run python examples/legacy_scripts/cost_optimization.py

# Query with semantic cache & token budgeting:
uv run python examples/legacy_scripts/cost_optimization.py -q "What is ChromaDB used for?"
```

Python usage:
```python
from langchain_rag.cost_optimization import create_cost_optimizer

# Configure semantic cache and query token budget (e.g. max 150 tokens per query)
optimizer = create_cost_optimizer(
    cache_threshold=0.80,
    max_query_tokens=150,
)

# 1. Normal query (executes through model & populates cache)
res1 = optimizer.execute_optimized_query("What is ChromaDB?")
print("Cold run latency:", res1["latency_ms"], "ms")

# 2. Semantically equivalent query (instant cache hit, $0.00 token cost!)
res2 = optimizer.execute_optimized_query("Tell me what ChromaDB is used for?")
print("Cache hit:", res2["cache_hit"], "| Latency:", res2["latency_ms"], "ms")

# 3. Overly long / abusive query (instantly revoked in <1ms, blocking LLM consumption!)
res3 = optimizer.execute_optimized_query("Very long prompt exceeding budget...")
if res3.get("revoked"):
    print("Guardrail blocked query:", res3["revocation_reason"])

print("Summary:", optimizer.tracker.summary())
```


