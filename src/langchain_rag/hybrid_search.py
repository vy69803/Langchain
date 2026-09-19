"""Production Hybrid Search module for LangChain RAG applications.

Combines Dense Semantic Search (ChromaDB vector store) with Sparse Lexical Search
(Okapi BM25 index) using Reciprocal Rank Fusion (RRF) and Relative Score Fusion.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from langchain_rag.document_loader import load_document
from langchain_rag.text_splitter import TextSplitter
from langchain_rag.vector_stores import VectorStore, create_vector_store


def default_tokenizer(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric and hypen-joined tokens.
    
    Preserves acronyms, model names, error codes (e.g., 'err-404-auth', 'gpt-4o').
    """
    return [match.group(0).lower() for match in re.finditer(r"\b[\w]+(?:[-_][\w]+)*\b", text)]


class BM25Index:
    """Production-grade Okapi BM25 lexical index with metadata filtering and persistence.
    
    Uses Lucene's non-negative IDF formulation:
        IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or default_tokenizer

        # Storage structures
        self.doc_ids: list[str] = []
        self.docs: dict[str, str] = {}
        self.metadatas: dict[str, dict[str, Any]] = {}
        self.doc_lens: dict[str, int] = {}
        self.avg_doc_len: float = 0.0

        # Inverted index: term -> {doc_id: term_frequency}
        self.inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        # Document frequency: term -> number of documents containing term
        self.doc_frequencies: dict[str, int] = defaultdict(int)

    def count(self) -> int:
        """Return total number of documents in the index."""
        return len(self.doc_ids)

    def add_texts(
        self,
        texts: Sequence[str],
        ids: Sequence[str] | None = None,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add text documents into the BM25 index."""
        if not texts:
            return []

        if ids is None:
            existing_count = self.count()
            assigned_ids = [f"bm25_doc_{existing_count + i}" for i in range(len(texts))]
        else:
            assigned_ids = list(ids)

        for i, (text, doc_id) in enumerate(zip(texts, assigned_ids)):
            # If document already exists, remove old counts first
            if doc_id in self.docs:
                self.delete([doc_id])

            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            tokens = self.tokenizer(text)
            doc_len = len(tokens)

            self.doc_ids.append(doc_id)
            self.docs[doc_id] = text
            self.metadatas[doc_id] = meta or {}
            self.doc_lens[doc_id] = doc_len

            term_counts = Counter(tokens)
            for term, freq in term_counts.items():
                self.inverted_index[term][doc_id] = freq
                self.doc_frequencies[term] += 1

        self._update_avg_doc_len()
        return assigned_ids

    def _update_avg_doc_len(self) -> None:
        total_len = sum(self.doc_lens.values())
        self.avg_doc_len = (total_len / len(self.doc_lens)) if self.doc_lens else 0.0

    def delete(self, ids: Sequence[str]) -> None:
        """Remove documents by ID from the BM25 index."""
        ids_to_remove = set(ids)
        for doc_id in ids_to_remove:
            if doc_id not in self.docs:
                continue

            text = self.docs[doc_id]
            tokens = self.tokenizer(text)
            unique_terms = set(tokens)

            for term in unique_terms:
                if doc_id in self.inverted_index[term]:
                    del self.inverted_index[term][doc_id]
                    self.doc_frequencies[term] -= 1
                    if self.doc_frequencies[term] <= 0:
                        del self.doc_frequencies[term]
                        del self.inverted_index[term]

            self.doc_ids.remove(doc_id)
            del self.docs[doc_id]
            del self.metadatas[doc_id]
            del self.doc_lens[doc_id]

        self._update_avg_doc_len()

    def reset(self) -> None:
        """Clear all indexed documents."""
        self.doc_ids.clear()
        self.docs.clear()
        self.metadatas.clear()
        self.doc_lens.clear()
        self.avg_doc_len = 0.0
        self.inverted_index.clear()
        self.doc_frequencies.clear()

    def _compute_idf(self, term: str) -> float:
        """Calculate non-negative Lucene-style IDF."""
        n_q = self.doc_frequencies.get(term, 0)
        n = len(self.doc_ids)
        if n == 0 or n_q == 0:
            return 0.0
        return math.log(1.0 + (n - n_q + 0.5) / (n_q + 0.5))

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for documents matching query using Okapi BM25 scoring.
        
        Args:
            query: The search query string.
            n_results: Maximum number of results to return.
            where: Optional metadata key-value filters.
            
        Returns:
            List of dicts: {'id', 'text', 'metadata', 'score'}.
        """
        if not self.doc_ids or not query.strip():
            return []

        tokens = self.tokenizer(query)
        if not tokens:
            return []

        scores: dict[str, float] = defaultdict(float)
        k1 = self.k1
        b = self.b
        avgdl = self.avg_doc_len or 1.0

        for term in tokens:
            if term not in self.inverted_index:
                continue

            idf = self._compute_idf(term)
            for doc_id, tf in self.inverted_index[term].items():
                dl = self.doc_lens.get(doc_id, avgdl)
                # Standard BM25 term weight
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (dl / avgdl))
                scores[doc_id] += idf * (numerator / denominator)

        # Filter by metadata if requested
        ranked_items: list[tuple[str, float]] = []
        for doc_id, score in scores.items():
            if score <= 0.0:
                continue

            if where:
                doc_meta = self.metadatas.get(doc_id, {})
                matches = True
                for k, v in where.items():
                    if doc_meta.get(k) != v:
                        matches = False
                        break
                if not matches:
                    continue

            ranked_items.append((doc_id, score))

        ranked_items.sort(key=lambda item: item[1], reverse=True)
        top_items = ranked_items[:n_results]

        return [
            {
                "id": doc_id,
                "text": self.docs[doc_id],
                "metadata": self.metadatas.get(doc_id, {}),
                "score": round(score, 4),
            }
            for doc_id, score in top_items
        ]

    def save_state(self, filepath: str | Path) -> None:
        """Serialize BM25 index state to a JSON file."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "docs": self.docs,
            "metadatas": self.metadatas,
            "doc_lens": self.doc_lens,
            "avg_doc_len": self.avg_doc_len,
            "inverted_index": {t: dict(docs) for t, docs in self.inverted_index.items()},
            "doc_frequencies": dict(self.doc_frequencies),
        }
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load_state(self, filepath: str | Path) -> None:
        """Load serialized BM25 index state from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index state file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)
        self.doc_ids = data.get("doc_ids", [])
        self.docs = data.get("docs", {})
        self.metadatas = data.get("metadatas", {})
        self.doc_lens = data.get("doc_lens", {})
        self.avg_doc_len = data.get("avg_doc_len", 0.0)
        self.inverted_index = defaultdict(dict, data.get("inverted_index", {}))
        self.doc_frequencies = defaultdict(int, data.get("doc_frequencies", {}))


class HybridSearchEngine:
    """Production Hybrid Search Engine combining ChromaDB Dense Search and BM25 Sparse Search.
    
    Supports:
      - Reciprocal Rank Fusion (RRF) (Industry standard, robust across different score scales)
      - Weighted Score Fusion (Convex combination with min-max normalization)
      - Metadata filtering
      - Persistence (saving & loading index state)
      - LangChain Document and raw text ingestion
      - Text splitting & chunking integration
    """

    def __init__(
        self,
        collection_name: str = "hybrid_search_collection",
        persist_directory: str | None = None,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
        text_splitter: TextSplitter | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.vector_store = vector_store or create_vector_store(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        self.bm25_index = bm25_index or BM25Index()
        self.text_splitter = text_splitter or TextSplitter(chunk_size=500, chunk_overlap=50)

        # In-memory mapping of doc_id -> (text, metadata)
        self._doc_map: dict[str, dict[str, Any]] = {}

        # Auto-load existing BM25 state if persist directory exists
        if self.persist_directory:
            bm25_path = Path(self.persist_directory) / f"{self.collection_name}_bm25.json"
            if bm25_path.exists():
                try:
                    self.bm25_index.load_state(bm25_path)
                    for doc_id in self.bm25_index.doc_ids:
                        self._doc_map[doc_id] = {
                            "text": self.bm25_index.docs.get(doc_id, ""),
                            "metadata": self.bm25_index.metadatas.get(doc_id, {}),
                        }
                except Exception as err:
                    print(f"Notice: Could not load BM25 state from {bm25_path}: {err}")

    def count(self) -> int:
        """Return total document count in the hybrid search engine."""
        return max(self.vector_store.count(), self.bm25_index.count())

    def add_texts(
        self,
        texts: list[str],
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Index raw texts into both dense (ChromaDB) and sparse (BM25) search indices."""
        if not texts:
            return []

        if ids is None:
            existing_count = self.count()
            ids = [f"hdoc_{existing_count + i}" for i in range(len(texts))]

        # 1. Add to ChromaDB vector store
        self.vector_store.add_texts(texts, ids=ids, metadatas=metadatas)

        # 2. Add to BM25 sparse index
        self.bm25_index.add_texts(texts, ids=ids, metadatas=metadatas)

        # 3. Cache doc map
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            self._doc_map[doc_id] = {"text": texts[i], "metadata": meta}

        # Auto-save if persisted
        self._persist_bm25_if_needed()
        return ids

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add LangChain Document objects into both dense and sparse indices."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, ids=ids, metadatas=metadatas)

    def index_file(
        self,
        file_path_or_url: str | Path,
        *,
        chunk: bool = True,
    ) -> list[str]:
        """Load a file or URL, optionally chunk it, and index into the hybrid engine."""
        docs = load_document(file_path_or_url)
        if chunk and self.text_splitter:
            docs = self.text_splitter.split_documents(docs)
        return self.add_documents(docs)

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID from both vector store and BM25 index."""
        self.vector_store.delete(ids)
        self.bm25_index.delete(ids)
        for doc_id in ids:
            self._doc_map.pop(doc_id, None)
        self._persist_bm25_if_needed()

    def reset(self) -> None:
        """Clear both dense and sparse search indices completely."""
        self.vector_store.reset()
        self.bm25_index.reset()
        self._doc_map.clear()
        self._persist_bm25_if_needed()

    def _persist_bm25_if_needed(self) -> None:
        if self.persist_directory:
            bm25_path = Path(self.persist_directory) / f"{self.collection_name}_bm25.json"
            self.bm25_index.save_state(bm25_path)

    def search_dense(
        self,
        query: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform dense semantic similarity search only (via ChromaDB)."""
        raw_results = self.vector_store.query(query, n_results=k, where=where)
        formatted: list[dict[str, Any]] = []
        for rank, r in enumerate(raw_results, start=1):
            distance = r.get("distance", 1.0) or 0.0
            # Convert cosine/L2 distance to normalized similarity score [0, 1]
            sim_score = max(0.0, 1.0 / (1.0 + distance))
            formatted.append({
                "id": r["id"],
                "text": r.get("text") or self._get_text(r["id"]),
                "metadata": r.get("metadata") or self._get_metadata(r["id"]),
                "dense_rank": rank,
                "dense_score": round(sim_score, 4),
                "distance": distance,
            })
        return formatted

    def search_sparse(
        self,
        query: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform sparse lexical keyword search only (via BM25)."""
        raw_results = self.bm25_index.search(query, n_results=k, where=where)
        formatted: list[dict[str, Any]] = []
        for rank, r in enumerate(raw_results, start=1):
            formatted.append({
                "id": r["id"],
                "text": r["text"],
                "metadata": r["metadata"],
                "sparse_rank": rank,
                "sparse_score": r["score"],
            })
        return formatted

    def search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        fusion_mode: str = "rrf",
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform Production Hybrid Search combining Dense and Sparse retrievers.
        
        Args:
            query: The user query string.
            k: Top results to return.
            alpha: Weight for dense search vs sparse search (0.0 = pure sparse, 1.0 = pure dense).
                   Used in 'weighted' fusion mode, or as rank weights in 'rrf' mode.
            fusion_mode: 'rrf' (Reciprocal Rank Fusion, default) or 'weighted' (Score Fusion).
            rrf_k: Constant for Reciprocal Rank Fusion smoothing (standard is 60).
            candidate_multiplier: Multiplier to retrieve more candidate chunks from each retriever
                                  before reranking/fusion (default: 4x).
            where: Optional metadata filter dict.
            
        Returns:
            List of hybrid-ranked document dicts with full scoring details.
        """
        fetch_k = max(k * candidate_multiplier, 10)

        # 1. Fetch dense candidates
        dense_results = self.search_dense(query, k=fetch_k, where=where)

        # 2. Fetch sparse candidates
        sparse_results = self.search_sparse(query, k=fetch_k, where=where)

        if fusion_mode.lower() == "weighted":
            return self._fuse_weighted(
                dense_results=dense_results,
                sparse_results=sparse_results,
                alpha=alpha,
                top_k=k,
            )
        else:
            return self._fuse_rrf(
                dense_results=dense_results,
                sparse_results=sparse_results,
                rrf_k=rrf_k,
                alpha=alpha,
                top_k=k,
            )

    def _fuse_rrf(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        rrf_k: int,
        alpha: float,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF) algorithm."""
        # Dense weight: alpha, Sparse weight: (1 - alpha)
        w_dense = alpha
        w_sparse = 1.0 - alpha

        doc_data: dict[str, dict[str, Any]] = {}
        rrf_scores: dict[str, float] = defaultdict(float)

        for item in dense_results:
            doc_id = item["id"]
            rank = item["dense_rank"]
            doc_data[doc_id] = {
                "id": doc_id,
                "text": item["text"],
                "metadata": item["metadata"],
                "dense_rank": rank,
                "dense_score": item["dense_score"],
                "sparse_rank": None,
                "sparse_score": None,
            }
            rrf_scores[doc_id] += w_dense * (1.0 / (rrf_k + rank))

        for item in sparse_results:
            doc_id = item["id"]
            rank = item["sparse_rank"]
            if doc_id in doc_data:
                doc_data[doc_id]["sparse_rank"] = rank
                doc_data[doc_id]["sparse_score"] = item["sparse_score"]
            else:
                doc_data[doc_id] = {
                    "id": doc_id,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": rank,
                    "sparse_score": item["sparse_score"],
                }
            rrf_scores[doc_id] += w_sparse * (1.0 / (rrf_k + rank))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
            entry = dict(doc_data[doc_id])
            entry["hybrid_rank"] = rank
            entry["hybrid_score"] = round(score, 6)
            entry["fusion_method"] = "rrf"
            results.append(entry)

        return results

    def _fuse_weighted(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        alpha: float,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Convex combination / min-max normalized score fusion."""
        # Normalize dense scores
        dense_scores_raw = [item["dense_score"] for item in dense_results]
        max_dense = max(dense_scores_raw) if dense_scores_raw else 1.0
        min_dense = min(dense_scores_raw) if dense_scores_raw else 0.0
        dense_range = (max_dense - min_dense) or 1.0

        # Normalize sparse scores
        sparse_scores_raw = [item["sparse_score"] for item in sparse_results]
        max_sparse = max(sparse_scores_raw) if sparse_scores_raw else 1.0
        min_sparse = min(sparse_scores_raw) if sparse_scores_raw else 0.0
        sparse_range = (max_sparse - min_sparse) or 1.0

        doc_data: dict[str, dict[str, Any]] = {}
        combined_scores: dict[str, float] = defaultdict(float)

        for item in dense_results:
            doc_id = item["id"]
            norm_dense = (item["dense_score"] - min_dense) / dense_range
            doc_data[doc_id] = {
                "id": doc_id,
                "text": item["text"],
                "metadata": item["metadata"],
                "dense_rank": item["dense_rank"],
                "dense_score": item["dense_score"],
                "sparse_rank": None,
                "sparse_score": None,
            }
            combined_scores[doc_id] += alpha * norm_dense

        for item in sparse_results:
            doc_id = item["id"]
            norm_sparse = (item["sparse_score"] - min_sparse) / sparse_range
            if doc_id in doc_data:
                doc_data[doc_id]["sparse_rank"] = item["sparse_rank"]
                doc_data[doc_id]["sparse_score"] = item["sparse_score"]
            else:
                doc_data[doc_id] = {
                    "id": doc_id,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": item["sparse_rank"],
                    "sparse_score": item["sparse_score"],
                }
            combined_scores[doc_id] += (1.0 - alpha) * norm_sparse

        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
            entry = dict(doc_data[doc_id])
            entry["hybrid_rank"] = rank
            entry["hybrid_score"] = round(score, 4)
            entry["fusion_method"] = "weighted"
            results.append(entry)

        return results

    def _get_text(self, doc_id: str) -> str:
        if doc_id in self._doc_map:
            return self._doc_map[doc_id]["text"]
        return self.bm25_index.docs.get(doc_id, "")

    def _get_metadata(self, doc_id: str) -> dict[str, Any]:
        if doc_id in self._doc_map:
            return self._doc_map[doc_id]["metadata"]
        return self.bm25_index.metadatas.get(doc_id, {})

    def as_retriever(
        self,
        k: int = 4,
        alpha: float = 0.5,
        fusion_mode: str = "rrf",
    ) -> BaseRetriever:
        """Wrap this HybridSearchEngine as a LangChain BaseRetriever."""
        return LangChainHybridRetriever(
            hybrid_engine=self,
            k=k,
            alpha=alpha,
            fusion_mode=fusion_mode,
        )


class LangChainHybridRetriever(BaseRetriever):
    """LangChain BaseRetriever adapter for HybridSearchEngine."""

    hybrid_engine: HybridSearchEngine = Field(description="Underlying HybridSearchEngine")
    k: int = 4
    alpha: float = 0.5
    fusion_mode: str = "rrf"

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        results = self.hybrid_engine.search(
            query=query,
            k=self.k,
            alpha=self.alpha,
            fusion_mode=self.fusion_mode,
        )
        docs: list[Document] = []
        for r in results:
            metadata = dict(r.get("metadata", {}))
            metadata.update({
                "doc_id": r["id"],
                "hybrid_rank": r["hybrid_rank"],
                "hybrid_score": r["hybrid_score"],
                "fusion_method": r["fusion_method"],
                "dense_rank": r.get("dense_rank"),
                "sparse_rank": r.get("sparse_rank"),
            })
            docs.append(Document(page_content=r["text"], metadata=metadata))
        return docs


def create_hybrid_search_engine(
    collection_name: str = "hybrid_knowledge_base",
    persist_directory: str | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> HybridSearchEngine:
    """Convenience factory function to create a configured HybridSearchEngine."""
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return HybridSearchEngine(
        collection_name=collection_name,
        persist_directory=persist_directory,
        text_splitter=splitter,
    )
