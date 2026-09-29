"""Reranking module for RAG pipelines.

Implements high-accuracy Cross-Encoder and local fast reranking (FlashRank)
to re-score and re-order candidate document chunks retrieved by vector search
or hybrid search before passing them to the LLM.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Sequence

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract base class for document rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: Sequence[dict[str, Any] | Document],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank a sequence of documents against a query.

        Args:
            query: The user query string.
            documents: Candidate documents (dicts or LangChain Document objects).
            top_k: Optional maximum number of top results to return.

        Returns:
            List of reranked document dicts sorted descending by rerank score.
        """
        pass


class FlashRankReranker(BaseReranker):
    """Ultra-fast local reranker using FlashRank (ONNX-accelerated Cross-Encoder).
    
    Default model is 'ms-marco-TinyBERT-L-2-v2' (~4MB ONNX, <20ms latency on CPU).
    Other models supported by FlashRank include:
      - 'ms-marco-MiniLM-L-12-v2' (~130MB)
      - 'rank-T5-flan' (~110MB)
      - 'ms-marco-MultiBERT-L-12' (multilingual)
    """

    def __init__(
        self,
        model_name: str = "ms-marco-TinyBERT-L-2-v2",
        cache_dir: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self._ranker: Any = None

    @property
    def ranker(self) -> Any:
        """Lazily initialize the FlashRank Ranker instance."""
        if self._ranker is None:
            try:
                from flashrank import Ranker
                kwargs: dict[str, Any] = {"model_name": self.model_name, "max_length": self.max_length}
                if self.cache_dir:
                    kwargs["cache_dir"] = self.cache_dir
                self._ranker = Ranker(**kwargs)
            except ImportError as e:
                logger.error("flashrank is not installed. Run `uv add flashrank`.")
                raise ImportError(
                    "flashrank is required for FlashRankReranker. Install with `uv add flashrank`."
                ) from e
        return self._ranker

    def rerank(
        self,
        query: str,
        documents: Sequence[dict[str, Any] | Document],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank candidate document chunks using local Cross-Encoder.

        Args:
            query: The search query.
            documents: Candidates from ChromaDB, BM25, or hybrid search.
            top_k: Number of highest-ranking documents to return (e.g. 3).

        Returns:
            List of reranked documents with rerank scores and ranks.
        """
        if not documents:
            return []

        if not query or not query.strip():
            # If query is empty, return original documents as standard dicts
            return self._to_doc_dicts(documents)[:top_k]

        from flashrank import RerankRequest

        passages: list[dict[str, Any]] = []
        for idx, doc in enumerate(documents):
            if isinstance(doc, Document):
                passages.append({
                    "id": doc.metadata.get("id") or f"doc_{idx}",
                    "text": doc.page_content,
                    "metadata": dict(doc.metadata),
                    "original_rank": idx + 1,
                    "original_score": doc.metadata.get("score") or doc.metadata.get("distance"),
                })
            elif isinstance(doc, dict):
                entry = dict(doc)
                entry.setdefault("id", doc.get("id") or f"doc_{idx}")
                entry.setdefault("text", doc.get("text") or doc.get("content") or "")
                entry.setdefault("metadata", doc.get("metadata") or {})
                entry["original_rank"] = idx + 1
                entry["original_score"] = doc.get("score") or doc.get("distance")
                passages.append(entry)
            else:
                passages.append({
                    "id": f"doc_{idx}",
                    "text": str(doc),
                    "metadata": {},
                    "original_rank": idx + 1,
                    "original_score": None,
                })

        try:
            req = RerankRequest(query=query.strip(), passages=passages)
            raw_reranked = self.ranker.rerank(req)
        except Exception as e:
            logger.warning(f"FlashRank reranking failed with error: {e}. Falling back to original order.", exc_info=True)
            return passages[:top_k]

        results: list[dict[str, Any]] = []
        for rank_idx, item in enumerate(raw_reranked, start=1):
            doc_dict = dict(item)
            score_val = float(item.get("score", 0.0))
            doc_dict["score"] = score_val
            doc_dict["rerank_score"] = score_val
            doc_dict["rerank_rank"] = rank_idx
            results.append(doc_dict)

        if top_k is not None:
            results = results[:top_k]

        return results

    def _to_doc_dicts(self, documents: Sequence[dict[str, Any] | Document]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for idx, doc in enumerate(documents):
            if isinstance(doc, Document):
                results.append({
                    "id": doc.metadata.get("id") or f"doc_{idx}",
                    "text": doc.page_content,
                    "metadata": dict(doc.metadata),
                    "original_rank": idx + 1,
                })
            elif isinstance(doc, dict):
                results.append(dict(doc))
        return results


def get_reranker(
    reranker_type: str = "flashrank",
    model_name: str | None = None,
    **kwargs: Any,
) -> BaseReranker:
    """Factory function to instantiate a configured reranker.

    Args:
        reranker_type: Type of reranker ('flashrank', etc.).
        model_name: Name of model for reranking (defaults to RERANKER_MODEL env var or 'ms-marco-MiniLM-L-12-v2').
        **kwargs: Additional parameters passed to reranker constructor.

    Returns:
        Instance of BaseReranker.
    """
    import os
    resolved_model = model_name or os.environ.get("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")
    reranker_type_lower = reranker_type.lower()
    if reranker_type_lower == "flashrank":
        return FlashRankReranker(model_name=resolved_model, **kwargs)
    raise ValueError(f"Unknown reranker type: '{reranker_type}'. Supported: 'flashrank'.")
