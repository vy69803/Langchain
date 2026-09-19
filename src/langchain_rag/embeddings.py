"""Embeddings module for LangChain RAG applications.

Provides local (ONNX all-MiniLM-L6-v2) and cloud-based embedding generation
compatible with LangChain's Embeddings interface.
"""

from __future__ import annotations

import math
import os
from typing import Any, Sequence

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()


class LocalEmbeddings(Embeddings):
    """Local, fast, free embeddings using ChromaDB's cached all-MiniLM-L6-v2 ONNX model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the local embedding model."""
        self.model_name = model_name
        self._ef: Any = None

    def _get_ef(self) -> Any:
        if self._ef is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._ef = DefaultEmbeddingFunction()
        return self._ef

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document strings into float vectors.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors (384 dimensions).
        """
        if not texts:
            return []
        ef = self._get_ef()
        raw_embeddings = ef(texts)
        return [[float(val) for val in vec] for vec in raw_embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string into a float vector.

        Args:
            text: Query string.

        Returns:
            Embedding vector (384 dimensions).
        """
        ef = self._get_ef()
        raw = ef([text])
        return [float(val) for val in raw[0]]


class APIEmbeddings(Embeddings):
    """Cloud-based embeddings via OpenRouter or OpenAI."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize API embeddings."""
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.kwargs = kwargs
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key or self.api_key == "your_openrouter_api_key_here":
                raise ValueError("Valid OPENROUTER_API_KEY is required for API embeddings.")
            from langchain_openai import OpenAIEmbeddings

            self._client = OpenAIEmbeddings(
                model=self.model,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                **self.kwargs,
            )
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_client().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get_client().embed_query(text)


def get_embeddings(
    model_type: str = "local",
    **kwargs: Any,
) -> Embeddings:
    """Factory function to get an Embeddings instance.

    Args:
        model_type: "local" (default, free offline ONNX model) or "api" (OpenRouter/OpenAI).
        **kwargs: Arguments forwarded to the embeddings constructor.

    Returns:
        Configured LangChain Embeddings instance.
    """
    if model_type.lower() == "local":
        return LocalEmbeddings(**kwargs)
    elif model_type.lower() in ("api", "openai", "openrouter"):
        try:
            return APIEmbeddings(**kwargs)
        except Exception:
            # Gracefully fallback to local embeddings if API key is missing
            print("Notice: API key not configured for embeddings. Falling back to local embeddings.")
            return LocalEmbeddings()
    else:
        raise ValueError(f"Unknown embeddings model_type: '{model_type}'. Choose 'local' or 'api'.")


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Compute cosine similarity between two numeric vectors.

    Returns a value between -1.0 and 1.0 (1.0 = identical direction).
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have the same dimension: {len(vec1)} vs {len(vec2)}")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


def calculate_similarity(
    text1: str,
    text2: str,
    embeddings: Embeddings | None = None,
) -> float:
    """Calculate cosine similarity score between two strings using embeddings."""
    model = embeddings or get_embeddings("local")
    vec1 = model.embed_query(text1)
    vec2 = model.embed_query(text2)
    return cosine_similarity(vec1, vec2)


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("        LangChain RAG - Embeddings Demonstration        ")
    print("=" * 60)

    # 1. Initialize local embeddings
    emb = get_embeddings("local")
    print(f"\n[1] Initialized: {emb.__class__.__name__} (all-MiniLM-L6-v2)")

    # 2. Embed a single query
    sample_text = "What is Retrieval-Augmented Generation?"
    vec = emb.embed_query(sample_text)
    print(f"\n[2] Embedded query: '{sample_text}'")
    print(f"    Vector dimensions: {len(vec)}")
    print(f"    Preview (first 5 numbers): {[round(x, 4) for x in vec[:5]]}")

    # 3. Test semantic similarity comparison
    phrase_a = "Artificial intelligence helps doctors diagnose diseases."
    phrase_b = "Machine learning assists healthcare physicians with patient diagnosis."
    phrase_c = "Delicious chocolate cake recipe with fresh strawberries."

    score_ab = calculate_similarity(phrase_a, phrase_b, emb)
    score_ac = calculate_similarity(phrase_a, phrase_c, emb)

    print("\n[3] Semantic Similarity Comparisons:")
    print(f"    Text A: \"{phrase_a}\"")
    print(f"    Text B: \"{phrase_b}\"")
    print(f"    Text C: \"{phrase_c}\"")
    print(f"\n    Similarity (A vs B - similar topics):   {score_ab:.4f} (High)")
    print(f"    Similarity (A vs C - unrelated topics): {score_ac:.4f} (Low)")

    print("\n[✔] Embeddings module executed successfully!")
