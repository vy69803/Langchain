"""Vector store module for LangChain RAG applications.

Provides a ChromaDB-backed vector store with helpers for adding documents,
querying by similarity, and managing collections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document


class VectorStore:
    """ChromaDB-backed vector store for document storage and similarity search."""

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str | None = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Path to persist the database on disk.
                If None, uses an in-memory (ephemeral) store.
        """
        if persist_directory:
            path = Path(persist_directory).resolve()
            path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(path))
        else:
            self.client = chromadb.Client()

        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_texts(
        self,
        texts: list[str],
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add text strings to the vector store.

        Args:
            texts: List of text strings to embed and store.
            ids: Optional custom IDs. Auto-generated if not provided.
            metadatas: Optional metadata dicts for each text.

        Returns:
            List of IDs assigned to the added documents.
        """
        if not texts:
            return []

        if ids is None:
            # Generate unique IDs based on the current collection count
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(texts))]

        kwargs: dict[str, Any] = {"documents": texts, "ids": ids}
        if metadatas:
            kwargs["metadatas"] = metadatas

        self.collection.upsert(**kwargs)
        return ids

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add LangChain Document objects to the vector store.

        Args:
            documents: List of LangChain Document objects.
            ids: Optional custom IDs. Auto-generated if not provided.

        Returns:
            List of IDs assigned to the added documents.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, ids=ids, metadatas=metadatas)

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for documents similar to the query text.

        Args:
            query_text: The text to search for.
            n_results: Number of results to return (default: 3).
            where: Optional metadata filter (e.g. {"source": "README.md"}).

        Returns:
            List of result dicts with keys: id, text, metadata, distance.
        """
        total_count = self.collection.count()
        if total_count == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(n_results, total_count),
        }
        if where:
            kwargs["where"] = where

        raw = self.collection.query(**kwargs)

        results = []
        for i in range(len(raw["ids"][0])):
            results.append({
                "id": raw["ids"][0][i],
                "text": raw["documents"][0][i] if raw["documents"] else None,
                "metadata": raw["metadatas"][0][i] if raw["metadatas"] else None,
                "distance": raw["distances"][0][i] if raw["distances"] else None,
            })
        return results

    def delete(self, ids: list[str]) -> None:
        """Delete documents by their IDs."""
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """Return the total number of documents in the collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete the collection and recreate it (clears all data)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def list_collections(self) -> list[str]:
        """List all collection names in this ChromaDB instance."""
        return [col.name for col in self.client.list_collections()]

    def __repr__(self) -> str:
        return (
            f"VectorStore(collection='{self.collection_name}', "
            f"documents={self.collection.count()})"
        )


# --- Convenience functions ---

def create_vector_store(
    collection_name: str = "default",
    persist_directory: str | None = None,
) -> VectorStore:
    """Create and return a new VectorStore instance."""
    return VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 55)
    print("  LangChain RAG - Vector Store Demonstration")
    print("=" * 55)

    # 1. Create an in-memory vector store
    store = create_vector_store("demo_collection")
    print(f"\n[1] Created: {store}")

    # 2. Add plain text documents
    texts = [
        "Artificial intelligence is transforming healthcare.",
        "Machine learning models require large datasets.",
        "Natural language processing understands human text.",
        "Deep learning uses neural networks with many layers.",
        "Reinforcement learning trains agents through rewards.",
    ]
    ids = store.add_texts(texts)
    print(f"\n[2] Added {len(ids)} documents: {ids}")
    print(f"    Total in store: {store.count()}")

    # 3. Query by similarity
    query = "How does AI help in medicine?"
    print(f"\n[3] Query: \"{query}\"")
    results = store.query(query, n_results=3)
    for rank, r in enumerate(results, 1):
        print(f"    #{rank} (distance: {r['distance']:.4f}): {r['text']}")

    # 4. Add LangChain Documents
    from langchain_core.documents import Document

    lc_docs = [
        Document(page_content="ChromaDB is a vector database.", metadata={"source": "docs"}),
        Document(page_content="LangChain simplifies LLM apps.", metadata={"source": "readme"}),
    ]
    store.add_documents(lc_docs)
    print(f"\n[4] Added {len(lc_docs)} LangChain Documents. Total: {store.count()}")

    # 5. Query with metadata filter
    query2 = "vector database"
    print(f"\n[5] Query with filter (source='docs'): \"{query2}\"")
    filtered = store.query(query2, n_results=1, where={"source": "docs"})
    for r in filtered:
        print(f"    Result: {r['text']} | metadata: {r['metadata']}")

    # 6. Delete a document
    store.delete(["doc_0"])
    print(f"\n[6] Deleted doc_0. Remaining: {store.count()}")

    print(f"\n[✔] Final state: {store}")
