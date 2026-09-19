"""Text splitter and chunking module for LangChain RAG applications.

Splits long documents into smaller, semantically coherent chunks with overlap,
preserving metadata for vector store embedding.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class TextSplitter:
    """Configurable text and document chunker optimized for RAG retrieval."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ) -> None:
        """Initialize the TextSplitter.

        Args:
            chunk_size: Maximum character length for each chunk.
            chunk_overlap: Number of overlapping characters between consecutive chunks.
            separators: Optional custom separator hierarchy. Defaults to ["\n\n", "\n", " ", ""].
            keep_separator: Whether to keep separators in chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""],
            keep_separator=keep_separator,
        )

    def split_text(self, text: str) -> list[str]:
        """Split a raw text string into chunks.

        Args:
            text: Input string.

        Returns:
            List of chunk strings.
        """
        if not text:
            return []
        return self.splitter.split_text(text)

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        """Split LangChain Document objects into smaller chunk Documents.

        Enriches chunk metadata with 'chunk_index' and 'total_chunks'.

        Args:
            documents: List of Document objects to chunk.

        Returns:
            List of chunked Document objects with preserved and enriched metadata.
        """
        if not documents:
            return []

        chunked_docs = self.splitter.split_documents(list(documents))

        # Enrich chunk metadata with index and parent reference
        for idx, doc in enumerate(chunked_docs):
            doc.metadata["chunk_index"] = idx
            doc.metadata["chunk_size"] = len(doc.page_content)

        return chunked_docs

    @staticmethod
    def split_markdown(
        markdown_text: str,
        headers_to_split_on: list[tuple[str, str]] | None = None,
    ) -> list[Document]:
        """Split markdown text by header tags (# Header 1, ## Header 2, etc.).

        Args:
            markdown_text: Raw markdown string.
            headers_to_split_on: List of tuples mapping header syntax to metadata keys.
                Defaults to [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")].

        Returns:
            List of Document objects with headers preserved in metadata.
        """
        headers = headers_to_split_on or [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers,
            strip_headers=False,
        )
        return md_splitter.split_text(markdown_text)

    @staticmethod
    def split_code(
        code: str,
        language: Language = Language.PYTHON,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> list[Document]:
        """Split source code using language-specific syntax boundaries."""
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return code_splitter.create_documents([code])


# --- Convenience Functions ---

def split_documents(
    documents: Sequence[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split documents into chunks using default RecursiveCharacterTextSplitter."""
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split raw text into chunks."""
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("      LangChain RAG - Text Splitter Demonstration      ")
    print("=" * 60)

    # 1. Splitting plain text
    sample_text = (
        "Retrieval-Augmented Generation (RAG) is an AI framework for improving the quality "
        "of LLM responses by grounding the model on external sources of knowledge. "
        "Text splitters play a vital role in RAG: they take long documents and break them "
        "into smaller, coherent passages called chunks. Chunks allow vector search to find "
        "the precise paragraph answering a user's question without overwhelming the LLM's "
        "context window. Chunk overlap ensures that semantic boundaries aren't accidentally "
        "cut in half."
    )

    splitter = TextSplitter(chunk_size=160, chunk_overlap=30)
    chunks = splitter.split_text(sample_text)

    print(f"\n[1] Splitting text (chunk_size=160, overlap=30):")
    print(f"    Original length: {len(sample_text)} characters")
    print(f"    Created {len(chunks)} chunk(s):")
    for idx, c in enumerate(chunks, 1):
        print(f"    --- Chunk #{idx} ({len(c)} chars) ---")
        print(f"    {c.strip()}")

    # 2. Splitting a local file (README.md) into LangChain Documents
    readme_path = Path("README.md")
    if readme_path.exists():
        from langchain_rag.document_loader import load_document

        raw_docs = load_document(readme_path)
        chunked_docs = splitter.split_documents(raw_docs)
        print(f"\n[2] Chunked '{readme_path.name}':")
        print(f"    Input: {len(raw_docs)} doc ({len(raw_docs[0].page_content)} chars)")
        print(f"    Output: {len(chunked_docs)} chunk Documents")
        if chunked_docs:
            print(f"    Sample Chunk #1 metadata: {chunked_docs[0].metadata}")

    print("\n[✔] Text splitting test completed successfully!")
