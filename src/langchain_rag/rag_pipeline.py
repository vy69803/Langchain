"""RAG (Retrieval-Augmented Generation) pipeline module.

Combines DocumentLoader, VectorStore, and LLM to retrieve relevant knowledge
and generate grounded responses with citations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from langchain_rag.document_loader import DocumentLoader, load_document
from langchain_rag.llm import get_llm
from langchain_rag.text_splitter import TextSplitter
from langchain_rag.vector_stores import VectorStore, create_vector_store

load_dotenv()

DEFAULT_SYSTEM_PROMPT = """You are a helpful and accurate assistant.
Answer the user's question using ONLY the provided context below.
If the context does not contain the answer, state clearly that the information is not available in the provided documents.
Do not make up facts or hallucinate. Always be concise and factual.

Context:
{context}

Question: {question}

Helpful Answer:"""

SOURCES_SYSTEM_PROMPT = """You are a helpful and accurate assistant.
Answer the user's question using ONLY the provided numbered source documents below.
When stating facts, cite the corresponding source number using brackets, e.g. [1], [2].
At the end of your answer, include a 'Sources:' section listing the referenced sources.
If the context does not contain the answer, state clearly that the information is not available in the provided documents.

Context:
{context}

Question: {question}

Helpful Answer (with [citations]):"""


def format_docs_with_sources(docs: Sequence[Document | dict[str, Any]]) -> str:
    """Format a sequence of documents into a numbered string with source citations.

    Args:
        docs: Sequence of LangChain Document objects or vector store result dictionaries.

    Returns:
        Formatted multi-line string with [Source #X] headers.
    """
    if not docs:
        return "No relevant context found in documents."

    formatted_parts: list[str] = []
    for idx, item in enumerate(docs, start=1):
        if isinstance(item, Document):
            source = item.metadata.get("source") or item.metadata.get("filename") or f"doc_{idx}"
            content = item.page_content.strip()
        elif isinstance(item, dict):
            meta = item.get("metadata") or {}
            source = meta.get("source") or meta.get("filename") or item.get("id") or f"doc_{idx}"
            content = (item.get("text") or "").strip()
        else:
            source = f"doc_{idx}"
            content = str(item).strip()

        formatted_parts.append(f"--- [Source #{idx}: {source}] ---\n{content}")

    return "\n\n".join(formatted_parts)


def create_rag_with_sources_chain(
    retriever: Any,
    llm: Any = None,
    prompt_template: str = SOURCES_SYSTEM_PROMPT,
) -> Any:
    """Create a modern LangChain Expression Language (LCEL) RAG chain that returns answers alongside cited source documents.

    Args:
        retriever: LangChain BaseRetriever or object with .invoke(query) -> list[Document].
        llm: Configured LangChain chat model. If None, initialized via get_llm().
        prompt_template: Custom prompt template string containing {context} and {question}.

    Returns:
        LCEL Runnable chain returning dict with 'question', 'answer', 'context', and 'source_documents'.
    """
    active_llm = llm or get_llm()
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def extract_sources(input_data: dict[str, Any]) -> list[dict[str, Any]]:
        docs = input_data.get("source_documents") or []
        sources = []
        for i, d in enumerate(docs, 1):
            if isinstance(d, Document):
                sources.append({
                    "citation": f"[{i}]",
                    "source": d.metadata.get("source", "unknown"),
                    "metadata": d.metadata,
                    "content_preview": " ".join(d.page_content.split())[:120] + "...",
                })
            elif isinstance(d, dict):
                meta = d.get("metadata") or {}
                sources.append({
                    "citation": f"[{i}]",
                    "source": meta.get("source", "unknown"),
                    "metadata": meta,
                    "content_preview": " ".join((d.get("text") or "").split())[:120] + "...",
                })
        return sources

    # LCEL pipeline
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs_with_sources(x["source_documents"]))
        | prompt
        | active_llm
        | StrOutputParser()
    )

    rag_with_sources_chain = (
        RunnableParallel(
            source_documents=lambda x: retriever.invoke(x["question"]) if hasattr(retriever, "invoke") else retriever(x["question"]),
            question=lambda x: x["question"],
        )
        .assign(answer=rag_chain_from_docs)
        .assign(sources=extract_sources)
    )

    return rag_with_sources_chain


class RAGPipeline:
    """End-to-end RAG pipeline managing indexing, chunking, retrieval, and LLM response generation."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        text_splitter: TextSplitter | None = None,
        llm: Any = None,
        collection_name: str = "rag_knowledge_base",
        persist_directory: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        """Initialize the RAG pipeline.

        Args:
            vector_store: Optional existing VectorStore instance.
            text_splitter: Optional TextSplitter instance for chunking documents.
            llm: Optional configured LangChain LLM instance. If None, initialized lazily.
            collection_name: Name of the ChromaDB collection if creating a new store.
            persist_directory: Directory path to persist embeddings on disk.
            chunk_size: Default chunk size if text_splitter not provided (default: 500).
            chunk_overlap: Default chunk overlap if text_splitter not provided (default: 50).
            system_prompt_template: Template string containing {context} and {question}.
        """
        self.vector_store = vector_store or create_vector_store(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        self.text_splitter = text_splitter or TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._llm = llm
        self.system_prompt_template = system_prompt_template

    @property
    def llm(self) -> Any:
        """Lazily initialize the LLM if not provided during init."""
        if self._llm is None:
            try:
                self._llm = get_llm()
            except Exception as e:
                # Return None if API key is not configured, allowing retrieval-only mode
                self._llm = None
        return self._llm

    def index_documents(
        self,
        documents: Sequence[Document],
        *,
        chunk: bool = True,
    ) -> list[str]:
        """Index LangChain Document objects into the vector store.

        Args:
            documents: Sequence of Document objects to index.
            chunk: Whether to split documents into smaller chunks before indexing (default: True).

        Returns:
            List of IDs assigned to indexed documents.
        """
        docs_to_index = list(documents)
        if chunk and self.text_splitter:
            docs_to_index = self.text_splitter.split_documents(docs_to_index)
        return self.vector_store.add_documents(docs_to_index)

    def index_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        chunk: bool = False,
    ) -> list[str]:
        """Index raw text strings into the vector store.

        Args:
            texts: List of strings.
            metadatas: Optional metadata dicts.
            chunk: Whether to split texts into chunks before indexing (default: False).

        Returns:
            List of IDs assigned to indexed texts.
        """
        if chunk and self.text_splitter:
            chunked_texts: list[str] = []
            chunked_metas: list[dict[str, Any]] | None = [] if metadatas else None
            for idx, text in enumerate(texts):
                sub_chunks = self.text_splitter.split_text(text)
                chunked_texts.extend(sub_chunks)
                if chunked_metas is not None and metadatas:
                    meta = metadatas[idx] if idx < len(metadatas) else {}
                    chunked_metas.extend([dict(meta) for _ in sub_chunks])
            return self.vector_store.add_texts(chunked_texts, metadatas=chunked_metas)
        return self.vector_store.add_texts(texts, metadatas=metadatas)

    def index_file(
        self,
        file_path_or_url: str | Path,
        *,
        chunk: bool = True,
    ) -> list[str]:
        """Load a file or URL via DocumentLoader, chunk it, and index it into the vector store.

        Args:
            file_path_or_url: Path to file or URL.
            chunk: Whether to chunk loaded documents (default: True).

        Returns:
            List of IDs assigned to the indexed documents.
        """
        docs = load_document(file_path_or_url)
        return self.index_documents(docs, chunk=chunk)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k most relevant document chunks for a query.

        Args:
            query: The question or search string.
            k: Maximum number of chunks to return.
            where: Optional metadata filter.

        Returns:
            List of result dicts with keys: id, text, metadata, distance.
        """
        return self.vector_store.query(query_text=query, n_results=k, where=where)

    def format_context(self, retrieved_chunks: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a clean context string for the prompt."""
        if not retrieved_chunks:
            return "No relevant context found in documents."

        formatted_parts = []
        for idx, item in enumerate(retrieved_chunks, start=1):
            source = item.get("metadata", {}).get("source", "unknown source") if item.get("metadata") else "unknown"
            text = item.get("text", "").strip()
            formatted_parts.append(f"--- Document Chunk #{idx} [Source: {source}] ---\n{text}")

        return "\n\n".join(formatted_parts)

    def generate_prompt(self, question: str, context: str) -> str:
        """Build the full prompt by filling the template."""
        return self.system_prompt_template.format(context=context, question=question)

    def query(
        self,
        question: str,
        k: int = 3,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the full RAG query: retrieve context and generate answer.

        Args:
            question: The user query.
            k: Number of context chunks to retrieve.
            where: Optional metadata filter.

        Returns:
            Dict containing 'question', 'answer', 'sources', 'context', and 'raw_results'.
        """
        retrieved = self.retrieve(query=question, k=k, where=where)
        context = self.format_context(retrieved)
        prompt = self.generate_prompt(question=question, context=context)

        sources = []
        for r in retrieved:
            meta = r.get("metadata") or {}
            source_id = meta.get("source") or meta.get("filename") or r.get("id")
            if source_id and source_id not in sources:
                sources.append(source_id)

        # Generate response using LLM if configured
        answer = ""
        llm_instance = self.llm
        if llm_instance is not None:
            try:
                response = llm_instance.invoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)
            except Exception as err:
                answer = f"[LLM Error: {err}] (Retrieved {len(retrieved)} relevant context chunks successfully)."
        else:
            answer = (
                "[LLM not configured: OPENROUTER_API_KEY missing]. "
                f"Retrieved {len(retrieved)} relevant context chunk(s) from vector store."
            )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "context": context,
            "retrieved_count": len(retrieved),
            "raw_results": retrieved,
        }


    def as_retriever(self, k: int = 3, where: dict[str, Any] | None = None) -> Any:
        """Convert the internal vector store to a LangChain BaseRetriever."""
        class _PipelineRetriever:
            def __init__(self, pipeline: RAGPipeline, top_k: int, filter_dict: dict[str, Any] | None):
                self.pipeline = pipeline
                self.top_k = top_k
                self.filter_dict = filter_dict

            def invoke(self, query_text: str) -> list[Document]:
                results = self.pipeline.retrieve(query_text, k=self.top_k, where=self.filter_dict)
                docs = []
                for r in results:
                    docs.append(
                        Document(
                            page_content=r.get("text") or "",
                            metadata=r.get("metadata") or {"id": r.get("id")},
                        )
                    )
                return docs

            def __call__(self, query_text: str) -> list[Document]:
                return self.invoke(query_text)

        return _PipelineRetriever(self, top_k=k, filter_dict=where)

    def query_with_sources(
        self,
        question: str,
        k: int = 3,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute RAG query with formatted citations, source attribution, and metadata.

        Args:
            question: The user query string.
            k: Number of source document chunks to retrieve.
            where: Optional metadata filter dict.

        Returns:
            Dictionary containing:
              - 'question': Original user query.
              - 'answer': Generated answer with inline bracketed citations (e.g. [1], [2]).
              - 'sources': Structured list of cited sources with source path, snippet, and citation key.
              - 'context': Numbered context string fed to LLM.
              - 'raw_results': Underlying vector store hits.
        """
        retriever = self.as_retriever(k=k, where=where)
        chain = create_rag_with_sources_chain(retriever=retriever, llm=self.llm)
        try:
            return chain.invoke({"question": question})
        except Exception as e:
            # Fallback if LLM is unavailable or missing API key
            retrieved = self.retrieve(question, k=k, where=where)
            context = format_docs_with_sources(retrieved)
            sources = []
            for i, r in enumerate(retrieved, 1):
                meta = r.get("metadata") or {}
                sources.append({
                    "citation": f"[{i}]",
                    "source": meta.get("source") or meta.get("filename") or f"doc_{i}",
                    "metadata": meta,
                    "content_preview": " ".join((r.get("text") or "").split())[:120] + "...",
                })
            return {
                "question": question,
                "answer": f"[LLM Invocation Notice: {e}]. Retrieved {len(retrieved)} source documents successfully.",
                "sources": sources,
                "context": context,
                "source_documents": retrieved,
            }


def rag_with_sources(
    question: str,
    documents: Sequence[Document | str] | None = None,
    collection_name: str = "rag_sources_default",
    k: int = 3,
    llm: Any = None,
) -> dict[str, Any]:
    """One-line convenience function to execute RAG with citations and source attribution.

    Args:
        question: The user query.
        documents: Optional documents or strings to index into vector store first.
        collection_name: Name of the vector collection.
        k: Number of source documents to retrieve (default: 3).
        llm: Configured LLM instance.

    Returns:
        Dict containing 'question', 'answer', 'sources', 'context', and 'source_documents'.
    """
    pipeline = create_rag_pipeline(collection_name=collection_name, llm=llm)
    if documents:
        if isinstance(documents[0], str):
            pipeline.index_texts(list(documents), chunk=True)  # type: ignore[arg-type]
        else:
            pipeline.index_documents(list(documents), chunk=True)  # type: ignore[arg-type]
    return pipeline.query_with_sources(question=question, k=k)


def create_rag_pipeline(
    collection_name: str = "rag_knowledge_base",
    persist_directory: str | None = None,
    llm: Any = None,
) -> RAGPipeline:
    """Create and return a configured RAGPipeline instance."""
    return RAGPipeline(
        collection_name=collection_name,
        persist_directory=persist_directory,
        llm=llm,
    )


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("        LangChain RAG Pipeline Demonstration        ")
    print("=" * 60)

    pipeline = create_rag_pipeline(collection_name="demo_rag_pipeline")

    # Sample knowledge base
    sample_kb = [
        "Thinking Machines: Inkling is a fast, free LLM available on OpenRouter.",
        "ChromaDB is an open-source vector database designed for building AI apps with embeddings.",
        "RAG (Retrieval-Augmented Generation) combines external document retrieval with an LLM.",
        "LangChain provides abstractions for document loaders, vector stores, and prompt chains.",
        "Vector embeddings represent the semantic meaning of text as high-dimensional coordinates.",
    ]

    print(f"\n[1] Indexing {len(sample_kb)} knowledge documents into ChromaDB...")
    pipeline.index_texts(sample_kb)
    print(f"    Indexed successfully! Collection count: {pipeline.vector_store.count()}")

    test_question = "What is ChromaDB used for?"
    print(f"\n[2] Asking question: '{test_question}'")
    result = pipeline.query(test_question, k=2)

    print(f"\n[3] Retrieved {result['retrieved_count']} relevant chunks:")
    for idx, r in enumerate(result["raw_results"], 1):
        print(f"    #{idx} [Distance: {r['distance']:.4f}]: {r['text']}")

    print(f"\n[4] Answer:")
    print(f"    {result['answer']}")
    print(f"\n[✔] RAG pipeline test complete.")
