"""Query Expansion, Query Rewriting, and Hypothetical Document Embeddings (HyDE).

Provides strategies to optimize retrieval recall and precision:
  1. Multi-Query Expansion: Generates semantically diverse queries and merges results via RRF.
  2. Query Rewriting: Resolves ambiguous references and strips conversational noise using chat history.
  3. Hypothetical Document Embeddings (HyDE): Generates a plausible answer passage to search in vector space.
  4. Query Decomposition: Breaks multi-part queries into standalone sub-queries.
  5. Step-Back Prompting: Generates a higher-level foundational question.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Sequence

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pydantic import ConfigDict, Field
from langchain_core.retrievers import BaseRetriever

from langchain_rag.llm import get_llm

logger = logging.getLogger("langchain_rag.query_expansion")

# ═════════════════════════════════════════════════════════════════════════════
# Prompts
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_MULTI_QUERY_PROMPT = """You are an AI language model assistant specialized in search and retrieval.
Your task is to generate {num_queries} different versions or perspectives of the given user query to retrieve relevant documents from a knowledge base.
Provide these alternative queries separated by newlines.
Do not number them, use bullet points, or add introductory/concluding remarks.
Output ONLY the alternative queries, one per line.

Original query: {query}"""

DEFAULT_REWRITE_PROMPT = """You are an AI assistant specialized in optimizing search queries for retrieval systems.
Your task is to reformulate the user's latest query into a clear, standalone, search-optimized query.
- Resolve any ambiguous pronouns or references (e.g. 'it', 'they', 'the previous tool') using the conversation history if present.
- Strip conversational filler (e.g. 'Can you please tell me', 'I want to know', 'Hello').
- Retain specific keywords, technical terms, error codes, and entity names.
- Output ONLY the rewritten query, without quotes or explanation.

Conversation History:
{history}

User Query: {query}

Rewritten Search Query:"""

DEFAULT_HYDE_PROMPT = """You are an expert technical documentation assistant.
Write a concise, factual hypothetical passage that answers the following question.
Focus on likely terminology, entities, and mechanisms that would appear in authoritative documentation.
Do not say 'Here is a passage' or 'Based on the question'.
Output ONLY the passage.

Question: {query}

Passage:"""

DEFAULT_DECOMPOSE_PROMPT = """You are an AI assistant that decomposes complex multi-part questions into simpler, independent sub-questions.
Decompose the following question into 2 to 4 distinct sub-questions that together answer the whole question.
Provide each sub-question on a new line without numbers or bullet points.
Output ONLY the sub-questions.

Question: {query}"""

DEFAULT_STEP_BACK_PROMPT = """You are an expert at problem-solving and abstraction.
Given a specific question, generate a broader, more abstract 'step-back' question that retrieves the foundational concepts, background principles, or overarching architecture.
Output ONLY the step-back question without quotes or introductory text.

Specific Question: {query}

Step-Back Question:"""


def _clean_query_lines(raw_text: str) -> list[str]:
    """Parse newline-delimited queries, removing numbering, bullets, and blank lines."""
    lines = []
    for line in raw_text.strip().splitlines():
        # Remove numbers like "1. ", "1) ", bullets like "- ", "* "
        cleaned = re.sub(r"^(?:\d+[\.\)]|\*|-|\u2022)\s*", "", line).strip()
        # Remove surrounding quotes
        cleaned = re.sub(r"^[\"']|[\"']$", "", cleaned).strip()
        if cleaned and len(cleaned) > 2:
            lines.append(cleaned)
    return lines


# ═════════════════════════════════════════════════════════════════════════════
# Core Query Expander / Transformer
# ═════════════════════════════════════════════════════════════════════════════

class QueryExpander:
    """Production Query Expander and Transformer for RAG systems."""

    def __init__(self, llm: Any = None) -> None:
        """Initialize QueryExpander with an optional LLM.

        Args:
            llm: Configured LangChain Chat/LLM instance. If None, loaded lazily.
        """
        self._llm = llm

    @property
    def llm(self) -> Any:
        """Lazily initialize LLM if needed."""
        if self._llm is None:
            try:
                self._llm = get_llm()
            except Exception as e:
                logger.warning(f"Could not initialize LLM for QueryExpander: {e}")
                self._llm = None
        return self._llm

    def _call_llm(self, prompt: str) -> str:
        """Helper to invoke LLM safely and return stripped string."""
        active_llm = self.llm
        if active_llm is None:
            raise RuntimeError("LLM is not configured or unavailable.")

        response = active_llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return str(content).strip()

    def expand_query(
        self,
        query: str,
        num_queries: int = 3,
        include_original: bool = True,
    ) -> list[str]:
        """Generate multiple semantically diverse perspectives of a query.

        Args:
            query: The initial user query.
            num_queries: Number of alternative queries to generate.
            include_original: Whether to include the original query as the first item.

        Returns:
            List of search queries starting with original if include_original=True.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        variations: list[str] = [clean_query] if include_original else []

        try:
            prompt = DEFAULT_MULTI_QUERY_PROMPT.format(
                query=clean_query,
                num_queries=num_queries,
            )
            raw_output = self._call_llm(prompt)
            parsed_lines = _clean_query_lines(raw_output)

            for line in parsed_lines:
                if line.lower() != clean_query.lower() and line not in variations:
                    variations.append(line)
        except Exception as exc:
            logger.warning(f"Query expansion failed ({exc}); falling back to original query.")
            if not variations:
                variations = [clean_query]

        return variations

    def rewrite_query(
        self,
        query: str,
        chat_history: list[dict[str, str]] | Sequence[tuple[str, str]] | str | None = None,
    ) -> str:
        """Rewrite a conversational or ambiguous query into a standalone search query.

        Args:
            query: The latest user question.
            chat_history: Optional prior messages (e.g. list of dicts with role/content,
                          list of (user, assistant) tuples, or raw string).

        Returns:
            Optimized, standalone query string.
        """
        clean_query = query.strip()
        if not clean_query:
            return ""

        formatted_history = "None provided."
        if chat_history:
            if isinstance(chat_history, str):
                formatted_history = chat_history.strip()
            elif isinstance(chat_history, (list, tuple)):
                history_lines = []
                for turn in chat_history:
                    if isinstance(turn, dict):
                        role = turn.get("role", "User").capitalize()
                        content = turn.get("content", "")
                        history_lines.append(f"{role}: {content}")
                    elif isinstance(turn, (tuple, list)) and len(turn) >= 2:
                        history_lines.append(f"User: {turn[0]}\nAssistant: {turn[1]}")
                if history_lines:
                    formatted_history = "\n".join(history_lines)

        try:
            prompt = DEFAULT_REWRITE_PROMPT.format(
                history=formatted_history,
                query=clean_query,
            )
            rewritten = self._call_llm(prompt)
            cleaned = re.sub(r"^[\"']|[\"']$", "", rewritten.strip())
            return cleaned if cleaned else clean_query
        except Exception as exc:
            logger.warning(f"Query rewrite failed ({exc}); using original query.")
            return clean_query

    def generate_hyde_doc(self, query: str) -> str:
        """Generate a hypothetical document passage for HyDE (Hypothetical Document Embeddings).

        Args:
            query: The user's question.

        Returns:
            A hypothetical answer passage to be embedded for similarity search.
        """
        clean_query = query.strip()
        if not clean_query:
            return ""

        try:
            prompt = DEFAULT_HYDE_PROMPT.format(query=clean_query)
            passage = self._call_llm(prompt)
            return passage.strip() if passage.strip() else clean_query
        except Exception as exc:
            logger.warning(f"HyDE generation failed ({exc}); using original query.")
            return clean_query

    def decompose_query(self, query: str) -> list[str]:
        """Decompose a complex question into independent sub-queries.

        Args:
            query: The user query.

        Returns:
            List of sub-queries.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        try:
            prompt = DEFAULT_DECOMPOSE_PROMPT.format(query=clean_query)
            raw = self._call_llm(prompt)
            lines = _clean_query_lines(raw)
            return lines if lines else [clean_query]
        except Exception as exc:
            logger.warning(f"Query decomposition failed ({exc}); using original query.")
            return [clean_query]

    def step_back_query(self, query: str) -> str:
        """Generate a higher-level step-back concept query.

        Args:
            query: The specific question.

        Returns:
            Broader step-back query.
        """
        clean_query = query.strip()
        if not clean_query:
            return ""

        try:
            prompt = DEFAULT_STEP_BACK_PROMPT.format(query=clean_query)
            result = self._call_llm(prompt)
            cleaned = re.sub(r"^[\"']|[\"']$", "", result.strip())
            return cleaned if cleaned else clean_query
        except Exception as exc:
            logger.warning(f"Step-back query failed ({exc}); using original query.")
            return clean_query


# ═════════════════════════════════════════════════════════════════════════════
# Reciprocal Rank Fusion for Multi-Query Results
# ═════════════════════════════════════════════════════════════════════════════

def fuse_multiquery_results(
    all_results: list[tuple[str, list[dict[str, Any]]]],
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Fuse retrieved documents from multiple query variations using Reciprocal Rank Fusion (RRF).

    Args:
        all_results: List of (query_string, result_list) tuples where each result_list
                     contains dicts with keys like 'id', 'text', 'metadata'.
        rrf_k: Smoothing constant for RRF (default: 60).
        top_k: Number of fused results to return.

    Returns:
        List of deduplicated result dicts ranked by combined RRF score.
    """
    scores: dict[str, float] = {}
    doc_data: dict[str, dict[str, Any]] = {}
    query_sources: dict[str, list[str]] = {}
    best_ranks: dict[str, int] = {}

    for query_text, results in all_results:
        for rank, item in enumerate(results, start=1):
            doc_id = str(item.get("id") or item.get("text", "")[:64])
            rrf_score = 1.0 / (rrf_k + rank)

            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = item
                query_sources[doc_id] = []
                best_ranks[doc_id] = rank
            else:
                best_ranks[doc_id] = min(best_ranks[doc_id], rank)

            if query_text not in query_sources[doc_id]:
                query_sources[doc_id].append(query_text)

    # Sort doc_ids descending by fused score
    sorted_ids = sorted(scores.keys(), key=lambda did: scores[did], reverse=True)[:top_k]

    fused_results: list[dict[str, Any]] = []
    for rank, doc_id in enumerate(sorted_ids, start=1):
        item = dict(doc_data[doc_id])
        item["fused_rank"] = rank
        item["fused_score"] = round(scores[doc_id], 6)
        item["matched_queries"] = query_sources[doc_id]
        item["query_count"] = len(query_sources[doc_id])
        item["best_individual_rank"] = best_ranks[doc_id]
        fused_results.append(item)

    return fused_results


# ═════════════════════════════════════════════════════════════════════════════
# Multi-Query Retriever & Adapters
# ═════════════════════════════════════════════════════════════════════════════

class MultiQueryRetriever:
    """Production Multi-Query Retriever that expands queries and fuses hits with RRF."""

    def __init__(
        self,
        retriever: Any,
        expander: QueryExpander | None = None,
        num_queries: int = 3,
        rrf_k: int = 60,
    ) -> None:
        """Initialize MultiQueryRetriever.

        Args:
            retriever: Search target. Can be:
                       - An object with `.query(query_text, n_results=...)` (e.g. VectorStore)
                       - An object with `.search(query, k=...)` (e.g. HybridSearchEngine)
                       - An object with `.retrieve(query, k=...)` (e.g. RAGPipeline)
                       - A LangChain BaseRetriever or object with `.invoke(query)`
                       - A callable: `search_fn(query: str, k: int)`
            expander: QueryExpander instance. If None, initialized with default LLM.
            num_queries: Number of query variations to generate (default: 3).
            rrf_k: Constant for Reciprocal Rank Fusion (default: 60).
        """
        self.retriever = retriever
        self.expander = expander or QueryExpander()
        self.num_queries = num_queries
        self.rrf_k = rrf_k

    def _execute_single_query(self, query: str, k: int) -> list[dict[str, Any]]:
        """Dispatch query to the underlying retriever and normalize results into dict format."""
        target = self.retriever

        # 1. LangChain BaseRetriever
        if isinstance(target, BaseRetriever):
            raw = target.invoke(query)
        # 2. HybridSearchEngine
        elif hasattr(target, "search") and callable(getattr(target, "search", None)):
            raw = target.search(query, k=k)
        # 3. VectorStore
        elif hasattr(target, "query") and callable(getattr(target, "query", None)):
            raw = target.query(query_text=query, n_results=k)
        # 4. RAGPipeline
        elif hasattr(target, "retrieve") and callable(getattr(target, "retrieve", None)):
            raw = target.retrieve(query=query, k=k)
        # 5. Other Runnable / LangChain object with .invoke
        elif hasattr(target, "invoke") and callable(getattr(target, "invoke", None)):
            raw = target.invoke(query)
        # 6. Custom search callable
        elif callable(target):
            raw = target(query, k)
        else:
            raise TypeError(f"Unsupported retriever object: {type(target)}")


        # Normalize to list of dicts: {"id": ..., "text": ..., "metadata": ...}
        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(raw or []):
            if isinstance(item, Document):
                normalized.append({
                    "id": item.metadata.get("id") or item.metadata.get("doc_id") or f"doc_{idx}",
                    "text": item.page_content,
                    "metadata": item.metadata,
                })
            elif isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({
                    "id": f"doc_{idx}",
                    "text": str(item),
                    "metadata": {},
                })
        return normalized

    def retrieve(
        self,
        query: str,
        k: int = 5,
        num_queries: int | None = None,
        return_expansion_details: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[str]]:
        """Retrieve documents by generating multiple queries and fusing with RRF.

        Args:
            query: The user input query.
            k: Top-k fused documents to return.
            num_queries: Optional override for number of queries to generate.
            return_expansion_details: If True, returns (results, expanded_queries).

        Returns:
            Ranked list of fused document dicts (or tuple with queries if requested).
        """
        n_queries = num_queries or self.num_queries
        expanded_queries = self.expander.expand_query(
            query=query,
            num_queries=n_queries,
            include_original=True,
        )

        all_results: list[tuple[str, list[dict[str, Any]]]] = []
        for q in expanded_queries:
            results = self._execute_single_query(q, k=k)
            all_results.append((q, results))

        fused = fuse_multiquery_results(all_results, rrf_k=self.rrf_k, top_k=k)

        if return_expansion_details:
            return fused, expanded_queries
        return fused

    def as_retriever(self, k: int = 5) -> BaseRetriever:
        """Convert this MultiQueryRetriever into a LangChain BaseRetriever."""
        return LangChainMultiQueryRetrieverAdapter(multi_query_retriever=self, k=k)


class LangChainMultiQueryRetrieverAdapter(BaseRetriever):
    """LangChain BaseRetriever adapter for MultiQueryRetriever."""

    multi_query_retriever: MultiQueryRetriever = Field(description="Underlying MultiQueryRetriever")
    k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        results = self.multi_query_retriever.retrieve(query=query, k=self.k)
        docs: list[Document] = []
        for r in results:
            meta = dict(r.get("metadata") or {})
            meta.update({
                "fused_rank": r.get("fused_rank"),
                "fused_score": r.get("fused_score"),
                "matched_queries": r.get("matched_queries"),
                "query_count": r.get("query_count"),
            })
            docs.append(Document(page_content=r.get("text", ""), metadata=meta))
        return docs


# ═════════════════════════════════════════════════════════════════════════════
# HyDE Retriever & Adapters
# ═════════════════════════════════════════════════════════════════════════════

class HyDERetriever:
    """Hypothetical Document Embeddings (HyDE) Retriever.

    Generates a plausible answer passage using an LLM and queries the underlying
    vector store using the passage's dense embedding.
    """

    def __init__(
        self,
        vector_retriever: Any,
        expander: QueryExpander | None = None,
    ) -> None:
        """Initialize HyDERetriever.

        Args:
            vector_retriever: Dense retriever (VectorStore, RAGPipeline, or LangChain retriever).
            expander: QueryExpander instance for generating the hypothetical passage.
        """
        self.vector_retriever = vector_retriever
        self.expander = expander or QueryExpander()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        return_hypothetical_doc: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
        """Generate a hypothetical document and retrieve matching chunks.

        Args:
            query: The user question.
            k: Top-k chunks to return.
            return_hypothetical_doc: If True, returns (results, hypothetical_passage).

        Returns:
            List of matching document dicts (or tuple with hypothetical doc).
        """
        hypothetical_doc = self.expander.generate_hyde_doc(query)

        # Retrieve using the generated passage
        target = self.vector_retriever
        if hasattr(target, "query") and callable(target.query):
            raw = target.query(query_text=hypothetical_doc, n_results=k)
        elif hasattr(target, "retrieve") and callable(target.retrieve):
            raw = target.retrieve(query=hypothetical_doc, k=k)
        elif hasattr(target, "invoke") and callable(target.invoke):
            docs = target.invoke(hypothetical_doc)
            raw = [
                {
                    "id": d.metadata.get("id", f"doc_{i}"),
                    "text": d.page_content,
                    "metadata": d.metadata,
                }
                for i, d in enumerate(docs)
            ]
        else:
            raise TypeError(f"Unsupported retriever for HyDE: {type(target)}")

        for r in raw:
            r["retrieval_mode"] = "hyde"
            r["original_query"] = query

        if return_hypothetical_doc:
            return raw, hypothetical_doc
        return raw

    def as_retriever(self, k: int = 5) -> BaseRetriever:
        """Convert this HyDERetriever into a LangChain BaseRetriever."""
        return LangChainHyDERetrieverAdapter(hyde_retriever=self, k=k)


class LangChainHyDERetrieverAdapter(BaseRetriever):
    """LangChain BaseRetriever adapter for HyDERetriever."""

    hyde_retriever: HyDERetriever = Field(description="Underlying HyDERetriever")
    k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        results = self.hyde_retriever.retrieve(query=query, k=self.k)
        docs: list[Document] = []
        for r in results:
            meta = dict(r.get("metadata") or {})
            meta.update({"retrieval_mode": "hyde"})
            docs.append(Document(page_content=r.get("text", ""), metadata=meta))
        return docs


# ═════════════════════════════════════════════════════════════════════════════
# Convenience Factory Functions
# ═════════════════════════════════════════════════════════════════════════════

def create_query_expander(llm: Any = None) -> QueryExpander:
    """Create a configured QueryExpander instance."""
    return QueryExpander(llm=llm)


def create_multi_query_retriever(
    retriever: Any,
    llm: Any = None,
    num_queries: int = 3,
    rrf_k: int = 60,
) -> MultiQueryRetriever:
    """Create a configured MultiQueryRetriever instance."""
    expander = QueryExpander(llm=llm)
    return MultiQueryRetriever(
        retriever=retriever,
        expander=expander,
        num_queries=num_queries,
        rrf_k=rrf_k,
    )


def create_hyde_retriever(
    vector_retriever: Any,
    llm: Any = None,
) -> HyDERetriever:
    """Create a configured HyDERetriever instance."""
    expander = QueryExpander(llm=llm)
    return HyDERetriever(vector_retriever=vector_retriever, expander=expander)
