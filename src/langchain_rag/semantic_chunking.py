"""Production-Ready Semantic & Multi-Strategy Chunking Engine for LangChain RAG.

Provides enterprise-grade document chunking with:
  1. Adaptive Semantic Chunking with dynamic statistical thresholds (percentile,
     std-dev, IQR, gradient, fixed similarity) and min/max chunk guardrails.
  2. Hierarchical / Parent-Child Chunking (small chunks for vector precision,
     large parent chunks for complete LLM context).
  3. Contextual & Markdown-Aware Chunking (header breadcrumb preservation,
     table/code protection, contextual prefix injection).
  4. Embedding caching, batching, and chunk quality distribution diagnostics.
"""

from __future__ import annotations

import math
import re
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_rag.embeddings import cosine_similarity, get_embeddings


@dataclass
class ChunkMetrics:
    """Statistical metrics for chunked document distributions."""

    total_chunks: int
    total_characters: int
    total_estimated_tokens: int
    min_chunk_chars: int
    max_chunk_chars: int
    mean_chunk_chars: float
    median_chunk_chars: float
    std_dev_chunk_chars: float
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_chunks": self.total_chunks,
            "total_characters": self.total_characters,
            "total_estimated_tokens": self.total_estimated_tokens,
            "min_chunk_chars": self.min_chunk_chars,
            "max_chunk_chars": self.max_chunk_chars,
            "mean_chunk_chars": round(self.mean_chunk_chars, 1),
            "median_chunk_chars": round(self.median_chunk_chars, 1),
            "std_dev_chunk_chars": round(self.std_dev_chunk_chars, 1),
            "duration_ms": round(self.duration_ms, 2),
        }


def estimate_tokens(text: str) -> int:
    """Rough token estimation (~4 characters per token for English)."""
    return max(1, math.ceil(len(text) / 4))


class SentenceSplitter:
    """Splits text into coherent sentences while respecting code blocks, abbreviations, and list items."""

    # Common abbreviations to protect
    KNOWN_ABBREVS = {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.",
        "e.g.", "i.e.", "approx.", "inc.", "ltd.", "dept.", "fig.", "al.",
        "no.", "vol.", "est.", "jan.", "feb.", "mar.", "apr.", "jun.", "jul.",
        "aug.", "sep.", "oct.", "nov.", "dec.",
    }

    @classmethod
    def split(cls, text: str) -> list[str]:
        """Split text into a list of sentence strings."""
        if not text:
            return []

        # Split on double newlines (paragraphs) first to maintain macro structure
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        sentences: list[str] = []

        for p in paragraphs:
            # If paragraph contains markdown code blocks, do not aggressively split inside them
            if p.startswith("```") and p.endswith("```"):
                sentences.append(p)
                continue

            # Tokenize into potential sentences
            # Match sentence ending punctuation followed by space and uppercase/quote/bracket
            raw_parts = re.split(r"(?<=[.?!])\s+(?=[A-Z0-9\"'(\[#])", p)

            # Rejoin erroneously split abbreviations (e.g., "Dr." or "e.g.")
            reconstructed: list[str] = []
            for part in raw_parts:
                part = part.strip()
                if not part:
                    continue

                if reconstructed:
                    prev_last_word = reconstructed[-1].split()[-1].lower() if reconstructed[-1].split() else ""
                    if prev_last_word in cls.KNOWN_ABBREVS or re.match(r"^[A-Za-z]\.$", prev_last_word):
                        # Merge with previous sentence
                        reconstructed[-1] = reconstructed[-1] + " " + part
                        continue

                reconstructed.append(part)

            sentences.extend(reconstructed)

        return sentences


class EmbeddingCache:
    """In-memory cache for sentence and text embeddings to prevent duplicate computations."""

    def __init__(self, embeddings: Embeddings) -> None:
        self.embeddings = embeddings
        self.cache: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed list of texts with cache lookup and batch embedding for misses."""
        results: list[Optional[list[float]]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if text in self.cache:
                results[i] = self.cache[text]
                self.hits += 1
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
                self.misses += 1

        if uncached_texts:
            new_vectors = self.embeddings.embed_documents(uncached_texts)
            for idx, vec, text in zip(uncached_indices, new_vectors, uncached_texts):
                self.cache[text] = vec
                results[idx] = vec

        return [v for v in results if v is not None]

    def clear(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0


BreakpointType = Literal["percentile", "standard_deviation", "interquartile", "gradient", "fixed"]


class ProductionSemanticChunker:
    """Enterprise-grade semantic chunker with adaptive statistical thresholding and size guardrails."""

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        breakpoint_threshold_type: BreakpointType = "percentile",
        breakpoint_threshold_amount: float = 95.0,
        buffer_size: int = 1,
        min_chunk_chars: int = 100,
        max_chunk_chars: int = 1800,
        embedding_cache: Optional[EmbeddingCache] = None,
    ) -> None:
        """Initialize the ProductionSemanticChunker.

        Args:
            embeddings: LangChain Embeddings instance. Defaults to local ONNX model.
            breakpoint_threshold_type: Method to calculate distance threshold:
                - 'percentile': Cut at specified percentile (e.g., 95.0) of distance spikes.
                - 'standard_deviation': Cut at mean + (amount * std_dev) of distances.
                - 'interquartile': Cut at Q3 + (amount * IQR) (Tukey outlier detection).
                - 'gradient': Cut where rate of change in distance exceeds threshold.
                - 'fixed': Direct cosine distance threshold [0.0 - 1.0].
            breakpoint_threshold_amount: Parameter value for chosen threshold type.
            buffer_size: Number of neighboring sentences combined into context window before embedding.
            min_chunk_chars: Hard minimum characters per chunk (merges small fragments).
            max_chunk_chars: Hard maximum characters per chunk (splits oversized chunks).
            embedding_cache: Optional shared EmbeddingCache instance.
        """
        self.embeddings = embeddings or get_embeddings("local")
        self.cache = embedding_cache or EmbeddingCache(self.embeddings)
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.buffer_size = max(1, buffer_size)
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    def _create_sentence_buffers(self, sentences: list[str]) -> list[dict[str, Any]]:
        """Group sentences with adjacent context to produce smoother semantic representations."""
        buffered_sentences: list[dict[str, Any]] = []

        for i in range(len(sentences)):
            start_idx = max(0, i - self.buffer_size)
            end_idx = min(len(sentences), i + 1 + self.buffer_size)
            combined_window = " ".join(sentences[start_idx:end_idx])

            buffered_sentences.append({
                "index": i,
                "sentence": sentences[i],
                "combined_window": combined_window,
            })

        return buffered_sentences

    def _calculate_distances(self, buffered_sentences: list[dict[str, Any]]) -> list[float]:
        """Compute cosine distance (1.0 - similarity) between consecutive sentence windows."""
        if len(buffered_sentences) <= 1:
            return []

        window_texts = [b["combined_window"] for b in buffered_sentences]
        vectors = self.cache.embed_texts(window_texts)

        distances: list[float] = []
        for i in range(len(vectors) - 1):
            sim = cosine_similarity(vectors[i], vectors[i + 1])
            # Cosine distance: 0.0 means identical, 1.0 means orthogonal
            dist = max(0.0, 1.0 - sim)
            distances.append(dist)

        return distances

    def _determine_threshold(self, distances: list[float]) -> float:
        """Calculate dynamic distance cutoff threshold based on selected statistical method."""
        if not distances:
            return 0.0

        if len(distances) == 1:
            return distances[0]

        if self.breakpoint_threshold_type == "percentile":
            p = min(100.0, max(0.0, self.breakpoint_threshold_amount))
            sorted_d = sorted(distances)
            k = (len(sorted_d) - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_d[int(k)]
            d0 = sorted_d[int(f)] * (c - k)
            d1 = sorted_d[int(c)] * (k - f)
            return d0 + d1

        elif self.breakpoint_threshold_type == "standard_deviation":
            mean_dist = statistics.mean(distances)
            std_dist = statistics.stdev(distances) if len(distances) > 1 else 0.0
            return mean_dist + (self.breakpoint_threshold_amount * std_dist)

        elif self.breakpoint_threshold_type == "interquartile":
            sorted_d = sorted(distances)
            n = len(sorted_d)
            q1 = sorted_d[int(n * 0.25)]
            q3 = sorted_d[int(n * 0.75)]
            iqr = q3 - q1
            multiplier = self.breakpoint_threshold_amount if self.breakpoint_threshold_amount != 95.0 else 1.5
            return q3 + (multiplier * iqr)

        elif self.breakpoint_threshold_type == "gradient":
            # Identify sharp sudden spikes in distance change (derivative)
            gradients = [abs(distances[i + 1] - distances[i]) for i in range(len(distances) - 1)]
            if not gradients:
                return statistics.mean(distances)
            mean_grad = statistics.mean(gradients)
            std_grad = statistics.stdev(gradients) if len(gradients) > 1 else 0.0
            return mean_grad + (self.breakpoint_threshold_amount * 0.01 * std_grad)

        elif self.breakpoint_threshold_type == "fixed":
            return float(self.breakpoint_threshold_amount)

        return statistics.mean(distances)

    def _split_into_raw_chunks(self, sentences: list[str]) -> Tuple[list[str], list[float], float]:
        """Split sentences into raw semantic chunks based on threshold breakpoints."""
        if not sentences:
            return [], [], 0.0

        if len(sentences) == 1:
            return sentences, [], 0.0

        buffered = self._create_sentence_buffers(sentences)
        distances = self._calculate_distances(buffered)
        threshold = self._determine_threshold(distances)

        chunks: list[str] = []
        current_sentences: list[str] = [sentences[0]]

        for i, dist in enumerate(distances):
            if dist > threshold:
                # Semantic boundary detected!
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentences[i + 1]]
            else:
                current_sentences.append(sentences[i + 1])

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks, distances, threshold

    def _apply_guardrails(self, raw_chunks: list[str]) -> list[str]:
        """Apply min/max chunk size constraints to enforce production chunking safety."""
        if not raw_chunks:
            return []

        # 1. Enforce min_chunk_chars by merging tiny fragments with adjacent chunks
        merged_chunks: list[str] = []
        accumulator = ""

        for chunk in raw_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            if len(chunk) < self.min_chunk_chars:
                if accumulator:
                    accumulator = accumulator + " " + chunk
                else:
                    accumulator = chunk
            else:
                if accumulator:
                    if len(accumulator) + len(chunk) + 1 <= self.max_chunk_chars:
                        merged_chunks.append(accumulator + " " + chunk)
                        accumulator = ""
                    else:
                        merged_chunks.append(accumulator)
                        accumulator = chunk
                else:
                    merged_chunks.append(chunk)

        if accumulator:
            if merged_chunks and (len(merged_chunks[-1]) + len(accumulator) + 1 <= self.max_chunk_chars):
                merged_chunks[-1] = merged_chunks[-1] + " " + accumulator
            else:
                merged_chunks.append(accumulator)

        # 2. Enforce max_chunk_chars by sub-splitting oversized chunks recursively
        final_chunks: list[str] = []
        for chunk in merged_chunks:
            if len(chunk) <= self.max_chunk_chars:
                final_chunks.append(chunk)
            else:
                # Sub-split oversized chunk cleanly
                sub_sentences = SentenceSplitter.split(chunk)
                sub_acc: list[str] = []
                sub_len = 0
                for s in sub_sentences:
                    if sub_len + len(s) + 1 > self.max_chunk_chars and sub_acc:
                        final_chunks.append(" ".join(sub_acc))
                        sub_acc = [s]
                        sub_len = len(s)
                    else:
                        sub_acc.append(s)
                        sub_len += len(s) + 1
                if sub_acc:
                    final_chunks.append(" ".join(sub_acc))

        return final_chunks

    def split_text(self, text: str) -> list[str]:
        """Split text string into production semantic chunks."""
        sentences = SentenceSplitter.split(text)
        if not sentences:
            return []
        raw_chunks, _, _ = self._split_into_raw_chunks(sentences)
        return self._apply_guardrails(raw_chunks)

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        """Split LangChain Documents into semantically coherent Document chunks with rich metadata."""
        output_docs: list[Document] = []

        for doc_idx, doc in enumerate(documents):
            text = doc.page_content
            parent_metadata = dict(doc.metadata) if doc.metadata else {}
            doc_id = parent_metadata.get("id", f"doc_{doc_idx}_{uuid.uuid4().hex[:6]}")

            chunks = self.split_text(text)
            total_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_meta = dict(parent_metadata)
                chunk_meta.update({
                    "parent_id": doc_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": total_chunks,
                    "char_count": len(chunk),
                    "token_estimate": estimate_tokens(chunk),
                    "chunk_strategy": "semantic",
                    "threshold_type": self.breakpoint_threshold_type,
                })

                output_docs.append(Document(page_content=chunk, metadata=chunk_meta))

        return output_docs


class ParentChildChunker:
    """Hierarchical chunker that pairs fine-grained child chunks with full parent context documents."""

    def __init__(
        self,
        parent_chunk_size: int = 1600,
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 350,
        child_chunk_overlap: int = 50,
        semantic_child_chunking: bool = True,
        embeddings: Optional[Embeddings] = None,
    ) -> None:
        """Initialize Hierarchical Parent-Child Chunker.

        Args:
            parent_chunk_size: Character size for parent context blocks.
            parent_chunk_overlap: Overlap between parent blocks.
            child_chunk_size: Character size for child dense retrieval blocks.
            child_chunk_overlap: Overlap between child blocks.
            semantic_child_chunking: Use semantic splitting for child chunks if True.
            embeddings: Optional embeddings instance for semantic child splitting.
        """
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.semantic_child_chunking = semantic_child_chunking
        self.embeddings = embeddings or get_embeddings("local")

        if self.semantic_child_chunking:
            self.child_splitter = ProductionSemanticChunker(
                embeddings=self.embeddings,
                min_chunk_chars=100,
                max_chunk_chars=child_chunk_size,
            )

    def split_document_hierarchical(
        self,
        document: Document,
    ) -> Tuple[list[Document], list[Document]]:
        """Split a document into both parent Documents and linked child Documents.

        Returns:
            Tuple of (parent_documents, child_documents).
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        )

        parent_docs = parent_splitter.split_documents([document])
        child_docs: list[Document] = []

        for p_idx, p_doc in enumerate(parent_docs):
            p_id = f"parent_{uuid.uuid4().hex[:8]}"
            p_doc.metadata["parent_id"] = p_id
            p_doc.metadata["parent_index"] = p_idx
            p_doc.metadata["role"] = "parent_context"

            if self.semantic_child_chunking:
                child_texts = self.child_splitter.split_text(p_doc.page_content)
            else:
                child_rec = RecursiveCharacterTextSplitter(
                    chunk_size=self.child_chunk_size,
                    chunk_overlap=self.child_chunk_overlap,
                )
                child_texts = child_rec.split_text(p_doc.page_content)

            for c_idx, c_text in enumerate(child_texts):
                c_meta = dict(document.metadata) if document.metadata else {}
                c_meta.update({
                    "parent_id": p_id,
                    "parent_text": p_doc.page_content,
                    "child_index": c_idx,
                    "total_siblings": len(child_texts),
                    "role": "child_dense_search",
                    "char_count": len(c_text),
                    "token_estimate": estimate_tokens(c_text),
                    "chunk_strategy": "parent_child",
                })
                child_docs.append(Document(page_content=c_text, metadata=c_meta))

        return parent_docs, child_docs


class ContextualMarkdownChunker:
    """Structure-aware chunker that preserves header breadcrumbs, tables, and code fences."""

    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(
        self,
        max_chunk_chars: int = 1200,
        inject_breadcrumbs: bool = True,
    ) -> None:
        """Initialize ContextualMarkdownChunker.

        Args:
            max_chunk_chars: Maximum characters allowed per section before sub-splitting.
            inject_breadcrumbs: If True, prepends document title & section hierarchy as context headers.
        """
        self.max_chunk_chars = max_chunk_chars
        self.inject_breadcrumbs = inject_breadcrumbs

    def split_markdown(self, markdown_text: str, source_name: str = "Document") -> list[Document]:
        """Parse markdown into hierarchy-aware documents with injected context breadcrumbs."""
        lines = markdown_text.split("\n")
        sections: list[dict[str, Any]] = []

        current_headers: dict[int, str] = {1: source_name}
        current_content_lines: list[str] = []
        current_level = 1
        in_code_block = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                current_content_lines.append(line)
                continue

            header_match = self.HEADER_PATTERN.match(line) if not in_code_block else None
            if header_match:
                # Save previous section if it has content
                if current_content_lines:
                    content = "\n".join(current_content_lines).strip()
                    if content:
                        breadcrumb_trail = " > ".join(
                            current_headers[lvl] for lvl in sorted(current_headers.keys())
                        )
                        sections.append({
                            "breadcrumbs": breadcrumb_trail,
                            "content": content,
                            "level": current_level,
                        })
                    current_content_lines = []

                hashes, title = header_match.groups()
                level = len(hashes)
                current_level = level

                # Clean header stack for levels >= current level
                current_headers = {lvl: h for lvl, h in current_headers.items() if lvl < level}
                current_headers[level] = title.strip()
            else:
                current_content_lines.append(line)

        # Append last section
        if current_content_lines:
            content = "\n".join(current_content_lines).strip()
            if content:
                breadcrumb_trail = " > ".join(
                    current_headers[lvl] for lvl in sorted(current_headers.keys())
                )
                sections.append({
                    "breadcrumbs": breadcrumb_trail,
                    "content": content,
                    "level": current_level,
                })

        # Process sections into chunk documents
        docs: list[Document] = []
        for idx, sec in enumerate(sections):
            content = sec["content"]
            breadcrumbs = sec["breadcrumbs"]

            # If section is within size limits
            if len(content) <= self.max_chunk_chars:
                text_to_store = f"[{breadcrumbs}]\n{content}" if self.inject_breadcrumbs else content
                docs.append(
                    Document(
                        page_content=text_to_store,
                        metadata={
                            "breadcrumbs": breadcrumbs,
                            "section_index": idx,
                            "char_count": len(text_to_store),
                            "token_estimate": estimate_tokens(text_to_store),
                            "chunk_strategy": "contextual_markdown",
                        },
                    )
                )
            else:
                # Sub-split oversized section using sentences
                sentences = SentenceSplitter.split(content)
                acc: list[str] = []
                acc_len = 0
                sub_idx = 0

                for s in sentences:
                    if acc_len + len(s) + 1 > self.max_chunk_chars and acc:
                        part_text = " ".join(acc)
                        text_to_store = (
                            f"[{breadcrumbs} (Part {sub_idx + 1})]\n{part_text}"
                            if self.inject_breadcrumbs
                            else part_text
                        )
                        docs.append(
                            Document(
                                page_content=text_to_store,
                                metadata={
                                    "breadcrumbs": breadcrumbs,
                                    "section_index": idx,
                                    "sub_index": sub_idx,
                                    "char_count": len(text_to_store),
                                    "token_estimate": estimate_tokens(text_to_store),
                                    "chunk_strategy": "contextual_markdown",
                                },
                            )
                        )
                        acc = [s]
                        acc_len = len(s)
                        sub_idx += 1
                    else:
                        acc.append(s)
                        acc_len += len(s) + 1

                if acc:
                    part_text = " ".join(acc)
                    text_to_store = (
                        f"[{breadcrumbs} (Part {sub_idx + 1})]\n{part_text}"
                        if self.inject_breadcrumbs
                        else part_text
                    )
                    docs.append(
                        Document(
                            page_content=text_to_store,
                            metadata={
                                "breadcrumbs": breadcrumbs,
                                "section_index": idx,
                                "sub_index": sub_idx,
                                "char_count": len(text_to_store),
                                "token_estimate": estimate_tokens(text_to_store),
                                "chunk_strategy": "contextual_markdown",
                            },
                        )
                    )

        return docs


class ProductionChunker:
    """Unified high-level facade for all production chunking strategies and analytics."""

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        default_strategy: Literal["semantic", "parent_child", "markdown", "recursive"] = "semantic",
    ) -> None:
        self.embeddings = embeddings or get_embeddings("local")
        self.cache = EmbeddingCache(self.embeddings)
        self.default_strategy = default_strategy
        self.semantic_chunker = ProductionSemanticChunker(
            embeddings=self.embeddings,
            embedding_cache=self.cache,
        )
        self.parent_child_chunker = ParentChildChunker(
            embeddings=self.embeddings,
        )
        self.markdown_chunker = ContextualMarkdownChunker()

    def chunk_document(
        self,
        document: Document,
        strategy: Optional[Literal["semantic", "parent_child", "markdown", "recursive"]] = None,
        **kwargs: Any,
    ) -> Tuple[list[Document], ChunkMetrics]:
        """Chunk a document using the specified strategy and compute detailed distribution metrics."""
        strat = strategy or self.default_strategy
        start_time = time.perf_counter()

        if strat == "semantic":
            chunker = ProductionSemanticChunker(
                embeddings=self.embeddings,
                embedding_cache=self.cache,
                breakpoint_threshold_type=kwargs.get("threshold_type", "percentile"),
                breakpoint_threshold_amount=kwargs.get("threshold_amount", 95.0),
                min_chunk_chars=kwargs.get("min_chunk_chars", 100),
                max_chunk_chars=kwargs.get("max_chunk_chars", 1800),
            )
            chunks = chunker.split_documents([document])

        elif strat == "parent_child":
            _, child_chunks = self.parent_child_chunker.split_document_hierarchical(document)
            chunks = child_chunks

        elif strat == "markdown":
            chunks = self.markdown_chunker.split_markdown(
                document.page_content,
                source_name=document.metadata.get("source", "Document"),
            )

        elif strat == "recursive":
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=kwargs.get("chunk_size", 500),
                chunk_overlap=kwargs.get("chunk_overlap", 50),
            )
            chunks = splitter.split_documents([document])
            for idx, c in enumerate(chunks):
                c.metadata["chunk_index"] = idx
                c.metadata["chunk_strategy"] = "recursive"
        else:
            raise ValueError(f"Unknown chunking strategy: '{strat}'")

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        metrics = self.calculate_metrics(chunks, duration_ms)
        return chunks, metrics

    @staticmethod
    def calculate_metrics(chunks: Sequence[Document], duration_ms: float = 0.0) -> ChunkMetrics:
        """Compute statistical distribution metrics over a sequence of chunks."""
        if not chunks:
            return ChunkMetrics(
                total_chunks=0,
                total_characters=0,
                total_estimated_tokens=0,
                min_chunk_chars=0,
                max_chunk_chars=0,
                mean_chunk_chars=0.0,
                median_chunk_chars=0.0,
                std_dev_chunk_chars=0.0,
                duration_ms=duration_ms,
            )

        lengths = [len(c.page_content) for c in chunks]
        total_chars = sum(lengths)
        total_tokens = sum(estimate_tokens(c.page_content) for c in chunks)

        return ChunkMetrics(
            total_chunks=len(chunks),
            total_characters=total_chars,
            total_estimated_tokens=total_tokens,
            min_chunk_chars=min(lengths),
            max_chunk_chars=max(lengths),
            mean_chunk_chars=statistics.mean(lengths),
            median_chunk_chars=statistics.median(lengths),
            std_dev_chunk_chars=statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
            duration_ms=duration_ms,
        )


def create_production_chunker(embeddings: Optional[Embeddings] = None) -> ProductionChunker:
    """Factory helper to create a configured ProductionChunker."""
    return ProductionChunker(embeddings=embeddings)


def smart_chunker(
    text: str,
    use_semantic: bool = True,
    fallback_chunk_size: int = 500,
) -> list[str]:
    """
    Production chunking with semantic as primary, recursive as fallback.
    """
    import os
    try:
        from langchain_openai import OpenAIEmbeddings
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key or api_key.startswith("your_"):
            raise ValueError("No valid API key for OpenAIEmbeddings")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception:
        embeddings = get_embeddings("local")

    if use_semantic:
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            chunker = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90,
            )
            return chunker.split_text(text)
        except Exception as e:
            print(f"Semantic chunking failed ({e}), falling back to recursive...")

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=fallback_chunk_size,
        chunk_overlap=50,
    )
    return fallback.split_text(text)

