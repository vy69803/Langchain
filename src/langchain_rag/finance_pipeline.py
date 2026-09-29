"""Enterprise Financial Ingestion and Retrieval Pipeline for FinanceBench.

Orchestrates SEC PDF filing discovery, financial document parsing, table structure
detection, dense ChromaDB vector indexing, sparse Okapi BM25 indexing, and two-stage
reranking with FlashRank.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from langchain_core.documents import Document

from langchain_rag.adapters.storage_adapters import BaseGraphStoreAdapter, Neo4jGraphAdapter
from langchain_rag.finance_parser import (
    FinanceBenchParser,
    compute_file_hash,
    sanitize_metadata_for_chroma,
)
from langchain_rag.graph.ingest_graph import FinancialGraphIngestor
from langchain_rag.hybrid_search import BM25Index
from langchain_rag.parsers.layout_table_parser import LayoutTableParser
from langchain_rag.vector_stores import VectorStore

logger = logging.getLogger("langchain_rag.finance_pipeline")


class FinanceBenchIngestionPipeline:
    """Enterprise Ingestion and Retrieval Pipeline for SEC Filings and FinanceBench."""

    def __init__(
        self,
        pdf_dir: str | Path = "data/financebench/pdfs",
        metadata_path: str | Path = "data/financebench/financebench_document_information.jsonl",
        persist_directory: str | Path = "./data/chroma_db",
        collection_name: str = "financebench",
        manifest_path: str | Path = "./data/financebench_manifest.json",
        bm25_path: str | Path = "./data/bm25_financebench.json",
        max_chunk_chars: int = 1200,
        batch_size: int = 50,
        enable_bm25: bool = True,
        enable_graph: bool = True,
        vector_store: VectorStore | None = None,
        graph_adapter: BaseGraphStoreAdapter | None = None,
        reranker: Any = None,
        reranker_model: str | None = None,
    ) -> None:
        """Initialize the pipeline configuration and storage backends.

        Args:
            pdf_dir: Directory containing SEC filing PDF documents.
            metadata_path: Path to the FinanceBench document metadata JSONL.
            persist_directory: Directory to persist ChromaDB vector embeddings.
            collection_name: Name of the ChromaDB collection.
            manifest_path: Path to JSON file tracking indexed file hashes for incremental updates.
            bm25_path: Path to serialized BM25 index state.
            max_chunk_chars: Maximum character length per chunk.
            batch_size: Number of chunks per database write batch.
            enable_bm25: Maintain a sparse lexical BM25 index alongside ChromaDB.
            enable_graph: Maintain a structured Knowledge Graph in Neo4j.
            vector_store: Optional pre-configured VectorStore.
            graph_adapter: Optional pre-configured Knowledge Graph adapter.
            reranker: Optional pre-configured FlashRank reranker instance.
            reranker_model: Model name for FlashRank reranker.
        """
        self.pdf_dir = Path(pdf_dir).resolve()
        self.metadata_path = Path(metadata_path).resolve()
        self.persist_directory = Path(persist_directory).resolve()
        self.collection_name = collection_name
        self.manifest_path = Path(manifest_path).resolve()
        self.bm25_path = Path(bm25_path).resolve()
        self.max_chunk_chars = max_chunk_chars
        self.batch_size = max(10, batch_size)
        self.enable_bm25 = enable_bm25
        self.enable_graph = enable_graph
        self._reranker = reranker
        self.reranker_model = reranker_model or os.environ.get("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")

        self.parser = FinanceBenchParser(metadata_path=self.metadata_path)
        self.table_parser = LayoutTableParser()
        self.graph_ingestor = FinancialGraphIngestor(graph_adapter=graph_adapter)
        self._vector_store: VectorStore | None = vector_store
        self._bm25_index: BM25Index | None = None

    @property
    def vector_store(self) -> VectorStore:
        """Lazy-loaded ChromaDB vector store instance."""
        if self._vector_store is None:
            self._vector_store = VectorStore(
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory),
            )
        return self._vector_store

    @property
    def bm25_index(self) -> BM25Index:
        """Lazy-loaded BM25 sparse index instance."""
        if self._bm25_index is None:
            self._bm25_index = BM25Index()
            if self.bm25_path.exists():
                try:
                    self._bm25_index.load_state(self.bm25_path)
                except Exception as e:
                    logger.warning("Failed to load existing BM25 index from %s: %s. Creating new index.", self.bm25_path, e)
        return self._bm25_index

    @property
    def reranker(self) -> Any:
        """Lazy-loaded FlashRank reranker instance."""
        if self._reranker is None:
            from langchain_rag.reranker import get_reranker
            self._reranker = get_reranker(model_name=self.reranker_model)
        return self._reranker

    def load_manifest(self) -> dict[str, Any]:
        """Load manifest tracking file content hashes and indexed metadata."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not read manifest at %s: %s. Rebuilding.", self.manifest_path, e)
        return {"version": 1, "collection": self.collection_name, "files": {}}

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        """Persist manifest atomically to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.manifest_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        tmp_path.replace(self.manifest_path)

    def discover_pdfs(
        self,
        company: str | None = None,
        doc_type: str | None = None,
        sample: int | None = None,
    ) -> list[Path]:
        """Discover SEC filing PDF documents matching optional company and type filters."""
        if not self.pdf_dir.exists():
            logger.warning("PDF directory does not exist: %s", self.pdf_dir)
            return []

        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))

        if company:
            c_lower = company.lower()
            pdf_files = [p for p in pdf_files if c_lower in p.name.lower()]

        if doc_type:
            dt_lower = doc_type.lower()
            pdf_files = [p for p in pdf_files if dt_lower in p.name.lower()]

        if sample and sample > 0:
            pdf_files = pdf_files[:sample]

        return pdf_files

    def ingest(
        self,
        force: bool = False,
        dry_run: bool = False,
        sample: int | None = None,
        company: int | str | None = None,
        doc_type: str | None = None,
        max_pages_per_doc: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Ingest FinanceBench SEC filings into ChromaDB and BM25 index.

        Args:
            force: Force re-indexing all documents, ignoring manifest hash.
            dry_run: Parse and chunk files without writing to storage backends.
            sample: Limit the total number of documents processed.
            company: Optional filter for company name.
            doc_type: Optional filter for document type (e.g. '10k', '10q', '8k').
            max_pages_per_doc: Optional limit of pages to parse per filing.
            progress_callback: Optional status callback function.

        Returns:
            Dict containing ingestion execution statistics.
        """
        start_time = time.perf_counter()
        manifest = self.load_manifest()
        known_files = manifest.get("files", {})

        files = self.discover_pdfs(company=company, doc_type=doc_type, sample=sample)
        logger.info("Discovered %d PDF files in %s", len(files), self.pdf_dir)

        to_index: list[tuple[Path, str]] = []
        unchanged_count = 0

        for fpath in files:
            rel_str = str(fpath.relative_to(self.pdf_dir))
            current_hash = compute_file_hash(fpath)
            if not force and rel_str in known_files and known_files[rel_str].get("hash") == current_hash:
                unchanged_count += 1
            else:
                to_index.append((fpath, current_hash))

        logger.info("To index: %d files | Unchanged: %d files", len(to_index), unchanged_count)

        stats: dict[str, Any] = {
            "total_files_discovered": len(files),
            "files_to_index": len(to_index),
            "files_unchanged": unchanged_count,
            "chunks_created": 0,
            "chunks_upserted": 0,
            "graph_entities_ingested": 0,
            "dry_run": dry_run,
            "duration_seconds": 0.0,
            "errors": [],
        }

        if not to_index:
            stats["duration_seconds"] = round(time.perf_counter() - start_time, 2)
            return stats

        batch_texts: list[str] = []
        batch_ids: list[str] = []
        batch_metadatas: list[dict[str, Any]] = []
        updated_manifest_entries: dict[str, dict[str, Any]] = {}

        for idx, (fpath, file_hash) in enumerate(to_index, start=1):
            rel_str = str(fpath.relative_to(self.pdf_dir))
            try:
                page_docs = self.parser.parse_pdf(fpath, max_pages=max_pages_per_doc)
                chunks: list[Document] = []
                for pdoc in page_docs:
                    table = None
                    if pdoc.metadata.get("has_table"):
                        table = self.table_parser.parse_table_from_text(
                            page_text=pdoc.page_content,
                            page_num=pdoc.metadata.get("page", 1),
                            statement_name=pdoc.metadata.get("section", "Financial Statement"),
                            file_path=str(fpath),
                        )
                    if table:
                        parent_doc, child_docs = self.table_parser.create_parent_child_chunks(
                            table, pdoc.metadata
                        )
                        chunks.append(parent_doc)
                        chunks.extend(child_docs)
                        if self.enable_graph and not dry_run:
                            try:
                                entities_count = self.graph_ingestor.ingest_table(
                                    table, pdoc.metadata
                                )
                                stats["graph_entities_ingested"] += entities_count
                            except Exception as e:
                                logger.debug("Graph ingestion note: %s", e)
                    else:
                        chunks.extend(
                            self.parser.chunk_document(
                                pdoc, max_chunk_chars=self.max_chunk_chars
                            )
                        )

                stats["chunks_created"] += len(chunks)

                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = f"{fpath.stem}_p{chunk.metadata.get('page', 1)}_c{chunk_idx}"
                    batch_texts.append(chunk.page_content)
                    batch_ids.append(doc_id)
                    batch_metadatas.append(chunk.metadata)

                    if len(batch_texts) >= self.batch_size:
                        if not dry_run:
                            self.vector_store.add_texts(
                                texts=batch_texts,
                                ids=batch_ids,
                                metadatas=batch_metadatas,
                            )
                            if self.enable_bm25:
                                self.bm25_index.add_texts(
                                    texts=batch_texts,
                                    ids=batch_ids,
                                    metadatas=batch_metadatas,
                                )
                        stats["chunks_upserted"] += len(batch_texts)
                        batch_texts.clear()
                        batch_ids.clear()
                        batch_metadatas.clear()

                updated_manifest_entries[rel_str] = {
                    "hash": file_hash,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                    "chunks": len(chunks),
                    "pages": len(page_docs),
                }

                if progress_callback:
                    progress_callback({
                        "current": idx,
                        "total": len(to_index),
                        "file": fpath.name,
                        "chunks": len(chunks),
                    })

            except Exception as e:
                err_msg = f"Error indexing {fpath.name}: {e}"
                logger.error(err_msg, exc_info=True)
                stats["errors"].append({"file": str(fpath), "error": str(e)})

        # Flush remaining batch
        if batch_texts and not dry_run:
            self.vector_store.add_texts(
                texts=batch_texts,
                ids=batch_ids,
                metadatas=batch_metadatas,
            )
            if self.enable_bm25:
                self.bm25_index.add_texts(
                    texts=batch_texts,
                    ids=batch_ids,
                    metadatas=batch_metadatas,
                )
            stats["chunks_upserted"] += len(batch_texts)

        # Save BM25 and manifest if not dry run
        if not dry_run:
            if self.enable_bm25:
                self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
                self.bm25_index.save_state(self.bm25_path)

            manifest["files"].update(updated_manifest_entries)
            self.save_manifest(manifest)

        stats["duration_seconds"] = round(time.perf_counter() - start_time, 2)
        logger.info(
            "Ingestion completed in %.2fs. Indexed %d chunks across %d files.",
            stats["duration_seconds"],
            stats["chunks_upserted"],
            len(updated_manifest_entries),
        )
        return stats

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        rerank: bool = True,
        company: str | None = None,
        doc_name: str | None = None,
        candidate_k: int = 15,
        score_threshold: float = 0.05,
    ) -> list[dict[str, Any]]:
        """Query the financial index using Dense, BM25, or Hybrid search with two-stage reranking.

        Args:
            query_text: The user query string.
            top_k: Number of final results to return.
            search_type: 'hybrid', 'dense', or 'sparse'.
            rerank: Whether to apply FlashRank cross-encoder reranking.
            company: Optional filter by company name.
            doc_name: Optional filter by exact document name (e.g. '3M_2018_10K').
            candidate_k: Number of candidate documents retrieved before reranking.
            score_threshold: Minimum reranker relevance score threshold.

        Returns:
            List of result dictionaries sorted by relevance.
        """
        k_retrieve = max(candidate_k, top_k * 3) if rerank else top_k

        # 1. Dense search
        dense_results: list[dict[str, Any]] = []
        if search_type in ("dense", "hybrid"):
            filter_dict = {}
            if company:
                filter_dict["company"] = company
            if doc_name:
                filter_dict["doc_name"] = doc_name
            dense_where = filter_dict if filter_dict else None
            try:
                raw_dense = self.vector_store.query(
                    query_text=query_text,
                    n_results=k_retrieve,
                    where=dense_where,
                )
                dense_results = raw_dense if isinstance(raw_dense, list) else []
            except Exception as e:
                logger.warning("Dense search failed: %s", e)

        # 2. Sparse BM25 search
        sparse_results: list[dict[str, Any]] = []
        if search_type in ("sparse", "hybrid") and getattr(self, "enable_bm25", True):
            try:
                bm25_matches = self.bm25_index.search(query_text, n_results=k_retrieve)
                for item in bm25_matches:
                    meta = item.get("metadata", {})
                    if company and meta.get("company", "").lower() != company.lower():
                        continue
                    if doc_name and meta.get("doc_name", "") != doc_name:
                        continue
                    sparse_results.append({
                        "id": item["id"],
                        "text": item.get("text", ""),
                        "metadata": meta,
                        "bm25_score": item.get("score", 0.0),
                    })
            except Exception as e:
                logger.warning("BM25 search failed: %s", e)

        # 3. Fusion
        if search_type == "dense":
            candidates = dense_results
        elif search_type == "sparse":
            candidates = sparse_results
        else:
            # Reciprocal Rank Fusion (RRF)
            rrf_scores: dict[str, float] = {}
            doc_lookup: dict[str, dict[str, Any]] = {}
            rrf_k = 60.0

            for rank, item in enumerate(dense_results, start=1):
                did = item["id"]
                rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (rrf_k + rank))
                doc_lookup[did] = item

            for rank, item in enumerate(sparse_results, start=1):
                did = item["id"]
                rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (rrf_k + rank))
                if did not in doc_lookup:
                    doc_lookup[did] = item

            sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            candidates = []
            for did in sorted_ids[:k_retrieve]:
                entry = dict(doc_lookup[did])
                entry["rrf_score"] = rrf_scores[did]
                candidates.append(entry)

        # 4. Rerank
        if rerank and candidates:
            try:
                passages = [
                    {"id": c["id"], "text": c.get("text", ""), "meta": c.get("metadata", {})}
                    for c in candidates
                ]
                reranked = self.reranker.rerank(query_text, passages, top_k=top_k)
                filtered_results = []
                for item in reranked:
                    score = item.get("rerank_score", 0.0)
                    if score >= score_threshold:
                        filtered_results.append({
                            "id": item["id"],
                            "text": item["text"],
                            "metadata": item.get("meta", {}),
                            "rerank_score": score,
                            "rerank_rank": item.get("rerank_rank", 1),
                        })
                return filtered_results if filtered_results else candidates[:top_k]
            except Exception as e:
                logger.warning("Reranking failed: %s. Falling back to candidates.", e)
                return candidates[:top_k]

        return candidates[:top_k]



