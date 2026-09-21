"""Handbook Ingestion Pipeline for GitLab Handbook dataset.

Orchestrates file discovery, Hugo shortcode & YAML frontmatter parsing,
markdown-aware semantic chunking with breadcrumb injection, SHA-256 change detection,
and dual-indexing into ChromaDB (dense vectors) and BM25 (sparse lexical index).
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

from langchain_rag.handbook_parser import HandbookParser, compute_content_hash
from langchain_rag.hybrid_search import BM25Index
from langchain_rag.semantic_chunking import ContextualMarkdownChunker
from langchain_rag.vector_stores import VectorStore

logger = logging.getLogger("langchain_rag.handbook_pipeline")


def sanitize_metadata_for_chroma(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Sanitize metadata values so they conform to ChromaDB's accepted types.

    ChromaDB only accepts str, int, float, and bool. Lists, dicts, or None values
    are converted to strings or safely omitted/defaulted.
    """
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (bool, int, float)):
            clean[key] = value
        elif isinstance(value, str):
            clean[key] = value
        elif isinstance(value, list):
            # Convert list of strings or objects to comma-separated string or JSON
            if all(isinstance(x, str) for x in value):
                clean[key] = ", ".join(value)
            else:
                clean[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, dict):
            clean[key] = json.dumps(value, ensure_ascii=False)
        else:
            clean[key] = str(value)
    return clean


class HandbookIngestionPipeline:
    """Enterprise Ingestion Pipeline for the GitLab Handbook dataset."""

    def __init__(
        self,
        handbook_dir: str | Path = "handbook",
        persist_directory: str | Path = "./data/chroma_db",
        collection_name: str = "gitlab_handbook",
        manifest_path: str | Path = "./data/handbook_manifest.json",
        bm25_path: str | Path = "./data/bm25_handbook.json",
        max_chunk_chars: int = 1000,
        batch_size: int = 100,
        enable_bm25: bool = True,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Initialize the pipeline configuration and storage backends.

        Args:
            handbook_dir: Root path to the GitLab handbook repo.
            persist_directory: Disk storage directory for ChromaDB.
            collection_name: Name of ChromaDB collection.
            manifest_path: Path to the JSON manifest tracking indexed file hashes.
            bm25_path: Path to serialized BM25 index state.
            max_chunk_chars: Target maximum character length per chunk.
            batch_size: Number of chunks to upsert per database write batch.
            enable_bm25: Whether to maintain a sparse lexical BM25 index alongside ChromaDB.
            vector_store: Optional pre-configured VectorStore instance.
        """
        self.handbook_dir = Path(handbook_dir).resolve()
        self.content_dir = self._resolve_content_dir()
        self.persist_directory = Path(persist_directory).resolve()
        self.collection_name = collection_name
        self.manifest_path = Path(manifest_path).resolve()
        self.bm25_path = Path(bm25_path).resolve()
        self.max_chunk_chars = max_chunk_chars
        self.batch_size = max(10, batch_size)
        self.enable_bm25 = enable_bm25

        # Initialize parser and chunker
        self.parser = HandbookParser()
        self.chunker = ContextualMarkdownChunker(
            max_chunk_chars=max_chunk_chars,
            inject_breadcrumbs=True,
        )

        # Storage components (lazy-loaded or passed)
        self._vector_store = vector_store
        self._bm25_index: BM25Index | None = None

        # State manifest
        self.manifest = self._load_manifest()

    def _resolve_content_dir(self) -> Path:
        """Locate the actual content directory within the handbook tree."""
        candidates = [
            self.handbook_dir / "content" / "handbook",
            self.handbook_dir / "content",
            self.handbook_dir,
        ]
        for c in candidates:
            if c.is_dir() and any(c.rglob("*.md")):
                return c
        return self.handbook_dir

    @property
    def vector_store(self) -> VectorStore:
        """Return the persistent ChromaDB VectorStore."""
        if self._vector_store is None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._vector_store = VectorStore(
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory),
            )
        return self._vector_store

    @property
    def bm25_index(self) -> BM25Index:
        """Return or load the persistent BM25 lexical index."""
        if self._bm25_index is None:
            self._bm25_index = BM25Index()
            if self.bm25_path.is_file():
                try:
                    self._bm25_index.load_state(self.bm25_path)
                    logger.info(f"Loaded existing BM25 index with {self._bm25_index.count()} chunks.")
                except Exception as e:
                    logger.warning(f"Could not load BM25 state from {self.bm25_path}: {e}")
        return self._bm25_index

    def _load_manifest(self) -> dict[str, Any]:
        """Load the indexing manifest from disk if it exists."""
        if self.manifest_path.is_file():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read manifest file: {e}. Starting fresh.")
        return {
            "version": "1.0",
            "collection": self.collection_name,
            "files": {},
            "stats": {"total_files_indexed": 0, "total_chunks_indexed": 0},
        }

    def _save_manifest(self) -> None:
        """Save the updated indexing manifest to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def discover_files(
        self,
        department: str | None = None,
        sample: int | None = None,
    ) -> list[Path]:
        """Discover all handbook markdown files according to optional department and sample filters.

        Args:
            department: Optional department subdirectory name (e.g. 'values', 'engineering').
            sample: Optional integer limiting the number of files returned.

        Returns:
            Sorted list of resolved Path objects.
        """
        search_root = self.content_dir
        if department:
            dept_candidate = search_root / department
            if dept_candidate.is_dir():
                search_root = dept_candidate
            else:
                # Try relative to handbook/content/handbook
                alt_dept = self.handbook_dir / "content" / "handbook" / department
                if alt_dept.is_dir():
                    search_root = alt_dept
                else:
                    logger.warning(f"Department '{department}' not found under {self.content_dir}")

        files = [
            p
            for p in search_root.rglob("*.md")
            if p.is_file() and not p.name.startswith(".") and ".git" not in p.parts
        ]
        # Sort for deterministic processing order
        files.sort(key=lambda p: str(p))

        if sample and sample > 0:
            files = files[:sample]

        return files

    def chunk_document(self, doc: Document) -> list[Document]:
        """Chunk a parsed handbook document into enriched, context-bearing chunk Documents.

        Args:
            doc: Parsed LangChain Document from HandbookParser.

        Returns:
            List of chunk Document objects with contextual breadcrumbs and metadata.
        """
        title = doc.metadata.get("title") or doc.metadata.get("filename") or "Handbook"
        raw_chunks = self.chunker.split_markdown(doc.page_content, source_name=title)

        enriched_chunks: list[Document] = []
        doc_hash = doc.metadata.get("doc_hash", "")
        base_url = doc.metadata.get("url", "")
        file_source = doc.metadata.get("source", "")
        total_chunks = len(raw_chunks)

        for idx, chunk in enumerate(raw_chunks):
            chunk_id = f"{doc_hash[:12]}_{idx}"
            chunk_meta = dict(doc.metadata)

            # Update chunk-specific metadata
            chunk_meta.update({
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_char_count": len(chunk.page_content),
                "breadcrumbs": chunk.metadata.get("breadcrumbs", doc.metadata.get("breadcrumbs", "")),
            })

            # Clean and flatten metadata for ChromaDB compatibility
            sanitized_meta = sanitize_metadata_for_chroma(chunk_meta)

            enriched_chunks.append(
                Document(page_content=chunk.page_content, metadata=sanitized_meta)
            )

        return enriched_chunks

    def run(
        self,
        department: str | None = None,
        sample: int | None = None,
        force: bool = False,
        dry_run: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Execute the ingestion pipeline.

        Args:
            department: Optional department name filter.
            sample: Optional sample size limit.
            force: If True, re-indexes files even if their SHA-256 hash has not changed.
            dry_run: If True, parses and chunks without writing to ChromaDB/BM25.
            progress_callback: Optional callback invoked with periodic progress updates.

        Returns:
            Summary dictionary containing statistics on files scanned, skipped,
            processed, total chunks, execution time, and any errors encountered.
        """
        start_time = time.perf_counter()
        files = self.discover_files(department=department, sample=sample)
        total_files = len(files)

        scanned = 0
        skipped = 0
        processed_files = 0
        total_chunks = 0
        errors: list[dict[str, str]] = []

        chunk_batch: list[Document] = []
        file_records_batch: list[tuple[str, str, int]] = []  # (rel_path, hash, chunk_count)

        def flush_batch() -> None:
            nonlocal chunk_batch, file_records_batch
            if not chunk_batch or dry_run:
                chunk_batch = []
                file_records_batch = []
                return

            texts = [c.page_content for c in chunk_batch]
            ids = [str(c.metadata.get("chunk_id")) for c in chunk_batch]
            metadatas = [c.metadata for c in chunk_batch]

            # 1. Upsert into ChromaDB
            try:
                self.vector_store.collection.upsert(
                    documents=texts,
                    ids=ids,
                    metadatas=metadatas,
                )
            except Exception as e:
                logger.error(f"Error during ChromaDB batch upsert: {e}")
                raise

            # 2. Add to BM25 index if enabled
            if self.enable_bm25:
                try:
                    self.bm25_index.add_texts(
                        texts=texts,
                        ids=ids,
                        metadatas=metadatas,
                    )
                except Exception as e:
                    logger.warning(f"Error adding batch to BM25 index: {e}")

            # 3. Record in manifest
            for rel_path, f_hash, f_chunks in file_records_batch:
                self.manifest["files"][rel_path] = {
                    "hash": f_hash,
                    "chunks": f_chunks,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }

            chunk_batch = []
            file_records_batch = []

        try:
            from tqdm import tqdm
            pbar = tqdm(files, desc="Ingesting GitLab Handbook", unit="file")
        except ImportError:
            pbar = files

        for file_path in pbar:
            scanned += 1
            rel_path = file_path.relative_to(self.handbook_dir).as_posix()

            try:
                doc = self.parser.parse_file(file_path)
                doc_hash = doc.metadata.get("doc_hash", "")

                # Check if file has changed
                existing_entry = self.manifest.get("files", {}).get(rel_path)
                if not force and existing_entry and existing_entry.get("hash") == doc_hash:
                    skipped += 1
                    continue

                # Generate chunks
                chunks = self.chunk_document(doc)
                processed_files += 1
                total_chunks += len(chunks)

                if not dry_run:
                    chunk_batch.extend(chunks)
                    file_records_batch.append((rel_path, doc_hash, len(chunks)))

                    # Flush when batch size threshold is reached
                    if len(chunk_batch) >= self.batch_size:
                        flush_batch()

            except Exception as exc:
                err_msg = f"Failed processing {file_path.name}: {exc}"
                logger.error(err_msg)
                errors.append({"file": rel_path, "error": str(exc)})

            if progress_callback:
                progress_callback({
                    "scanned": scanned,
                    "total": total_files,
                    "processed": processed_files,
                    "skipped": skipped,
                    "chunks": total_chunks,
                })

        # Flush any remaining items in the buffer
        flush_batch()

        # Persist BM25 and manifest if not dry-run
        if not dry_run:
            if self.enable_bm25:
                try:
                    self.bm25_index.save_state(self.bm25_path)
                except Exception as e:
                    logger.error(f"Failed to save BM25 index: {e}")

            self.manifest["stats"]["total_files_indexed"] = len(self.manifest["files"])
            self.manifest["stats"]["total_chunks_indexed"] = sum(
                entry.get("chunks", 0) for entry in self.manifest["files"].values()
            )
            self._save_manifest()

        duration = time.perf_counter() - start_time

        summary = {
            "total_scanned": scanned,
            "skipped_unchanged": skipped,
            "processed_files": processed_files,
            "total_chunks_generated": total_chunks,
            "duration_seconds": round(duration, 2),
            "dry_run": dry_run,
            "errors": errors,
            "chroma_collection_count": (
                self.vector_store.count() if not dry_run else 0
            ),
            "bm25_chunk_count": (
                self.bm25_index.count() if (not dry_run and self.enable_bm25) else 0
            ),
        }

        return summary

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        department: str | None = None,
        search_type: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Query the ingested handbook using dense, sparse (BM25), or hybrid search.

        Args:
            query_text: User question or search query.
            top_k: Number of top documents to return.
            department: Optional metadata filter for department.
            search_type: 'dense' (ChromaDB), 'sparse' (BM25), or 'hybrid' (combined).

        Returns:
            List of result dictionaries with 'text', 'metadata', 'score', and 'source'.
        """
        where_filter: dict[str, Any] | None = None
        if department:
            where_filter = {"department": department}

        if search_type == "dense":
            chroma_results = self.vector_store.query(
                query_text=query_text,
                n_results=top_k,
                where=where_filter,
            )
            return [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "distance": r.get("distance"),
                    "source": r.get("metadata", {}).get("url") or r.get("metadata", {}).get("source"),
                }
                for r in chroma_results
            ]

        elif search_type == "sparse":
            bm25_results = self.bm25_index.search(
                query=query_text,
                n_results=top_k,
                where=where_filter,
            )
            return [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "bm25_score": r.get("score"),
                    "source": r.get("metadata", {}).get("url") or r.get("metadata", {}).get("source"),
                }
                for r in bm25_results
            ]

        else:
            # Hybrid search combining dense and sparse using Reciprocal Rank Fusion
            chroma_results = self.vector_store.query(
                query_text=query_text,
                n_results=top_k * 2,
                where=where_filter,
            )
            bm25_results = self.bm25_index.search(
                query=query_text,
                n_results=top_k * 2,
                where=where_filter,
            )

            rrf_scores: dict[str, float] = {}
            doc_map: dict[str, dict[str, Any]] = {}
            k_rrf = 60

            for rank, item in enumerate(chroma_results):
                doc_id = item["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + rank + 1))
                doc_map[doc_id] = {
                    "id": doc_id,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "dense_distance": item.get("distance"),
                }

            for rank, item in enumerate(bm25_results):
                doc_id = item["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + rank + 1))
                if doc_id not in doc_map:
                    doc_map[doc_id] = {
                        "id": doc_id,
                        "text": item["text"],
                        "metadata": item["metadata"],
                        "bm25_score": item.get("score"),
                    }
                else:
                    doc_map[doc_id]["bm25_score"] = item.get("score")

            sorted_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
            results = []
            for doc_id in sorted_ids[:top_k]:
                entry = doc_map[doc_id]
                entry["hybrid_score"] = round(rrf_scores[doc_id], 5)
                entry["source"] = entry.get("metadata", {}).get("url") or entry.get("metadata", {}).get("source")
                results.append(entry)

            return results
