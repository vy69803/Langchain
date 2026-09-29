"""FinanceBench parser and document loader for SEC financial filings.

Extracts financial statements, tables, and narrative disclosures from 10-K, 10-Q,
and 8-K filings with company metadata, page-level tracking, and table structure detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator, Sequence

from langchain_core.documents import Document

logger = logging.getLogger("langchain_rag.finance_parser")

# Regex heuristics for detecting SEC financial statement sections
FINANCIAL_STATEMENT_PATTERNS = [
    (re.compile(r"consolidated\s+statements?\s+of\s+(?:income|operations|earnings)", re.IGNORECASE), "Income Statement"),
    (re.compile(r"consolidated\s+statements?\s+of\s+comprehensive\s+income", re.IGNORECASE), "Comprehensive Income"),
    (re.compile(r"consolidated\s+balance\s+sheets?", re.IGNORECASE), "Balance Sheet"),
    (re.compile(r"consolidated\s+statements?\s+of\s+cash\s+flows?", re.IGNORECASE), "Cash Flow Statement"),
    (re.compile(r"consolidated\s+statements?\s+of\s+stockholders?['\u2019]?\s+equity", re.IGNORECASE), "Stockholders Equity"),
    (re.compile(r"notes?\s+to\s+consolidated\s+financial\s+statements?", re.IGNORECASE), "Notes to Financial Statements"),
    (re.compile(r"item\s+1a[.:\s]+risk\s+factors", re.IGNORECASE), "Item 1A. Risk Factors"),
    (re.compile(r"item\s+7[.:\s]+management['\u2019]?s\s+discussion\s+and\s+analysis", re.IGNORECASE), "Item 7. MD&A"),
    (re.compile(r"item\s+1[.:\s]+business", re.IGNORECASE), "Item 1. Business"),
    (re.compile(r"item\s+8[.:\s]+financial\s+statements", re.IGNORECASE), "Item 8. Financial Statements"),
]

# Patterns indicating tabular financial data
TABLE_INDICATOR_PATTERN = re.compile(
    r"(?:\$\s*\(?\d[\d,.]*\)?|\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\b\d+\s+%\b)",
    re.IGNORECASE,
)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for incremental change detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_financebench_metadata(
    metadata_path: str | Path = "data/financebench/financebench_document_information.jsonl",
) -> dict[str, dict[str, Any]]:
    """Load document metadata from the FinanceBench document information JSONL file.

    Returns:
        Mapping of doc_name (e.g. '3M_2018_10K') to metadata dictionary.
    """
    path = Path(metadata_path)
    if not path.exists():
        logger.warning("FinanceBench metadata file not found at %s", path)
        return {}

    mapping: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc_name = item.get("doc_name")
            if doc_name:
                mapping[doc_name] = item
    return mapping


def load_financebench_qa(
    qa_path: str | Path = "data/financebench/financebench_open_source.jsonl",
) -> list[dict[str, Any]]:
    """Load evaluation QA pairs from the FinanceBench open-source dataset."""
    path = Path(qa_path)
    if not path.exists():
        raise FileNotFoundError(f"FinanceBench QA file not found at: {path}")

    qa_list: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_list.append(json.loads(line))
    return qa_list


def detect_section_header(text: str) -> str:
    """Detect common SEC financial statement or Item sections in page text."""
    # Look in the first 800 characters of the page
    snippet = text[:800]
    for pattern, section_name in FINANCIAL_STATEMENT_PATTERNS:
        if pattern.search(snippet):
            return section_name
    return "General"


def detect_has_table(text: str) -> bool:
    """Detect whether a page or text block contains tabular financial numbers."""
    matches = TABLE_INDICATOR_PATTERN.findall(text)
    # If 4 or more financial figures/currencies are present, highly likely a table
    return len(matches) >= 4


def sanitize_metadata_for_chroma(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Sanitize metadata values so they conform to ChromaDB's accepted types (str, int, float, bool)."""
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (bool, int, float)):
            clean[key] = value
        elif isinstance(value, str):
            clean[key] = value
        elif isinstance(value, (list, dict)):
            clean[key] = json.dumps(value, ensure_ascii=False)
        else:
            clean[key] = str(value)
    return clean


class FinanceBenchParser:
    """Parses FinanceBench SEC PDF filings with rich metadata and financial structure awareness."""

    def __init__(
        self,
        metadata_path: str | Path = "data/financebench/financebench_document_information.jsonl",
    ) -> None:
        self.doc_metadata = load_financebench_metadata(metadata_path)

    def resolve_doc_metadata(self, file_path: Path) -> dict[str, Any]:
        """Resolve metadata for a filing from its filename and document info table."""
        doc_name = file_path.stem
        meta = dict(self.doc_metadata.get(doc_name, {}))

        # Parse from filename fallback if not in JSONL mapping
        if not meta:
            parts = doc_name.split("_")
            company = parts[0] if parts else "Unknown"
            year = ""
            doc_type = "10k"
            for p in parts[1:]:
                if re.match(r"^\d{4}", p):
                    year = p
                elif any(t in p.lower() for t in ("10k", "10q", "8k", "report")):
                    doc_type = p.lower()
            meta = {
                "doc_name": doc_name,
                "company": company,
                "doc_type": doc_type,
                "doc_period": year,
                "gics_sector": "Unknown",
                "doc_link": "",
            }

        return meta

    def parse_pdf(
        self,
        file_path: str | Path,
        max_pages: int | None = None,
    ) -> list[Document]:
        """Parse an SEC PDF filing into page-level Document objects with financial metadata.

        Args:
            file_path: Path to the SEC filing PDF.
            max_pages: Optional maximum number of pages to parse (for quick tests).

        Returns:
            List of LangChain Document objects, one per page.
        """
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required to parse FinanceBench PDFs. "
                "Install it using: uv pip install pypdf"
            )

        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        doc_meta = self.resolve_doc_metadata(path)
        reader = pypdf.PdfReader(str(path))
        total_pages = len(reader.pages)
        pages_to_read = min(total_pages, max_pages) if max_pages else total_pages

        documents: list[Document] = []
        for page_idx in range(pages_to_read):
            page_num = page_idx + 1
            page_obj = reader.pages[page_idx]
            text = page_obj.extract_text() or ""
            text = text.strip()

            if not text:
                continue

            section = detect_section_header(text)
            has_table = detect_has_table(text)

            # Metadata conforms to ChromaDB constraints
            metadata: dict[str, Any] = {
                "source": str(path),
                "filename": path.name,
                "doc_name": doc_meta.get("doc_name", path.stem),
                "company": doc_meta.get("company", "Unknown"),
                "doc_type": doc_meta.get("doc_type", "10k"),
                "doc_period": str(doc_meta.get("doc_period", "")),
                "gics_sector": doc_meta.get("gics_sector", "Unknown"),
                "doc_link": doc_meta.get("doc_link", ""),
                "page": page_num,
                "total_pages": total_pages,
                "section": section,
                "has_table": has_table,
            }

            # Enriched contextual header injected for better dense and sparse retrieval
            header = (
                f"[Company: {metadata['company']} | Filing: {metadata['doc_name']} "
                f"| Page: {page_num}/{total_pages} | Section: {section}]\n\n"
            )
            enriched_content = header + text

            documents.append(
                Document(
                    page_content=enriched_content,
                    metadata=sanitize_metadata_for_chroma(metadata),
                )
            )

        return documents

    def chunk_document(
        self,
        doc: Document,
        max_chunk_chars: int = 1200,
        chunk_overlap: int = 150,
    ) -> list[Document]:
        """Split a page into smaller chunk documents while preserving breadcrumb context.

        If a page is smaller than max_chunk_chars, it is returned as a single chunk.
        Otherwise, text is split across paragraph and sentence boundaries.
        """
        text = doc.page_content
        if len(text) <= max_chunk_chars:
            return [doc]

        # Extract breadcrumb prefix if present
        prefix = ""
        body = text
        if text.startswith("[Company:"):
            prefix_end = text.find("]\n\n")
            if prefix_end != -1:
                prefix = text[: prefix_end + 3]
                body = text[prefix_end + 3 :]

        paragraphs = body.split("\n\n")
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)
            if current_len + p_len + 2 > max_chunk_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0

            current_chunk.append(p)
            current_len += p_len + 2

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        result: list[Document] = []
        for idx, chunk_text in enumerate(chunks):
            meta = dict(doc.metadata)
            meta["chunk_index"] = idx
            meta["total_chunks_in_page"] = len(chunks)
            chunk_content = f"{prefix}{chunk_text}" if prefix and not chunk_text.startswith(prefix) else chunk_text
            result.append(Document(page_content=chunk_content, metadata=meta))

        return result
