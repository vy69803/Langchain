"""Layout-aware Table Parser and Footnote Resolution Engine.

Extracts 2D financial tables from SEC filings, resolves footnote disclosures
to line items, performs parent-child document chunking, and prepares structured
entities for Knowledge Graph (Neo4j) and Vector Store ingestion.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from langchain_rag.graph.schema import (
    CanonicalMetricDef,
    normalize_line_item_name,
    resolve_canonical_metric,
)

logger = logging.getLogger("langchain_rag.parsers.layout_table_parser")

# Regex to detect footnote reference markers in table cells/labels
FOOTNOTE_REF_PATTERN = re.compile(
    r"(?:\((\d+|[a-z]|\*|\†)\)|\[(\d+|[a-z])\]|\bNote\s+(\d+|[A-Z]+)\b)",
    re.IGNORECASE,
)

# Regex to detect footnote disclosure definitions at the bottom of pages or in Notes section
FOOTNOTE_DEF_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\((\d+|[a-z]|\*|\†)\)|\[(\d+|[a-z])\]|Note\s+(\d+|[A-Z]+)[.:\s\-]+)\s*([^\n]+(?:\n(?!\s*(?:\(\d+\)|\[\d+\]|Note\s+\d+))[^\n]+)*)",
    re.IGNORECASE,
)

# Regex to extract numeric table rows: line item description followed by numbers
TABLE_ROW_PATTERN = re.compile(
    r"^([A-Za-z\s,\(\)\'\/\-\&]{3,80}?)\s+([\$\(\)\d,\.\s\-]{3,80})$",
    re.MULTILINE,
)

# Individual number extractor within a row
NUM_EXTRACTOR = re.compile(r"\(?\s*[\$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*\)?")


@dataclass
class ExtractedLineItem:
    """A single financial statement line item."""

    raw_name: str
    canonical_metric: Optional[str]
    category: str
    value: float
    raw_value: str
    unit: str = "USD"
    footnote_refs: List[str] = field(default_factory=list)
    footnote_texts: List[str] = field(default_factory=list)


@dataclass
class StructuredFinancialTable:
    """A structured 2D financial table extracted from a filing page."""

    statement_name: str
    page: int
    headers: List[str]
    markdown_grid: str
    line_items: List[ExtractedLineItem]
    footnote_definitions: Dict[str, str] = field(default_factory=dict)


def parse_numeric_value(raw: str) -> Optional[float]:
    """Parse a financial string like '(29,915)' or '$1,200.5' into float (-29915.0)."""
    raw = raw.strip()
    if not raw or raw in ("-", "—", "–", "N/A", "None"):
        return 0.0

    is_negative = raw.startswith("(") and raw.endswith(")")
    match = NUM_EXTRACTOR.search(raw)
    if not match:
        return None

    num_str = match.group(1).replace(",", "")
    try:
        val = float(num_str)
        return -val if is_negative else val
    except ValueError:
        return None


class LayoutTableParser:
    """Parses financial tables, resolves footnotes, and constructs parent-child chunks."""

    def __init__(self, dlq_path: str | Path = "data/ingestion_dlq.jsonl") -> None:
        self.dlq_path = Path(dlq_path)
        self.dlq_path.parent.mkdir(parents=True, exist_ok=True)

    def log_dlq_error(self, file_path: str, page_num: int, error_msg: str, raw_snippet: str = "") -> None:
        """Record an ingestion failure in the Dead-Letter Queue."""
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "file": str(file_path),
            "page": page_num,
            "error": error_msg,
            "snippet": raw_snippet[:300],
        }
        try:
            with open(self.dlq_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to DLQ: {e}")

    def extract_footnote_definitions(self, page_text: str) -> Dict[str, str]:
        """Extract footnote definitions (e.g. '(1) Represents capitalized software...') from page text."""
        defs: Dict[str, str] = {}
        for match in FOOTNOTE_DEF_PATTERN.finditer(page_text):
            marker = match.group(1) or match.group(2) or f"Note {match.group(3)}"
            content = match.group(0).strip()
            # Clean up leading marker
            clean_text = re.sub(r"^(?:\(\w+\)|\[\w+\]|Note\s+\w+[.:\s\-]+)\s*", "", content)
            defs[marker.strip()] = clean_text.strip()
        return defs

    def parse_table_from_text(
        self,
        page_text: str,
        page_num: int,
        statement_name: str = "Financial Statement",
        file_path: str = "",
    ) -> Optional[StructuredFinancialTable]:
        """Parse 2D table rows and attach footnote definitions."""
        try:
            footnote_defs = self.extract_footnote_definitions(page_text)
            lines = page_text.splitlines()
            table_rows: List[ExtractedLineItem] = []
            grid_lines: List[str] = [f"### {statement_name} (Page {page_num})", ""]

            for line in lines:
                line_str = line.strip()
                match = TABLE_ROW_PATTERN.match(line_str)
                if not match:
                    continue

                raw_label = match.group(1).strip()
                raw_numbers = match.group(2).strip()

                # Extract footnote refs across label and numbers (e.g. "Net Sales (1) $ 383,285")
                refs = [m.group(0).strip("()[] ") for m in FOOTNOTE_REF_PATTERN.finditer(line_str)]

                # Clean footnote markers from raw_numbers so '(1)' isn't parsed as a negative number
                clean_numbers = FOOTNOTE_REF_PATTERN.sub("", raw_numbers).strip()

                # Extract latest numerical value
                num_matches = NUM_EXTRACTOR.findall(clean_numbers)
                if not num_matches:
                    continue

                primary_val = parse_numeric_value(clean_numbers.split()[-1])
                if primary_val is None:
                    continue

                # Match against Canonical XBRL Ontology
                metric_def = resolve_canonical_metric(raw_label)
                canonical_name = metric_def.canonical_name if metric_def else None
                category = metric_def.category if metric_def else "General"

                # Resolve footnote texts if definitions found
                resolved_texts = [footnote_defs[r] for r in refs if r in footnote_defs]

                item = ExtractedLineItem(
                    raw_name=raw_label,
                    canonical_metric=canonical_name,
                    category=category,
                    value=primary_val,
                    raw_value=raw_numbers,
                    footnote_refs=refs,
                    footnote_texts=resolved_texts,
                )
                table_rows.append(item)

                # Append to Markdown Grid
                fn_note = f" *[Footnote: {', '.join(resolved_texts)}]*" if resolved_texts else ""
                grid_lines.append(f"| {raw_label} | {raw_numbers} |{fn_note}")

            if not table_rows:
                return None

            # Add header to markdown grid
            grid_lines.insert(2, "| Line Item | Value (in Millions / USD) | Disclosures |")
            grid_lines.insert(3, "| :--- | :--- | :--- |")

            return StructuredFinancialTable(
                statement_name=statement_name,
                page=page_num,
                headers=["Line Item", "Value", "Disclosures"],
                markdown_grid="\n".join(grid_lines),
                line_items=table_rows,
                footnote_definitions=footnote_defs,
            )

        except Exception as e:
            self.log_dlq_error(file_path, page_num, str(e), page_text)
            return None

    def create_parent_child_chunks(
        self,
        table: StructuredFinancialTable,
        doc_metadata: Dict[str, Any],
    ) -> Tuple[Document, List[Document]]:
        """Create parent-child document hierarchy for vector retrieval and graph linking.

        Returns:
            Tuple of (Parent Document, List of Child Line-Item Documents).
        """
        parent_id = f"{doc_metadata.get('doc_name', 'DOC')}_p{table.page}_{table.statement_name.replace(' ', '_')}"

        # 1. Parent Chunk (Full Statement / Markdown Grid)
        parent_doc = Document(
            page_content=table.markdown_grid,
            metadata={
                **doc_metadata,
                "chunk_id": parent_id,
                "chunk_type": "parent_financial_statement",
                "page": table.page,
                "statement_name": table.statement_name,
                "line_item_count": len(table.line_items),
            },
        )

        # 2. Child Chunks (Individual Line Items with resolved footnotes)
        child_docs: List[Document] = []
        for idx, item in enumerate(table.line_items):
            child_id = f"{parent_id}_row_{idx}"
            fn_context = (
                f"\n[Footnote Disclosure: {' | '.join(item.footnote_texts)}]"
                if item.footnote_texts
                else ""
            )

            child_content = (
                f"Filing: {doc_metadata.get('doc_name')} | Period: {doc_metadata.get('doc_period')}\n"
                f"Statement: {table.statement_name} | Page: {table.page}\n"
                f"Line Item: {item.raw_name}\n"
                f"Canonical Metric: {item.canonical_metric or 'Non-GAAP / Custom'}\n"
                f"Value: {item.value} {item.unit}{fn_context}"
            )

            child_doc = Document(
                page_content=child_content,
                metadata={
                    **doc_metadata,
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "chunk_type": "child_line_item",
                    "page": table.page,
                    "statement_name": table.statement_name,
                    "canonical_metric": item.canonical_metric or "",
                    "raw_name": item.raw_name,
                    "value": item.value,
                    "footnote_refs": json.dumps(item.footnote_refs),
                    "neo4j_lineitem_id": f"{doc_metadata.get('doc_name')}_{item.canonical_metric or idx}",
                },
            )
            child_docs.append(child_doc)

        return parent_doc, child_docs
