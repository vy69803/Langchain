"""Document loader module for LangChain RAG applications.

Provides flexible document loading capabilities for single files, directories,
web URLs, and raw text, producing LangChain Document objects.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterator, Sequence
import urllib.request
import urllib.error

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


# Common directories to skip during recursive directory loading
DEFAULT_IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
    "site-packages",
}

# Supported file extensions for text-based decoding
DEFAULT_SUPPORTED_EXTENSIONS = {
    ".txt",
    ".text",
    ".md",
    ".markdown",
    ".py",
    ".json",
    ".csv",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".xml",
    ".rst",
    ".log",
    ".ini",
    ".toml",
    ".cfg",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".asciidoc",
    ".adoc",
}

# Formats that IBM Docling can parse with layout and table analysis
DOCLING_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".asciidoc",
    ".adoc",
    ".md",
}


def is_docling_available() -> bool:
    """Check if the Docling library is installed and importable."""
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        return False


class DocumentLoader(BaseLoader):
    """Versatile document loader supporting single files, directories, and URLs."""

    def __init__(
        self,
        path_or_url: str | Path,
        *,
        encoding: str = "utf-8",
        recursive: bool = True,
        extensions: Sequence[str] | None = None,
        ignore_dirs: set[str] | None = None,
        use_docling: bool = False,
        docling_export_format: str = "markdown",
    ) -> None:
        """Initialize the DocumentLoader.

        Args:
            path_or_url: File path, directory path, or web URL.
            encoding: Text encoding to use when reading files (default: utf-8).
            recursive: Whether to scan directories recursively (default: True).
            extensions: Allowed extensions when scanning a directory (e.g. ['.md', '.txt']).
            ignore_dirs: Directory names to ignore during traversal.
            use_docling: Whether to use IBM Docling for advanced document parsing (default: False).
            docling_export_format: Export format when using Docling ('markdown' or 'text').
        """
        self.target = str(path_or_url)
        self.encoding = encoding
        self.recursive = recursive
        self.extensions = {ext.lower() for ext in extensions} if extensions else None
        self.ignore_dirs = ignore_dirs if ignore_dirs is not None else DEFAULT_IGNORE_DIRS
        self.use_docling = use_docling
        self.docling_export_format = docling_export_format

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents from the specified target."""
        if self._is_url(self.target):
            yield from self._load_url(self.target)
            return

        target_path = Path(self.target).resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"Target path does not exist: {target_path}")

        if target_path.is_file():
            yield from self._load_single_file(target_path)
        elif target_path.is_dir():
            yield from self._load_directory(target_path)
        else:
            raise ValueError(f"Target is neither a regular file nor a directory: {target_path}")

    def _is_url(self, target: str) -> bool:
        return target.startswith(("http://", "https://"))

    def _read_file_text(self, file_path: Path) -> str:
        """Read file text with fallback encodings for robustness."""
        for enc in (self.encoding, "utf-8", "utf-8-sig", "latin-1"):
            try:
                return file_path.read_text(encoding=enc)
            except (UnicodeDecodeError, LookupError):
                continue
        raise UnicodeDecodeError(
            "utf-8",
            b"",
            0,
            1,
            f"Failed to decode {file_path} using {self.encoding} and fallback encodings.",
        )

    def _load_single_file(self, file_path: Path) -> Iterator[Document]:
        """Load a single file based on its extension."""
        suffix = file_path.suffix.lower()
        stat = file_path.stat()
        base_metadata: dict[str, Any] = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": suffix if suffix else "unknown",
            "file_size": stat.st_size,
        }

        # Docling parser support for rich document formats (tables, layout, OCR)
        if self.use_docling and suffix in DOCLING_SUPPORTED_EXTENSIONS:
            yield from self._load_with_docling(file_path, base_metadata)
            return

        # PDF file support (lightweight fallback via pypdf)
        if suffix == ".pdf":
            yield from self._load_pdf(file_path, base_metadata)
            return

        # CSV file support
        if suffix == ".csv":
            yield from self._load_csv(file_path, base_metadata)
            return

        # JSON file support
        if suffix == ".json":
            yield from self._load_json(file_path, base_metadata)
            return

        # General text/code/markdown file support
        content = self._read_file_text(file_path)
        metadata = dict(base_metadata)
        metadata["encoding"] = self.encoding
        yield Document(page_content=content, metadata=metadata)

    def _load_pdf(self, file_path: Path, base_metadata: dict[str, Any]) -> Iterator[Document]:
        """Load PDF documents page-by-page using pypdf if available."""
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                f"Cannot load PDF file '{file_path}'. "
                "Please install pypdf to enable PDF support: uv add pypdf"
            )

        reader = pypdf.PdfReader(str(file_path))
        total_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            metadata = dict(base_metadata)
            metadata["page"] = page_idx
            metadata["total_pages"] = total_pages
            metadata["parser"] = "pypdf"
            yield Document(page_content=text, metadata=metadata)

    def _load_with_docling(self, file_path: Path, base_metadata: dict[str, Any]) -> Iterator[Document]:
        """Convert documents using IBM Docling for advanced layout, table, and structure preservation.

        Docling natively extracts layout, reading order, and complex tables into clean Markdown.
        """
        if not is_docling_available():
            raise ImportError(
                f"Docling is required to load '{file_path.name}' with use_docling=True. "
                "Install it using: uv add docling (or pip install docling)"
            )

        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(str(file_path))

            if self.docling_export_format == "markdown":
                content = result.document.export_to_markdown()
            else:
                content = result.document.export_to_text()

            metadata = dict(base_metadata)
            metadata["parser"] = "docling"

            doc_obj = result.document
            if hasattr(doc_obj, "pages") and doc_obj.pages:
                metadata["total_pages"] = len(doc_obj.pages)
            if hasattr(doc_obj, "tables") and doc_obj.tables:
                metadata["table_count"] = len(doc_obj.tables)
            if hasattr(doc_obj, "pictures") and doc_obj.pictures:
                metadata["picture_count"] = len(doc_obj.pictures)

            yield Document(page_content=content, metadata=metadata)
        except Exception as e:
            if isinstance(e, ImportError):
                raise
            raise RuntimeError(f"Docling failed to parse document '{file_path}': {e}") from e

    def _load_csv(self, file_path: Path, base_metadata: dict[str, Any]) -> Iterator[Document]:
        """Load a CSV file, formatting each row into a structured document."""
        content = self._read_file_text(file_path)
        reader = csv.DictReader(content.splitlines())
        rows = list(reader)

        if rows:
            formatted_lines = []
            for idx, row in enumerate(rows, start=1):
                row_str = ", ".join(f"{k}: {v}" for k, v in row.items())
                formatted_lines.append(f"Row {idx}: {row_str}")
            page_content = "\n".join(formatted_lines)
        else:
            page_content = content

        metadata = dict(base_metadata)
        metadata["row_count"] = len(rows)
        yield Document(page_content=page_content, metadata=metadata)

    def _load_json(self, file_path: Path, base_metadata: dict[str, Any]) -> Iterator[Document]:
        """Load a JSON file, pretty printing structured data."""
        raw_text = self._read_file_text(file_path)
        try:
            parsed = json.loads(raw_text)
            page_content = json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            page_content = raw_text

        metadata = dict(base_metadata)
        yield Document(page_content=page_content, metadata=metadata)

    def _load_directory(self, dir_path: Path) -> Iterator[Document]:
        """Recursively or shallowly scan a directory for supported files."""
        for root, dirs, files in os.walk(dir_path):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs and not d.startswith(".")]

            for file_name in files:
                if file_name.startswith("."):
                    continue

                file_path = Path(root) / file_name
                suffix = file_path.suffix.lower()

                if self.extensions is not None:
                    if suffix not in self.extensions:
                        continue
                elif suffix not in DEFAULT_SUPPORTED_EXTENSIONS:
                    continue

                try:
                    yield from self._load_single_file(file_path)
                except Exception as err:
                    # Gracefully skip unreadable or locked files with a log or note
                    print(f"Warning: Failed to load {file_path}: {err}")
                    continue

            if not self.recursive:
                break

    def _load_url(self, url: str) -> Iterator[Document]:
        """Fetch content from a web URL."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LangChainRAG/0.1"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            html_or_text = response.read().decode("utf-8", errors="replace")

        # Strip HTML tags simply if HTML content
        content = self._clean_html_content(html_or_text)

        metadata: dict[str, Any] = {
            "source": url,
            "file_type": "webpage",
            "url": url,
        }
        yield Document(page_content=content, metadata=metadata)

    @staticmethod
    def _clean_html_content(raw_html: str) -> str:
        """Strip basic script/style tags and clean HTML text."""
        import re

        # Remove scripts and styles
        cleaned = re.sub(r"<(script|style).*?</\1>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
        # Remove remaining tags
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        # Normalize whitespace
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        return "\n".join(lines)


# Convenient helper functions
def load_document(
    path_or_url: str | Path,
    *,
    encoding: str = "utf-8",
    use_docling: bool = False,
) -> list[Document]:
    """Load a single document or URL and return a list of Documents."""
    loader = DocumentLoader(path_or_url, encoding=encoding, use_docling=use_docling)
    return loader.load()


def load_directory(
    dir_path: str | Path,
    *,
    recursive: bool = True,
    extensions: Sequence[str] | None = None,
    encoding: str = "utf-8",
    use_docling: bool = False,
) -> list[Document]:
    """Load documents from a directory matching specified extensions."""
    loader = DocumentLoader(
        dir_path,
        encoding=encoding,
        recursive=recursive,
        extensions=extensions,
        use_docling=use_docling,
    )
    return loader.load()


def load_with_docling(path_or_url: str | Path) -> list[Document]:
    """Convenience helper to parse a document or URL using Docling."""
    return load_document(path_or_url, use_docling=True)


def load_text(text: str, metadata: dict[str, Any] | None = None) -> Document:
    """Wrap raw string text into a LangChain Document."""
    doc_metadata = metadata or {}
    if "source" not in doc_metadata:
        doc_metadata["source"] = "string"
    return Document(page_content=text, metadata=doc_metadata)


def load_url(url: str) -> list[Document]:
    """Fetch and return documents from a web URL."""
    loader = DocumentLoader(url)
    return loader.load()


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    print("--- DocumentLoader Demo ---")

    # 1. Test loading local README.md
    readme_path = Path("README.md")
    if readme_path.exists():
        docs = load_document(readme_path)
        print(f"\n[1] Loaded README.md: {len(docs)} document(s)")
        print(f"    Source: {docs[0].metadata.get('source')}")
        print(f"    Size: {docs[0].metadata.get('file_size')} bytes")
        print(f"    Preview:\n{docs[0].page_content[:180]}...\n")

    # 2. Test directory scanning for Python files
    src_path = Path("src")
    if src_path.exists():
        py_docs = load_directory(src_path, extensions=[".py"])
        print(f"[2] Scanned 'src' folder: found {len(py_docs)} Python document(s)")
        for doc in py_docs:
            print(f"    - {doc.metadata.get('filename')} ({doc.metadata.get('file_size')} bytes)")

    # 3. Test raw string loader
    raw_doc = load_text("LangChain RAG document loader test string.", {"topic": "demo"})
    print(f"\n[3] Raw text Document created: metadata={raw_doc.metadata}, content='{raw_doc.page_content}'")
