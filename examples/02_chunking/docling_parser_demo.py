"""Example: Advanced Document Parsing with Docling & Semantic Chunking.

Demonstrates:
1. Loading a document using IBM Docling layout parsing (with table/figure preservation).
2. Inspecting extracted structured Markdown and metadata.
3. Passing Docling-parsed documents directly to LangChain RAG text splitters.
"""

from pathlib import Path
import sys

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from langchain_rag.document_loader import (
    DocumentLoader,
    is_docling_available,
    load_with_docling,
)
from langchain_rag.text_splitter import TextSplitter


def main():
    print("=" * 60)
    print("Docling Document Parser & RAG Integration Demo")
    print("=" * 60)

    # Check if docling is installed in current environment
    if not is_docling_available():
        print("\n[!] Docling is not installed in the current environment.")
        print("    To install Docling:")
        print("      uv add docling")
        print("    Or as an optional extra:")
        print("      uv sync --extra docling\n")
        print("Demonstrating fallback/mock workflow:")
        print("Docling enables extraction of:")
        print(" - Hierarchical Markdown headers (#, ##, ###)")
        print(" - Markdown formatted tables (| col1 | col2 |)")
        print(" - Embedded images / figures metadata")
        print(" - Clean reading order for multi-column academic/business PDFs")
        return

    # If docling is installed, demonstrate parsing a file
    print("\n[✓] Docling is installed and ready to parse documents.")
    sample_file = Path("examples/sample_document.pdf")

    if not sample_file.exists():
        print(f"Sample PDF '{sample_file}' not found. You can parse any PDF, DOCX, or PPTX:")
        print("  docs = load_with_docling('path/to/your/document.pdf')")
        return

    print(f"Parsing '{sample_file}' with Docling...")
    docs = load_with_docling(sample_file)

    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Document {i} ---")
        print(f"Parser: {doc.metadata.get('parser')}")
        print(f"Total Pages: {doc.metadata.get('total_pages')}")
        print(f"Tables Extracted: {doc.metadata.get('table_count', 0)}")
        print(f"Content Preview:\n{doc.page_content[:300]}...\n")

        # Split document with TextSplitter
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents([doc])
        print(f"Created {len(chunks)} chunks from Docling-parsed document.")


if __name__ == "__main__":
    main()
