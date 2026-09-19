import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src directory is in sys.path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
from langchain_rag.document_loader import (
    DocumentLoader,
    load_directory,
    load_document,
    load_text,
    load_url,
)

load_dotenv()


def main() -> None:
    print("==================================================")
    print("  LangChain RAG - Document Loader Demonstration   ")
    print("==================================================")

    # 1. Loading a single file (README.md)
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        print(f"\n[1] Loading single file: {readme_path.name}")
        docs = load_document(readme_path)
        print(f"    - Loaded: {len(docs)} document(s)")
        for doc in docs:
            print(f"    - Source: {doc.metadata.get('source')}")
            print(f"    - Type: {doc.metadata.get('file_type')}")
            print(f"    - Size: {doc.metadata.get('file_size')} bytes")
            print(f"    - Preview (first 150 chars):\n      {doc.page_content[:150].strip()}...\n")

    # 2. Loading project source files from directory
    src_path = Path(__file__).parent / "src"
    if src_path.exists():
        print(f"[2] Scanning directory recursively: {src_path.name}")
        py_docs = load_directory(src_path, extensions=[".py"])
        print(f"    - Found {len(py_docs)} Python source document(s):")
        for doc in py_docs:
            print(f"      * {doc.metadata.get('filename')} ({doc.metadata.get('file_size')} bytes)")

    # 3. Loading in-memory text
    print("\n[3] In-memory text loading:")
    raw_doc = load_text(
        "RAG stands for Retrieval-Augmented Generation. It enhances LLM capabilities.",
        metadata={"topic": "rag_definition", "author": "system"},
    )
    print(f"    - Content: {raw_doc.page_content}")
    print(f"    - Metadata: {raw_doc.metadata}")

    print("\n[✔] Document loading completed successfully!")


if __name__ == "__main__":
    main()
