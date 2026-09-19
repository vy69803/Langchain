import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src directory is in sys.path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langchain_rag.document_loader import load_document
from langchain_rag.text_splitter import (
    TextSplitter,
    split_documents,
    split_text,
)


def main() -> None:
    print("============================================================")
    print("            LangChain RAG - Text Splitter Demo              ")
    print("============================================================")

    # 1. Text chunking
    sample_text = (
        "Retrieval-Augmented Generation (RAG) is an AI framework for improving the quality "
        "of LLM responses by grounding the model on external sources of knowledge. "
        "Text splitters take long documents and break them into smaller, coherent passages "
        "called chunks. Chunks allow vector search to find the precise paragraph answering "
        "a user's question without overwhelming the LLM's context window. "
        "Chunk overlap ensures that semantic boundaries aren't accidentally lost."
    )

    splitter = TextSplitter(chunk_size=180, chunk_overlap=30)
    chunks = splitter.split_text(sample_text)

    print(f"\n[1] Chunking plain text (chunk_size=180, overlap=30):")
    print(f"    Original length: {len(sample_text)} chars -> {len(chunks)} chunks")
    for idx, c in enumerate(chunks, 1):
        print(f"    - Chunk #{idx} ({len(c)} chars): {c.strip()}")

    # 2. Chunking a real document (README.md)
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        raw_docs = load_document(readme_path)
        doc_chunks = splitter.split_documents(raw_docs)
        print(f"\n[2] Chunking '{readme_path.name}':")
        print(f"    Input: 1 file ({len(raw_docs[0].page_content)} characters)")
        print(f"    Output: {len(doc_chunks)} chunks")
        if doc_chunks:
            print(f"    Sample chunk #1:")
            print(f"      Metadata: {doc_chunks[0].metadata}")
            print(f"      Preview: {doc_chunks[0].page_content[:100]}...\n")

    print("============================================================")
    print("[✔] Text splitter executed successfully!")


if __name__ == "__main__":
    main()
