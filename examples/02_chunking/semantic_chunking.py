import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root and src directory are in sys.path for direct execution
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_rag.document_loader import load_document
from langchain_rag.embeddings import get_embeddings

load_dotenv()


def demonstrate_semantic_chunking() -> None:
    """Demonstrate how Semantic Chunking splits text based on semantic similarity rather than character count."""
    print("=" * 65)
    print("      LangChain RAG - Semantic Chunking Demonstration")
    print("=" * 65)

    # 1. Initialize embeddings model (Local ONNX all-MiniLM-L6-v2 by default)
    print("\n[1] Initializing Embeddings Model...")
    embeddings = get_embeddings("local")
    print(f"    ✓ Model: {embeddings.__class__.__name__} (local & offline)")

    # 2. Sample text containing multiple distinct semantic topics
    sample_text = (
        "Artificial intelligence and machine learning have revolutionized modern medicine. "
        "Deep learning models can analyze medical images, MRI scans, and X-rays to detect abnormalities early. "
        "Physicians and radiologists use these AI diagnostic assistants to provide more accurate treatments.\n\n"
        "Baking sourdough bread at home requires patience and attention to fermentation. "
        "The starter is a symbiotic culture of wild yeast and lactic acid bacteria. "
        "Proper hydration and temperature control are crucial for developing an open, airy crumb structure.\n\n"
        "Quantum computing leverages the principles of quantum mechanics like superposition and entanglement. "
        "Unlike classical bits that exist strictly as 0 or 1, qubits can represent multiple states simultaneously. "
        "This makes quantum computers exceptionally powerful for cryptographic algorithms and complex molecular simulations."
    )

    # 3. Recursive Character Text Splitter (Rule-based chunking)
    print("\n[2] Splitting text with RecursiveCharacterTextSplitter (chunk_size=400, overlap=50)...")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    recursive_chunks = recursive_splitter.split_text(sample_text)
    print(f"    Total recursive chunks generated: {len(recursive_chunks)}")
    for idx, chunk in enumerate(recursive_chunks, 1):
        print(f"    - [Recursive Chunk #{idx}] ({len(chunk)} chars): {chunk.strip()[:80]}...")

    # 4. Semantic Chunker (Embedding similarity-based chunking)
    print("\n[3] Splitting text with SemanticChunker (breakpoint_threshold_type='percentile')...")
    semantic_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )

    semantic_chunks = semantic_chunker.split_text(sample_text)

    print(f"    Input text length: {len(sample_text)} characters")
    print(f"    Total semantic chunks generated: {len(semantic_chunks)}\n")

    for idx, chunk in enumerate(semantic_chunks, 1):
        print(f"    --- Semantic Chunk #{idx} ({len(chunk)} chars) ---")
        print(f"    {chunk.strip()}\n")

    # 5. Store in Chroma Vector Stores
    print("[4] Creating Chroma Vector Stores for both chunking strategies...")
    recursive_vectorstore = Chroma.from_texts(
        texts=recursive_chunks,
        embedding=embeddings,
        collection_name="recursive_chunks",
    )
    print("    ✓ Created 'recursive_chunks' collection in Chroma")

    semantic_vectorstore = Chroma.from_texts(
        texts=semantic_chunks,
        embedding=embeddings,
        collection_name="semantic_chunks",
    )
    print("    ✓ Created 'semantic_chunks' collection in Chroma")

    # Quick similarity search comparison
    test_query = "How do physicians and radiologists diagnose diseases using AI?"
    print(f"\n    Querying Vector Stores: '{test_query}'")

    rec_result = recursive_vectorstore.similarity_search(test_query, k=1)
    if rec_result:
        print(f"    - Top Recursive Match: {rec_result[0].page_content[:100]}...")

    sem_result = semantic_vectorstore.similarity_search(test_query, k=1)
    if sem_result:
        print(f"    - Top Semantic Match : {sem_result[0].page_content[:100]}...\n")

    # 6. Compare different breakpoint threshold types
    print("[5] Comparing Breakpoint Threshold Types:")
    threshold_types = ["percentile", "standard_deviation", "interquartile", "gradient"]

    for threshold in threshold_types:
        try:
            chunker = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=threshold,
            )
            result_chunks = chunker.split_text(sample_text)
            print(f"    - Threshold Type '{threshold:<18}': {len(result_chunks)} chunks generated")
        except Exception as e:
            print(f"    - Threshold Type '{threshold:<18}': Error ({e})")

    # 7. Chunking a document from file if available
    readme_path = project_root / "README.md"
    if readme_path.exists():
        print(f"\n[6] Chunking Document: '{readme_path.name}'...")
        docs = load_document(readme_path)
        if docs:
            doc_chunks = semantic_chunker.split_documents(docs)
            print(f"    Original document size : {len(docs[0].page_content)} characters")
            print(f"    Semantic chunks created: {len(doc_chunks)}")
            if doc_chunks:
                print("\n    Sample Chunk #1 Preview:")
                print(f"    {doc_chunks[0].page_content[:180]}...")

    # 8. Demonstrate smart_chunker with fallback
    print("\n[7] Testing smart_chunker (Semantic with Recursive fallback):")
    from langchain_rag.semantic_chunking import smart_chunker
    demo_chunks = smart_chunker(sample_text, use_semantic=True)
    print(f"    Total chunks generated by smart_chunker: {len(demo_chunks)}")

    print("\n" + "=" * 65)
    print("[✔] Semantic chunking completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    demonstrate_semantic_chunking()
