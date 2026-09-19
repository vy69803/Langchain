import os
import sys

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src directory is in sys.path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langchain_rag.embeddings import (
    LocalEmbeddings,
    calculate_similarity,
    cosine_similarity,
    get_embeddings,
)


def main() -> None:
    print("============================================================")
    print("            LangChain RAG - Embeddings Demo                 ")
    print("============================================================")

    # 1. Initialize local embeddings (all-MiniLM-L6-v2, runs offline, 100% free)
    embeddings = get_embeddings("local")
    print(f"\n[1] Model: {embeddings.__class__.__name__} (384-dimensional ONNX)")

    # 2. Embed sample text
    query = "Retrieval-Augmented Generation in LangChain"
    vector = embeddings.embed_query(query)
    print(f"\n[2] Embedding text: '{query}'")
    print(f"    - Dimensions: {len(vector)}")
    print(f"    - Sample coordinates: {[round(x, 4) for x in vector[:6]]}...")

    # 3. Semantic similarity test
    text_1 = "Python vector database for AI agents"
    text_2 = "ChromaDB stores embeddings for LLM applications"
    text_3 = "The quick brown fox jumps over the lazy dog"

    sim_1_2 = calculate_similarity(text_1, text_2, embeddings)
    sim_1_3 = calculate_similarity(text_1, text_3, embeddings)

    print(f"\n[3] Cosine Similarity Test:")
    print(f"    1. '{text_1}'")
    print(f"    2. '{text_2}'")
    print(f"    3. '{text_3}'")
    print(f"\n    Similarity (1 & 2 - AI/Database related): {sim_1_2:.4f}")
    print(f"    Similarity (1 & 3 - Unrelated):           {sim_1_3:.4f}")

    print("\n============================================================")
    print("[✔] Embeddings executed successfully!")


if __name__ == "__main__":
    main()
