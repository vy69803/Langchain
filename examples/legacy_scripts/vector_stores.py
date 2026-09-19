import os
import sys

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure src directory is in sys.path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langchain_rag.vector_stores import VectorStore, create_vector_store

if __name__ == "__main__":
    print("=" * 55)
    print("  LangChain RAG - Vector Store Demonstration")
    print("=" * 55)

    # 1. Create an in-memory vector store
    store = create_vector_store("demo_collection")
    print(f"\n[1] Created: {store}")

    # 2. Add documents
    texts = [
        "Artificial intelligence is transforming healthcare.",
        "Machine learning models require large datasets.",
        "Natural language processing understands human text.",
        "Deep learning uses neural networks with many layers.",
        "Reinforcement learning trains agents through rewards.",
    ]
    ids = store.add_texts(texts)
    print(f"\n[2] Added {len(ids)} documents. Total: {store.count()}")

    # 3. Query by similarity
    query = "How does AI help in medicine?"
    print(f"\n[3] Query: \"{query}\"")
    results = store.query(query, n_results=3)
    for rank, r in enumerate(results, 1):
        print(f"    #{rank} (distance: {r['distance']:.4f}): {r['text']}")

    print(f"\n[✔] Done: {store}")
