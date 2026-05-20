"""
vector_store.py
---------------
ChromaDB vector store – persist, add, search, and manage document chunks.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional

# Path where ChromaDB stores its data on disk
CHROMA_PERSIST_PATH = "./chroma_db"


def get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)


def get_or_create_collection(collection_name: str = "rag_documents"):
    """
    Get an existing ChromaDB collection or create it if it doesn't exist.

    Args:
        collection_name: Name of the collection.

    Returns:
        ChromaDB Collection object.
    """
    client = get_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )
    safe_name = collection_name.encode("ascii", errors="replace").decode("ascii")
    print(f"[VectorStore] Collection '{safe_name}' ready ({collection.count()} docs).")
    return collection


def add_chunks(
    chunks: List[Dict],
    embeddings: List[List[float]],
    collection_name: str = "rag_documents",
) -> None:
    """
    Add chunks and their embeddings to the ChromaDB collection.

    Args:
        chunks:          List of chunk dicts (must contain 'chunk_id', 'text', 'source', 'page').
        embeddings:      Corresponding list of embedding vectors.
        collection_name: Target collection name.
    """
    collection = get_or_create_collection(collection_name)

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

    # Add in batches to avoid memory issues with large corpora
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    safe_name = collection_name.encode("ascii", errors="replace").decode("ascii")
    print(f"[VectorStore] Added {len(chunks)} chunks to '{safe_name}'.")


def search(
    query_embedding: List[float],
    n_results: int = 5,
    collection_name: str = "rag_documents",
) -> List[Dict]:
    """
    Search the vector store for the most similar chunks.

    Args:
        query_embedding: Embedding vector for the user query.
        n_results:       Number of results to return.
        collection_name: Collection to search in.

    Returns:
        List of result dicts: [{"text": str, "source": str, "page": int, "score": float}, ...]
    """
    collection = get_or_create_collection(collection_name)

    if collection.count() == 0:
        print("[VectorStore] Collection is empty – no results.")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", 0),
            "score": round(1 - dist, 4),  # Convert cosine distance to similarity
        })

    return output


def delete_collection(collection_name: str = "rag_documents") -> None:
    """Delete all documents in a collection (reset the knowledge base)."""
    client = get_client()
    try:
        client.delete_collection(collection_name)
        print(f"[VectorStore] Collection '{collection_name}' deleted.")
    except Exception as e:
        print(f"[VectorStore] Could not delete collection: {e}")


def list_sources(collection_name: str = "rag_documents") -> List[str]:
    """Return unique source filenames currently stored in the collection."""
    collection = get_or_create_collection(collection_name)
    if collection.count() == 0:
        return []

    all_items = collection.get(include=["metadatas"])
    sources = list({m.get("source", "unknown") for m in all_items["metadatas"]})
    return sorted(sources)


def collection_count(collection_name: str = "rag_documents") -> int:
    """Return total number of chunks stored."""
    collection = get_or_create_collection(collection_name)
    return collection.count()
