"""
embeddings.py
-------------
Generate sentence embeddings locally using sentence-transformers.
No API key required.
"""

from typing import List
from sentence_transformers import SentenceTransformer

# Using a lightweight, fast model good for retrieval tasks
MODEL_NAME = "all-MiniLM-L6-v2"

# Singleton model instance (loaded once, reused)
_model: SentenceTransformer = None


def get_model() -> SentenceTransformer:
    """Load the embedding model (singleton pattern)."""
    global _model
    if _model is None:
        print(f"[Embeddings] Loading model '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[Embeddings] Model loaded.")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Generate an embedding for a single query string.

    Args:
        query: The user question.

    Returns:
        Embedding vector as list of floats.
    """
    model = get_model()
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding[0].tolist()
