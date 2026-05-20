"""
reranker.py
-----------
Re-rank retrieved chunks using a cross-encoder model for higher precision.
Uses: cross-encoder/ms-marco-MiniLM-L-6-v2 (~70 MB, already in sentence-transformers)
"""

from typing import List, Dict

_cross_encoder = None
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder():
    """Load cross-encoder singleton."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            print(f"[Reranker] Loading '{RERANKER_MODEL}'...")
            _cross_encoder = CrossEncoder(RERANKER_MODEL)
            print("[Reranker] Model ready.")
        except Exception as e:
            print(f"[Reranker] Could not load model: {e}")
            _cross_encoder = None
    return _cross_encoder


def rerank(query: str, chunks: List[Dict], top_k: int = None) -> List[Dict]:
    """
    Re-rank chunks using a cross-encoder.

    Args:
        query:  The user question.
        chunks: List of chunk dicts from the vector store.
        top_k:  How many to keep after re-ranking (None = keep all).

    Returns:
        Chunks sorted by cross-encoder score (descending), with 'rerank_score' added.
    """
    if not chunks:
        return chunks

    model = _get_cross_encoder()
    if model is None:
        # Graceful fallback: return original order
        print("[Reranker] Falling back to original retrieval order.")
        return chunks

    pairs = [(query, c["text"]) for c in chunks]

    try:
        scores = model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = round(float(score), 4)

        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k] if top_k else reranked

    except Exception as e:
        print(f"[Reranker] Prediction failed: {e}. Returning original order.")
        return chunks
