"""
pipeline.py
-----------
Orchestrates the full RAG pipeline (Option 2 – RAG Amélioré):
  PDF → Extract → Chunk → Embed → Store → Retrieve → Re-rank → Generate

Improvements over Option 1:
  - Re-ranking step after retrieval (cross-encoder)
  - Conversation history passed to generator
  - use_reranker toggle
"""

from typing import List, Dict, Optional, Callable
from rag.document_loader import load_pdf
from rag.chunker import chunk_pages
from rag.embeddings import embed_texts, embed_query
from rag.vector_store import (
    add_chunks,
    search,
    delete_collection,
    list_sources,
    collection_count,
)
from rag.generator import generate_response, DEFAULT_MODEL
from rag.reranker import rerank


class RAGPipeline:
    """
    End-to-end RAG pipeline (Option 2 – RAG Amélioré).

    Steps:
      1. Load PDF        → document_loader.py
      2. Chunk text      → chunker.py
      3. Embed chunks    → embeddings.py  (sentence-transformers, local)
      4. Store vectors   → vector_store.py (ChromaDB, persistent)
      5. Search          → vector_store.py (cosine similarity, top-k)
      6. Re-rank         → reranker.py    (cross-encoder, optional)
      7. Generate answer → generator.py   (Ollama, with history)
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
        model_name: str = DEFAULT_MODEL,
        use_reranker: bool = True,
    ):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.model_name = model_name
        self.use_reranker = use_reranker

    # ------------------------------------------------------------------
    # STEP 1–4 : INGESTION
    # ------------------------------------------------------------------

    def ingest(
        self,
        pdf_path: str,
        progress_callback: Optional[Callable[[str, int], None]] = None,
    ) -> Dict:
        """Full ingestion pipeline for a single PDF."""

        def _progress(msg: str, pct: int):
            safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
            print(f"[Pipeline] ({pct}%) {safe_msg}")
            if progress_callback:
                progress_callback(msg, pct)

        # Step 1 – Extract text
        _progress("[1/4] Extraction du texte PDF...", 10)
        pages = load_pdf(pdf_path)
        if not pages:
            return {"success": False, "error": "Aucun texte extrait du PDF."}

        # Step 2 – Chunk
        _progress("[2/4] Decoupage en chunks...", 30)
        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return {"success": False, "error": "Aucun chunk cree."}

        # Step 3 – Generate embeddings
        _progress("[3/4] Generation des embeddings...", 55)
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        # Step 4 – Store in ChromaDB
        _progress("[4/4] Stockage dans ChromaDB...", 80)
        add_chunks(chunks, embeddings, self.collection_name)

        _progress("Indexation terminee !", 100)

        return {
            "success": True,
            "pages": len(pages),
            "chunks": len(chunks),
            "source": pdf_path,
        }

    # ------------------------------------------------------------------
    # STEP 5–7 : QUERYING
    # ------------------------------------------------------------------

    def query(self, question: str, history_text: str = "") -> Dict:
        """
        Run a full RAG query with optional re-ranking and conversation history.

        Args:
            question:     User's natural language question.
            history_text: Formatted conversation history string from ConversationMemory.

        Returns:
            Dict: {
              "answer": str,
              "sources": List[str],
              "context": List[Dict],
              "reranked": bool,
            }
        """
        if collection_count(self.collection_name) == 0:
            return {
                "answer": "Aucun document indexe. Importez d'abord un PDF.",
                "sources": [],
                "context": [],
                "reranked": False,
            }

        # Step 5 – Embed query & search (retrieve more candidates for re-ranking)
        fetch_k = self.top_k * 3 if self.use_reranker else self.top_k
        q_embedding = embed_query(question)
        results = search(q_embedding, fetch_k, self.collection_name)

        if not results:
            return {
                "answer": "Aucun passage pertinent trouve dans les documents.",
                "sources": [],
                "context": [],
                "reranked": False,
            }

        # Step 6 – Re-rank (optional)
        reranked = False
        if self.use_reranker and len(results) > 1:
            results = rerank(question, results, top_k=self.top_k)
            reranked = True
        else:
            results = results[: self.top_k]

        # Step 7 – Generate answer with history
        answer = generate_response(
            question,
            results,
            model_name=self.model_name,
            history_text=history_text,
        )

        sources = sorted({
            f"{r['source']} (page {r['page']})" for r in results
        })

        return {
            "answer": answer,
            "sources": sources,
            "context": results,
            "reranked": reranked,
        }

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the entire vector store."""
        delete_collection(self.collection_name)
        print("[Pipeline] Base de connaissances reinitialisee.")

    def get_indexed_sources(self) -> List[str]:
        """Return list of indexed source filenames."""
        return list_sources(self.collection_name)

    def get_chunk_count(self) -> int:
        """Return total number of indexed chunks."""
        return collection_count(self.collection_name)
