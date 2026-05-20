"""
chunker.py
----------
Split extracted page text into overlapping chunks for better retrieval.
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict]:
    """
    Split page-level text into smaller overlapping chunks.

    Args:
        pages:         List of page dicts from document_loader.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunk dicts:
        [{"chunk_id": str, "text": str, "source": str, "page": int}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = []
    for page in pages:
        raw_chunks = splitter.split_text(page["text"])
        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{page['source']}_p{page['page']}_c{i}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text.strip(),
                "source": page["source"],
                "page": page["page"],
            })

    print(
        f"[Chunker] Created {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})."
    )
    return chunks
