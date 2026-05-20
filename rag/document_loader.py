"""
document_loader.py
------------------
Load PDF files and extract text content page by page.
"""

import os
import pdfplumber
from typing import List, Dict


def load_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from a PDF file, returning a list of page dicts.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of dicts: [{"page": int, "text": str, "source": str}, ...]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    filename = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page": i + 1,
                    "text": text.strip(),
                    "source": filename,
                })

    safe_name = filename.encode("ascii", errors="replace").decode("ascii")
    print(f"[DocumentLoader] Loaded '{safe_name}': {len(pages)} pages extracted.")
    return pages


def load_pdfs_from_folder(folder_path: str) -> List[Dict]:
    """
    Load all PDFs found in a directory.

    Args:
        folder_path: Path to folder containing PDFs.

    Returns:
        Combined list of page dicts from all PDFs.
    """
    all_pages = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        safe_folder = folder_path.encode("ascii", errors="replace").decode("ascii")
        print(f"[DocumentLoader] No PDFs found in '{safe_folder}'.")
        return []

    for filename in pdf_files:
        full_path = os.path.join(folder_path, filename)
        try:
            pages = load_pdf(full_path)
            all_pages.extend(pages)
        except Exception as e:
            safe_fn = filename.encode("ascii", errors="replace").decode("ascii")
            print(f"[DocumentLoader] Error loading '{safe_fn}': {e}")

    print(f"[DocumentLoader] Total pages extracted: {len(all_pages)}")  # ASCII-safe
    return all_pages
