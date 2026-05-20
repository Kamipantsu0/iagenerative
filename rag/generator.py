"""
generator.py
------------
Generate answers via Ollama local HTTP API (Option 2 – RAG Amélioré).
Improvements over Option 1:
  - Conversation history injected into prompt (memory-aware)
  - Structured system prompt for better context adherence
  - Source attribution in prompt

Requires Ollama running: `ollama serve`
Model must be pulled: `ollama pull mistral`
"""

import requests
from typing import List, Dict

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"


def build_prompt(
    question: str,
    context_chunks: List[Dict],
    history_text: str = "",
) -> str:
    """
    Build an enhanced RAG prompt with conversation history and source attribution.

    Args:
        question:       The user's current question.
        context_chunks: Relevant chunks from the vector store (already re-ranked).
        history_text:   Formatted conversation history from ConversationMemory.

    Returns:
        Full prompt string for Ollama.
    """
    # Build context block with source labels
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        rerank = chunk.get("rerank_score")
        score_label = f", re-rank score: {rerank:.2f}" if rerank is not None else ""
        src = f"[Passage {i} | Source: {chunk['source']}, Page {chunk['page']}{score_label}]"
        context_parts.append(f"{src}\n{chunk['text']}")

    context_text = "\n\n".join(context_parts)

    # System instruction
    system = (
        "You are a precise document assistant. "
        "Answer questions using ONLY the provided document context. "
        "If the answer is not found in the context, say exactly: "
        "'Information not found in the documents.' "
        "If the user refers to a previous question, use the conversation history."
    )

    # Conversation history block
    history_block = ""
    if history_text:
        history_block = f"\n\n[Conversation History]\n{history_text}"

    prompt = (
        f"{system}"
        f"{history_block}\n\n"
        f"[Document Context]\n{context_text}\n\n"
        f"[Current Question]\n{question}\n\n"
        f"[Answer]"
    )
    return prompt


def generate_response(
    question: str,
    context_chunks: List[Dict],
    model_name: str = DEFAULT_MODEL,
    history_text: str = "",
) -> str:
    """
    Generate an answer using the Ollama local API.

    Args:
        question:       User's question.
        context_chunks: Retrieved (and re-ranked) context chunks.
        model_name:     Ollama model name.
        history_text:   Formatted conversation history string.

    Returns:
        Generated answer string.
    """
    if not context_chunks:
        return "Aucun passage pertinent trouve dans les documents indexes."

    prompt = build_prompt(question, context_chunks, history_text)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        return answer if answer else "Je n'ai pas pu generer de reponse."

    except requests.exceptions.ConnectionError:
        return (
            "Erreur : Ollama n'est pas demarre. "
            "Lancez 'ollama serve' dans un terminal, puis reessayez."
        )
    except requests.exceptions.Timeout:
        return "Erreur : le modele a mis trop de temps a repondre (timeout 120s)."
    except Exception as e:
        return f"Erreur lors de la generation : {e}"


def list_available_models() -> List[str]:
    """Return locally installed Ollama models, or a safe fallback list."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if models:
            return models
    except Exception:
        pass
    return ["mistral", "llama3", "phi3", "gemma:2b"]
