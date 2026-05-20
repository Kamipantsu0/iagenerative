"""
app.py
------
Streamlit interface for the RAG system (Option 2 – RAG Amélioré).
Adds: conversation memory, re-ranking toggle, score display, history panel.
Run: streamlit run app.py
"""

import os
import streamlit as st
from rag.pipeline import RAGPipeline
from rag.generator import list_available_models, DEFAULT_MODEL
from rag.memory import ConversationMemory

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG – IA Générative",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS – dark premium theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 60%, #0f3460 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: rgba(15,15,35,0.95);
    border-right: 1px solid rgba(99,102,241,0.2);
}
h1 { color: #818cf8 !important; letter-spacing: -1px; }
h2, h3 { color: #a5b4fc !important; }

.user-bubble {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    padding: 14px 20px;
    border-radius: 20px 20px 4px 20px;
    margin: 10px 0 10px auto;
    max-width: 78%;
    box-shadow: 0 4px 20px rgba(79,70,229,0.35);
    line-height: 1.6;
}
.assistant-bubble {
    background: rgba(255,255,255,0.06);
    color: #e2e8f0;
    padding: 14px 20px;
    border-radius: 20px 20px 20px 4px;
    margin: 10px 0;
    max-width: 85%;
    border: 1px solid rgba(129,140,248,0.25);
    line-height: 1.6;
}
.source-pill {
    display: inline-block;
    background: rgba(79,70,229,0.2);
    color: #a5b4fc;
    font-size: 0.72rem;
    padding: 3px 12px;
    border-radius: 20px;
    margin: 3px 3px 0 0;
    border: 1px solid rgba(129,140,248,0.3);
}
.rerank-badge {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    color: #6ee7b7;
    font-size: 0.70rem;
    padding: 2px 10px;
    border-radius: 20px;
    border: 1px solid rgba(16,185,129,0.3);
    margin-left: 6px;
}
.step-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(129,140,248,0.2);
    border-radius: 12px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
}
.memory-bubble {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid rgba(129,140,248,0.4);
    padding: 8px 14px;
    margin: 4px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #94a3b8;
}
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(79,70,229,0.45) !important;
}
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(129,140,248,0.35) !important;
    border-radius: 14px !important;
    background: rgba(79,70,229,0.05) !important;
}
.stSelectbox > div, .stSlider > div { color: #e2e8f0 !important; }
[data-testid="stMetricValue"] { color: #818cf8 !important; font-size: 2rem !important; }
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 14px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "memory_list" not in st.session_state:
    st.session_state.memory_list = []   # serialized ConversationMemory


def get_memory() -> ConversationMemory:
    mem = ConversationMemory(max_turns=st.session_state.get("cfg", {}).get("max_turns", 5))
    mem.load_from_list(st.session_state.memory_list)
    return mem


def get_pipeline() -> RAGPipeline:
    cfg = st.session_state.get("cfg", {})
    if st.session_state.pipeline is None:
        st.session_state.pipeline = RAGPipeline(
            chunk_size=cfg.get("chunk_size", 500),
            chunk_overlap=cfg.get("chunk_overlap", 50),
            top_k=cfg.get("top_k", 5),
            model_name=cfg.get("model", DEFAULT_MODEL),
            use_reranker=cfg.get("use_reranker", True),
        )
    return st.session_state.pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RAG – IA Générative")
    st.caption("Option 2 · RAG Amélioré · Ollama + ChromaDB + Re-ranking")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Pipeline steps legend
    st.markdown("### 📐 Pipeline RAG")
    steps = [
        ("1", "📄", "Chargement PDF"),
        ("2", "✂️", "Chunking du texte"),
        ("3", "🧬", "Génération embeddings"),
        ("4", "💾", "Stockage ChromaDB"),
        ("5", "🔍", "Recherche similarité (top-k×3)"),
        ("6", "🎯", "Re-ranking (cross-encoder)"),
        ("7", "💬", "Génération réponse (LLM + mémoire)"),
    ]
    for num, icon, label in steps:
        st.markdown(
            f"<div class='step-card'><b>{num}.</b> {icon} {label}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Settings
    st.markdown("### ⚙️ Paramètres")
    models = list_available_models()
    selected_model = st.selectbox("Modèle Ollama", models, index=0)
    chunk_size    = st.slider("Taille des chunks (chars)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chevauchement", 0, 200, 50, 10)
    top_k         = st.slider("Passages récupérés (top-k)", 1, 10, 5)
    max_turns     = st.slider("Mémoire (nb de tours)", 1, 10, 5)
    use_reranker  = st.toggle("🎯 Re-ranking activé", value=True)

    # Reset pipeline on config change
    prev_cfg = st.session_state.get("cfg", {})
    if (
        prev_cfg.get("model") != selected_model
        or prev_cfg.get("chunk_size") != chunk_size
        or prev_cfg.get("use_reranker") != use_reranker
    ):
        st.session_state.pipeline = None

    st.session_state.cfg = {
        "model": selected_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "max_turns": max_turns,
        "use_reranker": use_reranker,
    }

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # PDF upload
    st.markdown("### 📂 Importer des PDFs")
    uploaded_files = st.file_uploader(
        "Déposez vos PDFs ici",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("📥 Indexer les documents", use_container_width=True):
            os.makedirs("./data/pdfs", exist_ok=True)
            pipeline = get_pipeline()

            for uf in uploaded_files:
                save_path = os.path.join("./data/pdfs", uf.name)
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())

                bar = st.progress(0, text=f"⏳ {uf.name}")

                def cb(msg, pct, b=bar):
                    b.progress(pct / 100, text=msg)

                result = pipeline.ingest(save_path, cb)

                if result["success"]:
                    st.success(
                        f"✅ **{uf.name}** — {result['pages']} pages, {result['chunks']} chunks"
                    )
                else:
                    st.error(f"❌ {result.get('error')}")

            st.rerun()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Knowledge base stats
    st.markdown("### 📊 Base de connaissances")
    pipeline = get_pipeline()
    n_chunks = pipeline.get_chunk_count()
    sources  = pipeline.get_indexed_sources()

    c1, c2 = st.columns(2)
    c1.metric("Chunks", n_chunks)
    c2.metric("Documents", len(sources))

    if sources:
        st.markdown("**Documents indexés :**")
        for s in sources:
            st.markdown(f"- 📄 `{s}`")

    if n_chunks > 0:
        if st.button("🗑️ Réinitialiser la base", use_container_width=True, key="btn_reset_db"):
            pipeline.reset()
            st.session_state.pipeline = None
            st.session_state.messages = []
            st.session_state.memory_list = []
            st.rerun()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Conversation", use_container_width=True, key="btn_clear_conv"):
            st.session_state.messages = []
            st.session_state.memory_list = []
            st.toast("Conversation effacée.", icon="🧹")
            st.rerun()
    with col2:
        if st.button("🧠 Mémoire", use_container_width=True, key="btn_clear_mem"):
            st.session_state.memory_list = []
            st.toast("Mémoire réinitialisée.", icon="🧠")
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Main — chat interface
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 💬 Système RAG – Question & Réponse")
st.markdown(
    "**Option 2 – RAG Amélioré** · Re-ranking · Mémoire de conversation · Prompt engineering"
)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Memory status bar
mem = get_memory()
if len(mem) > 0:
    with st.expander(f"🧠 Mémoire de conversation — {len(mem)} tour(s) enregistré(s)", expanded=False):
        for item in st.session_state.memory_list:
            role_label = "👤 Vous" if item["role"] == "user" else "🤖 Assistant"
            st.markdown(
                f"<div class='memory-bubble'><b>{role_label}:</b> {item['content'][:200]}{'…' if len(item['content']) > 200 else ''}</div>",
                unsafe_allow_html=True,
            )

# Render message history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='user-bubble'>👤 {msg['content']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='assistant-bubble'>🤖 {msg['content']}</div>",
            unsafe_allow_html=True,
        )
        # Source pills with re-rank badge
        if msg.get("sources"):
            reranked = msg.get("reranked", False)
            badge = "<span class='rerank-badge'>🎯 re-ranked</span>" if reranked else ""
            pills = " ".join(
                f"<span class='source-pill'>📄 {s}</span>" for s in msg["sources"]
            )
            st.markdown(
                f"<div style='margin:6px 0 14px;'>Sources {badge}: {pills}</div>",
                unsafe_allow_html=True,
            )

# Chat input
question = st.chat_input("Posez votre question sur les documents indexés...")

if question:
    # Add to display
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(
        f"<div class='user-bubble'>👤 {question}</div>",
        unsafe_allow_html=True,
    )

    # Build memory & query
    mem = get_memory()
    history_text = mem.get_history_text()

    pipeline = get_pipeline()
    with st.spinner("🔍 Recherche + re-ranking + génération en cours..."):
        result  = pipeline.query(question, history_text=history_text)
        answer  = result["answer"]
        sources = result["sources"]
        reranked = result.get("reranked", False)

    # Display answer
    st.markdown(
        f"<div class='assistant-bubble'>🤖 {answer}</div>",
        unsafe_allow_html=True,
    )
    if sources:
        badge = "<span class='rerank-badge'>🎯 re-ranked</span>" if reranked else ""
        pills = " ".join(
            f"<span class='source-pill'>📄 {s}</span>" for s in sources
        )
        st.markdown(
            f"<div style='margin:6px 0 14px;'>Sources {badge}: {pills}</div>",
            unsafe_allow_html=True,
        )

    # Update memory
    mem.add("user", question)
    mem.add("assistant", answer)
    st.session_state.memory_list = mem.get_history_list()

    # Update message history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "reranked": reranked,
    })
