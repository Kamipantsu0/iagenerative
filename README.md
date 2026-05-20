# 🤖 Système RAG – IA Générative

Système RAG (Retrieval-Augmented Generation) complet permettant d'interroger des documents PDF via un LLM local.

## Architecture

```
PDF → Extraction texte → Chunking → Embeddings → ChromaDB
                                                      ↓
Question → Embedding → Recherche similarité → Contexte → Ollama → Réponse
```

## Prérequis

### 1. Python 3.10+

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Installer Ollama
Télécharger depuis [ollama.ai](https://ollama.ai) puis :
```bash
ollama pull llama3
ollama serve
```

## Lancement

```bash
streamlit run app.py
```

Ouvrir `http://localhost:8501` dans le navigateur.

## Utilisation

1. **Importer** vos PDFs via la barre latérale
2. Cliquer **Indexer les documents**
3. **Poser vos questions** dans le chat
4. Les réponses citent les **sources** et numéros de page

## Structure du projet

```
ia generative/
├── app.py                    # Interface Streamlit
├── rag/
│   ├── document_loader.py    # Extraction texte PDF
│   ├── chunker.py            # Découpage en chunks
│   ├── embeddings.py         # Génération d'embeddings (all-MiniLM-L6-v2)
│   ├── vector_store.py       # Base vectorielle ChromaDB
│   ├── generator.py          # Génération via Ollama
│   └── pipeline.py           # Orchestration complète
├── data/pdfs/                # PDFs importés
├── chroma_db/                # Stockage vectoriel persistant
└── requirements.txt
```

## Technologies utilisées

| Composant | Technologie |
|---|---|
| Extraction PDF | pdfplumber |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Base vectorielle | ChromaDB |
| LLM | Ollama (llama3 / mistral) |
| Interface | Streamlit |
"# iagenerative" 
