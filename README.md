# RAG Pipeline — PDF to Vector Store

A **Retrieval-Augmented Generation (RAG)** data ingestion pipeline that loads PDF documents, chunks them intelligently, generates semantic embeddings, and stores everything in a persistent ChromaDB vector database — ready to power an LLM-based Q&A system.

---

## What It Does

```
PDF Files  →  Document Loader  →  Text Splitter  →  Embeddings  →  ChromaDB Vector Store
```

1. **PDF Ingestion** — Recursively discovers and loads all `.pdf` files from a directory using LangChain's `PyPDFLoader`
2. **Smart Chunking** — Splits documents into overlapping chunks (1000 chars, 200 overlap) with `RecursiveCharacterTextSplitter` for context-aware retrieval
3. **Semantic Embeddings** — Encodes every chunk using `all-MiniLM-L6-v2` (384-dim vectors) via SentenceTransformers
4. **Vector Store** — Persists all embeddings + metadata in a local ChromaDB collection for fast similarity search

---

## Project Structure

```
RAG/
├── notebook/
│   └── pdf_loader.ipynb     # Full ingestion pipeline (step-by-step)
├── data/
│   ├── pdf_files/           # Drop your PDFs here
│   └── vector_store/        # Auto-created ChromaDB persistence dir
├── main.py                  # Entry point (pipeline runner — WIP)
├── pyproject.toml           # Dependencies
└── .python-version          # Python 3.13
```

---

## Tech Stack

| Layer | Library |
|---|---|
| PDF Loading | `langchain-community` (PyPDFLoader, PyMuPDFLoader) |
| Text Splitting | `langchain` RecursiveCharacterTextSplitter |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | `chromadb` (persistent local store) |
| Similarity Search | `faiss-cpu`, `scikit-learn` cosine similarity |
| Notebook | `ipykernel` / Jupyter |

---

## Getting Started

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repo
git clone https://github.com/mankarmanas/RAG.git
cd RAG

# Create virtual environment and install dependencies
uv sync
# or
pip install -e .
```

### Run the Pipeline

**Option 1 — Jupyter Notebook (recommended for exploration)**
```bash
jupyter notebook notebook/pdf_loader.ipynb
```

**Option 2 — Python script (WIP)**
```bash
python main.py
```

### Add Your PDFs

Drop your PDF files into `data/pdf_files/` and run the pipeline. The vector store will be created automatically at `data/vector_store/`.

---

## Pipeline Walkthrough

### 1. Load PDFs
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/pdf_files/your_document.pdf")
documents = loader.load()  # Returns list of LangChain Document objects
```

### 2. Split into Chunks
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
# 44 pages → 260 chunks
```

### 3. Generate Embeddings
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([doc.page_content for doc in chunks])
# Shape: (260, 384)
```

### 4. Store in ChromaDB
```python
import chromadb

client = chromadb.PersistentClient(path="data/vector_store")
collection = client.get_or_create_collection("pdf_documents")
collection.add(ids=[...], embeddings=[...], documents=[...], metadatas=[...])
```

---

## Sample Output

```
Found 4 PDF files to process
Processing: pdf1.pdf  ✓ Loaded 11 pages
Processing: pdf2.pdf  ✓ Loaded 11 pages
Processing: pdf3.pdf  ✓ Loaded 11 pages
Processing: pdf4.pdf  ✓ Loaded 11 pages

Split 44 documents into 260 chunks

Loading embedding model: all-MiniLM-L6-v2
Model loaded successfully. Embedding dimension: 384
Generating embeddings for 260 texts...

Vector store initialized. Collection: pdf_documents
Successfully added 260 documents to vector store
```

---

## Roadmap

- [x] PDF loading and preprocessing
- [x] Recursive text chunking with overlap
- [x] Sentence embedding generation (all-MiniLM-L6-v2)
- [x] ChromaDB vector store with persistence
- [ ] Retrieval (similarity search / MMR)
- [ ] LLM integration (Claude / OpenAI) for Q&A
- [ ] REST API / FastAPI backend
- [ ] Frontend chat interface
- [ ] Multi-modal support (images in PDFs)

---

## License

MIT
