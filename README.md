# AMM-Bot — Aircraft Maintenance Manual Intelligence System

> A Retrieval-Augmented Generation (RAG) system engineered to query, retrieve, and synthesize technical knowledge from Aircraft Maintenance Manuals (AMMs). Powered by OpenAI GPT-4o, Pinecone vector database, FAISS local index, and a cross-encoder reranker for high-precision document retrieval.

---

## Overview

AMM-Bot is a domain-specific LLM application built for aviation maintenance engineers and technical publications teams. It ingests structured AMM PDF documentation, indexes them into a vector database, and enables natural language queries against the manual corpus — returning expert-level, context-grounded responses modelled after AMM documentation style.

---

## Architecture

```
PDF Docs (AMM)
     │
     ▼
[DocumentLoader]  ←── PyMuPDF (fitz)
     │
     ▼
[EmbeddingManager]  ←── RecursiveCharacterTextSplitter + OpenAI text-embedding-ada-002
     │
     ├──► [VectorStoreManager]  ──► FAISS local index  (database/)
     │
     └──► [PCVectorDB]          ──► Pinecone vector index (cloud)
                                         │
                                         ▼
                                  [RAGRetriever]
                                  ├── FAISS similarity_search (k nearest)
                                  └── CrossEncoder reranker (ms-marco-MiniLM-L-6-v2)
                                         │
                                         ▼
                               [ChatOpenAI — GPT-4o]
                                         │
                                         ▼
                               AMM-style natural language response
```

---

## Key Components

### `utils/documentloader.py` — PDF Ingestion
- Recursively scans a target directory for PDF files.
- Uses `PyMuPDFLoader` (via `langchain-community`) for high-fidelity text extraction from structured AMM content.
- Returns a flat `List[Document]` with page-level metadata preserved.

### `utils/embedding_pipeline.py` — Chunking and Embedding
- Splits documents using `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200`, preserving semantic boundaries across AMM sections.
- Instantiates `OpenAIEmbeddings` (text-embedding-ada-002) via a validated `SecretStr` API key.

### `utils/vectorstore_manager.py` — Local FAISS Index
- Builds and persists a FAISS vector store from document chunks.
- Saved locally under `database/` for offline retrieval without external API calls.

### `utils/pc_vectordb.py` — Pinecone Cloud Vector Index
- Connects to Pinecone and manages index operations.
- Converts LangChain `Document` chunks to `Vector` objects (UUID-keyed, with metadata).
- Upserts vectors into a named namespace under a specified Pinecone index.

### `utils/rag_retriever.py` — Retrieval and Reranking
- Loads FAISS index from disk and runs cosine similarity search.
- Applies `CrossEncoder (cross-encoder/ms-marco-MiniLM-L-6-v2)` for two-stage reranking of initial candidates.
- Returns documents ranked by cross-attention relevance score — critical for precise AMM section retrieval.

### `core/config.py` — Settings and Validation
- Pydantic `BaseSettings` with `.env` file support.
- Validates OpenAI API key format (`sk-` prefix check) at startup via `field_validator`.
- Configures LangSmith tracing for observability and query audit trail.

### `main.py` — Query Interface
- End-to-end query pipeline: retrieves top-k relevant AMM context, constructs a domain-scoped prompt, and invokes GPT-4o for the response.
- System prompt is AMM-manual-style: "Aircraft Maintenance Technical Expert" persona.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o (`gpt-4o-2024-08-06`) |
| Embedding | OpenAI `text-embedding-ada-002` |
| Vector DB (cloud) | Pinecone >= 8.1.0 |
| Vector DB (local) | FAISS CPU >= 1.13.2 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` |
| PDF Parsing | PyMuPDF (`pymupdf`) >= 1.27.2 |
| LLM Framework | LangChain >= 1.2.13 |
| Config | `pydantic-settings` >= 2.13.1 |
| Observability | LangSmith |
| Runtime | Python >= 3.12 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/amm-bot.git
cd amm-bot

# Create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate        # Linux/macOS

# Install dependencies
pip install -e .
```

---

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...

# LangSmith (optional but recommended for tracing)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=amm-bot

MODEL=gpt-4o-2024-08-06
```

---

## Usage

### 1. Ingest AMM documents into FAISS (local)

Place PDF files inside the `doc/` directory, then run:

```bash
python utils/vectorstore_manager.py
```

This will produce `database/index.faiss` and `database/index.pkl`.

### 2. Ingest into Pinecone (cloud)

```bash
python utils/pc_vectordb.py
```

Configure `index_name` and `namespace` in the `__main__` block to match your Pinecone index and ATA chapter scope (e.g., `32-XX-XX-Landing-Gear-Docs`).

### 3. Query the AMM

```bash
python main.py
```

Example query:

```
"Summarize the operation of landing gears and doors."
```

---

## Project Structure

```
amm-bot/
├── core/
│   └── config.py               # Pydantic settings, API key validation, LangSmith setup
├── database/
│   └── index.faiss             # Persisted FAISS vector index
├── doc/                        # Drop AMM PDFs here for ingestion
├── logs/
│   └── metadata.py
├── schema/
│   └── schemas.py              # Pydantic output models
├── utils/
│   ├── documentloader.py       # PyMuPDF PDF ingestion
│   ├── embedding_pipeline.py   # Chunking + OpenAI embedding
│   ├── pc_vectordb.py          # Pinecone vector DB operations
│   ├── rag_retriever.py        # FAISS retrieval + cross-encoder reranking
│   └── vectorstore_manager.py  # FAISS build and persist
├── main.py                     # Query entrypoint
├── pyproject.toml
└── README.md
```

---

## Roadmap

- [ ] Batch upsert with configurable chunk size for large AMM volumes
- [ ] ATA chapter-aware namespace routing in Pinecone
- [ ] REST API layer (FastAPI) for integration with MRO systems
- [ ] Streaming response support
- [ ] Hybrid retrieval (BM25 + dense vectors)
- [ ] MMR (Maximum Marginal Relevance) for result diversity

---

## License

MIT
