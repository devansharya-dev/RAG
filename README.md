# LocalRAG

A modular, testable, and local-first Retrieval-Augmented Generation (RAG) system built with FastAPI, Ollama, and Supabase pgvector.

## Overview

LocalRAG is designed to run local-first, keeping document parsing, chunking, embeddings, and LLM inference local via Ollama while persisting vectors securely in Supabase PostgreSQL with pgvector. It ingests local documents (PDF, DOCX, TXT), splits them into traceable chunks, generates semantic embeddings, persists vectors in Supabase, and performs grounded question answering using local LLMs.

## Technology Stack

- **Python**: 3.10+
- **API Framework**: FastAPI & Uvicorn
- **Local LLM & Embedding Inference**: [Ollama](https://ollama.ai/)
  - Initial Chat Model: `llama3.2:3b`
  - Embedding Model: `nomic-embed-text`
- **Vector Storage**: [Supabase](https://supabase.com/) (PostgreSQL with pgvector)
- **Document Parsers**: PyMuPDF (`fitz`), `python-docx`
- **Validation & Settings**: Pydantic v2 & `pydantic-settings`
- **Testing & Quality**: pytest, pytest-asyncio, Ruff

## Project Structure

```text
local-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Application entry point and lifecycle
│   ├── api/                     # API routing and dependencies
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   ├── router.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py        # Liveness & readiness probes
│   │       ├── documents.py     # Document upload & management
│   │       └── chat.py          # Grounded QA endpoint
│   ├── core/                    # Core configuration and cross-cutting concerns
│   │   ├── __init__.py
│   │   ├── config.py            # Environment-driven settings
│   │   ├── logging.py           # Structured logging setup
│   │   ├── exceptions.py        # Domain-specific exceptions
│   │   └── constants.py         # Global constants
│   ├── schemas/                 # Pydantic schemas (Deduplicated DTOs)
│   │   ├── __init__.py
│   │   ├── document.py
│   │   ├── chat.py
│   │   └── retrieval.py
│   ├── ingestion/               # Document ingestion pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── text_cleaner.py
│   │   ├── chunker.py
│   │   └── loaders/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── pdf_loader.py
│   │       ├── text_loader.py
│   │       └── docx_loader.py
│   ├── embeddings/              # Embedding generation
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── ollama_embeddings.py
│   ├── vectorstore/             # Vector database abstraction and storage
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── supabase_store.py
│   ├── retrieval/               # Retrieval logic, filtering, and reranking
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   ├── filters.py
│   │   └── reranker.py
│   ├── generation/              # Local LLM client and answer generation
│   │   ├── __init__.py
│   │   ├── llm_client.py
│   │   ├── prompt_builder.py
│   │   └── answer_generator.py
│   ├── services/                # Application coordination services
│   │   ├── __init__.py
│   │   ├── ingestion_service.py
│   │   └── rag_service.py
│   └── utils/                   # File system and cryptographic utilities
│       ├── __init__.py
│       ├── file_utils.py
│       └── hashing.py
├── scripts/                     # Operational and maintenance scripts
│   ├── ingest_documents.py
│   ├── reindex_documents.py
│   └── check_environment.py
├── tests/                       # Test suite
│   ├── conftest.py
│   ├── unit/                    # Fast isolated tests
│   ├── integration/             # Component and store integration tests
│   └── evaluation/              # Grounding and retrieval quality tests
├── data/                        # Persistent local data (git-ignored)
│   ├── documents/
│   ├── chroma_db/
│   ├── evaluation/
│   └── cache/
├── logs/                        # Application log files (git-ignored)
├── docs/                        # Architecture & development documentation
│   ├── architecture.md
│   ├── development.md
│   └── prd.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── README.md
└── LICENSE
```

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally:
  ```bash
  ollama serve
  ollama pull nomic-embed-text
  ollama pull llama3.2:3b
  ```

### 2. Environment Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements-dev.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and adjust settings as needed:
```bash
cp .env.example .env
```

### 4. Running the Development Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Access the interactive API documentation at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Health check: `http://127.0.0.1:8000/health`

## Development & Testing

Run tests with pytest:
```bash
pytest
```

Run linter with Ruff:
```bash
ruff check .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
