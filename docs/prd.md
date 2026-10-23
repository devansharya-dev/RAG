# Product Requirements Document (PRD)
## LocalRAG — Modular Offline Retrieval-Augmented Generation System

**Version:** 1.0  
**Project type:** Local AI / RAG infrastructure  
**Primary objective:** Build a modular, reliable, testable, and production-oriented RAG engine that runs entirely on the user's computer.  
**Target environment:** Windows 11, Python 3.10+, Ollama, ChromaDB  
**Development approach:** Incremental implementation with testing at every stage.

---

# 1. Project Overview

LocalRAG is a locally hosted Retrieval-Augmented Generation (RAG) system that enables users to query their own documents using AI.

The system must ingest documents, extract and clean text, split text into chunks, generate embeddings, store vectors locally, retrieve relevant information, and generate answers grounded in the retrieved evidence.

The architecture must support future extensions such as hybrid retrieval, reranking, conversational memory, agentic workflows, automation, and a frontend without requiring a complete rewrite.

## 1.1 Core objectives

- Run inference and document retrieval locally without cloud AI APIs.
- Maintain a modular and maintainable Python codebase.
- Support multiple document formats.
- Persist the vector database between application restarts.
- Generate answers using retrieved document context.
- Provide source attribution for generated answers.
- Handle missing evidence and retrieval failures explicitly.
- Include automated tests and evaluation tools.
- Make the system easy to configure, debug, extend, and deploy locally.

## 1.2 Non-goals for the initial version

The initial release will not include:

- Cloud-hosted inference or vector databases.
- Multi-user authentication and authorization.
- Distributed processing.
- Fine-tuning or training an LLM.
- Autonomous agents that execute external actions.
- A complex frontend before the core RAG engine is validated.

These features may be considered after the core system is reliable.

# 2. Technology Stack

| Component | Technology | Responsibility |
|---|---|---|
| Language | Python 3.10+ | Core application |
| API framework | FastAPI | HTTP API |
| Local model runtime | Ollama | Local inference |
| Embedding model | `nomic-embed-text` | Document and query embeddings |
| Initial LLM | `llama3.2:3b` | Answer generation |
| Vector database | ChromaDB | Persistent vector storage |
| Document parsing | PyMuPDF or pypdf, python-docx | PDF and DOCX extraction |
| Validation | Pydantic | Request and configuration validation |
| Testing | pytest | Unit and integration tests |
| HTTP client | httpx | API testing and supported local HTTP calls |
| Configuration | pydantic-settings | Environment-based configuration |
| Logging | Python logging | Diagnostics and application events |
| Code quality | Ruff | Linting and formatting |
| Packaging | `pyproject.toml` | Dependencies and project metadata |

### Technology requirements

1. All AI inference must use the locally running Ollama service.
2. ChromaDB must persist its data to a local directory.
3. Embeddings must be generated locally.
4. No cloud embedding provider, cloud LLM, or hosted vector service may be required.
5. Model names, storage paths, chunk parameters, retrieval settings, and service URLs must be configurable.
6. Avoid introducing LangChain or LlamaIndex as mandatory dependencies in the initial implementation. Build the core pipeline using explicit modules so that the developer understands and controls each stage.
7. Third-party libraries may be used where they provide a clear benefit, but unnecessary dependencies must be avoided.

**Offline requirement:** Internet access is permitted during initial installation and model downloads. Once dependencies and models are installed, the application must be able to operate without internet access. It must not silently fall back to cloud services.

# 3. Required Folder Structure

Create the following project structure before implementing the RAG logic.

```text
local-rag/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   ├── router.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       ├── documents.py
│   │       └── chat.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── exceptions.py
│   │   └── constants.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── document.py
│   │   ├── chat.py
│   │   └── retrieval.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── pdf_loader.py
│   │   │   ├── text_loader.py
│   │   │   └── docx_loader.py
│   │   ├── text_cleaner.py
│   │   └── chunker.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── ollama_embeddings.py
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── chroma_store.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   ├── filters.py
│   │   └── reranker.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_client.py
│   │   ├── prompt_builder.py
│   │   └── answer_generator.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingestion_service.py
│   │   └── rag_service.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py
│       └── hashing.py
│
├── scripts/
│   ├── ingest_documents.py
│   ├── reindex_documents.py
│   └── check_environment.py
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_text_cleaner.py
│   │   ├── test_hashing.py
│   │   ├── test_retriever.py
│   │   └── test_prompt_builder.py
│   ├── integration/
│   │   ├── test_ingestion_pipeline.py
│   │   ├── test_vectorstore.py
│   │   └── test_rag_service.py
│   └── evaluation/
│       ├── test_retrieval_quality.py
│       └── test_grounded_answers.py
│
├── data/
│   ├── documents/
│   │   └── .gitkeep
│   ├── chroma_db/
│   │   └── .gitkeep
│   ├── evaluation/
│   │   └── .gitkeep
│   └── cache/  
│       └── .gitkeep
│
├── logs/
│   └── .gitkeep
│
├── docs/
│   ├── architecture.md
│   └── development.md
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── README.md
└── LICENSE
```

### Folder structure rules

- Create all required directories and package initialization files.
- Empty modules may initially contain docstrings and TODOs, but must not contain fake implementations.
- Do not place the entire pipeline inside `main.py`.
- Do not place unrelated functionality inside one large Python file.
- Keep business logic independent of FastAPI routes.
- Keep document parsing independent of vector storage.
- Keep embeddings independent of retrieval and answer generation.
- Do not introduce circular imports.
- The `data/` and `logs/` directories are runtime storage, not source-code packages.
- ChromaDB files, original documents, logs, environment files, virtual environments, and caches must not be committed to Git.
- Store a stable document ID, source metadata, content hash, and chunk metadata so that reindexing and document updates can be managed safely.
- The reranker module may initially expose a documented interface only. Do not add a heavyweight reranking model until the basic retrieval system has been evaluated.

# 4. Module Responsibilities

## 4.1 `app/main.py`

The application entry point.

Requirements:

- Create the FastAPI application.
- Register API routes.
- Configure lifecycle startup and shutdown.
- Initialize required shared services through controlled dependency injection.
- Provide a `/health` endpoint through the health router.
- Avoid implementing ingestion, retrieval, or generation logic directly in this file.

## 4.2 `app/core/`

Responsible for shared application configuration and infrastructure.

### `config.py`

- Read settings from environment variables and `.env`.
- Define document directory, database directory, Ollama URL, model names, chunk settings, and retrieval parameters.
- Validate invalid configuration values at startup.
- Use sensible defaults for local development.
- Keep secrets out of source code.

### `logging.py`

- Configure consistent application logging.
- Log failures, ingestion results, latency, and operational events.
- Never log entire documents, prompts containing private content, or sensitive data by default.

### `exceptions.py`

Define clear application exceptions for unsupported file types, document parsing failures, embedding errors, vector-store errors, and generation errors.

### `constants.py`

Keep shared constants in one place when they genuinely need to be shared.

## 4.3 `app/ingestion/`

Responsible for transforming raw files into clean, traceable chunks.

### Loaders

- `base.py`: Define the common loader interface.
- `pdf_loader.py`: Extract text from PDFs and retain page numbers where available.
- `text_loader.py`: Load plain-text files with safe encoding handling.
- `docx_loader.py`: Extract text from DOCX files.

Each loader must return a consistent internal document representation containing text and metadata.

Unsupported file types must be rejected or explicitly skipped with a useful error.

Scanned PDFs may require OCR, but OCR is not part of the initial release.

### `text_cleaner.py`

- Normalize whitespace.
- Handle empty or malformed extracted text.
- Preserve meaningful paragraph and heading boundaries where possible.
- Avoid transformations that destroy code blocks, tables, or meaningful structure unnecessarily.

### `chunker.py`

Implement configurable chunking.

Requirements:

- Configurable chunk size and overlap.
- Prefer paragraph or sentence boundaries.
- Support a deterministic fallback for unusually long sections.
- Preserve source, page number, and chunk index metadata.
- Handle empty documents and very short documents.
- Avoid duplicate or empty chunks.
- Validate that overlap is smaller than chunk size.

### `pipeline.py`

Coordinate the ingestion stages:

1. Validate the input.
2. Select the correct loader.
3. Extract and clean content.
4. Create chunks.
5. Generate stable document and chunk identifiers.
6. Pass prepared chunks to the embedding and indexing layers.
7. Return an ingestion report containing successful files, skipped files, failed files, and chunk counts.

The pipeline must not contain FastAPI request-handling code.

## 4.4 `app/embeddings/`

### `base.py`

Define a provider-independent embedding interface.

### `ollama_embeddings.py`

- Connect to Ollama running on the local machine.
- Generate embeddings for documents and queries.
- Support batching where appropriate.
- Validate embedding dimensions and response lengths.
- Detect missing models and unavailable services.
- Return actionable errors without silently switching providers.

Document and query embeddings must use the same compatible embedding model and configuration.

## 4.5 `app/vectorstore/`

### `base.py`

Define the operations required from a vector-store implementation, such as adding chunks, searching, deleting, and checking collection status.

### `chroma_store.py`

- Initialize a persistent local ChromaDB collection.
- Store chunk text, embeddings, metadata, and stable IDs.
- Support similarity search and metadata filtering.
- Support safe document-level deletion and replacement.
- Detect collection and embedding configuration mismatches.
- Avoid reinitializing or deleting the collection on every application startup.

Database operations must be idempotent where practical.

## 4.6 `app/retrieval/`

### `retriever.py`

- Embed the user query.
- Search ChromaDB.
- Return ranked chunks with source metadata and distances or scores.
- Support configurable top-k retrieval.
- Handle an empty index and empty queries.
- Avoid returning more chunks than are available.

### `filters.py`

Apply validated metadata filters, including source and document-level filters where supported.

### `reranker.py`

Define an optional reranking interface. The initial system may use vector similarity alone. Implement a local reranker only after baseline evaluation demonstrates that it is needed.

## 4.7 `app/generation/`

### `llm_client.py`

- Communicate with the local Ollama chat model.
- Apply timeouts and handle unavailable services.
- Expose a consistent interface for text generation.
- Never call cloud inference APIs.

### `prompt_builder.py`

Construct prompts from the user question and retrieved chunks.

Requirements:

- Clearly delimit retrieved evidence.
- Treat document content as untrusted input rather than executable instructions.
- Instruct the model not to invent unsupported facts.
- Require source markers for claims that depend on retrieved evidence.
- Support an explicit insufficient-evidence response.

### `answer_generator.py`

- Call the local LLM.
- Generate the answer using retrieved evidence.
- Return answer text and source references.
- Handle model failures and malformed responses.
- Do not fabricate source references or claim that citations have been verified when they have not.

The first version may use prompt-based grounding. A later stage must evaluate answer faithfulness and implement additional evidence checks.

## 4.8 `app/services/`

### `ingestion_service.py`

Expose a high-level interface for adding, updating, and removing documents. Coordinate the ingestion pipeline and vector store.

### `rag_service.py`

Coordinate question answering:

1. Validate the question.
2. Retrieve relevant chunks.
3. Determine whether retrieval produced usable evidence.
4. Build the generation prompt.
5. Generate an answer.
6. Return the answer and source references in a consistent schema.

Keep orchestration logic here rather than duplicating it across API routes.

## 4.9 `app/api/`

### `health.py`

Expose health and readiness checks for the application and its local dependencies.

Distinguish between the API being alive and the complete RAG pipeline being ready.

### `documents.py`

Provide endpoints to ingest, list, inspect, and delete indexed documents.

Requirements:

- Validate file types and upload sizes.
- Avoid arbitrary filesystem path access.
- Prevent unsafe filenames and path traversal.
- Return clear operation status and errors.

### `chat.py`

Provide an endpoint to ask questions against indexed documents.

Return structured JSON containing:

- Answer text.
- Source references.
- Relevant metadata.
- A retrieval status or evidence indicator where appropriate.

### `dependencies.py`

Manage shared service dependencies and their lifecycle.

### `router.py`

Register and group the application's API routes.

# 5. Functional Requirements

## FR-01: Document ingestion

The system shall accept PDF, TXT, and DOCX documents.

It shall extract text, clean content, chunk the text, generate embeddings, and persist the resulting chunks and metadata.

Empty or unsupported files shall not be silently indexed as valid documents.

## FR-02: Persistent storage

The vector index shall survive application restarts.

Reindexing one document shall not accidentally delete unrelated documents.

The system shall support safe deletion and reingestion of individual documents.

## FR-03: Semantic retrieval

The system shall retrieve relevant document chunks based on semantic similarity.

The number of returned chunks shall be configurable.

Retrieved results shall retain sufficient metadata to identify their original sources.

## FR-04: Grounded answer generation

The system shall construct answers using the retrieved context and the local LLM.

When evidence is missing or inadequate, it shall be able to return an insufficient-evidence response rather than confidently inventing an answer.

## FR-05: Source citations

The system shall return source filenames and chunk references.

Where page-level information exists, citations should include the page number.

Citation references must be generated from actual retrieved records.

## FR-06: Local-only operation

The system shall run using local models, local files, and a local vector database.

No cloud fallback is permitted.

The system shall fail clearly when a required local model or service is unavailable.

## FR-07: API access

The application shall expose documented FastAPI endpoints for health checks, document management, and question answering.

## FR-08: Observability

The system shall record useful diagnostic information, including operation status, elapsed time, and failures.

Sensitive document contents shall not be written to logs by default.

# 6. API Requirements

The following endpoints define the initial interface.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Basic application health |
| GET | `/health/ready` | Check whether required dependencies are ready |
| POST | `/api/v1/documents` | Upload and index a document |
| GET | `/api/v1/documents` | List indexed documents |
| GET | `/api/v1/documents/{document_id}` | Retrieve document metadata |
| DELETE | `/api/v1/documents/{document_id}` | Remove a document and its indexed chunks |
| POST | `/api/v1/chat` | Ask a question against indexed documents |

Use Pydantic request and response models.

Define a consistent error format. Avoid returning stack traces or internal filesystem paths to API clients.

# 7. Configuration Requirements

Create `.env.example` with configuration similar to the following:

```dotenv
APP_NAME=LocalRAG
APP_ENV=development
APP_HOST=127.0.0.1
APP_PORT=8000

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=llama3.2:3b

DOCUMENTS_DIR=data/documents
CHROMA_DB_DIR=data/chroma_db
CHROMA_COLLECTION_NAME=local_documents

CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_TOP_K=4

LOG_LEVEL=INFO
```

These are initial defaults, not final performance recommendations. All settings must be validated and documented.

Do not commit the actual `.env` file.

# 8. Testing Requirements

Tests are mandatory. Do not consider a module complete merely because its code executes once.

## Unit tests

Test:

- Text cleaning.
- Chunk boundaries and overlap.
- Empty document handling.
- File extension validation.
- Stable hashing and identifiers.
- Prompt construction.
- Retrieval result formatting.
- Configuration validation.

## Integration tests

Test:

- Loading sample documents.
- Ingesting chunks into a temporary ChromaDB collection.
- Persistence across client restarts.
- Retrieving a relevant chunk.
- Replacing a document without leaving stale chunks.
- Deleting a document without affecting others.
- Running the RAG service using local model dependencies or controlled test doubles.

Tests must not modify the user's actual document collection.

## Evaluation tests

Create a small evaluation dataset containing questions, expected source documents, and expected supporting facts.

Measure:

- Retrieval Recall@k.
- Mean Reciprocal Rank where applicable.
- Whether relevant evidence appears in retrieved chunks.
- Whether answers are supported by the retrieved evidence.
- Whether unanswerable questions produce appropriate fallback behavior.

Do not claim that a RAG system is accurate merely because its responses sound convincing.

# 9. Non-functional Requirements

## Performance

- Use embedding batches where practical.
- Avoid repeatedly embedding unchanged documents.
- Keep startup and indexing operations separate where appropriate.
- Measure retrieval and generation latency independently.
- Do not promise specific latency targets before benchmarking on the target hardware.

## Reliability

- Handle Ollama connection failures and missing models gracefully.
- Avoid partial or inconsistent indexing when an ingestion operation fails.
- Make reindexing repeatable.
- Use temporary test databases.
- Provide clear error messages and diagnostic logs.

## Security

- Bind the development API to localhost by default.
- Validate uploaded files and enforce size limits.
- Prevent arbitrary path access.
- Treat retrieved content as untrusted input.
- Do not expose unrestricted filesystem operations through the API.
- Do not expose the local service to a public network by default.

## Maintainability

- Use type hints and docstrings for public interfaces.
- Keep modules focused on one responsibility.
- Use dependency injection where it improves testability.
- Keep business logic independent of HTTP.
- Avoid circular imports, global mutable application state, and unnecessary abstractions.
- Pin or constrain dependencies and document the supported environment.

# 10. Implementation Roadmap

The coding AI must follow these phases in order.

## Phase 1 — Project foundation

- Create the complete directory structure.
- Create `pyproject.toml`, dependency files, `.gitignore`, `.env.example`, and README.
- Configure Python package imports, settings, and logging.
- Create a working FastAPI application with health endpoints.
- Add the initial test setup.
- Verify the application starts successfully.

**Exit criteria:** Clean project structure, validated configuration, passing initial tests, and a working health endpoint.

## Phase 2 — Document ingestion

- Implement loader interfaces and PDF/TXT/DOCX loaders.
- Implement text cleaning and chunking.
- Add metadata and stable IDs.
- Write unit and integration tests.
- Produce an ingestion report.

**Exit criteria:** Sample documents can be loaded and chunked with reliable metadata.

## Phase 3 — Embeddings and vector storage

- Implement the embedding interface.
- Integrate the Ollama embedding model.
- Implement persistent ChromaDB storage.
- Add indexing, replacement, deletion, and metadata operations.
- Test persistence and failure handling.

**Exit criteria:** Documents can be indexed and searched without losing data across restarts.

## Phase 4 — Retrieval engine

- Implement semantic search.
- Add top-k configuration and metadata filters.
- Return ranked chunks with source information.
- Create the initial retrieval evaluation dataset.
- Establish baseline retrieval metrics.

**Exit criteria:** Queries retrieve relevant evidence, and retrieval quality is measured.

## Phase 5 — Grounded generation

- Implement the Ollama chat client.
- Build the prompt construction module.
- Implement the answer generator.
- Connect retrieval and generation through `rag_service.py`.
- Add citations and insufficient-evidence handling.
- Test answer grounding.

**Exit criteria:** The complete document-question-answering pipeline works locally and returns source references.

## Phase 6 — API and production hardening

- Complete document and chat endpoints.
- Add request validation, error handling, logging, and timeouts.
- Add incremental indexing and safe document replacement.
- Add regression and evaluation tests.
- Document setup, offline usage, backups, and troubleshooting.

**Exit criteria:** The service is testable, maintainable, and reliable enough for local deployment.

## Phase 7 — Future extensions

Only after the core pipeline passes its tests, consider:

- Hybrid keyword and vector retrieval.
- Local reranking models.
- Better PDF layout and table extraction.
- Conversational memory.
- Streaming responses.
- Frontend integration.
- n8n automation.
- More advanced retrieval evaluation.

Do not implement these prematurely.

# 11. Acceptance Criteria

The initial release is considered complete when:

1. The project follows the specified modular folder structure.
2. Each major responsibility has its own module.
3. The application can ingest PDF, TXT, and DOCX documents.
4. Document chunks and embeddings persist locally.
5. A relevant question retrieves supporting document passages.
6. The local LLM generates an answer grounded in retrieved evidence.
7. Responses include source references that correspond to actual indexed chunks.
8. The system handles missing evidence, invalid input, and unavailable dependencies.
9. The API works without cloud AI services.
10. The system operates offline after initial setup and model downloads.
11. Automated unit and integration tests pass.
12. Retrieval and answer-grounding evaluation can be repeated.
13. Setup and operational instructions are documented.

# 12. Instructions for the Coding AI

The following development rules are mandatory.

1. **Start with the project structure.** Create the directories and configuration files before implementing application logic.
2. **Do not generate the entire application in one giant file.** Every module must live in its specified location.
3. **Work phase by phase.** Complete the current phase, run its tests, and report the results before proceeding.
4. **Explain each module.** Briefly describe its responsibility, dependencies, and interfaces.
5. **Write actual implementations.** Do not replace required functionality with placeholder methods that return fake successful results.
6. **Preserve existing work.** Before modifying a module, inspect its current implementation and tests.
7. **Do not silently change the technology stack.** Any proposed substitution must include a clear reason and receive approval.
8. **Use local inference exclusively.** No OpenAI API, cloud embeddings, hosted vector database, or automatic cloud fallback.
9. **Validate integration boundaries.** Verify imports, model availability, embedding dimensions, vector-store operations, and request/response schemas.
10. **Run tests after each phase.** Report which tests passed, which failed, and what remains incomplete.
11. **Avoid unnecessary abstractions.** Implement only what the current phase needs while preserving clear extension points.
12. **Never claim a feature works without verifying it.** If the environment prevents execution, state that limitation explicitly.
13. **Keep documentation updated.** Update README and relevant architecture documents when interfaces or setup steps change.
14. **Do not proceed blindly after a failure.** Diagnose the failure, correct the implementation, and rerun the relevant tests before moving forward.

## Required output for each implementation phase

For every phase, provide:

- Files created or modified.
- The purpose of each file.
- Complete code for the files being implemented.
- Installation or configuration commands when necessary.
- Tests executed and their actual results.
- Known limitations and the next phase's prerequisites.

**Begin with Phase 1 only.** Create the project structure and foundational configuration, verify the application starts, and establish the test environment. Do not implement ingestion, embeddings, retrieval, or generation until Phase 1 is complete and verified.

---

**End of PRD**