# LocalRAG Architecture Documentation

## Architectural Overview

LocalRAG is structured around clear separation of concerns, strict boundary isolation, and modular dependency injection. The design forbids monolithic script patterns and decouples ingestion, embedding, vector storage, retrieval, and LLM answer generation.

```mermaid
flowchart TD
    subgraph Ingestion Pipeline
        RawDocs[Raw Files: PDF / TXT / DOCX] --> Loaders[Document Loaders]
        Loaders --> Cleaner[Text Cleaner]
        Cleaner --> Chunker[Deterministic Chunker]
        Chunker --> HashGen[Stable Hash & ID Generator]
    end

    subgraph Vector Storage & Embeddings
        HashGen --> Embeddings[Local Ollama Embeddings\nnomic-embed-text]
        Embeddings --> ChromaStore[Local Persistent ChromaDB\ndata/chroma_db]
    end

    subgraph Query & Retrieval
        UserQuery[User Question] --> QueryEmbed[Embed Query]
        QueryEmbed --> Retriever[Retriever Engine]
        ChromaStore --> Retriever
        Retriever --> Filters[Metadata Filters]
        Filters --> Reranker[Optional Reranker]
    end

    subgraph Grounded Generation
        Reranker --> PromptBuilder[Prompt Builder\nTreat context as untrusted data]
        UserQuery --> PromptBuilder
        PromptBuilder --> LLMClient[Local Ollama Chat LLM\nllama3.2:3b]
        LLMClient --> AnswerGen[Answer Generator\nCitations & Grounding Verification]
    end

    subgraph API Layer
        API[FastAPI Endpoints\n/health, /documents, /chat] --> Services[IngestionService & RAGService]
        Services --> IngestionPipeline
        Services --> GroundedGeneration
    end
```

## Core Architectural Principles

1. **100% Offline & Private**: All inference and storage are performed locally. Cloud APIs and fallback services are forbidden.
2. **Explicit Interfaces**: Each subsystem exposes an abstract interface (`base.py`) decoupled from specific vendor implementations.
3. **Idempotent Ingestion**: Documents generate deterministic content hashes. Duplicate ingestion is prevented, and updates safely supersede previous chunks.
4. **Untrusted Evidence Boundary**: Retrieved document text is treated as untrusted data in prompts to mitigate indirect prompt injection.
5. **Traceable Citations**: Every generated statement must be traceable to specific chunk IDs and document sources.
6. **No Heavy Framework Coupling**: The core pipeline relies on explicit modules rather than black-box framework abstractions (e.g. LangChain or LlamaIndex).
