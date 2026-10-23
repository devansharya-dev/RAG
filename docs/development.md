# LocalRAG Development Guide

## Environment Setup

### 1. Python Environment
Ensure Python 3.10+ is installed:
```bash
python --version
```

Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

### 2. Local Ollama Models
Download the required models using the Ollama CLI:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Verify Ollama is reachable:
```bash
curl http://127.0.0.1:11434/api/tags
```

## Coding Standards

- **Formatting & Linting**: Ruff is the standard linter and formatter.
  ```bash
  ruff check .
  ruff format .
  ```
- **Type Annotations**: All public function and method signatures must include Python type hints.
- **Documentation**: All modules, classes, and public functions must feature descriptive docstrings.
- **Async Safety**: Asynchronous endpoints and clients must avoid blocking operations on the main event loop.

## Testing Strategy

- **Unit Tests (`tests/unit/`)**: Fast, deterministic, isolated tests covering text cleaners, chunkers, hashing, prompt construction, and config parsing.
- **Integration Tests (`tests/integration/`)**: Verification of file loaders, local ChromaDB persistence, and mock Ollama service pipelines.
- **Evaluation Tests (`tests/evaluation/`)**: Retrieval quality tests (Recall@k, MRR) and grounding checks on evaluation datasets.

Run the test suite:
```bash
pytest
```
