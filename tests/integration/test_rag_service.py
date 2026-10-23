"""Integration tests for RAG orchestration service."""

import pytest
import app.services.rag_service as rag_service_module


@pytest.mark.integration
def test_rag_service_module_exists() -> None:
    """Validate RAG service module exists and is importable."""
    assert rag_service_module is not None
