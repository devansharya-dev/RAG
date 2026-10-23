"""Unit tests for retriever module."""

import pytest
import app.retrieval.retriever as retriever_module


@pytest.mark.unit
def test_retriever_module_exists() -> None:
    """Validate retriever module exists and is importable."""
    assert retriever_module is not None
