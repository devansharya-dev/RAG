"""Unit tests for text chunker module."""

import pytest
import app.ingestion.chunker as chunker_module


@pytest.mark.unit
def test_chunker_module_exists() -> None:
    """Validate chunker module exists and is importable."""
    assert chunker_module is not None
