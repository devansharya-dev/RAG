"""Integration tests for document ingestion pipeline."""

import pytest
import app.ingestion.pipeline as pipeline_module


@pytest.mark.integration
def test_ingestion_pipeline_module_exists() -> None:
    """Validate ingestion pipeline module exists and is importable."""
    assert pipeline_module is not None
