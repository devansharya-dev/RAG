"""Unit tests for text cleaner module."""

import pytest
import app.ingestion.text_cleaner as cleaner_module


@pytest.mark.unit
def test_cleaner_module_exists() -> None:
    """Validate cleaner module exists and is importable."""
    assert cleaner_module is not None
