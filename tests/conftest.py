"""Pytest configuration, shared fixtures, and environment hooks."""

import pytest


@pytest.fixture
def sample_text() -> str:
    """Fixture providing a standard multi-paragraph test document."""
    return (
        "LocalRAG is an offline Retrieval-Augmented Generation system.\n\n"
        "It runs completely locally without cloud API dependencies.\n\n"
        "Data privacy is preserved across ingestion, vector storage, and generation."
    )
