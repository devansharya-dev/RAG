"""Integration tests for Supabase vector store."""

import pytest

import app.vectorstore.supabase_store as supabase_store_module
from app.vectorstore.base import BaseVectorStore
from app.vectorstore.supabase_store import SupabaseVectorStore


@pytest.mark.integration
def test_vectorstore_module_exists() -> None:
    """Validate Supabase vectorstore module exists and implements BaseVectorStore."""
    assert supabase_store_module is not None
    assert issubclass(SupabaseVectorStore, BaseVectorStore)
