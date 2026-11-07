
"""Tests for the RAG retriever and hybrid search."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.retrieval.retriever import Retriever


@pytest.fixture
def fake_store():
    store = Mock()
    store.get_stats.return_value = {"total_chunks": 5}

    store.similarity_search.return_value = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "content": "Project Nebula uses 48 processing nodes.",
            "metadata": {"filename": "rag_test.pdf"},
            "similarity": 0.89,
        }
    ]

    store.hybrid_search.return_value = [
        {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "content": "Project Nebula uses 48 processing nodes.",
            "metadata": {"filename": "rag_test.pdf"},
            "similarity": 0.03,
        }
    ]

    return store


@pytest.mark.unit
def test_retrieve_returns_relevant_chunks(fake_store, monkeypatch):
    mock_embed = Mock(
        return_value=SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])
    )
    monkeypatch.setattr(
        "app.retrieval.retriever.ollama.embed",
        mock_embed,
    )

    retriever = Retriever(vector_store=fake_store, top_k=4)
    results = retriever.retrieve("How many processing nodes?")

    assert len(results) == 1
    assert results[0]["content"] == (
        "Project Nebula uses 48 processing nodes."
    )

    mock_embed.assert_called_once()
    fake_store.hybrid_search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        query_text="How many processing nodes?",
        top_k=4,
        filters=None,
    )
    fake_store.similarity_search.assert_not_called()


@pytest.mark.unit
def test_retrieve_strips_query_whitespace(fake_store, monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.retriever.ollama.embed",
        Mock(return_value=SimpleNamespace(embeddings=[[0.1, 0.2]])),
    )

    retriever = Retriever(vector_store=fake_store)
    retriever.retrieve("  test query  ")

    assert fake_store.hybrid_search.call_args.kwargs["query_text"] == (
        "test query"
    )


@pytest.mark.unit
def test_retrieve_uses_custom_top_k(fake_store, monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.retriever.ollama.embed",
        Mock(return_value=SimpleNamespace(embeddings=[[0.1]])),
    )

    retriever = Retriever(vector_store=fake_store, top_k=4)
    retriever.retrieve("test query", top_k=2)

    assert fake_store.hybrid_search.call_args.kwargs["top_k"] == 2


@pytest.mark.unit
def test_retrieve_passes_metadata_filters(fake_store, monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.retriever.ollama.embed",
        Mock(return_value=SimpleNamespace(embeddings=[[0.1]])),
    )

    retriever = Retriever(vector_store=fake_store)
    filters = {"filename": "rag_test.pdf"}

    retriever.retrieve("test query", filters=filters)

    assert fake_store.hybrid_search.call_args.kwargs["filters"] == filters


@pytest.mark.unit
def test_retrieve_returns_empty_when_store_has_no_chunks(
    fake_store, monkeypatch
):
    fake_store.get_stats.return_value = {"total_chunks": 0}
    mock_embed = Mock()

    monkeypatch.setattr(
        "app.retrieval.retriever.ollama.embed",
        mock_embed,
    )

    retriever = Retriever(vector_store=fake_store)
    results = retriever.retrieve("test query")

    assert results == []
    mock_embed.assert_not_called()
    fake_store.hybrid_search.assert_not_called()


@pytest.mark.unit
def test_retrieve_rejects_empty_query(fake_store):
    retriever = Retriever(vector_store=fake_store)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve("   ")

    fake_store.get_stats.assert_not_called()


@pytest.mark.unit
def test_retrieve_rejects_empty_embedding(fake_store, monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.retriever.ollama.embed",
        Mock(return_value=SimpleNamespace(embeddings=[])),
    )

    retriever = Retriever(vector_store=fake_store)

    with pytest.raises(RuntimeError, match="no query embedding"):
        retriever.retrieve("test query")


@pytest.mark.unit
def test_retrieve_falls_back_to_vector_search(monkeypatch):
    class VectorOnlyStore:
        """A store that supports vector search but not hybrid search."""

        def __init__(self):
            self.stats = {"total_chunks": 5}
            self.search_called = False

        def get_stats(self):
            return self.stats

        def similarity_search(
            self, query_embedding, top_k, filters=None
        ):
            self.search_called = True
            return [{"content": "Vector result"}]

    vector_store = VectorOnlyStore()

    monkeypatch.setattr(
        "app.retrieval.retriever.ollama.embed",
        Mock(return_value=SimpleNamespace(embeddings=[[0.1, 0.2]])),
    )

    retriever = Retriever(vector_store=vector_store)
    result = retriever.retrieve("test query")

    assert result == [{"content": "Vector result"}]
    assert vector_store.search_called is True


@pytest.mark.unit
def test_retrieve_with_reranking_enabled(fake_store, monkeypatch):
    settings = SimpleNamespace(
        OLLAMA_EMBED_MODEL="nomic-embed-text",
        RETRIEVAL_TOP_K=4,
        RERANKER_ENABLED=True,
        RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2",
        RERANKER_CANDIDATES=20,
    )

    monkeypatch.setattr(
        "app.retrieval.retriever.get_settings",
        lambda: settings,
    )

    candidates = [
        {
            "chunk_id": f"chunk-{i}",
            "content": f"Document content {i}",
            "metadata": {},
            "similarity": 0.01 * i,
        }
        for i in range(5)
    ]

    fake_store.hybrid_search.return_value = candidates

    mock_reranker = Mock()
    mock_reranker.rerank.return_value = candidates[:4]

    monkeypatch.setattr(
        "app.retrieval.retriever.CrossEncoderReranker",
        Mock(return_value=mock_reranker),
    )

    monkeypatch.setattr(
        "app.retrieval.retriever.ollama.embed",
        Mock(return_value=SimpleNamespace(embeddings=[[0.1, 0.2]])),
    )

    retriever = Retriever(vector_store=fake_store)
    results = retriever.retrieve("test query")

    assert len(results) == 4

    fake_store.hybrid_search.assert_called_once_with(
        query_embedding=[0.1, 0.2],
        query_text="test query",
        top_k=20,
        filters=None,
    )

    mock_reranker.rerank.assert_called_once_with(
        query="test query",
        documents=candidates,
        top_k=4,
    )


@pytest.mark.unit
def test_retrieve_does_not_initialize_reranker_when_disabled(
    fake_store, monkeypatch
):
    settings = SimpleNamespace(
        OLLAMA_EMBED_MODEL="nomic-embed-text",
        RETRIEVAL_TOP_K=4,
        RERANKER_ENABLED=False,
        RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2",
        RERANKER_CANDIDATES=20,
    )

    monkeypatch.setattr(
        "app.retrieval.retriever.get_settings",
        lambda: settings,
    )

    mock_reranker_class = Mock()

    monkeypatch.setattr(
        "app.retrieval.retriever.CrossEncoderReranker",
        mock_reranker_class,
    )

    monkeypatch.setattr(
        "app.retrieval.retriever.ollama.embed",
        Mock(return_value=SimpleNamespace(embeddings=[[0.1, 0.2]])),
    )

    retriever = Retriever(vector_store=fake_store)
    results = retriever.retrieve("test query")

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-001"

    fake_store.hybrid_search.assert_called_once_with(
        query_embedding=[0.1, 0.2],
        query_text="test query",
        top_k=4,
        filters=None,
    )

    mock_reranker_class.assert_not_called()
