


from unittest.mock import Mock

import pytest

from app.retrieval.reranker import CrossEncoderReranker


@pytest.fixture
def documents():
    return [
        {
            "chunk_id": "chunk-001",
            "content": "Redis is an in-memory database.",
            "similarity": 0.05,
        },
        {
            "chunk_id": "chunk-002",
            "content": "FastAPI builds Python REST APIs.",
            "similarity": 0.03,
        },
        {
            "chunk_id": "chunk-003",
            "content": "Python is a programming language.",
            "similarity": 0.04,
        },
    ]


@pytest.fixture
def mock_model(monkeypatch):
    model = Mock()
    model.predict.return_value = [1.0, 10.0, 3.0]

    monkeypatch.setattr(
        "app.retrieval.reranker.CrossEncoder",
        Mock(return_value=model),
    )

    return model


@pytest.mark.unit
def test_reranker_sorts_by_relevance(documents, mock_model):
    reranker = CrossEncoderReranker(device="cpu")

    results = reranker.rerank(
        query="Which framework builds Python REST APIs?",
        documents=documents,
    )

    assert [item["chunk_id"] for item in results] == [
        "chunk-002",
        "chunk-003",
        "chunk-001",
    ]


@pytest.mark.unit
def test_reranker_preserves_original_similarity(documents, mock_model):
    reranker = CrossEncoderReranker(device="cpu")

    results = reranker.rerank(
        query="Python REST APIs",
        documents=documents,
    )

    original_scores = {
        item["chunk_id"]: item["similarity"]
        for item in documents
    }

    for result in results:
        assert result["similarity"] == original_scores[result["chunk_id"]]
        assert "reranker_score" in result


@pytest.mark.unit
def test_reranker_respects_top_k(documents, mock_model):
    reranker = CrossEncoderReranker(device="cpu")

    results = reranker.rerank(
        query="Python REST APIs",
        documents=documents,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-002"


@pytest.mark.unit
def test_reranker_returns_empty_for_no_documents(mock_model):
    reranker = CrossEncoderReranker(device="cpu")

    assert reranker.rerank("test query", []) == []
    mock_model.predict.assert_not_called()


@pytest.mark.unit
def test_reranker_rejects_empty_query(documents):
    reranker = CrossEncoderReranker(device="cpu")

    with pytest.raises(ValueError, match="Query cannot be empty"):
        reranker.rerank("   ", documents)


@pytest.mark.unit
def test_reranker_rejects_invalid_top_k(documents):
    reranker = CrossEncoderReranker(device="cpu")

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        reranker.rerank("test query", documents, top_k=0)

