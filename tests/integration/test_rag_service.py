
"""Tests for RAG orchestration service."""

import pytest

from app.services.rag_service import RAGService
from app.schemas.chat import ChatRequest, ChatResponse


class FakeRetriever:
    """Fake retriever for testing RAG orchestration."""

    def __init__(self, contexts=None, error=None):
        self.contexts = contexts if contexts is not None else []
        self.error = error
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append({
            "query": query,
            "top_k": top_k,
        })

        if self.error:
            raise self.error

        return self.contexts


class FakeAnswerGenerator:
    """Fake answer generator for testing."""

    def __init__(self, result=None, error=None):
        self.result = result or {
            "answer": "Default answer [1]",
            "sources": [{"passage_number": 1}],
            "grounded": True,
        }
        self.error = error
        self.calls = []

    def generate(self, question, contexts):
        self.calls.append({
            "question": question,
            "contexts": contexts,
        })

        if self.error:
            raise self.error

        return self.result


@pytest.fixture
def sample_contexts():
    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "content": "Project Orion uses FastAPI.",
            "metadata": {
                "filename": "orion.txt",
                "file_type": ".txt",
            },
            "similarity": 0.92,
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-2",
            "content": "Project Nebula uses PostgreSQL.",
            "metadata": {
                "filename": "nebula.txt",
                "file_type": ".txt",
            },
            "similarity": 0.87,
        },
    ]


@pytest.fixture
def fake_components(sample_contexts):
    retriever = FakeRetriever(contexts=sample_contexts)
    generator = FakeAnswerGenerator()

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
    )

    return service, retriever, generator


@pytest.mark.integration
def test_successful_question_answer_flow(fake_components):
    service, retriever, generator = fake_components

    request = ChatRequest(
        question="Which framework does Project Orion use?"
    )

    response = service.answer(request)

    assert isinstance(response, ChatResponse)
    assert response.question == request.question
    assert response.answer == "Default answer [1]"
    assert response.grounded is True

    assert len(response.sources) == 1
    assert response.sources[0].document_id == "doc-1"
    assert response.sources[0].chunk_id == "chunk-1"
    assert response.sources[0].filename == "orion.txt"

    assert retriever.calls[0]["query"] == request.question
    assert retriever.calls[0]["top_k"] == 4

    assert generator.calls[0]["question"] == request.question
    assert generator.calls[0]["contexts"] == retriever.contexts


@pytest.mark.integration
def test_question_whitespace_is_stripped(fake_components):
    service, retriever, generator = fake_components

    request = ChatRequest(
        question="   What is the backend framework?   "
    )

    response = service.answer(request)

    assert response.question == "What is the backend framework?"
    assert retriever.calls[0]["query"] == (
        "What is the backend framework?"
    )
    assert generator.calls[0]["question"] == (
        "What is the backend framework?"
    )


@pytest.mark.integration
def test_custom_top_k_is_passed_to_retriever(fake_components):
    service, retriever, generator = fake_components

    request = ChatRequest(
        question="Explain Project Orion.",
        top_k=10,
    )

    service.answer(request)

    assert retriever.calls[0]["top_k"] == 10


@pytest.mark.integration
def test_empty_retrieval_returns_fallback_answer():
    retriever = FakeRetriever(contexts=[])
    generator = FakeAnswerGenerator()

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
    )

    request = ChatRequest(
        question="What is Project Orion?"
    )

    response = service.answer(request)

    assert response.answer == (
        "The provided documents do not contain enough "
        "information to answer this."
    )

    assert response.sources == []
    assert response.grounded is False

    # Generator must not run without retrieved evidence.
    assert generator.calls == []


@pytest.mark.integration
def test_source_citations_are_mapped_correctly(sample_contexts):
    retriever = FakeRetriever(contexts=sample_contexts)

    generator = FakeAnswerGenerator(
        result={
            "answer": "The projects use different technologies [1] [2].",
            "sources": [
                {"passage_number": 1},
                {"passage_number": 2},
            ],
            "grounded": True,
        }
    )

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
    )

    response = service.answer(
        ChatRequest(question="Compare the projects.")
    )

    assert len(response.sources) == 2

    assert response.sources[0].document_id == "doc-1"
    assert response.sources[0].filename == "orion.txt"

    assert response.sources[1].document_id == "doc-2"
    assert response.sources[1].filename == "nebula.txt"

    assert response.grounded is True


@pytest.mark.integration
def test_invalid_passage_numbers_are_ignored(sample_contexts):
    retriever = FakeRetriever(contexts=sample_contexts)

    generator = FakeAnswerGenerator(
        result={
            "answer": "Some answer.",
            "sources": [
                {"passage_number": 0},
                {"passage_number": 99},
                {"passage_number": "1"},
                {"passage_number": None},
            ],
            "grounded": True,
        }
    )

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
    )

    response = service.answer(
        ChatRequest(question="Test invalid citations.")
    )

    assert response.sources == []
    assert response.grounded is False


@pytest.mark.integration
def test_generator_grounded_false_is_preserved(sample_contexts):
    retriever = FakeRetriever(contexts=sample_contexts)

    generator = FakeAnswerGenerator(
        result={
            "answer": "Uncertain answer.",
            "sources": [{"passage_number": 1}],
            "grounded": False,
        }
    )

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
    )

    response = service.answer(
        ChatRequest(question="Test grounding.")
    )

    assert len(response.sources) == 1
    assert response.grounded is False


@pytest.mark.integration
def test_whitespace_only_question_is_rejected():
    service = RAGService(
        retriever=FakeRetriever(),
        answer_generator=FakeAnswerGenerator(),
    )

    request = ChatRequest(question="   ")

    with pytest.raises(ValueError, match="Question cannot be empty"):
        service.answer(request)


@pytest.mark.integration
def test_retriever_errors_are_propagated():
    retriever = FakeRetriever(
        error=RuntimeError("Retrieval service unavailable")
    )

    service = RAGService(
        retriever=retriever,
        answer_generator=FakeAnswerGenerator(),
    )

    request = ChatRequest(question="Test retrieval error.")

    with pytest.raises(
        RuntimeError,
        match="Retrieval service unavailable",
    ):
        service.answer(request)


@pytest.mark.integration
def test_generator_errors_are_propagated(sample_contexts):
    retriever = FakeRetriever(contexts=sample_contexts)

    generator = FakeAnswerGenerator(
        error=RuntimeError("Generation service unavailable")
    )

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
    )

    request = ChatRequest(question="Test generation error.")

    with pytest.raises(
        RuntimeError,
        match="Generation service unavailable",
    ):
        service.answer(request)
