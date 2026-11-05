"""Tests for RAG orchestration service."""

import pytest

from app.services.rag_service import RAGService
from app.schemas.chat import ChatRequest, ChatResponse
from app.generation.answer_generator import INSUFFICIENT_ANSWER


class FakeRetriever:
    def __init__(self, contexts=None, error=None):
        self.contexts = contexts if contexts is not None else []
        self.error = error
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append({"query": query, "top_k": top_k})

        if self.error:
            raise self.error

        return self.contexts


class FakeAnswerGenerator:
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


class FakeGroundingVerifier:
    def __init__(self, result=None):
        self.result = result or {
            "supported": True,
            "claims": [
                {
                    "claim": "Default answer",
                    "verdict": "SUPPORTED",
                    "evidence_valid": True,
                }
            ],
            "reason": "All factual claims passed verification.",
        }
        self.calls = []

    def verify(self, answer, contexts, question=""):
        self.calls.append({
            "answer": answer,
            "contexts": contexts,
        })
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
    verifier = FakeGroundingVerifier()

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
        grounding_verifier=verifier,
    )

    return service, retriever, generator, verifier


@pytest.mark.integration
def test_successful_question_answer_flow(fake_components):
    service, retriever, generator, verifier = fake_components

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

    assert len(verifier.calls) == 1
    assert verifier.calls[0]["answer"] == "Default answer [1]"


@pytest.mark.integration
def test_question_whitespace_is_stripped(fake_components):
    service, retriever, generator, verifier = fake_components

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
    service, retriever, generator, verifier = fake_components

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
    verifier = FakeGroundingVerifier()

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
        grounding_verifier=verifier,
    )

    response = service.answer(
        ChatRequest(question="What is Project Orion?")
    )

    assert response.answer == INSUFFICIENT_ANSWER
    assert response.sources == []
    assert response.grounded is False
    assert generator.calls == []
    assert verifier.calls == []


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

    verifier = FakeGroundingVerifier()

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
        grounding_verifier=verifier,
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

    verifier = FakeGroundingVerifier()

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
        grounding_verifier=verifier,
    )

    response = service.answer(
        ChatRequest(question="Test invalid citations.")
    )

    assert response.sources == []
    assert response.grounded is False
    assert verifier.calls == []


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

    verifier = FakeGroundingVerifier()

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
        grounding_verifier=verifier,
    )

    response = service.answer(
        ChatRequest(question="Test grounding.")
    )

    assert len(response.sources) == 1
    assert response.grounded is False
    assert verifier.calls == []


@pytest.mark.integration
def test_unsupported_answer_is_rejected(sample_contexts):
    retriever = FakeRetriever(contexts=sample_contexts)
    generator = FakeAnswerGenerator()

    verifier = FakeGroundingVerifier(
        result={
            "supported": False,
            "claims": [
                {
                    "claim": "Unsupported factual claim",
                    "verdict": "NOT_SUPPORTED",
                    "evidence_valid": False,
                }
            ],
            "reason": "Claim is unsupported.",
        }
    )

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
        grounding_verifier=verifier,
    )

    response = service.answer(
        ChatRequest(question="Test unsupported answer.")
    )

    assert response.answer == INSUFFICIENT_ANSWER
    assert response.sources == []
    assert response.grounded is False
    assert len(verifier.calls) == 1


@pytest.mark.integration
def test_verifier_exception_result_fails_closed(sample_contexts):
    retriever = FakeRetriever(contexts=sample_contexts)
    generator = FakeAnswerGenerator()

    verifier = FakeGroundingVerifier(
        result={
            "supported": False,
            "claims": [],
            "reason": "Verification failed: Ollama unavailable",
        }
    )

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
        grounding_verifier=verifier,
    )

    response = service.answer(
        ChatRequest(question="Test verifier failure.")
    )

    assert response.answer == INSUFFICIENT_ANSWER
    assert response.sources == []
    assert response.grounded is False


@pytest.mark.integration
def test_verifier_malformed_result_fails_closed(sample_contexts):
    retriever = FakeRetriever(contexts=sample_contexts)
    generator = FakeAnswerGenerator()

    verifier = FakeGroundingVerifier(result=None)
    verifier.result = None

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
        grounding_verifier=verifier,
    )

    response = service.answer(
        ChatRequest(question="Test malformed verification.")
    )

    assert response.answer == INSUFFICIENT_ANSWER
    assert response.sources == []
    assert response.grounded is False


@pytest.mark.integration
def test_whitespace_only_question_is_rejected():
    service = RAGService(
        retriever=FakeRetriever(),
        answer_generator=FakeAnswerGenerator(),
        grounding_verifier=FakeGroundingVerifier(),
    )

    with pytest.raises(ValueError, match="Question cannot be empty"):
        service.answer(ChatRequest(question="   "))


@pytest.mark.integration
def test_retriever_errors_are_propagated():
    retriever = FakeRetriever(
        error=RuntimeError("Retrieval service unavailable")
    )

    service = RAGService(
        retriever=retriever,
        answer_generator=FakeAnswerGenerator(),
        grounding_verifier=FakeGroundingVerifier(),
    )

    with pytest.raises(
        RuntimeError,
        match="Retrieval service unavailable",
    ):
        service.answer(ChatRequest(question="Test retrieval error."))


@pytest.mark.integration
def test_generator_errors_are_propagated(sample_contexts):
    retriever = FakeRetriever(contexts=sample_contexts)

    generator = FakeAnswerGenerator(
        error=RuntimeError("Generation service unavailable")
    )

    service = RAGService(
        retriever=retriever,
        answer_generator=generator,
        grounding_verifier=FakeGroundingVerifier(),
    )

    with pytest.raises(
        RuntimeError,
        match="Generation service unavailable",
    ):
        service.answer(ChatRequest(question="Test generation error."))

