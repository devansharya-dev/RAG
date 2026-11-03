
"""Behavioral tests for answer grounding and citation validation."""

import pytest

from app.generation.answer_generator import (
    AnswerGenerator,
    INSUFFICIENT_ANSWER,
)


class FakeLLMClient:
    """Mock LLM client that returns predefined responses."""

    def __init__(self, response: str):
        self.response = response

    def generate(self, messages, temperature=0.2, max_tokens=512):
        return self.response


@pytest.fixture
def sample_contexts():
    return [
        {
            "document_id": "doc-001",
            "chunk_id": "chunk-001",
            "content": "Project Nebula uses 48 processing nodes.",
            "similarity": 0.85,
            "metadata": {
                "filename": "rag_test.pdf",
                "page_number": 1,
            },
        },
        {
            "document_id": "doc-002",
            "chunk_id": "chunk-002",
            "content": "Project Orion uses FastAPI.",
            "similarity": 0.75,
            "metadata": {
                "filename": "rag_test.docx",
            },
        },
    ]


@pytest.mark.evaluation
def test_valid_citation_returns_correct_source(sample_contexts):
    generator = AnswerGenerator(
        llm_client=FakeLLMClient("Answer: 48 [1]")
    )

    result = generator.generate(
        "How many processing nodes does Project Nebula use?",
        sample_contexts,
    )

    assert result["answer"] == "48 [1]"
    assert result["grounded"] is True
    assert len(result["sources"]) == 1
    assert result["sources"][0]["filename"] == "rag_test.pdf"
    assert result["sources"][0]["document_id"] == "doc-001"


@pytest.mark.evaluation
def test_invalid_citation_is_rejected(sample_contexts):
    generator = AnswerGenerator(
        llm_client=FakeLLMClient("The answer is 48 [99]")
    )

    result = generator.generate("Question", sample_contexts)

    assert result["grounded"] is False
    assert result["sources"] == []
    assert result["answer"] == INSUFFICIENT_ANSWER


@pytest.mark.evaluation
def test_missing_citation_is_not_marked_grounded(sample_contexts):
    generator = AnswerGenerator(
        llm_client=FakeLLMClient("The answer is 48.")
    )

    result = generator.generate("Question", sample_contexts)

    assert result["grounded"] is False
    assert result["sources"] == []
    assert result["answer"] == INSUFFICIENT_ANSWER


@pytest.mark.evaluation
def test_insufficient_evidence_returns_fallback(sample_contexts):
    generator = AnswerGenerator(
        llm_client=FakeLLMClient(
            "The provided documents do not contain enough information to answer this."
        )
    )

    result = generator.generate("Unknown question", sample_contexts)

    assert result["grounded"] is False
    assert result["sources"] == []
    assert result["answer"] == INSUFFICIENT_ANSWER


@pytest.mark.evaluation
def test_empty_context_returns_fallback():
    generator = AnswerGenerator(
        llm_client=FakeLLMClient("This should never be generated.")
    )

    result = generator.generate("Any question?", [])

    assert result["grounded"] is False
    assert result["sources"] == []
    assert result["answer"] == INSUFFICIENT_ANSWER


@pytest.mark.evaluation
def test_invalid_citations_are_removed_from_answer(sample_contexts):
    generator = AnswerGenerator(
        llm_client=FakeLLMClient("Answer: 48 [1] [99]")
    )

    result = generator.generate("Question", sample_contexts)

    assert result["answer"] == "48 [1]"
    assert result["grounded"] is True
    assert len(result["sources"]) == 1
