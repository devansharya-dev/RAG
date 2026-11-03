
"""Unit tests for grounded prompt construction."""

import pytest

from app.generation.prompt_builder import PromptBuilder


@pytest.fixture
def sample_contexts():
    return [
        {
            "document_id": "doc-001",
            "content": "Project Nebula uses 48 processing nodes.",
            "metadata": {
                "filename": "rag_test.pdf",
                "page_number": 1,
            },
        }
    ]


@pytest.mark.unit
def test_prompt_contains_user_question(sample_contexts):
    messages = PromptBuilder().build(
        "How many processing nodes are used?",
        sample_contexts,
    )

    assert messages[1]["content"].find(
        "How many processing nodes are used?"
    ) != -1


@pytest.mark.unit
def test_prompt_contains_document_evidence(sample_contexts):
    messages = PromptBuilder().build(
        "What is the node count?",
        sample_contexts,
    )

    assert "Project Nebula uses 48 processing nodes." in messages[1]["content"]


@pytest.mark.unit
def test_prompt_preserves_source_metadata(sample_contexts):
    messages = PromptBuilder().build(
        "What is the node count?",
        sample_contexts,
    )

    prompt = messages[1]["content"]

    assert "rag_test.pdf" in prompt
    assert "PAGE NUMBER: 1" in prompt
    assert "doc-001" in prompt


@pytest.mark.unit
def test_system_prompt_contains_grounding_rules(sample_contexts):
    messages = PromptBuilder().build(
        "What is the node count?",
        sample_contexts,
    )

    system_prompt = messages[0]["content"]

    assert "ONLY the supplied document evidence" in system_prompt
    assert "Never invent facts" in system_prompt
    assert "Cite supporting passages" in system_prompt


@pytest.mark.unit
def test_prompt_handles_empty_context():
    messages = PromptBuilder().build("Any question?", [])

    assert "No passages retrieved." in messages[1]["content"]
