"""Unit tests for independent factual grounding verification."""

import json

import pytest

from app.generation.grounding_verifier import GroundingVerifier


class FakeLLMClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def generate(self, messages, temperature=0.0, max_tokens=1024):
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def sample_contexts():
    return [
        {
            "document_id": "doc-001",
            "chunk_id": "chunk-001",
            "content": "Project Nebula uses 48 processing nodes.",
            "metadata": {"filename": "rag_test.pdf"},
        },
        {
            "document_id": "doc-002",
            "chunk_id": "chunk-002",
            "content": "Project Orion uses FastAPI.",
            "metadata": {"filename": "rag_test.docx"},
        },
    ]


def make_response(
    claim="Project Nebula uses 48 processing nodes.",
    answer_quote="48",
    verdict="SUPPORTED",
    passage=1,
    quote="Project Nebula uses 48 processing nodes.",
):
    return json.dumps({
        "claims": [
            {
                "claim": claim,
                "answer_quote": answer_quote,
                "verdict": verdict,
                "evidence_passage": passage,
                "evidence_quote": quote,
            }
        ]
    })


def test_supported_claim_passes_verification(sample_contexts):
    verifier = GroundingVerifier(
        llm_client=FakeLLMClient(make_response())
    )

    result = verifier.verify(
        "Project Nebula uses 48 processing nodes.",
        sample_contexts,
        question="How many processing nodes does Project Nebula use?",
    )

    assert result["supported"] is True
    assert len(result["claims"]) == 1
    assert result["claims"][0]["verdict"] == "SUPPORTED"
    assert result["claims"][0]["evidence_valid"] is True


def test_unsupported_claim_is_rejected(sample_contexts):
    verifier = GroundingVerifier(
        llm_client=FakeLLMClient(
            make_response(
                claim="Project Nebula uses 500 processing nodes.",
                answer_quote="500",
                verdict="NOT_SUPPORTED",
                passage=None,
                quote="",
            )
        )
    )

    result = verifier.verify(
        "Project Nebula uses 500 processing nodes.",
        sample_contexts,
    )

    assert result["supported"] is False
    assert result["claims"][0]["verdict"] == "NOT_SUPPORTED"


def test_contradicted_claim_is_rejected(sample_contexts):
    verifier = GroundingVerifier(
        llm_client=FakeLLMClient(
            make_response(
                claim="Project Nebula uses 500 processing nodes.",
                answer_quote="500",
                verdict="CONTRADICTED",
                passage=1,
                quote="Project Nebula uses 48 processing nodes.",
            )
        )
    )

    result = verifier.verify(
        "Project Nebula uses 500 processing nodes.",
        sample_contexts,
    )

    assert result["supported"] is False
    assert result["claims"][0]["verdict"] == "CONTRADICTED"


def test_supported_verdict_with_invalid_evidence_quote_is_rejected(
    sample_contexts,
):
    verifier = GroundingVerifier(
        llm_client=FakeLLMClient(
            make_response(
                quote="Project Nebula uses 500 processing nodes."
            )
        )
    )

    result = verifier.verify(
        "Project Nebula uses 48 processing nodes.",
        sample_contexts,
    )

    assert result["supported"] is False
    assert result["claims"][0]["verdict"] == "NOT_SUPPORTED"
    assert result["claims"][0]["evidence_valid"] is False


def test_one_unsupported_claim_rejects_entire_answer(sample_contexts):
    response = {
        "claims": [
            {
                "claim": "Project Nebula uses 48 processing nodes.",
                "answer_quote": "Project Nebula uses 48 processing nodes",
                "verdict": "SUPPORTED",
                "evidence_passage": 1,
                "evidence_quote": "Project Nebula uses 48 processing nodes.",
            },
            {
                "claim": "Project Nebula uses 500 storage nodes.",
                "answer_quote": "500 storage nodes",
                "verdict": "NOT_SUPPORTED",
                "evidence_passage": None,
                "evidence_quote": "",
            },
        ]
    }

    verifier = GroundingVerifier(
        llm_client=FakeLLMClient(json.dumps(response))
    )

    result = verifier.verify(
        "Project Nebula uses 48 processing nodes and 500 storage nodes.",
        sample_contexts,
    )

    assert result["supported"] is False
    assert len(result["claims"]) == 2


def test_invalid_passage_reference_fails_closed(sample_contexts):
    verifier = GroundingVerifier(
        llm_client=FakeLLMClient(
            make_response(passage=99)
        )
    )

    result = verifier.verify(
        "Project Nebula uses 48 processing nodes.",
        sample_contexts,
    )

    assert result["supported"] is False
    assert "Verification failed" in result["reason"]


def test_malformed_json_fails_closed(sample_contexts):
    verifier = GroundingVerifier(
        llm_client=FakeLLMClient("This is not JSON")
    )

    result = verifier.verify("Some answer.", sample_contexts)

    assert result["supported"] is False
    assert "Verification failed" in result["reason"]


def test_empty_verifier_response_fails_closed(sample_contexts):
    verifier = GroundingVerifier(
        llm_client=FakeLLMClient("")
    )

    result = verifier.verify("Some answer.", sample_contexts)

    assert result["supported"] is False


def test_llm_exception_fails_closed(sample_contexts):
    verifier = GroundingVerifier(
        llm_client=FakeLLMClient(
            error=RuntimeError("Ollama unavailable")
        )
    )

    result = verifier.verify("Some answer.", sample_contexts)

    assert result["supported"] is False
    assert "Ollama unavailable" in result["reason"]


def test_empty_answer_is_rejected(sample_contexts):
    verifier = GroundingVerifier(
        llm_client=FakeLLMClient("{}")
    )

    result = verifier.verify("", sample_contexts)

    assert result["supported"] is False


def test_empty_context_is_rejected():
    verifier = GroundingVerifier(
        llm_client=FakeLLMClient("{}")
    )

    result = verifier.verify("Some answer.", [])

    assert result["supported"] is False


def test_markdown_json_fence_is_supported(sample_contexts):
    response = "```json\n" + make_response() + "\n```"

    verifier = GroundingVerifier(
        llm_client=FakeLLMClient(response)
    )

    result = verifier.verify(
        "Project Nebula uses 48 processing nodes.",
        sample_contexts,
    )

    assert result["supported"] is True


def test_case_and_whitespace_differences_in_quote_are_allowed(
    sample_contexts,
):
    verifier = GroundingVerifier(
        llm_client=FakeLLMClient(
            make_response(
                answer_quote="48",
                quote="PROJECT NEBULA   uses 48 processing nodes.",
            )
        )
    )

    result = verifier.verify(
        "Project Nebula uses 48 processing nodes.",
        sample_contexts,
    )

    assert result["supported"] is True


def test_claim_not_present_in_answer_is_rejected(sample_contexts):
    response = make_response(
        claim="Project Orion uses FastAPI.",
        answer_quote="FastAPI",
        passage=2,
        quote="Project Orion uses FastAPI.",
    )

    verifier = GroundingVerifier(
        llm_client=FakeLLMClient(response)
    )

    result = verifier.verify(
        "Project Nebula uses 48 processing nodes.",
        sample_contexts,
    )

    assert result["supported"] is False
    assert "Verification failed" in result["reason"]
