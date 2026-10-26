
"""Generate answers grounded in retrieved passages."""

import re
from typing import Any

from app.generation.llm_client import LLMClient
from app.generation.prompt_builder import PromptBuilder


class AnswerGenerator:
    """Coordinate prompt construction and LLM inference."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def generate(
        self,
        question: str,
        contexts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate an answer and identify citations to retrieved passages."""
        messages = self.prompt_builder.build(question, contexts)

        answer = self.llm_client.generate(
            messages,
            temperature=0.2,
            max_tokens=512,
        )

        cited_numbers = sorted(
            {
                int(number)
                for number in re.findall(r"\[(\d+)\]", answer)
                if 1 <= int(number) <= len(contexts)
            }
        )

        sources = [
            {
                "passage_number": number,
                "document_id": contexts[number - 1].get("document_id"),
                "chunk_id": contexts[number - 1].get("chunk_id"),
                "filename": (
                    contexts[number - 1].get("metadata") or {}
                ).get("filename", "Unknown"),
                "similarity": contexts[number - 1].get("similarity", 0.0),
            }
            for number in cited_numbers
        ]

        insufficient_evidence = (
            "do not contain enough information to answer" in answer.lower()
        )

        return {
            "answer": answer,
            "sources": sources,
            "grounded": bool(sources) and not insufficient_evidence,
        }
