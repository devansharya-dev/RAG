
"""Generate and validate answers grounded in retrieved passages."""

import re
from typing import Any

from app.generation.llm_client import LLMClient
from app.generation.prompt_builder import PromptBuilder


INSUFFICIENT_ANSWER = (
    "The provided documents do not contain enough information to answer this."
)


class AnswerGenerator:
    """Generate answers and validate citations against retrieved contexts."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.prompt_builder = prompt_builder or PromptBuilder()

    @staticmethod
    def _extract_citations(answer: str, context_count: int) -> list[int]:
        """Return unique, valid passage numbers cited by the LLM."""
        numbers = re.findall(r"\[(\d+)\]", answer)

        return sorted({
            int(number)
            for number in numbers
            if 1 <= int(number) <= context_count
        })

    @staticmethod
    def _clean_answer(answer: str, citations: list[int]) -> str:
        """Remove formatting labels and unsupported citation references."""

        # Remove the separate Sources section if the model generated one.
        answer = re.split(
            r"\n\s*Sources\s*:",
            answer,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        # Remove the leading Answer: label.
        answer = re.sub(
            r"^\s*Answer\s*:\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )

        # Remove citation numbers that do not refer to retrieved contexts.
        def replace_citation(match: re.Match) -> str:
            number = int(match.group(1))
            return match.group(0) if number in citations else ""

        answer = re.sub(
            r"\[(\d+)\]",
            replace_citation,
            answer,
        )

        # Normalize excessive whitespace.
        answer = re.sub(r"[ \t]+", " ", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer)

        answer = answer.strip()

        # Append validated citations to the final answer.
        if citations:
            citation_text = " ".join(
                f"[{number}]" for number in citations
            )

            if not any(
                re.search(rf"\[{number}\]", answer)
                for number in citations
            ):
                answer = f"{answer} {citation_text}".strip()

        return answer

    def generate(
        self,
        question: str,
        contexts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate an answer and return only validated source references."""

        if not contexts:
            return {
                "answer": INSUFFICIENT_ANSWER,
                "sources": [],
                "grounded": False,
            }

        messages = self.prompt_builder.build(question, contexts)

        raw_answer = self.llm_client.generate(
            messages,
            temperature=0.2,
            max_tokens=512,
        )

        if not raw_answer:
            return {
                "answer": INSUFFICIENT_ANSWER,
                "sources": [],
                "grounded": False,
            }

        insufficient_evidence = (
            "do not contain enough information to answer"
            in raw_answer.lower()
        )

        if insufficient_evidence:
            return {
                "answer": INSUFFICIENT_ANSWER,
                "sources": [],
                "grounded": False,
            }

        cited_numbers = self._extract_citations(
            raw_answer,
            len(contexts),
        )

        # Do not present an uncited model response as grounded.
        if not cited_numbers:
            return {
                "answer": INSUFFICIENT_ANSWER,
                "sources": [],
                "grounded": False,
            }

        cleaned_answer = self._clean_answer(
            raw_answer,
            cited_numbers,
        )

        sources = []

        for number in cited_numbers:
            context = contexts[number - 1]
            metadata = context.get("metadata") or {}

            sources.append({
                "passage_number": number,
                "document_id": context.get("document_id"),
                "chunk_id": context.get("chunk_id"),
                "filename": metadata.get("filename", "Unknown"),
                "similarity": float(
                    context.get("similarity", 0.0)
                ),
            })

        return {
            "answer": cleaned_answer,
            "sources": sources,
            # This means citations were validated against retrieved
            # contexts, not that factual entailment was independently proven.
            "grounded": bool(sources),
        }
