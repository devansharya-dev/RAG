from app.generation.answer_generator import (
    AnswerGenerator,
    INSUFFICIENT_ANSWER,
)
from app.generation.grounding_verifier import GroundingVerifier
from app.retrieval.retriever import Retriever
from app.schemas.chat import ChatRequest, ChatResponse, SourceReference


class RAGService:
    """Coordinate retrieval, generation, citation mapping and verification."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        answer_generator: AnswerGenerator | None = None,
        grounding_verifier: GroundingVerifier | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.answer_generator = answer_generator or AnswerGenerator()
        self.grounding_verifier = (
            grounding_verifier or GroundingVerifier()
        )

    def answer(self, request: ChatRequest) -> ChatResponse:
        """Generate an answer and independently verify its factual claims."""

        question = request.question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        # 1. Retrieve evidence.
        contexts = self.retriever.retrieve(
            query=question,
            top_k=request.top_k,
        )

        if not contexts:
            return ChatResponse(
                question=question,
                answer=INSUFFICIENT_ANSWER,
                sources=[],
                grounded=False,
            )

        # 2. Generate an answer using retrieved evidence.
        result = self.answer_generator.generate(
            question=question,
            contexts=contexts,
        )

        # 3. Map valid citation numbers to retrieved chunks.
        sources: list[SourceReference] = []

        for source in result.get("sources", []):
            passage_number = source.get("passage_number")

            if (
                isinstance(passage_number, bool)
                or not isinstance(passage_number, int)
            ):
                continue

            if not 1 <= passage_number <= len(contexts):
                continue

            context = contexts[passage_number - 1]
            metadata = context.get("metadata") or {}

            sources.append(
                SourceReference(
                    document_id=str(context.get("document_id", "")),
                    chunk_id=(
                        str(context["chunk_id"])
                        if context.get("chunk_id")
                        else None
                    ),
                    filename=metadata.get("filename", "Unknown"),
                    content=context.get("content", ""),
                    similarity=float(context.get("similarity", 0.0)),
                    metadata=metadata,
                )
            )

        # 4. Do not verify answers that already failed citation validation.
        citations_validated = (
            result.get("grounded") is True and bool(sources)
        )

        if not citations_validated:
            return ChatResponse(
                question=question,
                answer=result.get("answer", INSUFFICIENT_ANSWER),
                sources=sources,
                grounded=False,
            )

        # 5. Independently verify every factual claim.
        verification = self.grounding_verifier.verify(
            answer=result["answer"],
            contexts=contexts,
        )

        # 6. Fail closed if verification does not explicitly pass.
        if (
            not isinstance(verification, dict)
            or verification.get("supported") is not True
        ):
            return ChatResponse(
                question=question,
                answer=INSUFFICIENT_ANSWER,
                sources=[],
                grounded=False,
            )

        return ChatResponse(
            question=question,
            answer=result["answer"],
            sources=sources,
            grounded=True,
        )
