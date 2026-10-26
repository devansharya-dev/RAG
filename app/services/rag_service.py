from app.generation.answer_generator import AnswerGenerator
from app.retrieval.retriever import Retriever
from app.schemas.chat import ChatRequest, ChatResponse, SourceReference


class RAGService:
    """Coordinate semantic retrieval and document-grounded generation."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.answer_generator = answer_generator or AnswerGenerator()

    def answer(self, request: ChatRequest) -> ChatResponse:
        """Answer a question using retrieved document evidence."""
        question = request.question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        # 1. Retrieve evidence from Supabase using Ollama embeddings.
        contexts = self.retriever.retrieve(
            query=question,
            top_k=request.top_k,
        )

        # 2. Do not invent an answer when the index has no relevant results.
        if not contexts:
            return ChatResponse(
                question=question,
                answer=(
                    "The provided documents do not contain enough "
                    "information to answer this."
                ),
                sources=[],
                grounded=False,
            )

        # 3. Generate an answer using the retrieved passages.
        result = self.answer_generator.generate(
            question=question,
            contexts=contexts,
        )

        # 4. Map generated citation numbers back to retrieved chunks.
        sources: list[SourceReference] = []

        for source in result.get("sources", []):
            passage_number = source.get("passage_number")

            if not isinstance(passage_number, int):
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

        return ChatResponse(
            question=question,
            answer=result["answer"],
            sources=sources,
            grounded=bool(result.get("grounded", False)) and bool(sources),
        )
