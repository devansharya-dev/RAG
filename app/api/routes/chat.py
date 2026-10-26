
"""Grounded question-answering API endpoint."""

from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

rag_service = RAGService()


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question using indexed documents."""
    try:
        return rag_service.answer(request)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        # Keep internal exception details out of the public API response.
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer. Check server logs.",
        ) from exc
