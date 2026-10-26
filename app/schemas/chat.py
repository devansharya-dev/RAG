
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=4, ge=1, le=20)


class SourceReference(BaseModel):
    document_id: str
    chunk_id: str | None = None
    filename: str = "Unknown"
    content: str
    similarity: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    grounded: bool = True
