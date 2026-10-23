
"""Schemas for document ingestion, metadata, and chunk tracking."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with an uploaded document."""

    filename: str
    content_hash: str = ""
    file_type: str = ""
    file_size: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A text chunk ready for embedding and storage."""

    document_id: str | None = None
    chunk_index: int = Field(ge=0)
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionReport(BaseModel):
    """Summary of a document ingestion operation."""

    document_id: str | None = None
    filename: str
    status: Literal[
        "pending", "processing", "completed", "failed"
    ] = "pending"
    chunks_created: int = 0
    content_hash: str = ""
    message: str = ""
