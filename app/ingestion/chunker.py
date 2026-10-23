
"""Deterministic text chunking with overlap and metadata preservation."""

import re
from collections.abc import Mapping
from typing import Any

from app.core.config import get_settings
from app.schemas.document import DocumentChunk


class Chunker:
    """Split text into overlapping chunks while preserving source metadata."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()

        self.chunk_size = (
            chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        )
        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else settings.CHUNK_OVERLAP
        )

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller "
                "than chunk_size"
            )

    @staticmethod
    def _split_long_text(text: str, max_size: int) -> list[str]:
        """Split long text using whitespace when possible."""
        parts: list[str] = []

        while len(text) > max_size:
            split_at = text.rfind(" ", 0, max_size + 1)

            if split_at <= 0:
                split_at = max_size

            parts.append(text[:split_at].strip())
            text = text[split_at:].strip()

        if text:
            parts.append(text)

        return parts

    def _make_segments(self, text: str) -> list[str]:
        """Prefer paragraph and sentence boundaries."""
        paragraphs = re.split(r"\n\s*\n+", text.strip())
        segments: list[str] = []

        for paragraph in paragraphs:
            paragraph = re.sub(r"\s+", " ", paragraph).strip()

            if not paragraph:
                continue

            if len(paragraph) <= self.chunk_size:
                segments.append(paragraph)
                continue

            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            current = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(sentence) > self.chunk_size:
                    if current:
                        segments.append(current)
                        current = ""

                    segments.extend(
                        self._split_long_text(sentence, self.chunk_size)
                    )
                    continue

                candidate = f"{current} {sentence}".strip()

                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    if current:
                        segments.append(current)
                    current = sentence

            if current:
                segments.append(current)

        return segments

    def chunk_text(
        self,
        text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Convert text into indexed chunks with overlapping context."""
        if not text or not text.strip():
            return []

        base_metadata = dict(metadata or {})
        segments = self._make_segments(text)

        chunks: list[DocumentChunk] = []
        current = ""

        def save_chunk(content: str) -> None:
            content = content.strip()
            if content:
                chunks.append(
                    DocumentChunk(
                        chunk_index=len(chunks),
                        content=content,
                        metadata=base_metadata.copy(),
                    )
                )

        for segment in segments:
            candidate = f"{current} {segment}".strip()

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                save_chunk(current)

            overlap = current[-self.chunk_overlap:] if self.chunk_overlap else ""
            current = f"{overlap} {segment}".strip()

            if len(current) > self.chunk_size:
                current = current[:self.chunk_size].strip()

        if current:
            save_chunk(current)

        return chunks

    def chunk(
        self,
        text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Convenience alias for chunk_text."""
        return self.chunk_text(text, metadata)
