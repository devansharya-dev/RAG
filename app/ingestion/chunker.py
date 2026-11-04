
"""Deterministic text chunking with overlap and metadata preservation."""

import re
from collections.abc import Mapping
from typing import Any

from app.core.config import get_settings
from app.schemas.document import DocumentChunk


class Chunker:
    """Split text into overlapping chunks without losing source content."""

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
        """Split text at whitespace boundaries without dropping words."""
        parts: list[str] = []

        while len(text) > max_size:
            split_at = text.rfind(" ", 0, max_size + 1)

            if split_at <= 0:
                split_at = max_size

            part = text[:split_at].strip()

            if part:
                parts.append(part)

            text = text[split_at:].strip()

        if text:
            parts.append(text)

        return parts

    def _make_segments(self, text: str) -> list[tuple[str, str]]:
        """
        Create text segments with their preceding separators.

        Paragraph boundaries use double newlines.
        Sentence boundaries use normal spaces.
        """
        paragraphs = re.split(r"\n\s*\n+", text.strip())

        segments: list[tuple[str, str]] = []
        first_paragraph = True

        for paragraph in paragraphs:
            paragraph = re.sub(r"[ \t]+", " ", paragraph).strip()

            if not paragraph:
                continue

            paragraph_separator = (
                "" if first_paragraph else "\n\n"
            )

            first_paragraph = False

            if len(paragraph) <= self.chunk_size:
                segments.append((paragraph, paragraph_separator))
                continue

            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            first_segment_in_paragraph = True
            current = ""

            def add_segment(value: str) -> None:
                nonlocal first_segment_in_paragraph

                separator = (
                    paragraph_separator
                    if first_segment_in_paragraph
                    else " "
                )

                segments.append((value, separator))
                first_segment_in_paragraph = False

            for sentence in sentences:
                sentence = sentence.strip()

                if not sentence:
                    continue

                if len(sentence) > self.chunk_size:
                    if current:
                        add_segment(current)
                        current = ""

                    for part in self._split_long_text(
                        sentence, self.chunk_size
                    ):
                        add_segment(part)

                    continue

                candidate = f"{current} {sentence}".strip()

                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    if current:
                        add_segment(current)

                    current = sentence

            if current:
                add_segment(current)

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

        for segment, separator in segments:
            actual_separator = separator if current else ""
            candidate = f"{current}{actual_separator}{segment}"

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                save_chunk(current)

            # Preserve overlap without exceeding the chunk size.
            available_overlap = max(
                0,
                self.chunk_size - len(segment) - len(separator),
            )

            overlap_length = min(
                self.chunk_overlap,
                available_overlap,
                len(current),
            )

            overlap = (
                current[-overlap_length:]
                if overlap_length > 0
                else ""
            )

            # Keep the paragraph separator between previous context
            # and a segment belonging to a new paragraph.
            current = (
                f"{overlap}{separator}{segment}"
                if overlap
                else segment
            )

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
