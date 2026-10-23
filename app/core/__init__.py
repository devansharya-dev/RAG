"""Core configuration, logging, exceptions, and constants."""

from app.core.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RETRIEVAL_TOP_K,
    SUPPORTED_FILE_EXTENSIONS,
)
from app.core.exceptions import (
    DocumentParsingError,
    EmbeddingError,
    GenerationError,
    LocalRAGException,
    UnsupportedFileTypeError,
    VectorStoreError,
)

__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_RETRIEVAL_TOP_K",
    "SUPPORTED_FILE_EXTENSIONS",
    "LocalRAGException",
    "DocumentParsingError",
    "UnsupportedFileTypeError",
    "EmbeddingError",
    "VectorStoreError",
    "GenerationError",
]
