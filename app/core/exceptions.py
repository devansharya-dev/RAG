"""Domain-specific exceptions for LocalRAG."""


class LocalRAGException(Exception):
    """Base exception for all LocalRAG domain errors."""


class DocumentParsingError(LocalRAGException):
    """Raised when parsing or extracting text from a document fails."""


class UnsupportedFileTypeError(LocalRAGException):
    """Raised when an uploaded or ingested file format is not supported."""


class EmbeddingError(LocalRAGException):
    """Raised when embedding generation fails or the embedding service is unreachable."""


class VectorStoreError(LocalRAGException):
    """Raised when a vector store operation fails."""


class GenerationError(LocalRAGException):
    """Raised when text generation or LLM interaction fails."""
