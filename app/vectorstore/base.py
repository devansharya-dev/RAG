"""Abstract base interface for vector store persistence."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any


class BaseVectorStore(ABC):
    """Abstract base class defining the vector store interface for LocalRAG.

    All vector store implementations (such as Supabase PostgreSQL with pgvector)
    must implement this interface to support ingestion and similarity search.
    """

    @abstractmethod
    def add_chunks(
        self,
        chunks: Sequence[Any],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Add document chunks and their corresponding embeddings to the vector store.

        Args:
            chunks: Sequence of document chunks or chunk data containing text and metadata.
            embeddings: Sequence of vector embeddings corresponding 1-to-1 with chunks.
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Perform vector similarity search for chunks nearest to the query embedding.

        Args:
            query_embedding: Vector embedding of the search query.
            top_k: Maximum number of nearest matching chunks to return.
            filters: Optional metadata filters to constrain the search.

        Returns:
            List of retrieved chunks with similarity scores and metadata.
        """
        pass

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks associated with a specific document ID.

        Args:
            document_id: The unique identifier of the document to purge.
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Retrieve statistics about the vector store.

        Returns:
            Dictionary containing metrics such as total vector count and document count.
        """
        pass

    def get_collection_stats(self) -> dict[str, Any]:
        """Alias for get_stats to maintain backwards compatibility."""
        return self.get_stats()
