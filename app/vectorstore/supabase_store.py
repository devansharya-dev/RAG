
"""Supabase pgvector storage for LocalRAG."""

from collections.abc import Mapping, Sequence
from typing import Any

from app.core.config import get_settings
from app.vectorstore.base import BaseVectorStore

try:
    from supabase import Client, create_client
except ImportError as exc:
    raise ImportError(
        "Supabase package missing. Install it using: "
        "python -m pip install supabase"
    ) from exc


class SupabaseVectorStore(BaseVectorStore):
    """Store and retrieve document chunks using Supabase pgvector."""

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        table_name: str | None = None,
    ) -> None:
        settings = get_settings()

        self.supabase_url = supabase_url or settings.SUPABASE_URL
        self.supabase_key = supabase_key or settings.SUPABASE_KEY
        self.table_name = table_name or "document_chunks"
        self._client: Client | None = None

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be configured in .env"
            )

    @property
    def client(self) -> Client:
        """Create the client lazily, on first use."""
        if self._client is None:
            self._client = create_client(
                self.supabase_url,
                self.supabase_key,
            )
        return self._client

    @staticmethod
    def _value(item: Any, name: str, default: Any = None) -> Any:
        """Read a field from either a Pydantic model or a dictionary."""
        if isinstance(item, Mapping):
            return item.get(name, default)
        return getattr(item, name, default)

    @staticmethod
    def _validate_embedding(
        embedding: Sequence[float],
    ) -> list[float]:
        """Validate and normalize a 768-dimensional embedding."""
        vector = [float(value) for value in embedding]

        if len(vector) != 768:
            raise ValueError(
                f"Expected a 768-dimensional embedding, got {len(vector)}."
            )

        return vector

    @staticmethod
    def _validate_filters(
        filters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate metadata filters."""
        if filters is None:
            return {}

        if not isinstance(filters, dict):
            raise ValueError("filters must be a dictionary.")

        return filters

    @staticmethod
    def _format_results(
        matches: Sequence[dict[str, Any]],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Format database matches into application results."""
        output: list[dict[str, Any]] = []

        for match in matches:
            metadata = match.get("metadata") or {}

            # Defensive metadata verification.
            if filters and any(
                metadata.get(key) != value
                for key, value in filters.items()
            ):
                continue

            output.append(
                {
                    "chunk_id": match.get("chunk_id"),
                    "document_id": match.get("document_id"),
                    "content": match.get("content", ""),
                    "metadata": metadata,
                    "similarity": float(
                        match.get("similarity", 0.0)
                    ),
                }
            )

            if top_k is not None and len(output) >= top_k:
                break

        return output

    def add_chunks(
        self,
        chunks: Sequence[Any],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Insert chunks and their embeddings into document_chunks."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Each chunk must have exactly one embedding."
            )

        if not chunks:
            return

        rows: list[dict[str, Any]] = []

        for chunk, embedding in zip(chunks, embeddings):
            document_id = self._value(chunk, "document_id")
            content = self._value(chunk, "content")
            chunk_index = self._value(chunk, "chunk_index")
            metadata = self._value(chunk, "metadata", {}) or {}

            if not document_id:
                raise ValueError(
                    "Each chunk must have a document_id before insertion."
                )

            if content is None or chunk_index is None:
                raise ValueError(
                    "Each chunk must have content and chunk_index."
                )

            vector = self._validate_embedding(embedding)

            rows.append(
                {
                    "document_id": str(document_id),
                    "chunk_index": int(chunk_index),
                    "content": str(content),
                    "metadata": dict(metadata),
                    "embedding": vector,
                }
            )

        batch_size = 100

        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            self.client.table(self.table_name).insert(batch).execute()

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search chunks using vector similarity."""
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        vector = self._validate_embedding(query_embedding)
        metadata_filter = self._validate_filters(filters)

        if metadata_filter:
            rpc_name = "match_document_chunks_filtered"

            rpc_params = {
                "query_embedding": vector,
                "match_count": 20,
                "match_threshold": 0.0,
                "metadata_filter": metadata_filter,
            }
        else:
            rpc_name = "match_document_chunks"

            rpc_params = {
                "query_embedding": vector,
                "match_count": top_k,
                "match_threshold": 0.0,
            }

        result = self.client.rpc(
            rpc_name,
            rpc_params,
        ).execute()

        return self._format_results(
            result.data or [],
            filters=metadata_filter,
            top_k=top_k,
        )

    def hybrid_search(
        self,
        query_embedding: Sequence[float],
        query_text: str,
        top_k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search using combined vector and keyword retrieval.

        The database RPC combines both rankings using Reciprocal
        Rank Fusion (RRF). The returned similarity field represents
        the RRF score, not cosine similarity.
        """
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text cannot be empty.")

        vector = self._validate_embedding(query_embedding)
        metadata_filter = self._validate_filters(filters)

        rpc_params = {
            "p_query_embedding": vector,
            "p_query_text": query_text.strip(),
            "p_match_count": top_k,
            "p_metadata_filter": metadata_filter,
        }

        result = self.client.rpc(
            "hybrid_search_document_chunks",
            rpc_params,
        ).execute()

        return self._format_results(
            result.data or [],
            filters=metadata_filter,
            top_k=top_k,
        )

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete a document's chunks, then its document record."""
        self.client.table(self.table_name).delete().eq(
            "document_id", document_id
        ).execute()

        self.client.table("documents").delete().eq(
            "id", document_id
        ).execute()

    def get_stats(self) -> dict[str, Any]:
        """Return document and chunk counts from Supabase."""
        chunks_result = (
            self.client.table(self.table_name)
            .select("id", count="exact", head=True)
            .execute()
        )

        documents_result = (
            self.client.table("documents")
            .select("id", count="exact", head=True)
            .execute()
        )

        return {
            "table": self.table_name,
            "total_chunks": chunks_result.count or 0,
            "total_documents": documents_result.count or 0,
        }

    def get_collection_stats(self) -> dict[str, Any]:
        """Compatibility alias for get_stats."""
        return self.get_stats()


SupabaseStore = SupabaseVectorStore
