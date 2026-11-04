
"""Retrieve relevant document chunks for RAG."""

from typing import Any

import ollama

from app.core.config import get_settings
from app.vectorstore.supabase_store import SupabaseVectorStore


class Retriever:
    """Embed a query and retrieve relevant document chunks."""

    def __init__(
        self,
        vector_store: SupabaseVectorStore | None = None,
        top_k: int | None = None,
    ) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store or SupabaseVectorStore()
        self.top_k = top_k or self.settings.RETRIEVAL_TOP_K

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks using hybrid search when supported."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        clean_query = query.strip()

        stats = self.vector_store.get_stats()

        if stats.get("total_chunks", 0) == 0:
            return []

        response = ollama.embed(
            model=self.settings.OLLAMA_EMBED_MODEL,
            input=clean_query,
        )

        if not response.embeddings:
            raise RuntimeError("Ollama returned no query embedding.")

        query_embedding = response.embeddings[0]
        result_count = top_k or self.top_k

        # Prefer hybrid search when the store supports it.
        hybrid_method = getattr(
            self.vector_store,
            "hybrid_search",
            None,
        )

        if callable(hybrid_method):
            return hybrid_method(
                query_embedding=query_embedding,
                query_text=clean_query,
                top_k=result_count,
                filters=filters,
            )

        # Fallback for vector stores that only support vector search.
        return self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=result_count,
            filters=filters,
        )
