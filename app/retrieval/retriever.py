

from typing import Any

import ollama

from app.core.config import get_settings
from app.vectorstore.supabase_store import SupabaseVectorStore


class Retriever:
    """Embed a query and retrieve the most relevant document chunks."""

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
        """Return the most relevant chunks for a query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        stats = self.vector_store.get_stats()
        if stats.get("total_chunks", 0) == 0:
            return []

        response = ollama.embed(
            model=self.settings.OLLAMA_EMBED_MODEL,
            input=query.strip(),
        )
        query_embedding = response.embeddings[0]

        return self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k or self.top_k,
            filters=filters,
        )
