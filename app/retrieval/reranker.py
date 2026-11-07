
"""Cross-encoder reranking for retrieved document chunks."""

from typing import Any

import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Re-score query-document pairs using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: CrossEncoder | None = None

    def _load_model(self) -> CrossEncoder:
        """Load the model only when reranking is first requested."""

        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512,
            )

        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rank retrieved chunks by cross-encoder relevance score."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not documents:
            return []

        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1.")

        pairs = [
            (query.strip(), str(document.get("content", "")))
            for document in documents
        ]

        model = self._load_model()

        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if len(scores) != len(documents):
            raise RuntimeError(
                "Reranker returned an unexpected number of scores."
            )

        ranked_documents = []

        for document, score in zip(documents, scores):
            ranked_document = dict(document)
            ranked_document["reranker_score"] = float(score)
            ranked_documents.append(ranked_document)

        ranked_documents.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        if top_k is not None:
            ranked_documents = ranked_documents[:top_k]

        return ranked_documents
