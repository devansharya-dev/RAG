
"""End-to-end document ingestion pipeline."""

import hashlib
from pathlib import Path
from typing import Any

import ollama

from app.core.config import get_settings
from app.ingestion.chunker import Chunker
from app.ingestion.loaders import DOCXLoader, PDFLoader, TextLoader
from app.ingestion.text_cleaner import TextCleaner
from app.schemas.document import DocumentChunk, IngestionReport
from app.vectorstore.supabase_store import SupabaseVectorStore


class IngestionPipeline:
    """Load, clean, chunk, embed, and persist documents."""

    def __init__(
        self,
        vector_store: SupabaseVectorStore | None = None,
        chunker: Chunker | None = None,
        text_cleaner: TextCleaner | None = None,
    ) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store or SupabaseVectorStore()
        self.chunker = chunker or Chunker()
        self.text_cleaner = text_cleaner or TextCleaner()

        self.loaders = {
            ".txt": TextLoader(),
            ".md": TextLoader(),
            ".csv": TextLoader(),
            ".log": TextLoader(),
            ".pdf": PDFLoader(),
            ".docx": DOCXLoader(),
        }

    def ingest(self, file_path: str | Path) -> IngestionReport:
        """Ingest a supported file into Supabase."""
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        loader = self.loaders.get(path.suffix.lower())
        if loader is None:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        loaded_documents = loader.load(path)

        if not loaded_documents:
            raise ValueError("The loader returned no documents.")

        # Clean extracted text while preserving loader metadata.
        for loaded_doc in loaded_documents:
            loaded_doc.content = self.text_cleaner.clean(
                loaded_doc.content
            )

        # Remove empty pages/documents after cleaning.
        loaded_documents = [
            doc for doc in loaded_documents if doc.content.strip()
        ]

        if not loaded_documents:
            raise ValueError(
                "The document contains no usable text after cleaning."
            )

        # Hash the cleaned content for consistent duplicate detection.
        content_hash = hashlib.sha256(
            "\n".join(
                doc.content for doc in loaded_documents
            ).encode("utf-8")
        ).hexdigest()

        store = self.vector_store
        client = store.client

        existing = (
            client.table("documents")
            .select("id,status")
            .eq("content_hash", content_hash)
            .limit(1)
            .execute()
        ).data

        # Skip an already completed document with identical content.
        if existing and existing[0]["status"] == "completed":
            return IngestionReport(
                document_id=str(existing[0]["id"]),
                filename=path.name,
                status="completed",
                content_hash=content_hash,
                message="Identical document already ingested; skipped.",
            )

        document_id: str | None = None

        try:
            if existing:
                document_id = str(existing[0]["id"])

                # Remove old partial chunks before retrying.
                client.table("document_chunks").delete().eq(
                    "document_id", document_id
                ).execute()

                client.table("documents").update(
                    {"status": "processing"}
                ).eq("id", document_id).execute()

            else:
                created = (
                    client.table("documents")
                    .insert(
                        {
                            "filename": path.name,
                            "content_hash": content_hash,
                            "status": "processing",
                        }
                    )
                    .execute()
                )

                if not created.data:
                    raise RuntimeError(
                        "Supabase did not return the document ID."
                    )

                document_id = str(created.data[0]["id"])

            chunks: list[DocumentChunk] = []

            for loaded_doc in loaded_documents:
                metadata: dict[str, Any] = dict(
                    loaded_doc.metadata
                )

                page_chunks = self.chunker.chunk_text(
                    loaded_doc.content,
                    metadata=metadata,
                )

                for chunk in page_chunks:
                    chunk.document_id = document_id
                    chunk.chunk_index = len(chunks)
                    chunks.append(chunk)

            if not chunks:
                raise ValueError("No text chunks were created.")

            # Generate embeddings locally through Ollama.
            texts = [chunk.content for chunk in chunks]

            embedding_response = ollama.embed(
                model=self.settings.OLLAMA_EMBED_MODEL,
                input=texts,
            )

            embeddings = embedding_response.embeddings

            if len(embeddings) != len(chunks):
                raise RuntimeError(
                    "Ollama returned a different number of embeddings "
                    "than the number of chunks."
                )

            if any(len(vector) != 768 for vector in embeddings):
                raise RuntimeError(
                    "Embedding dimension mismatch: expected 768."
                )

            store.add_chunks(chunks, embeddings)

            client.table("documents").update(
                {"status": "completed"}
            ).eq("id", document_id).execute()

            return IngestionReport(
                document_id=document_id,
                filename=path.name,
                status="completed",
                chunks_created=len(chunks),
                content_hash=content_hash,
                message="Document successfully ingested into Supabase.",
            )

        except Exception as exc:
            cleanup_error: Exception | None = None

            if document_id is not None:
                # Compensating cleanup: remove any chunks inserted before
                # the failure, but preserve the document row for retries.
                try:
                    client.table("document_chunks").delete().eq(
                        "document_id", document_id
                    ).execute()
                except Exception as cleanup_exc:
                    cleanup_error = cleanup_exc

                # Keep the document record so the same content can retry.
                try:
                    client.table("documents").update(
                        {"status": "failed"}
                    ).eq("id", document_id).execute()
                except Exception:
                    pass

            error_message = (
                f"Ingestion failed for '{path.name}': {exc}"
            )

            if cleanup_error is not None:
                error_message += (
                    f". Partial chunk cleanup also failed: {cleanup_error}"
                )

            raise RuntimeError(error_message) from exc
