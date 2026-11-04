
"""Tests for the document ingestion pipeline."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.ingestion.pipeline import IngestionPipeline


class FakeQuery:
    """Minimal Supabase query builder for pipeline tests."""

    def __init__(self, client, table_name):
        self.client = client
        self.table_name = table_name
        self.operation = "select"
        self.payload = None
        self.filters = {}
        self.limit_value = None

    def select(self, *args, **kwargs):
        self.operation = "select"
        return self

    def insert(self, payload):
        self.operation = "insert"
        self.payload = payload
        return self

    def update(self, payload):
        self.operation = "update"
        self.payload = payload
        return self

    def delete(self):
        self.operation = "delete"
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def limit(self, value):
        self.limit_value = value
        return self

    def execute(self):
        rows = self.client.tables[self.table_name]

        if self.operation == "insert":
            new_row = dict(self.payload)

            if self.table_name == "documents":
                new_row["id"] = f"doc-{len(rows) + 1}"

            rows.append(new_row)
            return SimpleNamespace(data=[new_row])

        if self.operation == "select":
            result = [
                row for row in rows
                if all(
                    row.get(key) == value
                    for key, value in self.filters.items()
                )
            ]

            if self.limit_value is not None:
                result = result[:self.limit_value]

            return SimpleNamespace(data=result)

        if self.operation == "update":
            updated = []

            for row in rows:
                if all(
                    row.get(key) == value
                    for key, value in self.filters.items()
                ):
                    row.update(self.payload)
                    updated.append(row)

            return SimpleNamespace(data=updated)

        if self.operation == "delete":
            remaining = [
                row for row in rows
                if not all(
                    row.get(key) == value
                    for key, value in self.filters.items()
                )
            ]

            self.client.tables[self.table_name] = remaining
            return SimpleNamespace(data=[])

        raise NotImplementedError(self.operation)


class FakeSupabaseClient:
    """In-memory fake Supabase client."""

    def __init__(self):
        self.tables = {
            "documents": [],
            "document_chunks": [],
        }

    def table(self, table_name):
        return FakeQuery(self, table_name)


class FakeVectorStore:
    """Fake vector store that simulates batch insertion."""

    def __init__(self):
        self.client = FakeSupabaseClient()
        self.inserted_chunks = []
        self.inserted_embeddings = []
        self.fail_on_call = None
        self.add_chunks_calls = 0

    def add_chunks(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding count mismatch.")

        batch_size = 100

        for start in range(0, len(chunks), batch_size):
            self.add_chunks_calls += 1

            if self.add_chunks_calls == self.fail_on_call:
                raise RuntimeError(
                    "Simulated batch insertion failure"
                )

            chunk_batch = chunks[start:start + batch_size]
            embedding_batch = embeddings[start:start + batch_size]

            rows = []

            for chunk, embedding in zip(
                chunk_batch, embedding_batch
            ):
                rows.append({
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "embedding": embedding,
                })

            self.client.tables["document_chunks"].extend(rows)
            self.inserted_chunks.extend(chunk_batch)
            self.inserted_embeddings.extend(embedding_batch)


@pytest.fixture
def fake_store():
    return FakeVectorStore()


@pytest.fixture
def pipeline(fake_store):
    return IngestionPipeline(vector_store=fake_store)


def mock_embedding_response(count):
    return SimpleNamespace(
        embeddings=[[0.1] * 768 for _ in range(count)]
    )


@pytest.mark.integration
def test_successful_document_ingestion(
    pipeline, fake_store, tmp_path
):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "Hello RAG pipeline.",
        encoding="utf-8",
    )

    with patch(
        "app.ingestion.pipeline.ollama.embed",
        return_value=mock_embedding_response(1),
    ) as mock_embed:
        report = pipeline.ingest(file_path)

    assert report.status == "completed"
    assert report.filename == "sample.txt"
    assert report.chunks_created == 1
    assert report.document_id == "doc-1"

    assert len(fake_store.inserted_chunks) == 1
    assert len(fake_store.inserted_embeddings) == 1

    assert (
        fake_store.client.tables["documents"][0]["status"]
        == "completed"
    )

    mock_embed.assert_called_once()


@pytest.mark.integration
def test_cleaned_text_is_sent_to_chunker(
    pipeline, fake_store, tmp_path
):
    file_path = tmp_path / "dirty.txt"
    file_path.write_bytes(b"Hello\x00\n\n\nWorld")

    with patch(
        "app.ingestion.pipeline.ollama.embed",
        return_value=mock_embedding_response(1),
    ):
        pipeline.ingest(file_path)

    chunk = fake_store.inserted_chunks[0]

    assert chunk.content == "Hello\n\nWorld"
    assert chunk.metadata["filename"] == "dirty.txt"


@pytest.mark.integration
def test_duplicate_document_is_skipped(
    pipeline, fake_store, tmp_path
):
    file_path = tmp_path / "duplicate.txt"
    file_path.write_text(
        "Same document.",
        encoding="utf-8",
    )

    content_hash = hashlib.sha256(
        b"Same document."
    ).hexdigest()

    fake_store.client.tables["documents"].append({
        "id": "existing-doc",
        "filename": "original.txt",
        "content_hash": content_hash,
        "status": "completed",
    })

    with patch(
        "app.ingestion.pipeline.ollama.embed"
    ) as mock_embed:
        report = pipeline.ingest(file_path)

    assert report.document_id == "existing-doc"
    assert report.status == "completed"
    assert report.chunks_created == 0
    assert "skipped" in report.message.lower()

    mock_embed.assert_not_called()
    assert fake_store.inserted_chunks == []


@pytest.mark.integration
def test_empty_content_after_cleaning_is_rejected(
    pipeline, tmp_path
):
    file_path = tmp_path / "empty_after_cleaning.txt"
    file_path.write_bytes(b"\x00")

    with pytest.raises(
        ValueError,
        match="no usable text",
    ):
        pipeline.ingest(file_path)


@pytest.mark.integration
def test_unsupported_file_type_is_rejected(
    pipeline, tmp_path
):
    file_path = tmp_path / "sample.exe"
    file_path.write_text(
        "Some content",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported file type",
    ):
        pipeline.ingest(file_path)


@pytest.mark.integration
def test_missing_file_is_rejected(pipeline, tmp_path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        pipeline.ingest(file_path)


@pytest.mark.integration
def test_embedding_dimension_mismatch_marks_document_failed(
    pipeline, fake_store, tmp_path
):
    file_path = tmp_path / "dimension.txt"
    file_path.write_text(
        "Test embedding dimensions.",
        encoding="utf-8",
    )

    invalid_embeddings = SimpleNamespace(
        embeddings=[[0.1] * 512]
    )

    with patch(
        "app.ingestion.pipeline.ollama.embed",
        return_value=invalid_embeddings,
    ):
        with pytest.raises(
            RuntimeError,
            match="Embedding dimension mismatch",
        ):
            pipeline.ingest(file_path)

    document = fake_store.client.tables["documents"][0]

    assert document["status"] == "failed"
    assert fake_store.inserted_chunks == []
    assert fake_store.client.tables["document_chunks"] == []


@pytest.mark.integration
def test_ollama_failure_marks_document_failed(
    pipeline, fake_store, tmp_path
):
    file_path = tmp_path / "ollama_failure.txt"
    file_path.write_text(
        "Test Ollama failure.",
        encoding="utf-8",
    )

    with patch(
        "app.ingestion.pipeline.ollama.embed",
        side_effect=ConnectionError("Ollama unavailable"),
    ):
        with pytest.raises(
            RuntimeError,
            match="Ingestion failed",
        ):
            pipeline.ingest(file_path)

    document = fake_store.client.tables["documents"][0]

    assert document["status"] == "failed"
    assert fake_store.client.tables["document_chunks"] == []


@pytest.mark.integration
def test_embedding_count_mismatch_marks_document_failed(
    pipeline, fake_store, tmp_path
):
    file_path = tmp_path / "count_mismatch.txt"
    file_path.write_text(
        "Test embedding count.",
        encoding="utf-8",
    )

    invalid_response = SimpleNamespace(
        embeddings=[]
    )

    with patch(
        "app.ingestion.pipeline.ollama.embed",
        return_value=invalid_response,
    ):
        with pytest.raises(
            RuntimeError,
            match="different number of embeddings",
        ):
            pipeline.ingest(file_path)

    document = fake_store.client.tables["documents"][0]

    assert document["status"] == "failed"
    assert fake_store.client.tables["document_chunks"] == []


@pytest.mark.integration
def test_metadata_is_preserved_during_ingestion(
    pipeline, fake_store, tmp_path
):
    file_path = tmp_path / "metadata.txt"
    file_path.write_text(
        "Metadata preservation test.",
        encoding="utf-8",
    )

    with patch(
        "app.ingestion.pipeline.ollama.embed",
        return_value=mock_embedding_response(1),
    ):
        pipeline.ingest(file_path)

    chunk = fake_store.inserted_chunks[0]

    assert chunk.metadata["filename"] == "metadata.txt"
    assert chunk.metadata["file_type"] == ".txt"
    assert chunk.metadata["file_size"] > 0


@pytest.mark.integration
def test_partial_chunk_insertion_is_cleaned_up(
    pipeline, fake_store, tmp_path
):
    file_path = tmp_path / "partial_failure.txt"

    # Generate enough content for more than 100 chunks.
    file_path.write_text(
        "This is a test sentence for chunk insertion recovery. " * 5000,
        encoding="utf-8",
    )

    # Simulate failure during the second batch.
    fake_store.fail_on_call = 2

    with patch(
        "app.ingestion.pipeline.ollama.embed",
        side_effect=lambda model, input: mock_embedding_response(
            len(input)
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="Simulated batch insertion failure",
        ):
            pipeline.ingest(file_path)

    documents = fake_store.client.tables["documents"]
    chunks = fake_store.client.tables["document_chunks"]

    # Document record should remain available for retry.
    assert len(documents) == 1
    assert documents[0]["status"] == "failed"

    # Previously inserted batch must be removed.
    assert len(chunks) == 0

    # First batch succeeded and second batch failed.
    assert fake_store.add_chunks_calls == 2
