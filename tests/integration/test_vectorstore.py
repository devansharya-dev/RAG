
"""Tests for Supabase pgvector storage."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.schemas.document import DocumentChunk
from app.vectorstore.base import BaseVectorStore
from app.vectorstore.supabase_store import SupabaseVectorStore


class FakeTableQuery:
    """Minimal fake Supabase table query."""

    def __init__(self, client, table_name):
        self.client = client
        self.table_name = table_name
        self.operation = None
        self.payload = None
        self.filters = {}
        self.count_requested = False

    def insert(self, payload):
        self.operation = "insert"
        self.payload = payload
        return self

    def delete(self):
        self.operation = "delete"
        return self

    def select(self, *args, **kwargs):
        self.operation = "select"
        self.count_requested = kwargs.get("count") == "exact"
        return self

    def eq(self, column, value):
        self.filters[column] = value
        return self

    def execute(self):
        rows = self.client.tables.setdefault(self.table_name, [])

        if self.operation == "insert":
            if isinstance(self.payload, list):
                rows.extend(self.payload)
            else:
                rows.append(self.payload)

            return SimpleNamespace(data=self.payload)

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

        if self.operation == "select":
            return SimpleNamespace(
                data=[],
                count=len(rows) if self.count_requested else None,
            )

        raise NotImplementedError(self.operation)


class FakeRPCQuery:
    """Fake Supabase RPC query."""

    def __init__(self, client, function_name, params):
        self.client = client
        self.function_name = function_name
        self.params = params

    def execute(self):
        self.client.rpc_calls.append({
            "function": self.function_name,
            "params": self.params,
        })

        if self.client.rpc_error:
            raise self.client.rpc_error

        return SimpleNamespace(data=self.client.rpc_data)


class FakeSupabaseClient:
    """In-memory Supabase client."""

    def __init__(self):
        self.tables = {
            "document_chunks": [],
            "documents": [],
        }
        self.rpc_data = []
        self.rpc_calls = []
        self.rpc_error = None

    def table(self, table_name):
        return FakeTableQuery(self, table_name)

    def rpc(self, function_name, params):
        return FakeRPCQuery(self, function_name, params)


@pytest.fixture
def fake_client():
    return FakeSupabaseClient()


@pytest.fixture
def store(fake_client):
    vector_store = SupabaseVectorStore(
        supabase_url="https://example.supabase.co",
        supabase_key="test-key",
    )
    vector_store._client = fake_client
    return vector_store


def make_chunk(
    document_id="doc-1",
    chunk_index=0,
    content="Test document content",
    metadata=None,
):
    return DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        metadata=metadata or {},
    )


def make_embedding(dimension=768):
    return [0.1] * dimension


@pytest.mark.integration
def test_vectorstore_implements_base_class():
    assert issubclass(SupabaseVectorStore, BaseVectorStore)


@pytest.mark.integration
def test_client_is_created_lazily():
    store = SupabaseVectorStore(
        supabase_url="https://example.supabase.co",
        supabase_key="test-key",
    )

    with patch(
        "app.vectorstore.supabase_store.create_client"
    ) as mock_create:
        mock_create.return_value = object()

        assert store._client is None

        first_client = store.client
        second_client = store.client

        assert first_client is second_client
        mock_create.assert_called_once_with(
            "https://example.supabase.co",
            "test-key",
        )


@pytest.mark.integration
def test_constructor_rejects_missing_credentials():
    with pytest.raises(ValueError, match="must be configured"):
        SupabaseVectorStore(
            supabase_url="",
            supabase_key="",
        )


@pytest.mark.integration
def test_add_chunks_inserts_correct_data(store, fake_client):
    chunk = make_chunk(
        metadata={"filename": "test.txt"}
    )

    store.add_chunks([chunk], [make_embedding()])

    rows = fake_client.tables["document_chunks"]

    assert len(rows) == 1
    assert rows[0]["document_id"] == "doc-1"
    assert rows[0]["chunk_index"] == 0
    assert rows[0]["content"] == "Test document content"
    assert rows[0]["metadata"]["filename"] == "test.txt"
    assert len(rows[0]["embedding"]) == 768


@pytest.mark.integration
def test_add_chunks_accepts_dictionary_input(store, fake_client):
    chunk = {
        "document_id": "doc-2",
        "chunk_index": 1,
        "content": "Dictionary chunk",
        "metadata": {"source": "manual"},
    }

    store.add_chunks([chunk], [make_embedding()])

    rows = fake_client.tables["document_chunks"]

    assert len(rows) == 1
    assert rows[0]["document_id"] == "doc-2"
    assert rows[0]["content"] == "Dictionary chunk"


@pytest.mark.integration
def test_add_chunks_rejects_mismatched_counts(store):
    with pytest.raises(ValueError, match="exactly one embedding"):
        store.add_chunks(
            [make_chunk()],
            [],
        )


@pytest.mark.integration
def test_add_chunks_rejects_invalid_embedding_dimension(store):
    with pytest.raises(ValueError, match="768-dimensional"):
        store.add_chunks(
            [make_chunk()],
            [make_embedding(512)],
        )


@pytest.mark.integration
def test_add_chunks_rejects_missing_document_id(store):
    chunk = make_chunk(document_id=None)

    with pytest.raises(ValueError, match="document_id"):
        store.add_chunks(
            [chunk],
            [make_embedding()],
        )


@pytest.mark.integration
def test_add_chunks_rejects_missing_content(store):
    chunk = {
        "document_id": "doc-1",
        "chunk_index": 0,
        "content": None,
        "metadata": {},
    }

    with pytest.raises(ValueError, match="content and chunk_index"):
        store.add_chunks(
            [chunk],
            [make_embedding()],
        )


@pytest.mark.integration
def test_add_chunks_handles_empty_input(store, fake_client):
    store.add_chunks([], [])

    assert fake_client.tables["document_chunks"] == []


@pytest.mark.integration
def test_add_chunks_splits_large_batches(store, fake_client):
    chunks = [
        make_chunk(
            chunk_index=index,
            content=f"Chunk {index}",
        )
        for index in range(205)
    ]

    embeddings = [
        make_embedding()
        for _ in range(205)
    ]

    store.add_chunks(chunks, embeddings)

    rows = fake_client.tables["document_chunks"]

    assert len(rows) == 205
    assert rows[0]["chunk_index"] == 0
    assert rows[-1]["chunk_index"] == 204


@pytest.mark.integration
def test_similarity_search_returns_formatted_results(
    store, fake_client
):
    fake_client.rpc_data = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "content": "Relevant content",
            "metadata": {"filename": "test.txt"},
            "similarity": 0.91,
        }
    ]

    results = store.similarity_search(
        make_embedding(),
        top_k=4,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["document_id"] == "doc-1"
    assert results[0]["content"] == "Relevant content"
    assert results[0]["similarity"] == 0.91

    call = fake_client.rpc_calls[0]

    assert call["function"] == "match_document_chunks"
    assert call["params"]["match_count"] == 4
    assert call["params"]["match_threshold"] == 0.0


@pytest.mark.integration
def test_similarity_search_applies_metadata_filters(
    store, fake_client
):
    fake_client.rpc_data = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "content": "Wrong source",
            "metadata": {"filename": "other.txt"},
            "similarity": 0.95,
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-2",
            "content": "Correct source",
            "metadata": {"filename": "wanted.txt"},
            "similarity": 0.89,
        },
    ]

    results = store.similarity_search(
        make_embedding(),
        top_k=2,
        filters={"filename": "wanted.txt"},
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-2"

    call = fake_client.rpc_calls[0]

    assert call["params"]["match_count"] == 20


@pytest.mark.integration
@pytest.mark.parametrize("top_k", [0, -1, 21])
def test_similarity_search_rejects_invalid_top_k(store, top_k):
    with pytest.raises(ValueError, match="top_k"):
        store.similarity_search(
            make_embedding(),
            top_k=top_k,
        )


@pytest.mark.integration
def test_similarity_search_rejects_invalid_embedding(store):
    with pytest.raises(ValueError, match="768-dimensional"):
        store.similarity_search(
            make_embedding(384),
        )


@pytest.mark.integration
def test_similarity_search_handles_empty_results(
    store, fake_client
):
    fake_client.rpc_data = []

    results = store.similarity_search(
        make_embedding()
    )

    assert results == []


@pytest.mark.integration
def test_similarity_search_propagates_rpc_errors(
    store, fake_client
):
    fake_client.rpc_error = RuntimeError("RPC unavailable")

    with pytest.raises(RuntimeError, match="RPC unavailable"):
        store.similarity_search(make_embedding())


@pytest.mark.integration
def test_delete_by_document_id(store, fake_client):
    fake_client.tables["document_chunks"] = [
        {"document_id": "doc-1", "content": "A"},
        {"document_id": "doc-2", "content": "B"},
    ]

    fake_client.tables["documents"] = [
        {"id": "doc-1"},
        {"id": "doc-2"},
    ]

    store.delete_by_document_id("doc-1")

    assert fake_client.tables["document_chunks"] == [
        {"document_id": "doc-2", "content": "B"}
    ]

    assert fake_client.tables["documents"] == [
        {"id": "doc-2"}
    ]


@pytest.mark.integration
def test_get_stats_returns_document_and_chunk_counts(
    store, fake_client
):
    fake_client.tables["document_chunks"] = [
        {"id": "chunk-1"},
        {"id": "chunk-2"},
        {"id": "chunk-3"},
    ]

    fake_client.tables["documents"] = [
        {"id": "doc-1"},
        {"id": "doc-2"},
    ]

    stats = store.get_stats()

    assert stats == {
        "table": "document_chunks",
        "total_chunks": 3,
        "total_documents": 2,
    }


@pytest.mark.integration
def test_get_collection_stats_is_compatibility_alias(
    store, fake_client
):
    stats = store.get_collection_stats()

    assert stats == store.get_stats()
