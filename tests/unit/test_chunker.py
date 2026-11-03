
import pytest

from app.ingestion.chunker import Chunker


@pytest.mark.unit
def test_empty_text_returns_empty_list():
    chunker = Chunker(chunk_size=100, chunk_overlap=20)

    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   ") == []


@pytest.mark.unit
def test_short_text_creates_single_chunk():
    chunker = Chunker(chunk_size=100, chunk_overlap=20)

    chunks = chunker.chunk_text("This is a short document.")

    assert len(chunks) == 1
    assert chunks[0].content == "This is a short document."
    assert chunks[0].chunk_index == 0


@pytest.mark.unit
def test_chunk_indices_are_sequential():
    chunker = Chunker(chunk_size=50, chunk_overlap=10)

    text = (
        "First sentence is here. "
        "Second sentence is here. "
        "Third sentence is here. "
        "Fourth sentence is here."
    )

    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


@pytest.mark.unit
def test_chunk_preserves_metadata():
    chunker = Chunker(chunk_size=100, chunk_overlap=20)

    metadata = {
        "filename": "test.pdf",
        "page_number": 1,
    }

    chunks = chunker.chunk_text(
        "Project Nebula uses 48 processing nodes.",
        metadata=metadata,
    )

    assert chunks[0].metadata == metadata

    # Modifying one chunk's metadata must not affect the input.
    chunks[0].metadata["filename"] = "changed.pdf"
    assert metadata["filename"] == "test.pdf"


@pytest.mark.unit
def test_chunk_size_limit():
    chunker = Chunker(chunk_size=50, chunk_overlap=10)

    text = " ".join(
        f"Sentence number {i} contains some useful information."
        for i in range(1, 15)
    )

    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 50 for chunk in chunks)


@pytest.mark.unit
def test_overlap_preserves_previous_context():
    chunker = Chunker(chunk_size=40, chunk_overlap=10)

    text = (
        "Alpha sentence is here. "
        "Bravo sentence is here. "
        "Charlie sentence is here."
    )

    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1

    # Some text from the previous chunk should appear in the next.
    assert any(
        previous.content[-10:] in current.content
        for previous, current in zip(chunks, chunks[1:])
    )


@pytest.mark.unit
def test_long_sentence_is_split_without_exceeding_limit():
    chunker = Chunker(chunk_size=30, chunk_overlap=5)

    text = "A" * 100

    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 30 for chunk in chunks)


@pytest.mark.unit
def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        Chunker(chunk_size=0, chunk_overlap=0)


@pytest.mark.unit
def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        Chunker(chunk_size=100, chunk_overlap=100)

@pytest.mark.unit
def test_long_text_does_not_silently_drop_words():
    chunker = Chunker(chunk_size=32, chunk_overlap=8)

    words = [f"word{i:03d}" for i in range(30)]
    text = " ".join(words)

    chunks = chunker.chunk_text(text)
    combined_content = " ".join(chunk.content for chunk in chunks)

    missing_words = [
        word for word in words
        if word not in combined_content
    ]

    assert missing_words == [], (
        f"Chunking silently dropped these words: {missing_words}"
    )

