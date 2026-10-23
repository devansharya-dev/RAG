"""Unit tests for content hashing utilities."""

import pytest
from app.utils.hashing import compute_sha256


@pytest.mark.unit
def test_compute_sha256_deterministic() -> None:
    text = "LocalRAG test content"
    hash1 = compute_sha256(text)
    hash2 = compute_sha256(text)
    assert hash1 == hash2
    assert len(hash1) == 64


@pytest.mark.unit
def test_compute_sha256_bytes_and_str_match() -> None:
    text = "Identical content"
    assert compute_sha256(text) == compute_sha256(text.encode("utf-8"))
