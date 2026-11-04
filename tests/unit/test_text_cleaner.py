"""Unit tests for text cleaner module."""

import pytest

from app.ingestion.text_cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


@pytest.mark.unit
def test_cleaner_module_exists():
    assert TextCleaner is not None


@pytest.mark.unit
def test_empty_text(cleaner):
    assert cleaner.clean("") == ""


@pytest.mark.unit
def test_whitespace_only_text(cleaner):
    assert cleaner.clean("   \n   \n   ") == ""


@pytest.mark.unit
def test_normalizes_line_endings(cleaner):
    text = "First line\r\nSecond line\rThird line"

    assert cleaner.clean(text) == (
        "First line\nSecond line\nThird line"
    )


@pytest.mark.unit
def test_preserves_paragraphs(cleaner):
    text = "First paragraph.\n\nSecond paragraph."

    assert cleaner.clean(text) == text


@pytest.mark.unit
def test_preserves_headings(cleaner):
    text = "# Heading\n\nThis is content."

    assert cleaner.clean(text) == text


@pytest.mark.unit
def test_removes_excessive_blank_lines(cleaner):
    text = "First paragraph.\n\n\n\nSecond paragraph."

    assert cleaner.clean(text) == (
        "First paragraph.\n\nSecond paragraph."
    )


@pytest.mark.unit
def test_removes_null_bytes(cleaner):
    text = "Hello\x00 World"

    assert cleaner.clean(text) == "Hello World"


@pytest.mark.unit
def test_preserves_indentation(cleaner):
    text = "Heading\n    Indented content"

    assert cleaner.clean(text) == text


@pytest.mark.unit
def test_removes_trailing_whitespace(cleaner):
    text = "First line   \nSecond line  "

    assert cleaner.clean(text) == "First line\nSecond line"


@pytest.mark.unit
def test_normalizes_unicode(cleaner):
    text = "Cafe\u0301"

    assert cleaner.clean(text) == "Café"


@pytest.mark.unit
def test_rejects_non_string_input(cleaner):
    with pytest.raises(TypeError):
        cleaner.clean(None)


@pytest.mark.unit
def test_clean_text_alias(cleaner):
    assert cleaner.clean_text("Hello   World") == (
        cleaner.clean("Hello   World")
    )