"""Unit tests for prompt builder module."""

import pytest
import app.generation.prompt_builder as prompt_builder_module


@pytest.mark.unit
def test_prompt_builder_module_exists() -> None:
    """Validate prompt builder module exists and is importable."""
    assert prompt_builder_module is not None
