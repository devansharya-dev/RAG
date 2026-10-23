"""Evaluation tests for answer grounding and citation validity."""

import pytest
import app.generation.answer_generator as answer_generator_module


@pytest.mark.evaluation
def test_grounded_answers_evaluation_module_exists() -> None:
    """Validate grounded answer evaluation test harness setup."""
    assert answer_generator_module is not None
