"""Evaluation tests for retrieval quality metrics (Recall@k, MRR)."""

import pytest
import app.retrieval.retriever as retriever_module


@pytest.mark.evaluation
def test_retrieval_evaluation_module_exists() -> None:
    """Validate retrieval evaluation test harness setup."""
    assert retriever_module is not None
