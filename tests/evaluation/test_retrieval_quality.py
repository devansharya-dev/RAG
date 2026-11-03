
import pytest

from app.retrieval.metrics import (
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


@pytest.mark.evaluation
def test_recall_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = {"A", "C", "E"}

    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


@pytest.mark.evaluation
def test_recall_at_k_with_no_relevant_results():
    assert recall_at_k(["A", "B"], set(), 2) == 0.0


@pytest.mark.evaluation
def test_recall_at_k_with_empty_retrieval():
    assert recall_at_k([], {"A", "B"}, 3) == 0.0


@pytest.mark.evaluation
def test_recall_at_k_rejects_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k(["A"], {"A"}, 0)


@pytest.mark.evaluation
def test_reciprocal_rank():
    assert reciprocal_rank(["A", "B", "C"], {"C"}) == pytest.approx(1 / 3)


@pytest.mark.evaluation
def test_reciprocal_rank_returns_zero_when_no_match():
    assert reciprocal_rank(["A", "B"], {"C"}) == 0.0


@pytest.mark.evaluation
def test_mean_reciprocal_rank():
    results = [
        ["A", "B", "C"],
        ["X", "Y", "Z"],
        ["P", "Q", "R"],
    ]

    relevant_sets = [
        {"A"},
        {"Y"},
        {"M"},
    ]

    expected = (1 + 0.5 + 0) / 3

    assert mean_reciprocal_rank(
        results, relevant_sets
    ) == pytest.approx(expected)


@pytest.mark.evaluation
def test_mrr_rejects_mismatched_query_counts():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([["A"], ["B"]], [{"A"}])


@pytest.mark.evaluation
def test_mrr_with_empty_dataset():
    assert mean_reciprocal_rank([], []) == 0.0

