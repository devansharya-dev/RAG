
"""Metrics for evaluating information retrieval quality."""

from collections.abc import Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")


def recall_at_k(
    retrieved_ids: Sequence[T],
    relevant_ids: Iterable[T],
    k: int,
) -> float:
    """Calculate the fraction of relevant items retrieved in top K."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = set(retrieved_ids[:k])

    return len(retrieved & relevant) / len(relevant)


def precision_at_k(
    retrieved_ids: Sequence[T],
    relevant_ids: Iterable[T],
    k: int,
) -> float:
    """Calculate relevant retrieved results divided by K."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = retrieved_ids[:k]

    relevant_count = sum(
        1 for item in retrieved if item in relevant
    )

    return relevant_count / k


def hit_rate_at_k(
    retrieved_ids: Sequence[T],
    relevant_ids: Iterable[T],
    k: int,
) -> float:
    """Return 1 if at least one relevant result appears in top K."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    return float(any(
        item in relevant for item in retrieved_ids[:k]
    ))


def reciprocal_rank(
    retrieved_ids: Sequence[T],
    relevant_ids: Iterable[T],
) -> float:
    """Return the reciprocal rank of the first relevant result."""
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    results: Sequence[Sequence[T]],
    relevant_sets: Sequence[Iterable[T]],
) -> float:
    """Calculate MRR across multiple queries."""
    if len(results) != len(relevant_sets):
        raise ValueError(
            "Each query result must have a corresponding relevant set."
        )

    if not results:
        return 0.0

    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(results, relevant_sets)
    ]

    return sum(scores) / len(scores)
