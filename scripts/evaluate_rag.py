
"""Evaluate the LocalRAG pipeline using a labelled evaluation dataset."""

import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.metrics import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    reciprocal_rank,
)
from app.retrieval.retriever import Retriever
from app.schemas.chat import ChatRequest
from app.services.rag_service import RAGService


DATASET_PATH = (
    PROJECT_ROOT / "tests" / "evaluation" / "datasets" / "rag_eval.json"
)
REPORT_PATH = PROJECT_ROOT / "reports" / "rag_eval_report.json"

TOP_K = 4
EVALUATION_K_VALUES = (1, 2, 4)


class RecordingRetriever:
    """Capture retrieval results without performing retrieval twice."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.last_results = []

    def retrieve(self, query, top_k=None, filters=None):
        self.last_results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
        )
        return self.last_results


def normalize(value: str) -> str:
    """Normalize text for basic expected-answer matching."""
    return " ".join(value.casefold().split())


def get_filename(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    return metadata.get("filename", "Unknown")


def evaluate():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8-sig")
    )

    if not isinstance(dataset, list) or not dataset:
        raise ValueError(
            "Evaluation dataset must be a non-empty JSON list."
        )

    recording_retriever = RecordingRetriever(Retriever())
    service = RAGService(retriever=recording_retriever)

    results = []
    retrieval_scores = []
    reciprocal_ranks = []

    retrieval_metrics = {
        k: {
            "precision": [],
            "recall": [],
            "hit_rate": [],
        }
        for k in EVALUATION_K_VALUES
    }

    answerable_correct = 0
    answerable_total = 0
    unanswerable_correct = 0
    unanswerable_total = 0
    grounded_count = 0
    total_latency = 0.0

    print("\n========== LOCALRAG EVALUATION ==========\n")

    for item in dataset:
        question_id = item["id"]
        question = item["question"]
        expected_answer = item.get("expected_answer")
        relevant_filenames = item.get("relevant_filenames", [])
        answerable = item["answerable"]

        if answerable and not relevant_filenames:
            raise ValueError(
                f"{question_id}: Answerable question has no relevant filenames."
            )

        if not answerable and relevant_filenames:
            raise ValueError(
                f"{question_id}: Unanswerable question has relevant filenames."
            )

        start_time = perf_counter()

        response = service.answer(
            ChatRequest(
                question=question,
                top_k=TOP_K,
            )
        )

        latency = perf_counter() - start_time
        total_latency += latency

        retrieved_chunks = recording_retriever.last_results

        # Evaluate at document level, not individual chunk level.
        retrieved_filenames = list(dict.fromkeys(
            get_filename(chunk)
            for chunk in retrieved_chunks
        ))

        if answerable:
            recall = recall_at_k(
                retrieved_filenames,
                relevant_filenames,
                TOP_K,
            )

            rank = reciprocal_rank(
                retrieved_filenames,
                relevant_filenames,
            )

            retrieval_scores.append(recall)
            reciprocal_ranks.append(rank)

            answerable_total += 1

            for k in EVALUATION_K_VALUES:
                retrieval_metrics[k]["precision"].append(
                    precision_at_k(
                        retrieved_filenames,
                        relevant_filenames,
                        k,
                    )
                )

                retrieval_metrics[k]["recall"].append(
                    recall_at_k(
                        retrieved_filenames,
                        relevant_filenames,
                        k,
                    )
                )

                retrieval_metrics[k]["hit_rate"].append(
                    hit_rate_at_k(
                        retrieved_filenames,
                        relevant_filenames,
                        k,
                    )
                )

            correct = (
                response.grounded
                and expected_answer is not None
                and normalize(expected_answer)
                in normalize(response.answer)
            )

            if correct:
                answerable_correct += 1

            answer_status = (
                "CORRECT" if correct else "INCORRECT"
            )

        else:
            recall = None
            rank = None
            unanswerable_total += 1

            correct = response.grounded is False

            if correct:
                unanswerable_correct += 1

            answer_status = (
                "CORRECT_ABSTENTION"
                if correct
                else "HALLUCINATION_OR_FAILURE"
            )

        if response.grounded:
            grounded_count += 1

        result = {
            "id": question_id,
            "question": question,
            "answerable": answerable,
            "expected_answer": expected_answer,
            "generated_answer": response.answer,
            "grounded": response.grounded,
            "answer_correct": correct,
            "retrieved_filenames": retrieved_filenames,
            "relevant_filenames": relevant_filenames,
            "recall_at_k": recall,
            "reciprocal_rank": rank,
            "latency_seconds": round(latency, 4),
            "status": answer_status,
        }

        results.append(result)

        print(f"[{question_id}] {answer_status}")
        print(f"Question: {question}")
        print(f"Answer: {response.answer}")
        print(f"Retrieved: {retrieved_filenames}")
        print(f"Grounded: {response.grounded}")
        print(f"Latency: {latency:.2f}s")
        print("-" * 55)

    answerable_accuracy = (
        answerable_correct / answerable_total
        if answerable_total else 0.0
    )

    abstention_accuracy = (
        unanswerable_correct / unanswerable_total
        if unanswerable_total else 0.0
    )

    retrieval_metrics_report = {
        str(k): {
            metric: (
                sum(scores) / len(scores)
                if scores else 0.0
            )
            for metric, scores in values.items()
        }
        for k, values in retrieval_metrics.items()
    }

    report = {
        "dataset_size": len(dataset),
        "top_k": TOP_K,
        "metrics": {
            "recall_at_k": (
                sum(retrieval_scores) / len(retrieval_scores)
                if retrieval_scores else 0.0
            ),
            "mean_reciprocal_rank": (
                sum(reciprocal_ranks) / len(reciprocal_ranks)
                if reciprocal_ranks else 0.0
            ),
            "answer_accuracy": answerable_accuracy,
            "abstention_accuracy": abstention_accuracy,
            "grounded_answer_rate": (
                grounded_count / len(dataset)
            ),
            "average_latency_seconds": (
                total_latency / len(dataset)
            ),
        },
        "retrieval_metrics_at_k": retrieval_metrics_report,
        "counts": {
            "answerable_questions": answerable_total,
            "correct_answers": answerable_correct,
            "unanswerable_questions": unanswerable_total,
            "correct_abstentions": unanswerable_correct,
            "grounded_answers": grounded_count,
        },
        "results": results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== EVALUATION SUMMARY ==========\n")

    for metric, value in report["metrics"].items():
        print(f"{metric}: {value:.4f}")

    print("\n========== RETRIEVAL METRICS ==========\n")

    for k, values in retrieval_metrics_report.items():
        print(f"K = {k}")

        for metric, score in values.items():
            print(f"  {metric}: {score:.4f}")

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate()
