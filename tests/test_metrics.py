"""Hand-checkable tests for Track E CP3 ranking metrics."""

import math
import pytest

from src.common import hit_at_k, mrr_at_k
from src.metrics import (
    recall_at_k,
    ndcg_at_k,
    evaluate,
    summarize_metrics,
)


def test_hand_checkable_metrics():
    recommendations = ["A", "B", "C", "D"]
    relevant = {"B", "D", "X"}
    k = 4

    assert math.isclose(
        recall_at_k(recommendations, relevant, k),
        2 / 3,
    )

    assert math.isclose(
        ndcg_at_k(recommendations, relevant, k),
        0.4982,
        rel_tol=1e-3,
    )

    assert hit_at_k(recommendations, relevant, k) == 1.0
    assert mrr_at_k(recommendations, relevant, k) == 0.5


def test_perfect_ranking():
    recommendations = ["B", "D", "X"]
    relevant = {"B", "D", "X"}

    assert ndcg_at_k(recommendations, relevant, 3) == 1.0


def test_no_relevant_recommendation():
    assert recall_at_k(["A", "C"], {"B", "D"}, 2) == 0.0
    assert ndcg_at_k(["A", "C"], {"B", "D"}, 2) == 0.0


def test_evaluate_and_summary():
    recommendations = {
        "u1": ["A", "B", "C", "D"],
    }

    relevant = {
        "u1": {"B", "D", "X"},
    }

    per_user = evaluate(recommendations, relevant, k=4)
    summary = summarize_metrics(per_user)

    assert math.isclose(summary["Recall@4"], 2 / 3)
    assert math.isclose(summary["NDCG@4"], 0.4982, rel_tol=1e-3)


def test_missing_recommendations():
    with pytest.raises(ValueError):
        evaluate({}, {"u1": {"A"}}, k=10)


def test_invalid_k():
    with pytest.raises(ValueError):
        recall_at_k(["A"], {"A"}, 0)

    with pytest.raises(ValueError):
        ndcg_at_k(["A"], {"A"}, 0)