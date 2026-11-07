"""Unit tests for the CP3 evaluation protocol."""

import pandas as pd

from src.eval_protocol import (
    build_seen_items,
    build_relevant_items,
    build_candidate_set,
    validate_recommendations,
)


def test_seen_items():
    train = pd.DataFrame({
        "user_id": [1, 1, 2],
        "item_id": ["A", "B", "C"],
        "rating": [5, 3, 4],
        "timestamp": [1, 2, 3],
    })

    seen = build_seen_items(train)

    assert seen[1] == {"A", "B"}
    assert seen[2] == {"C"}


def test_relevant_items():
    val = pd.DataFrame({
        "user_id": [1, 1, 2],
        "item_id": ["A", "B", "C"],
        "rating": [5, 3, 4],
        "timestamp": [1, 2, 3],
    })

    relevant = build_relevant_items(val, [1, 2])

    assert relevant[1] == {"A"}
    assert relevant[2] == {"C"}


def test_candidate_filtering():
    seen = {1: {"A", "B"}}

    candidates = build_candidate_set(
        ["A", "B", "C", "D"],
        seen,
        1
    )

    assert candidates == {"C", "D"}


def test_recommendation_validation():
    recs = {1: ["C", "D"]}
    candidates = ["A", "B", "C", "D"]
    seen = {1: {"A", "B"}}

    assert validate_recommendations(
        recs, candidates, seen, k=2
    )


if __name__ == "__main__":
    test_seen_items()
    test_relevant_items()
    test_candidate_filtering()
    test_recommendation_validation()

    print("All evaluation protocol tests passed.")