"""Evaluation protocol for Track E CP3."""

import pandas as pd

RELEVANCE_THRESHOLD = 4.0

ACTIVITY_BINS = [0, 4, 9, float("inf")]

ACTIVITY_LABELS = [
    "low (1-4)",
    "medium (5-9)",
    "high (>=10)"
]


def build_seen_items(train):
    """
    Build user history from training interactions.

    All training interactions are considered seen,
    regardless of rating.
    """
    return (
        train.groupby("user_id")["item_id"]
        .agg(set)
        .to_dict()
    )


def build_relevant_items(val_warm, eval_users):
    """
    Identify relevant validation items.

    An item is relevant when rating >= 4.
    """
    eval_users = set(eval_users)

    relevant = val_warm[
        (val_warm["rating"] >= RELEVANCE_THRESHOLD)
        & (val_warm["user_id"].isin(eval_users))
    ]

    return (
        relevant.groupby("user_id")["item_id"]
        .agg(set)
        .to_dict()
    )


def get_evaluable_users(train, val_warm, eval_users):
    """
    Keep users with training history and
    at least one relevant warm-validation item.
    """
    train_users = set(train["user_id"].unique())

    relevant = build_relevant_items(val_warm, eval_users)

    valid_users = (
        set(eval_users)
        & train_users
        & set(relevant.keys())
    )

    return sorted(valid_users)


def build_activity_groups(train, users):
    """
    Group users based on the number of training interactions.

    Low: 1-4
    Medium: 5-9
    High: >=10
    """
    counts = (
        train.groupby("user_id")
        .size()
        .reindex(users)
    )

    groups = pd.cut(
        counts,
        bins=ACTIVITY_BINS,
        labels=ACTIVITY_LABELS
    )

    return groups


def build_candidate_set(candidates, seen_items, user_id):
    """
    Remove items already interacted with during training
    from the warm-start candidate universe.
    """
    candidate_set = set(candidates)
    seen = seen_items.get(user_id, set())

    return candidate_set - seen


def validate_recommendations(
    recommendations,
    candidates,
    seen_items,
    k=10
):
    """
    Check recommendation protocol:
    - exactly k items per user
    - all items belong to warm candidates
    - no training item is recommended again
    """
    candidate_set = set(candidates)

    for user_id, recs in recommendations.items():

        assert len(recs) == k, (
            f"User {user_id}: expected {k} recommendations, "
            f"got {len(recs)}"
        )

        assert len(set(recs)) == len(recs), (
            f"User {user_id}: duplicate recommendations found"
        )

        assert set(recs).issubset(candidate_set), (
            f"User {user_id}: recommendation outside candidate universe"
        )

        assert not (
            set(recs) & seen_items.get(user_id, set())
        ), (
            f"User {user_id}: training item leakage detected"
        )

    return True