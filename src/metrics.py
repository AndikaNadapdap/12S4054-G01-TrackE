"""Ranking evaluation metrics for Track E CP3."""

import numpy as np
import pandas as pd


def recall_at_k(recommendations, relevant_items, k=10):
    """Calculate Recall@K for one user."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_items:
        return 0.0

    recommended = set(recommendations[:k])

    return (
        len(recommended & set(relevant_items))
        / len(relevant_items)
    )


def ndcg_at_k(recommendations, relevant_items, k=10):
    """Calculate binary NDCG@K for one user."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_items:
        return 0.0

    relevant = set(relevant_items)

    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(recommendations[:k])
        if item in relevant
    )

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommendations, relevant_items, k=10):
    """1 if at least one relevant item is in the Top-K, else 0."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant = set(relevant_items)

    return float(any(item in relevant for item in recommendations[:k]))


def mrr_at_k(recommendations, relevant_items, k=10):
    """Reciprocal rank of the first relevant item in the Top-K (0 if none)."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant = set(relevant_items)

    for rank, item in enumerate(recommendations[:k]):
        if item in relevant:
            return 1.0 / (rank + 1)

    return 0.0


def catalog_coverage(recommendations, candidates, k=10):
    """Share of candidate items that appear in at least one user's Top-K."""

    shown = {
        item
        for recs in recommendations.values()
        for item in recs[:k]
    }

    return len(shown) / len(candidates)


def evaluate(recommendations, relevant, k=10):
    """
    Calculate per-user Recall@K, NDCG@K, HitRate@K and MRR@K.

    Returns a pandas DataFrame indexed by user_id.
    """

    missing_users = set(relevant) - set(recommendations)

    if missing_users:
        raise ValueError(
            f"Missing recommendations for {len(missing_users)} users."
        )

    rows = []

    for user_id, relevant_items in relevant.items():
        recs = recommendations[user_id]

        rows.append({
            "user_id": user_id,
            f"Recall@{k}": recall_at_k(recs, relevant_items, k),
            f"NDCG@{k}": ndcg_at_k(recs, relevant_items, k),
            f"HitRate@{k}": hit_rate_at_k(recs, relevant_items, k),
            f"MRR@{k}": mrr_at_k(recs, relevant_items, k),
        })

    return pd.DataFrame(rows).set_index("user_id")


def summarize_metrics(per_user):
    """Calculate mean ranking metrics across users."""

    return per_user.mean().to_dict()