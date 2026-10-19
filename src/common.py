"""Shared data loading and evaluation for the 12S4054 project (Track E).

Protocol (warm-start, validation):
  * models are fit on interactions_train.csv only
  * candidates  = candidate_items_warm.csv
  * evaluable   = evaluable_users_validation_warm.csv
  * relevant    = rating >= 4 in interactions_validation_warm.csv
  * items the user interacted with in training are excluded from the list
"""
import os
import numpy as np
import pandas as pd

RELEVANCE_THRESHOLD = 4.0
ACTIVITY_BINS = [0, 4, 9, np.inf]                      # provisional, frozen at CP3
ACTIVITY_LABELS = ["low (1-4)", "medium (5-9)", "high (>=10)"]


def load_data(data_dir):
    main = os.path.join(data_dir, "MAIN DATA")
    sup = os.path.join(data_dir, "SUPPORTING_DATA")
    return {
        "train": pd.read_csv(os.path.join(main, "interactions_train.csv")),
        "candidates": pd.read_csv(os.path.join(main, "candidate_items_warm.csv")).item_id.tolist(),
        "val_warm": pd.read_csv(os.path.join(sup, "interactions_validation_warm.csv")),
        "eval_users": pd.read_csv(os.path.join(sup, "evaluable_users_validation_warm.csv")).user_id.tolist(),
    }


def seen_items(train):
    """user_id -> set of items interacted with in training (any rating)."""
    return train.groupby("user_id").item_id.agg(set).to_dict()


def relevant_items(val, users):
    pos = val[(val.rating >= RELEVANCE_THRESHOLD) & (val.user_id.isin(set(users)))]
    return pos.groupby("user_id").item_id.agg(set).to_dict()


def activity_groups(train, users):
    counts = train.groupby("user_id").size().reindex(users)
    return pd.cut(counts, ACTIVITY_BINS, labels=ACTIVITY_LABELS)


# ---------- metrics for one user (binary relevance) ----------
def recall_at_k(rec, rel, k):
    return len(set(rec[:k]) & rel) / len(rel)

def ndcg_at_k(rec, rel, k):
    dcg = sum(1.0 / np.log2(i + 2) for i, it in enumerate(rec[:k]) if it in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg

def hit_at_k(rec, rel, k):
    return float(any(it in rel for it in rec[:k]))

def mrr_at_k(rec, rel, k):
    for i, it in enumerate(rec[:k]):
        if it in rel:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(recs, rel, k=10):
    """recs: user -> ranked list; rel: user -> set. Returns per-user DataFrame."""
    rows = []
    for u, r in rel.items():
        lst = recs[u]
        rows.append({"user_id": u,
                     f"Recall@{k}": recall_at_k(lst, r, k),
                     f"NDCG@{k}": ndcg_at_k(lst, r, k),
                     f"HitRate@{k}": hit_at_k(lst, r, k),
                     f"MRR@{k}": mrr_at_k(lst, r, k)})
    return pd.DataFrame(rows).set_index("user_id")


def catalog_coverage(recs, candidates, k=10):
    shown = {it for lst in recs.values() for it in lst[:k]}
    return len(shown) / len(candidates)
