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


# ---------- helpers shared by all baseline runners ----------
def load_items(data_dir):
    return pd.read_csv(os.path.join(data_dir, "MAIN DATA", "items.csv"))


def sample_users(users, n, seed=42):
    """Fixed random subset of evaluable users for quick sanity checks (n=0 -> all)."""
    if not n or n >= len(users):
        return list(users)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(sorted(users), size=n, replace=False).tolist())


def check_protocol(recs, candidates, seen, k):
    """Leakage / protocol assertions shared by every model."""
    cand = set(candidates)
    assert all(len(r) == k for r in recs.values()), "every list must contain exactly k items"
    assert all(it in cand for r in recs.values() for it in r), "items must be warm candidates"
    assert all(not (set(r) & seen.get(u, set())) for u, r in recs.items()), "training items leaked"


def save_report(name, recs, rel, train, candidates, k, out):
    """Evaluate, print, and save overall / per-group / per-user results."""
    import json
    per_user = evaluate(recs, rel, k)
    per_user["group"] = activity_groups(train, per_user.index.tolist()).values
    per_user.to_csv(f"{out}/{name}_per_user.csv")
    overall = per_user.drop(columns="group").mean().round(4).to_dict()
    overall.update({"catalog_coverage": round(catalog_coverage(recs, candidates, k), 4),
                    "n_users": len(per_user), "n_candidates": len(candidates)})
    by_group = per_user.groupby("group", observed=True).agg(
        n_users=("group", "size"), **{c: (c, "mean") for c in per_user.columns if "@" in c}).round(4)
    json.dump(overall, open(f"{out}/{name}_overall.json", "w"), indent=2)
    by_group.to_csv(f"{out}/{name}_by_group.csv")
    print(f"== {name} | warm validation ==")
    print(json.dumps(overall, indent=2))
    print(by_group.to_string())
    return overall, by_group


def write_examples(name, recs, train, items, rel, out, n_examples=3, seed=42, history=5):
    """Human-readable Top-10 examples (training history vs recommendations)."""
    title = items.set_index("item_id").title.fillna("(no title)").str.slice(0, 90).to_dict()
    users = sample_users(list(recs), n_examples, seed=seed + 1)
    lines = [f"# {name}: Top-10 sanity-check examples\n"]
    for u in users:
        hist = train[train.user_id == u].sort_values("timestamp").tail(history)
        lines.append(f"## User {u}\n\n**Last {len(hist)} training items (rating):**\n")
        lines += [f"1. {title.get(r.item_id, r.item_id)} ({r.rating:.0f})" for r in hist.itertuples()]
        lines.append("\n**Top-10 recommendations (✓ = relevant in validation):**\n")
        lines += [f"1. {'✓ ' if it in rel.get(u, set()) else ''}{title.get(it, it)}" for it in recs[u]]
        lines.append("")
    open(f"{out}/{name}_top10_examples.md", "w", encoding="utf-8").write("\n".join(lines))
