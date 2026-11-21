"""Jaccard@10 overlap between the Top-10 lists of two runs (Track E extra metric).

Shows whether TF-IDF and sentence embedding recommend different items or only
reorder the same ones. Reads results/<run_id>_top10.csv written by src/run.py.

Usage:  python scripts/jaccard_overlap.py E3-B_tfidf_subsetall E4-B_embedding_subsetall
"""

import json
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_lists(run_id):
    path = RESULTS / f"{run_id}_top10.csv"
    if not path.exists():
        sys.exit(f"Missing {path}. Re-run that experiment with src.run to create it.")
    df = pd.read_csv(path, dtype={"item_id": str})
    return df.groupby("user_id")["item_id"].apply(set)


def main(run_a, run_b):
    a, b = load_lists(run_a), load_lists(run_b)
    users = a.index.intersection(b.index)
    if len(users) == 0:
        sys.exit("The two runs share no users.")

    jac = pd.Series({u: len(a[u] & b[u]) / len(a[u] | b[u]) for u in users})

    result = {
        "run_a": run_a,
        "run_b": run_b,
        "n_users": int(len(users)),
        "jaccard@10_mean": round(float(jac.mean()), 4),
        "share_users_identical_lists": round(float((jac == 1).mean()), 4),
        "share_users_no_overlap": round(float((jac == 0).mean()), 4),
    }

    out = RESULTS / f"jaccard_{run_a}_vs_{run_b}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
