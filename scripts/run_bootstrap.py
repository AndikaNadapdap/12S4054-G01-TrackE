"""W11: paired bootstrap for the main model comparisons (all users and per activity group).

Reads the per-user results of the four selected runs (results/<run_id>_per_user.csv,
written by src.run and git-ignored), pairs users by id, and writes
results/bootstrap_pairs.json. 1,000 resamples, seed 42, 95% CI (src/bootstrap.py).
The pair E4-B vs E3-B is included so the output can be checked against
results/bootstrap_E4-B_vs_E3-B.json.

Usage:  python scripts/run_bootstrap.py [--results_dir results]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.bootstrap import paired_bootstrap

RUNS = {
    "E1": "E1_popularity_subsetall",
    "E2-07": "E2-07_als_subsetall",
    "E3-B": "E3-B_tfidf_subsetall",
    "E4-B": "E4-B_embedding_subsetall",
}
# (A, B): the reported difference is A minus B.
PAIRS = [("E2-07", "E1"), ("E2-07", "E3-B"), ("E2-07", "E4-B"),
         ("E3-B", "E1"), ("E4-B", "E1"), ("E4-B", "E3-B")]
METRICS = ["NDCG@10", "Recall@10"]
GROUPS = ["all", "low (1-4)", "medium (5-9)", "high (>=10)"]

p = argparse.ArgumentParser()
p.add_argument("--results_dir", default="results")
p.add_argument("--out", default="results/bootstrap_pairs.json")
a = p.parse_args()
res = Path(a.results_dir)

per_user = {e: pd.read_csv(res / f"{r}_per_user.csv").set_index("user_id") for e, r in RUNS.items()}
out = {"n_resamples": 1000, "seed": 42, "confidence_level": 0.95, "comparisons": {}}
for ea, eb in PAIRS:
    A, B = per_user[ea], per_user[eb]
    B = B.loc[A.index]
    block = {}
    for g in GROUPS:
        mask = (A["activity_group"] == g) if g != "all" else pd.Series(True, index=A.index)
        for m in METRICS:
            r = paired_bootstrap(A.loc[mask, m].values, B.loc[mask, m].values, n_resamples=1000, seed=42)
            block[f"{g}|{m}"] = {k: (round(v, 5) if isinstance(v, float) else v)
                                 for k, v in r.items() if k in ("mean_difference", "ci_lower", "ci_upper", "n_users", "significant")}
    out["comparisons"][f"{ea} minus {eb}"] = block
Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
print("Saved", a.out)
for name, block in out["comparisons"].items():
    v = block["all|NDCG@10"]
    print(f"{name:16s} NDCG@10 diff {v['mean_difference']:+.5f}  CI [{v['ci_lower']:+.5f}, {v['ci_upper']:+.5f}]  significant={v['significant']}")
