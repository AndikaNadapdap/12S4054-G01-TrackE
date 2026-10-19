"""Run the Popularity baseline on the warm-start validation protocol.

Usage: python run_popularity.py --data_dir /path/to/data --k 10
"""
import argparse, json, os
import pandas as pd
from src.common import (load_data, seen_items, relevant_items, activity_groups,
                        evaluate, catalog_coverage)
from src.popularity import PopularityRecommender

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default=".")
p.add_argument("--k", type=int, default=10)
p.add_argument("--out", default="results")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

d = load_data(a.data_dir)
seen = seen_items(d["train"])
rel = relevant_items(d["val_warm"], d["eval_users"])
assert set(rel) == set(d["eval_users"]), "every evaluable user must have >=1 relevant item"

model = PopularityRecommender().fit(d["train"], d["candidates"])
recs = {u: model.recommend(u, seen.get(u, set()), a.k) for u in d["eval_users"]}

# sanity checks (leakage / protocol)
cand = set(d["candidates"])
assert all(len(r) == a.k for r in recs.values())
assert all(it in cand for r in recs.values() for it in r)
assert all(not (set(r) & seen.get(u, set())) for u, r in recs.items())

per_user = evaluate(recs, rel, a.k)
per_user["group"] = activity_groups(d["train"], per_user.index.tolist()).values
per_user.to_csv(f"{a.out}/popularity_per_user.csv")          # kept for paired bootstrap later

overall = per_user.drop(columns="group").mean().round(4).to_dict()
overall.update({"catalog_coverage": round(catalog_coverage(recs, d["candidates"], a.k), 4),
                "n_users": len(per_user), "n_candidates": len(d["candidates"])})
by_group = per_user.groupby("group", observed=True).agg(
    n_users=("group", "size"), **{c: (c, "mean") for c in per_user.columns if "@" in c}).round(4)

json.dump(overall, open(f"{a.out}/popularity_overall.json", "w"), indent=2)
by_group.to_csv(f"{a.out}/popularity_by_group.csv")
print("== Popularity baseline | warm validation ==")
print(json.dumps(overall, indent=2))
print(by_group.to_string())
print("Top-10 global popular items:", model.ranking[:10])
