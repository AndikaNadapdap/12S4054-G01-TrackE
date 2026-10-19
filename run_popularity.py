"""Popularity baseline on the warm-start validation protocol.

Usage:
  python run_popularity.py --data_dir DATA_DIR               # all evaluable users (default)
  python run_popularity.py --data_dir DATA_DIR --subset 1000 # same sanity subset as other baselines
"""
import argparse, os
from src.common import (load_data, load_items, seen_items, relevant_items, sample_users,
                        check_protocol, save_report, write_examples)
from src.popularity import PopularityRecommender

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default=".")
p.add_argument("--k", type=int, default=10)
p.add_argument("--subset", type=int, default=0, help="number of evaluable users (0 = all)")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="results")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

d = load_data(a.data_dir)
seen = seen_items(d["train"])
users = sample_users(d["eval_users"], a.subset, a.seed)
rel = relevant_items(d["val_warm"], users)

model = PopularityRecommender().fit(d["train"], d["candidates"])
recs = {u: model.recommend(u, seen.get(u, set()), a.k) for u in users}
check_protocol(recs, d["candidates"], seen, a.k)

name = "popularity" if not a.subset else f"popularity_subset{a.subset}"
save_report(name, recs, rel, d["train"], d["candidates"], a.k, a.out)
write_examples(name, recs, d["train"], load_items(a.data_dir), rel, a.out)
print("Top-10 global popular items:", model.ranking[:10])
