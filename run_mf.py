"""W6: Matrix Factorization (implicit ALS) baseline on the warm-start validation protocol.

Usage:
  python run_mf.py --data_dir DATA_DIR                 # sanity check on 1,000 users (default)
  python run_mf.py --data_dir DATA_DIR --subset 0      # all evaluable users
"""
import argparse, os, time
from src.common import (load_data, load_items, seen_items, relevant_items, sample_users,
check_protocol, save_report, write_examples)
from src.mf_als import ALSRecommender

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default=".")
p.add_argument("--k", type=int, default=10)
p.add_argument("--subset", type=int, default=1000, help="number of evaluable users (0 = all)")
p.add_argument("--factors", type=int, default=64)
p.add_argument("--reg", type=float, default=0.1)
p.add_argument("--alpha", type=float, default=10.0)
p.add_argument("--iterations", type=int, default=15)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="results")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

d = load_data(a.data_dir)
seen = seen_items(d["train"])
users = sample_users(d["eval_users"], a.subset, a.seed)
rel = relevant_items(d["val_warm"], users)
print(f"Evaluating {len(users)} users | ALS factors={a.factors} reg={a.reg} alpha={a.alpha} "
      f"iterations={a.iterations} seed={a.seed}")

t = time.time()
model = ALSRecommender(a.factors, a.reg, a.alpha, a.iterations, a.seed).fit(d["train"], d["candidates"])
print(f"Training time: {time.time() - t:.1f}s")
recs = model.recommend_many(users, seen, a.k)
check_protocol(recs, d["candidates"], seen, a.k)

name = "mf" if not a.subset else f"mf_subset{a.subset}"
save_report(name, recs, rel, d["train"], d["candidates"], a.k, a.out)
write_examples(name, recs, d["train"], load_items(a.data_dir), rel, a.out)
print(f"Examples written to {a.out}/{name}_top10_examples.md")
