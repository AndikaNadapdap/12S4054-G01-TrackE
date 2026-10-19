"""W6: TF-IDF content-based filtering baseline on the warm-start validation protocol.

Usage:
  python run_tfidf.py --data_dir DATA_DIR                 # sanity check on 1,000 users (default)
  python run_tfidf.py --data_dir DATA_DIR --subset 0      # all evaluable users
"""
import argparse, os, time
from src.common import (load_data, load_items, seen_items, relevant_items, sample_users,
                        check_protocol, save_report, write_examples)
from src.tfidf_cbf import TfidfCBFRecommender

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default=".")
p.add_argument("--k", type=int, default=10)
p.add_argument("--subset", type=int, default=1000, help="number of evaluable users (0 = all)")
p.add_argument("--ngram_max", type=int, default=1, help="1 = unigrams, 2 = unigrams+bigrams")
p.add_argument("--min_df", type=int, default=2)
p.add_argument("--max_df", type=float, default=0.5)
p.add_argument("--max_features", type=int, default=0, help="0 = no limit")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="results")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

d = load_data(a.data_dir)
items = load_items(a.data_dir)
seen = seen_items(d["train"])
users = sample_users(d["eval_users"], a.subset, a.seed)
rel = relevant_items(d["val_warm"], users)

t = time.time()
model = TfidfCBFRecommender(ngram_range=(1, a.ngram_max), min_df=a.min_df, max_df=a.max_df,
                            max_features=a.max_features or None).fit(d["train"], d["candidates"], items)
print(f"Evaluating {len(users)} users | TF-IDF ngram=(1,{a.ngram_max}) min_df={a.min_df} "
      f"max_df={a.max_df} vocab={model.vocab_size} | fit {time.time() - t:.1f}s")
recs = model.recommend_many(users, seen, a.k)
check_protocol(recs, d["candidates"], seen, a.k)

name = "tfidf" if not a.subset else f"tfidf_subset{a.subset}"
save_report(name, recs, rel, d["train"], d["candidates"], a.k, a.out)
write_examples(name, recs, d["train"], items, rel, a.out)
print(f"Examples written to {a.out}/{name}_top10_examples.md")
