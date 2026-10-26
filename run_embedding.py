"""W7 (CP2): Sentence-embedding CBF prototype (Track E) on the warm-start validation protocol.

Uses exactly the same text field, profile rule, scoring, candidates and users as
run_tfidf.py; only the item representation changes. Also reports Jaccard@10 overlap
between the TF-IDF and embedding Top-10 lists of each user (Track E analysis E3).

Usage:
  python run_embedding.py --data_dir DATA_DIR                         # 1,000 users, all-MiniLM-L6-v2
  python run_embedding.py --data_dir DATA_DIR --model all-mpnet-base-v2
  python run_embedding.py --data_dir DATA_DIR --subset 0              # all evaluable users
First run downloads the model and encodes 22,753 items (a few minutes on CPU); the item
embeddings are then cached in cache/ and reused.
"""
import argparse, json, os, time
import numpy as np
from src.common import (load_data, load_items, seen_items, relevant_items, sample_users,
                        check_protocol, save_report, write_examples)
from src.embedding_cbf import EmbeddingCBFRecommender, sentence_transformer_encoder
from src.tfidf_cbf import TfidfCBFRecommender

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default=".")
p.add_argument("--k", type=int, default=10)
p.add_argument("--subset", type=int, default=1000, help="number of evaluable users (0 = all)")
p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--max_seq_length", type=int, default=0, help="0 = model default")
p.add_argument("--device", default=None, help="e.g. cpu or cuda (default: automatic)")
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
encoder = sentence_transformer_encoder(a.model, a.batch_size, a.max_seq_length or None, a.device)
short = a.model.split("/")[-1]
model = EmbeddingCBFRecommender(encoder, cache_name=f"{short}_len{encoder.max_seq_length}").fit(
    d["train"], d["candidates"], items)
print(f"Evaluating {len(users)} users | model={a.model} max_seq_length={encoder.max_seq_length} "
      f"dim={model.V.shape[1]} | fit {time.time() - t:.1f}s")
recs = model.recommend_many(users, seen, a.k)
check_protocol(recs, d["candidates"], seen, a.k)

name = f"emb_{short}" + ("" if not a.subset else f"_subset{a.subset}")
overall, _ = save_report(name, recs, rel, d["train"], d["candidates"], a.k, a.out)
write_examples(name, recs, d["train"], items, rel, a.out)

# --- Track E analysis (E3): how different are the lexical and semantic Top-10 lists? ---
tfidf = TfidfCBFRecommender().fit(d["train"], d["candidates"], items)
recs_tfidf = tfidf.recommend_many(users, seen, a.k)
jac = np.array([len(set(recs[u]) & set(recs_tfidf[u])) / len(set(recs[u]) | set(recs_tfidf[u]))
                for u in users])
overlap = {"jaccard@10_mean_vs_tfidf": round(float(jac.mean()), 4),
           "share_users_identical_lists": round(float((jac == 1).mean()), 4),
           "share_users_no_overlap": round(float((jac == 0).mean()), 4)}
json.dump(overlap, open(f"{a.out}/{name}_overlap_vs_tfidf.json", "w"), indent=2)
print("Overlap with TF-IDF Top-10:", json.dumps(overlap))
