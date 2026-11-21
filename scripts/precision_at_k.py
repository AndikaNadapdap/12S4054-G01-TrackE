"""W11: Precision@K (K = 10, 20, 50) for the four selected models.

Precision@K is derived from the saved per-user results: for each user,
Precision@K = Recall@K * (number of relevant items) / K, because src/metrics.py
computes Recall@K = hits / number of relevant items. The number of relevant
items per user comes from the warm validation file (rating >= 4).

Usage:  python scripts/precision_at_k.py --data_dir DATA_DIR [--results_dir results]
Writes results/tables/precision_at_k.csv
"""
import argparse
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True)
p.add_argument("--results_dir", default="results")
a = p.parse_args()
res = Path(a.results_dir)

val = pd.read_csv(Path(a.data_dir) / "SUPPORTING_DATA" / "interactions_validation_warm.csv")
n_rel = val[val["rating"] >= 4].groupby("user_id")["item_id"].nunique()

RUNS = {  # (model, K) -> per-user file
    ("Popularity", 10): "E1_popularity_subsetall", ("ALS", 10): "E2-07_als_subsetall",
    ("TF-IDF", 10): "E3-B_tfidf_subsetall", ("Embedding", 10): "E4-B_embedding_subsetall",
}
for k in (20, 50):
    RUNS.update({("Popularity", k): f"k_sweep/E1-K{k}_popularity_subsetall", ("ALS", k): f"k_sweep/E2-07-K{k}_als_subsetall",
                 ("TF-IDF", k): f"k_sweep/E3-B-K{k}_tfidf_subsetall", ("Embedding", k): f"k_sweep/E4-B-K{k}_embedding_subsetall"})
rows = []
for model in ("Popularity", "ALS", "TF-IDF", "Embedding"):
    row = {"Model": model}
    for k in (10, 20, 50):
        pu = pd.read_csv(res / f"{RUNS[(model, k)]}_per_user.csv").set_index("user_id")
        prec = pu[f"Recall@{k}"] * n_rel.reindex(pu.index) / k
        assert prec.notna().all() and (prec <= 1 + 1e-9).all()
        row[f"Precision@{k}"] = float(prec.mean())
    rows.append(row)
out = pd.DataFrame(rows)
Path(res / "tables").mkdir(exist_ok=True)
out.to_csv(res / "tables" / "precision_at_k.csv", index=False)
print(out.round(5).to_string(index=False))
