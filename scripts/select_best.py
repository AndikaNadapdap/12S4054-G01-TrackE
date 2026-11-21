"""Apply the pre-declared model-selection rule to results/runs.jsonl.

Rule (docs/search_space.md): within each tuning family, pick the configuration with
the highest validation NDCG@10 over ALL evaluable users (subset 0), Recall@10 breaks
ties. Only runs executed from committed code (git_dirty == False) are eligible.
Writes results/cp3_selection.json.
"""
import json
from pathlib import Path

FAMILIES = {"ALS": "E2-", "TF-IDF": ("E3-A", "E3-B", "E3-C"), "Embedding": ("E4-A", "E4-B", "E4-C")}

runs = [json.loads(l) for l in Path("results/runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
eligible = [r for r in runs if r.get("subset") == 0 and r.get("git_dirty") is False]

selection = {}
for family, key in FAMILIES.items():
    match = (lambda e: e.startswith(key)) if isinstance(key, str) else (lambda e: e in key)
    cands = {}
    for r in eligible:
        if match(r["experiment_id"]):
            cands[r["experiment_id"]] = r          # latest run per experiment id
    if not cands:
        print(f"{family}: no eligible runs yet"); continue
    best = max(cands.values(), key=lambda r: (r["ndcg@10"], r["recall@10"]))
    selection[family] = {k: best[k] for k in ("experiment_id", "run_id", "ndcg@10", "recall@10", "git_commit", "config_hash")}
    selection[family]["n_configs_compared"] = len(cands)
    print(f"{family}: {best['experiment_id']}  NDCG@10={best['ndcg@10']:.4f}  Recall@10={best['recall@10']:.4f}  ({len(cands)} configs)")

Path("results/cp3_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
print("Saved results/cp3_selection.json")
