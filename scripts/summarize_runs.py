"""Print a comparison table of the latest full-validation run per experiment.

Reads results/runs.jsonl (no data or model needed) and, for each experiment,
shows the most recent run on all evaluable users (subset 0), plus the
per-activity-group NDCG@10 from results/<run_id>_by_group.csv.

Usage:  python scripts/summarize_runs.py
"""

import csv
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
ORDER = ["E1", "E2", "E2-07", "E3-A", "E3-B", "E3-C", "E4-A", "E4-B", "E4-C"]


def load_latest_full_runs():
    latest = {}
    with open(RESULTS / "runs.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            run = json.loads(line)
            if run.get("subset") != 0:
                continue
            exp = run["experiment_id"]
            if exp not in latest or run["timestamp"] > latest[exp]["timestamp"]:
                latest[exp] = run
    return latest


def group_ndcg(run_id):
    path = RESULTS / f"{run_id}_by_group.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {row["activity_group"].split(" ")[0]: float(row["NDCG@10"])
                for row in csv.DictReader(f)}


def main():
    latest = load_latest_full_runs()
    exps = [e for e in ORDER if e in latest] + sorted(set(latest) - set(ORDER))
    n_users = {latest[e]["n_evaluable_users"] for e in exps}

    print("Track E - validation results, all evaluable users "
          f"(n = {', '.join(map(str, sorted(n_users)))}), K = 10")
    header = (f"{'Exp':<6}{'Model':<11}{'Recall@10':>10}{'NDCG@10':>9}"
              f"{'NDCG low':>11}{'NDCG med':>11}{'NDCG high':>11}  {'Commit':<8}{'Date':<11}")
    print(header)
    print("-" * len(header))
    for e in exps:
        r = latest[e]
        g = group_ndcg(r["run_id"])
        print(f"{e:<6}{r['model']:<11}{r['recall@10']:>10.4f}{r['ndcg@10']:>9.4f}"
              f"{g.get('low', float('nan')):>11.4f}{g.get('medium', float('nan')):>11.4f}"
              f"{g.get('high', float('nan')):>11.4f}  {(r.get('git_commit') or '-')[:7]:<8}"
              f"{r['timestamp'][:10]:<11}")

    # Additional metrics (only logged by runs made after they were added to src/run.py).
    extra = [e for e in exps if latest[e].get("hitrate@10") is not None]
    if extra:
        print("\nAdditional metrics (HitRate = share of users with >= 1 hit; "
              "Coverage = share of 22,753 candidates shown to anyone)")
        header = f"{'Exp':<6}{'Model':<11}{'HitRate@10':>11}{'MRR@10':>9}{'Coverage@10':>13}"
        print(header)
        print("-" * len(header))
        for e in extra:
            r = latest[e]
            print(f"{e:<6}{r['model']:<11}{r['hitrate@10']:>11.4f}{r['mrr@10']:>9.4f}"
                  f"{r['coverage@10']:>13.4f}")


if __name__ == "__main__":
    main()
