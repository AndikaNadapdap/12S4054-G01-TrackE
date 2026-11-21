"""W11: build result tables and figures from the saved validation results.

Reads only files under results/ and configs/ (no data, no model). Writes
  results/tables/*.csv   and   results/tables/*.tex
  results/figures/*.png

Inputs
  results/runs.jsonl            K = 10 runs on all evaluable users (subset 0)
  results/cp3_selection.json    best configuration per model family
  results/k_sweep/              K = 20 / 50 runs of the selected models
  configs/e2_grid/*.yaml        ALS grid (factors, reg, alpha) for the heatmap

Usage:  python scripts/make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
TAB = RES / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# One colour per model family, used in every figure (colour-blind safe).
COLOR = {"Popularity": "#7f7f7f", "ALS": "#0072B2", "TF-IDF": "#E69F00", "Embedding": "#009E73"}
LABEL = {"E1": "Popularity", "E2-07": "ALS", "E3-B": "TF-IDF", "E4-B": "Embedding"}
SELECTED = ["E1", "E2-07", "E3-B", "E4-B"]
GROUPS = ["low", "medium", "high"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def latest_full_runs():
    """Latest clean (git_dirty == False) run per experiment id on all users."""
    latest = {}
    for r in read_jsonl(RES / "runs.jsonl"):
        if r.get("subset") == 0 and r.get("git_dirty") is False:
            e = r["experiment_id"]
            if e not in latest or r["timestamp"] > latest[e]["timestamp"]:
                latest[e] = r
    return latest


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_groups(path, k):
    with open(path, encoding="utf-8") as f:
        return {row["activity_group"].split(" ")[0]: float(row[f"NDCG@{k}"]) for row in csv.DictReader(f)}


def write_csv(name, header, rows):
    with open(TAB / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_tex(name, header, rows, caption, label, fmt="{:.4f}"):
    cell = lambda v: fmt.format(v) if isinstance(v, float) else str(v)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{l" + "r" * (len(header) - 1) + "}", "\\hline",
             " & ".join(header) + " \\\\", "\\hline"]
    lines += [" & ".join(cell(v) for v in row) + " \\\\" for row in rows]
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    runs = latest_full_runs()
    n_users = next(iter(runs.values()))["n_evaluable_users"]

    # ---- Table 1: model x metric at K = 10 (selected configuration per model) ----
    rows = []
    for e in SELECTED:
        o = read_json(RES / f"{runs[e]['run_id']}_overall.json")
        rows.append([LABEL[e], e, o["Recall@10"], o["NDCG@10"], o["HitRate@10"], o["MRR@10"], o["Coverage@10"]])
    header = ["Model", "Config", "Recall@10", "NDCG@10", "HitRate@10", "MRR@10", "Coverage@10"]
    write_csv("main_results_k10", header, rows)
    write_tex("main_results_k10", header, rows,
              f"Validation results at K=10 on all {n_users:,} evaluable users (selected configuration per model).",
              "tab:main-k10")

    # ---- Table 2: all configurations tried (tuning evidence) ----
    allrows = []
    for e, r in sorted(runs.items()):
        o = read_json(RES / f"{r['run_id']}_overall.json")
        allrows.append([e, r["model"], o["Recall@10"], o["NDCG@10"], o.get("HitRate@10", ""), o.get("MRR@10", "")])
    write_csv("all_configs_k10", ["Config", "Model", "Recall@10", "NDCG@10", "HitRate@10", "MRR@10"], allrows)

    # ---- Figure 1: grouped bars of NDCG@K and Recall@K per model (K = 10, 20, 50) ----
    ks = {}
    sweep = RES / "k_sweep"
    for e in SELECTED:
        ks[e] = {10: read_json(RES / f"{runs[e]['run_id']}_overall.json")}
        for k in (20, 50):
            hits = sorted(sweep.glob(f"{e}-K{k}_*_overall.json"))
            if hits:
                ks[e][k] = read_json(hits[-1])
    have_k = all(len(v) == 3 for v in ks.values())
    if have_k:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
        w = 0.2
        for ax, metric in zip(axes, ["NDCG", "Recall"]):
            for i, e in enumerate(SELECTED):
                xs = [j + (i - 1.5) * w for j in range(3)]
                vals = [ks[e][k][f"{metric}@{k}"] for k in (10, 20, 50)]
                ax.bar(xs, vals, w, color=COLOR[LABEL[e]], label=LABEL[e])
            ax.set_xticks(range(3))
            ax.set_xticklabels(["K = 10", "K = 20", "K = 50"])
            ax.set_ylabel(f"{metric}@K")
            ax.set_title(f"{metric}@K by model")
        axes[0].legend(frameon=False, fontsize=8, title="Model", title_fontsize=8)
        fig.suptitle(f"Validation, all {n_users:,} evaluable users", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / "fig1_model_comparison.png", dpi=200)
        plt.close(fig)

    # ---- Figure 2: effect of K (needs results/k_sweep) ----
    if have_k:
        krows = []
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, metric in zip(axes, ["NDCG", "Recall"]):
            for e in SELECTED:
                ax.plot([10, 20, 50], [ks[e][k][f"{metric}@{k}"] for k in (10, 20, 50)],
                        marker="o", color=COLOR[LABEL[e]], label=LABEL[e])
            ax.set_xticks([10, 20, 50])
            ax.set_xlabel("K")
            ax.set_ylabel(f"{metric}@K")
            ax.set_title(f"{metric}@K vs K")
        axes[0].legend(frameon=False, fontsize=8, title="Model", title_fontsize=8, loc="lower right")
        fig.suptitle("Effect of K on the selected models (validation)", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / "fig2_effect_of_k.png", dpi=200)
        plt.close(fig)
        for e in SELECTED:
            krows.append([LABEL[e]] + [ks[e][k][f"{m}@{k}"] for k in (10, 20, 50) for m in ("Recall", "NDCG")])
        kh = ["Model", "Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "Recall@50", "NDCG@50"]
        write_csv("effect_of_k", kh, krows)
        write_tex("effect_of_k", kh, krows,
                  "Effect of K on the selected models (all evaluable users, validation).", "tab:effect-k")
        # Embedding (track model) versus each baseline: absolute and relative differences
        cmp_rows = []
        for base in ("E1", "E2-07", "E3-B"):
            for k in (10, 20, 50):
                row = [f"Embedding vs {LABEL[base]}", k]
                for m in ("NDCG", "Recall"):
                    a_, b_ = ks["E4-B"][k][f"{m}@{k}"], ks[base][k][f"{m}@{k}"]
                    row += [a_ - b_, (a_ / b_ - 1) * 100]
                cmp_rows.append(row)
        write_csv("baseline_vs_track", ["Comparison (track minus baseline)", "K", "dNDCG", "dNDCG_%", "dRecall", "dRecall_%"], cmp_rows)
        # Main table: model x metric for K = 10/20/50 incl. Precision@K when available
        prec_path = TAB / "precision_at_k.csv"
        prec = {r["Model"]: r for r in csv.DictReader(open(prec_path))} if prec_path.exists() else {}
        main_rows = []
        for e in SELECTED:
            row = [LABEL[e]] + [ks[e][k][f"NDCG@{k}"] for k in (10, 20, 50)] + [ks[e][k][f"Recall@{k}"] for k in (10, 20, 50)]
            if prec:
                row += [float(prec[LABEL[e]][f"Precision@{k}"]) for k in (10, 20, 50)]
            main_rows.append(row)
        mh = ["Model", "NDCG@10", "NDCG@20", "NDCG@50", "Recall@10", "Recall@20", "Recall@50"] + (["Precision@10", "Precision@20", "Precision@50"] if prec else [])
        write_csv("main_results_all_k", mh, main_rows)
    else:
        print("K sweep incomplete: skipping fig2_effect_of_k.png (run scripts/run_k_sweep.sh)")

    # ---- Figure 3: NDCG@10 by user activity group ----
    grp = {e: read_groups(RES / f"{runs[e]['run_id']}_by_group.csv", 10) for e in SELECTED}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.2
    for i, e in enumerate(SELECTED):
        xs = [j + (i - 1.5) * w for j in range(3)]
        vals = [grp[e][g] for g in GROUPS]
        ax.bar(xs, vals, w, color=COLOR[LABEL[e]], label=LABEL[e])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["low (1-4)", "medium (5-9)", "high (>=10)"])
    ax.set_xlabel("training interactions per user")
    ax.set_ylabel("NDCG@10")
    ax.set_title("NDCG@10 by user activity group (validation)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, title="Model", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_activity_groups.png", dpi=200)
    plt.close(fig)
    write_csv("ndcg10_by_activity_group", ["Model", "low", "medium", "high"],
              [[LABEL[e]] + [grp[e][g] for g in GROUPS] for e in SELECTED])

    # ---- Figure 4: ALS grid (factors x reg/alpha) ----
    cells = {}
    for p in sorted((ROOT / "configs" / "e2_grid").glob("*.yaml")):
        c = yaml.safe_load(p.read_text(encoding="utf-8"))
        e = c["experiment"]["id"]
        if e in runs:
            m = c["model"]
            cells[(m["factors"], m["reg"], m["alpha"])] = read_json(RES / f"{runs[e]['run_id']}_overall.json")["NDCG@10"]
    factors = sorted({k[0] for k in cells})
    cols = sorted({(k[1], k[2]) for k in cells})
    grid = [[cells[(f, r, a)] for (r, a) in cols] for f in factors]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    im = ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"reg {r:g}\nalpha {a:g}" for r, a in cols], fontsize=8)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels([f"{f} factors" for f in factors])
    for i in range(len(factors)):
        for j in range(len(cols)):
            ax.text(j, i, f"{grid[i][j]:.4f}", ha="center", va="center", fontsize=8,
                    color="white" if grid[i][j] > (max(map(max, grid)) + min(map(min, grid))) / 2 else "black")
    fig.colorbar(im, ax=ax, label="NDCG@10")
    ax.set_title("ALS grid search (18 configurations, validation)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_als_grid.png", dpi=200)
    plt.close(fig)
    write_csv("als_grid", ["factors", "reg", "alpha", "NDCG@10"],
              [[f, r, a, v] for (f, r, a), v in sorted(cells.items())])

    # ---- Figure 5: TF-IDF vs embedding configurations (equal tuning budget) ----
    names = ["E3-A", "E3-B", "E3-C", "E4-A", "E4-B", "E4-C"]
    vals = [read_json(RES / f"{runs[e]['run_id']}_overall.json")["NDCG@10"] for e in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(names, vals, color=[COLOR["TF-IDF"]] * 3 + [COLOR["Embedding"]] * 3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("TF-IDF vs sentence embedding: all tuned configurations", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_lexical_vs_semantic.png", dpi=200)
    plt.close(fig)

    print("Saved tables to", TAB.relative_to(ROOT))
    print("Saved figures to", FIG.relative_to(ROOT))


if __name__ == "__main__":
    main()
