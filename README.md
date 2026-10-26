# 12S4054 Track E – Lexical vs Semantic (Group 01)

Project: *Lexical or Semantic: A Controlled Comparison of TF-IDF and Sentence-Embedding
Item Representations for Top-N Recommendation on Amazon Musical Instruments*

Members: Joy Valeda Silalahi, Andika Immanuel Nadapdap, Pariama Valentino Sihombing, Gracia Pardede

## Repository layout
```
src/          shared evaluation code and models
tests/        hand-checkable metric tests
notebooks/    EDA (training data only)
results/      saved metrics and Top-10 examples (no raw data)
docs/         AI usage log
run_*.py      one runner per baseline
```

## Data layout
Extract both course data zips into one folder (`DATA_DIR`) **outside this repository**.
Course data must never be committed.
```
DATA_DIR/
  MAIN DATA/        interactions_train.csv, interactions_validation.csv, items.csv, candidate_items_warm.csv
  SUPPORTING_DATA/  interactions_validation_warm.csv, evaluable_users_validation_warm.csv, ...
```

## Setup
```
pip install -r requirements.txt
python -m tests.test_metrics          # metric check on a hand-computed example
```

## Baselines
| Baseline | Command | Runtime (approx.) |
|---|---|---|
| Popularity | `python run_popularity.py --data_dir DATA_DIR` | < 1 min |
| Matrix Factorization (implicit ALS) | `python run_mf.py --data_dir DATA_DIR` | ~2 min |
| TF-IDF content-based filtering | `python run_tfidf.py --data_dir DATA_DIR` | < 1 min |
| Sentence-embedding CBF (Track E prototype) | `python run_embedding.py --data_dir DATA_DIR` | 5–15 min first run (encoding), then < 1 min |

`--subset N` evaluates a fixed random sample of N evaluable users (seed 42) for quick sanity
checks; `--subset 0` evaluates all 11,659 users. `run_mf.py`, `run_tfidf.py` and `run_embedding.py` use `--subset 1000` by default (W6
sanity check). Initial MF settings: `--factors 64 --reg 0.1 --alpha 10 --iterations 15 --seed 42`
(not tuned yet; tuning on validation is planned for W9-W10).
Initial TF-IDF settings: text field `content_text_basic`, unigrams, `--min_df 2 --max_df 0.5`,
sublinear TF, English stop words; user profile = mean of the TF-IDF vectors of the user's
positive (rating >= 4) training items; score = cosine similarity.

Each run writes `results/<name>_overall.json`, `<name>_by_group.csv`, `<name>_per_user.csv`
(for paired bootstrap CIs later), and `<name>_top10_examples.md` (readable Top-10 examples).

## Protocol (warm-start validation)
- Fit on `interactions_train.csv` only (MF uses positive interactions, rating >= 4).
- Candidates: `candidate_items_warm.csv` (22,753 items); training items of each user are excluded.
- Evaluable users: `evaluable_users_validation_warm.csv` (11,659 users).
- Relevant item: rating >= 4 in `interactions_validation_warm.csv`.
- Metrics: Recall@10, NDCG@10, HitRate@10, MRR@10, catalog coverage.
- Activity groups (provisional, frozen at CP3): low 1-4, medium 5-9, high >=10 training interactions.
- Every runner asserts: exactly 10 items per list, all items are warm candidates, no training item recommended.

## Track E: sentence-embedding CBF (prototype)
`run_embedding.py` keeps **everything identical to TF-IDF CBF except the item representation**:
same text field (`content_text_basic`), same profile rule (mean of positive training items),
same cosine scoring, same candidates and users. Default model:
`sentence-transformers/all-MiniLM-L6-v2` (model's default max length); try another with
`--model sentence-transformers/all-mpnet-base-v2`. The first run downloads the model and encodes
all 22,753 items; embeddings are cached in `cache/` (git-ignored). The runner also writes
`results/<name>_overlap_vs_tfidf.json` with the Jaccard@10 overlap between the TF-IDF and
embedding Top-10 lists (analysis E3).
