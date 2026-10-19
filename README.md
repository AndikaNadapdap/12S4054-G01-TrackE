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

## Run order
```
pip install -r requirements.txt
python -m tests.test_metrics                      # 1. metric check on a hand-computed example
python run_popularity.py --data_dir DATA_DIR      # 2. Popularity baseline on warm validation
```

## Protocol (warm-start validation)
- Fit on `interactions_train.csv` only.
- Candidates: `candidate_items_warm.csv` (22,753 items); training items of each user are excluded.
- Evaluable users: `evaluable_users_validation_warm.csv` (11,659 users).
- Relevant item: rating >= 4 in `interactions_validation_warm.csv`.
- Metrics: Recall@10, NDCG@10, HitRate@10, MRR@10, catalog coverage.
- Activity groups (provisional, frozen at CP3): low 1-4, medium 5-9, high >=10 training interactions.
