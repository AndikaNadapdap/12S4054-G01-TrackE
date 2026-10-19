# AI Usage Log – Group 01 (Track E)

Record every AI use that affects code, experiments, analysis, or the manuscript.
Small autocomplete does not need to be recorded.

| Week | Member | Tool | Purpose | What was used | How it was verified |
|---|---|---|---|---|---|
| W5 | | Claude | EDA notebook for training data | Notebook code and summary statistics | Re-ran notebook locally; spot-checked counts against the dataset brief (427,957 training interactions) |
| W5 | | Claude | Popularity baseline and metric functions | `src/`, `run_popularity.py`, `tests/test_metrics.py` | Hand-computed metric test passes; re-ran on a second laptop with identical results |
| W5 | | Claude | Draft of title, problem statement, RQs, reference list, Introduction text | Structure and draft text; references | Checked each reference; Introduction to be rewritten in the group's own words |
| W6 | | Claude | MF (implicit ALS) baseline | `src/mf_als.py`, `run_mf.py` | Re-ran locally; checked Top-10 examples for plausibility; protocol assertions pass |
