#!/usr/bin/env bash
# W11: re-run the four selected models (E1, E2-07, E3-B, E4-B) at K = 20 and 50
# on ALL evaluable users. K = 10 results already exist in results/runs.jsonl.
# Output goes to results/k_sweep/ so the official K = 10 results are not overwritten.
# Usage:  bash scripts/run_k_sweep.sh "/path/to/data"
set -euo pipefail
DATA_DIR="${1:?give the data folder that contains 'MAIN DATA' and 'SUPPORTING_DATA'}"
ln -sfn "$DATA_DIR/MAIN DATA" "MAIN DATA"
ln -sfn "$DATA_DIR/SUPPORTING_DATA" "SUPPORTING_DATA"
for cfg in configs/k_sweep/*.yaml; do
  python -m src.run --config "$cfg"
done
rm -f "MAIN DATA" "SUPPORTING_DATA"
