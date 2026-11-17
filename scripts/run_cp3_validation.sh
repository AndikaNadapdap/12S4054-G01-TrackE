#!/usr/bin/env bash
# CP3: run every validation configuration on ALL evaluable users (subset: 0).
# Run from the repository root, from COMMITTED code (check `git status` first).
# Usage:  bash scripts/run_cp3_validation.sh "/path/to/data"
set -euo pipefail
DATA_DIR="${1:?give the data folder that contains 'MAIN DATA' and 'SUPPORTING_DATA'}"
ln -sfn "$DATA_DIR/MAIN DATA" "MAIN DATA"
ln -sfn "$DATA_DIR/SUPPORTING_DATA" "SUPPORTING_DATA"
for cfg in configs/e1_popularity.yaml configs/e2_grid/*.yaml \
           configs/e3a_*.yaml configs/e3b_*.yaml configs/e3c_*.yaml \
           configs/e4a_*.yaml configs/e4b_*.yaml configs/e4c_*.yaml; do
  python -m src.run --config "$cfg"
done
rm -f "MAIN DATA" "SUPPORTING_DATA"
python scripts/select_best.py
