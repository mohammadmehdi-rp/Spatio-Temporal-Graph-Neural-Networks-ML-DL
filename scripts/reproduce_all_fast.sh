#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

mkdir -p outputs/logs outputs/models outputs/summaries outputs/tables outputs/figures data/npz

echo "=== FAST mode: regenerating lightweight artifacts ==="

NPZ_BASE="data/npz/dataset_sparse_10.npz"
NPZ_FAIR="data/npz/dataset_sparse_10_no_next.npz"

# Build fair dataset variant (drops any *_next features, e.g., ar1_next)
if [ ! -f "$NPZ_FAIR" ]; then
  echo "[INFO] building fair dataset: $NPZ_FAIR"
  $PY src/make_npz_variant.py --in_npz "$NPZ_BASE" --out_npz "$NPZ_FAIR" --drop_suffix _next --mode drop
fi

# Nowcast calibration + simple baselines (uses cached calibration NPZ)
$PY src/eval_nowcast_baselines.py   --npz "$NPZ_BASE"   --calib_npz data/npz/calib_nowcast_10.npz   --out_csv outputs/summaries/nowcast_baselines.csv

$PY src/make_nowcast_calibration_artifacts.py   --in_csv outputs/summaries/nowcast_baselines.csv   --out_tex outputs/tables/nowcast_calibration_baselines.tex   --out_prefix outputs/figures/nowcast_calibration

echo ""
echo "[OK] FAST mode finished."
echo "Figures  : outputs/figures/"
echo "Tables   : outputs/tables/"
echo "Summaries: outputs/summaries/"
