#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

mkdir -p outputs/logs outputs/models outputs/summaries outputs/tables outputs/figures data/npz

echo "=== FULL mode: retraining and regenerating all artifacts ==="

SEEDS=(42 123 999)

NPZ_BASE="data/npz/dataset_sparse_10.npz"
NPZ_FAIR="data/npz/dataset_sparse_10_no_next.npz"

# Build fair dataset variant (drops any *_next features, e.g., ar1_next)
if [ ! -f "$NPZ_FAIR" ]; then
  echo "[INFO] building fair dataset: $NPZ_FAIR"
  $PY src/make_npz_variant.py --in_npz "$NPZ_BASE" --out_npz "$NPZ_FAIR" --drop_suffix _next --mode drop
fi

# 1) Sensor selection optimization (writes CSV in outputs/summaries)
$PY src/sensor_selection_optimization.py   --npz_full data/npz/dataset_full_v4.npz   --npz_sparse_fix data/npz/dataset_sparse_v4_fix.npz   --budgets 4,6,8   --l2 1e-2   |& tee outputs/logs/sensor_selection_optimization.log

# 2) Train NOWCAST and LEAD-1 on the k=10 sparse dataset (GraphSAGE) [original feature set]
for seed in "${SEEDS[@]}"; do
  $PY src/train_nowcast_sparse.py     --data "$NPZ_BASE"     --encoder sage     --seed "$seed"     --out "outputs/models/nowcast_sage_seed${seed}.pt"     |& tee "outputs/logs/nowcast_sage_seed${seed}.log"

  $PY src/train_lead1_sparse.py     --data "$NPZ_BASE"     --encoder sage     --temporal tcn     --seed "$seed"     --out "outputs/models/lead1_sage_seed${seed}.pt"     |& tee "outputs/logs/lead1_sage_seed${seed}.log"
done

# 3) Encoder variants (GraphSAGE vs RouteNet-lite) [original feature set]
for enc in sage routenet; do
  for seed in "${SEEDS[@]}"; do
    $PY src/train_nowcast_sparse.py       --data "$NPZ_BASE"       --encoder "$enc"       --seed "$seed"       --out "outputs/models/nowcast_${enc}_seed${seed}.pt"       |& tee "outputs/logs/nowcast_${enc}_seed${seed}.log"

    $PY src/train_lead1_sparse.py       --data "$NPZ_BASE"       --encoder "$enc"       --temporal tcn       --seed "$seed"       --out "outputs/models/lead1_${enc}_seed${seed}.pt"       |& tee "outputs/logs/lead1_${enc}_seed${seed}.log"
  done
done

echo "=== Benchmarking encoder latency ==="
$PY src/benchmark_encoder_latency.py --npz "$NPZ_BASE" --encoder sage     |& tee outputs/logs/encoder_latency_sage.log
$PY src/benchmark_encoder_latency.py --npz "$NPZ_BASE" --encoder routenet |& tee outputs/logs/encoder_latency_routenet.log

# 4) Calibration (produces calib_nowcast_10.npz) and calibration artifacts [original feature set]
$PY src/calibrate_nowcast.py   --npz "$NPZ_BASE"   --ckpt outputs/models/nowcast_sage_seed42.pt   --out data/npz/calib_nowcast_10.npz   |& tee outputs/logs/calibrate_nowcast.log

$PY src/eval_nowcast_baselines.py   --npz "$NPZ_BASE"   --calib_npz data/npz/calib_nowcast_10.npz   --out_csv outputs/summaries/nowcast_baselines.csv

$PY src/make_nowcast_calibration_artifacts.py   --in_csv outputs/summaries/nowcast_baselines.csv   --out_tex outputs/tables/nowcast_calibration_baselines.tex   --out_prefix outputs/figures/nowcast_calibration

# 5) Fair (no *_next) retrain for apples-to-apples comparison (SAGE only)
echo "=== Fair retrain for apples-to-apples (FAIR: no *_next) ==="
for seed in "${SEEDS[@]}"; do
  $PY src/train_nowcast_sparse.py     --data "$NPZ_FAIR"     --encoder sage     --seed "$seed"     --out "outputs/models/nowcast_sage_no_next_seed${seed}.pt"     |& tee "outputs/logs/nowcast_sage_no_next_seed${seed}.log"

  $PY src/train_lead1_sparse.py     --data "$NPZ_FAIR"     --encoder sage     --temporal tcn     --seed "$seed"     --out "outputs/models/lead1_sage_no_next_seed${seed}.pt"     |& tee "outputs/logs/lead1_sage_no_next_seed${seed}.log"
done

echo ""
echo "[OK] FULL mode finished."
echo "Check outputs/ for logs, models, figures, tables, and summaries."
