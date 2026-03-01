#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/_common.sh"

# Train + evaluate the dedicated path-conditioned RTT model (Option-2),
# including multi-probe RTT when latency_multi.csv is present in OUTDIR.

OUTDIR=${OUTDIR:-runs/nsfnet_seed1_plus}
TRAIN_SEEDS=(${TRAIN_SEEDS:-42 123 999})
DEVICE=${DEVICE:-cpu}  # cpu|cuda
HORIZON=${HORIZON:-0}  # 0=nowcast, 1=lead-1, 5=lead-5, ...
PY=${PY:-python3}

# Optional: restrict/reorder probes (must exist in dataset)
PROBES=${PROBES:-}  # e.g. "0-13,1-12"

require_file "$OUTDIR/dataset_multi.npz"
mkdir -p "$OUTDIR/models" "$OUTDIR/metrics"

log "=== NSFNET | RTT Graph Model (Option-2) | OUTDIR=$OUTDIR ==="

CKPTS=()
for SEED in "${TRAIN_SEEDS[@]}"; do
  CKPT="$OUTDIR/models/rtt_graph_seed${SEED}.pt"
  log "Training rtt-graph seed=${SEED} -> ${CKPT}"
  $PY src/train_nowcast_rtt_graph.py \
    --horizon "$HORIZON" \
    --data "$OUTDIR/dataset_multi.npz" \
    --out "$CKPT" \
    --seed "$SEED" \
    --device "$DEVICE" \
    ${PROBES:+--probes "$PROBES"}
  CKPTS+=("$CKPT")
done

OUTCSV="$OUTDIR/metrics/rtt_graph_metrics_H${HORIZON}.csv"
log "Evaluating (single + ensemble) -> $OUTCSV"
$PY src/eval_rtt_graph.py \
  --horizon "$HORIZON" \
  --npz "$OUTDIR/dataset_multi.npz" \
  --ckpts "${CKPTS[@]}" \
  --out_csv "$OUTCSV" \
  --device "$DEVICE"

log "OK: wrote $OUTCSV"
