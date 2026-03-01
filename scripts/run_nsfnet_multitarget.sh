#!/usr/bin/env bash
set -euo pipefail

# NSFNET multi-target experiment driver.
#
# This mirrors scripts/run_dumbbell_multitarget.sh but swaps in:
#   - src/run_nsfnet_capture_plus.py
#   - core-link links.txt is written automatically from the topology

DO_CAPTURE=${DO_CAPTURE:-1}
CAPTURE_SEEDS=(${CAPTURE_SEEDS:-1})
OUTROOT=${OUTROOT:-runs}

DOCKER_IMG=${DOCKER_IMG:-ndt/host:focal-nettools}
BW_CORE=${BW_CORE:-2}        # Mbps (all NSF core links)
BW_ACCESS=${BW_ACCESS:-1000} # Mbps (host access links)

UNDER_MBPS=${UNDER_MBPS:-6}
OVER_MBPS=${OVER_MBPS:-18}

TRAIN_SEEDS=(42 123 999)
BUSY_THR=${BUSY_THR:-50}
CALIB_Q=${CALIB_Q:-0.995}

# Loss weights (same semantics as dumbbell script)
W_QUEUE=${W_QUEUE:-1.0}
W_QUEUE_IDLE=${W_QUEUE_IDLE:-0.10}
W_THR=${W_THR:-1.0}
W_UTIL=${W_UTIL:-0.5}
W_RTT=${W_RTT:-0.0002}

PY=${PY:-python3}

log() { echo "[$(date +'%F %T')] $*"; }
require_file() { [[ -f "$1" ]] || { echo "ERROR: missing $1" >&2; exit 1; }; }

for CAPSEED in "${CAPTURE_SEEDS[@]}"; do
  OUTDIR="${OUTROOT}/nsfnet_seed${CAPSEED}_plus"
  mkdir -p "$OUTDIR"/{models,metrics}

  log "=== NSFNET | Capture seed=${CAPSEED} | OUTDIR=${OUTDIR} ==="

  if [[ "$DO_CAPTURE" -eq 1 ]]; then
    log "Running capture (requires sudo)..."
    sudo -E $PY src/run_nsfnet_capture_plus.py \
      --outdir "$OUTDIR" \
      --seed "$CAPSEED" \
      --img "$DOCKER_IMG" \
      --bw_access "$BW_ACCESS" \
      --bw_core "$BW_CORE" \
      --under_mbps "$UNDER_MBPS" \
      --over_mbps "$OVER_MBPS" \
      --ping

    log "Fixing ownership on OUTDIR after sudo capture..."
    sudo chown -R "$(id -un)":"$(id -gn)" "$OUTDIR"
    chmod -R u+rwX "$OUTDIR"
  else
    log "Skipping capture (DO_CAPTURE=0)."
  fi

  require_file "$OUTDIR/data.csv"
  require_file "$OUTDIR/processed_plus.csv"
  require_file "$OUTDIR/links.txt"
  require_file "$OUTDIR/dataset_multi.npz"

  log "Baselines NOWCAST"
  $PY src/eval_multitarget_baselines.py \
    --npz "$OUTDIR/dataset_multi.npz" \
    --mode nowcast \
    --busy_thr "$BUSY_THR" \
    --out_csv "$OUTDIR/metrics/baselines_nowcast.csv"

  log "Baselines LEAD1 + Kalman"
  $PY src/eval_multitarget_baselines.py \
    --npz "$OUTDIR/dataset_multi.npz" \
    --mode lead1 \
    --busy_thr "$BUSY_THR" --kalman \
    --out_csv "$OUTDIR/metrics/baselines_lead1.csv"

  for S in "${TRAIN_SEEDS[@]}"; do
    CKPT="$OUTDIR/models/nowcast_seed${S}.pt"
    log "Training NOWCAST (seed=${S})"
    $PY src/train_nowcast_multitarget.py \
      --data "$OUTDIR/dataset_multi.npz" \
      --seed "$S" \
      --busy_thr "$BUSY_THR" \
      --w_queue "$W_QUEUE" --w_queue_idle "$W_QUEUE_IDLE" \
      --w_thr "$W_THR" --w_util "$W_UTIL" --w_rtt "$W_RTT" \
      --out "$CKPT"

    log "Eval NOWCAST (seed=${S}, calibrated queue)"
    $PY src/eval_gnn_multitarget.py \
      --mode nowcast \
      --npz "$OUTDIR/dataset_multi.npz" \
      --ckpts "$CKPT" \
      --busy_thr "$BUSY_THR" \
      --calibrate_queue --calib_q "$CALIB_Q" \
      --out_csv "$OUTDIR/metrics/gnn_nowcast_seed${S}_cal.csv"
  done

  log "Eval NOWCAST ENSEMBLE (calibrated queue)"
  $PY src/eval_gnn_multitarget.py \
    --mode nowcast \
    --npz "$OUTDIR/dataset_multi.npz" \
    --ckpts \
      "$OUTDIR/models/nowcast_seed42.pt" \
      "$OUTDIR/models/nowcast_seed123.pt" \
      "$OUTDIR/models/nowcast_seed999.pt" \
    --busy_thr "$BUSY_THR" \
    --calibrate_queue --calib_q "$CALIB_Q" \
    --out_csv "$OUTDIR/metrics/gnn_nowcast_ens_cal.csv"

  # ------------------------------------------------------------------
  # Option 2: Dedicated graph-level RTT/qdelay model
  # ------------------------------------------------------------------
  log "Training RTT Graph NOWCAST (Option 2)"
  for S in "${TRAIN_SEEDS[@]}"; do
    CKPT="$OUTDIR/models/rtt_graph_seed${S}.pt"
    $PY src/train_nowcast_rtt_graph.py \
      --data "$OUTDIR/dataset_multi.npz" \
      --seed "$S" \
      --out "$CKPT" \
      --ping_src 0 --ping_dst 13 \
      --transform residual_log1p
  done

  log "Eval RTT Graph NOWCAST (Option 2)"
  $PY src/eval_rtt_graph.py \
    --npz "$OUTDIR/dataset_multi.npz" \
    --ckpts \
      "$OUTDIR/models/rtt_graph_seed42.pt" \
      "$OUTDIR/models/rtt_graph_seed123.pt" \
      "$OUTDIR/models/rtt_graph_seed999.pt" \
    --out_csv "$OUTDIR/metrics/rtt_graph_metrics.csv"

done

log "All done." 
