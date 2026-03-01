#!/usr/bin/env bash
set -euo pipefail

# run_dumbbell_multitarget_A_qdelay.sh
#
# build NO-MASK dataset with queueing delay target (qdelay_ms),
# train nowcast multitarget models (queue, throughput, util, qdelay),
# and evaluate baselines + GNN (per-seed + ensemble) with calibration.

PY=${PY:-python3}

CAPSEED=${CAPSEED:-1}
OUTROOT=${OUTROOT:-runs}
OUTDIR="${OUTROOT}/dumbbell_seed${CAPSEED}_plus_A_qdelay_nomask"

BW_ACCESS=${BW_ACCESS:-1000}
BW_BOTTLENECK=${BW_BOTTLENECK:-2}

BUSY_THR=${BUSY_THR:-50}
TRAIN_SEEDS=(42 123 999)

# Training weights (tune if needed)
W_QUEUE=${W_QUEUE:-1.0}
W_QUEUE_IDLE=${W_QUEUE_IDLE:-0.10}
W_THR=${W_THR:-1.0}
W_UTIL=${W_UTIL:-0.5}
W_DELAY=${W_DELAY:-0.3}

THR_IDLE_THR=${THR_IDLE_THR:-0.05}
W_THR_IDLE=${W_THR_IDLE:-0.5}
DELAY_IDLE_THR=${DELAY_IDLE_THR:-1.0}
W_DELAY_IDLE=${W_DELAY_IDLE:-0.2}

CALIB_Q=${CALIB_Q:-0.995}

mkdir -p "$OUTDIR"/{models,metrics}

echo "[INFO] OUTDIR=$OUTDIR"

# --- A1) process data (assumes data.csv and latency.csv already exist)
$PY src/process_data_plus.py "$OUTDIR/data.csv" \
  --out "$OUTDIR/processed_plus.csv" \
  --latency_csv "$OUTDIR/latency.csv" \
  --rtt_clip_q 0.99 \
  --make_qdelay --qdelay_base_q 0.05 --qdelay_clip_q 0.99

# --- A2) build NO-MASK NPZ (qdelay if present)
$PY src/gnn_prep_multitarget.py \
  --processed "$OUTDIR/processed_plus.csv" \
  --sensors_file "$OUTDIR/sensors.txt" \
  --links_file "$OUTDIR/links.txt" \
  --bw_access "$BW_ACCESS" \
  --bw_bottleneck "$BW_BOTTLENECK" \
  --out "$OUTDIR/dataset_multi_qdelay_nomask.npz"

# --- A3) baselines nowcast + lead1
$PY src/eval_multitarget_baselines.py \
  --npz "$OUTDIR/dataset_multi_qdelay_nomask.npz" \
  --mode nowcast --busy_thr "$BUSY_THR" \
  --thr_idle_thr "$THR_IDLE_THR" --delay_idle_thr "$DELAY_IDLE_THR" \
  --out_csv "$OUTDIR/metrics/baselines_nowcast_qdelay_nomask.csv"

$PY src/eval_multitarget_baselines.py \
  --npz "$OUTDIR/dataset_multi_qdelay_nomask.npz" \
  --mode lead1 --busy_thr "$BUSY_THR" --kalman \
  --thr_idle_thr "$THR_IDLE_THR" --delay_idle_thr "$DELAY_IDLE_THR" \
  --out_csv "$OUTDIR/metrics/baselines_lead1_qdelay_nomask.csv"

# --- A4) train/eval nowcast (seeds)
for S in "${TRAIN_SEEDS[@]}"; do
  CKPT="$OUTDIR/models/nowcast_qdelay_nomask_seed${S}.pt"
  $PY src/train_nowcast_multitarget.py \
    --data "$OUTDIR/dataset_multi_qdelay_nomask.npz" \
    --seed "$S" --busy_thr "$BUSY_THR" \
    --w_queue "$W_QUEUE" --w_queue_idle "$W_QUEUE_IDLE" \
    --w_thr "$W_THR" --w_util "$W_UTIL" --w_delay "$W_DELAY" \
    --thr_idle_thr "$THR_IDLE_THR" --w_thr_idle "$W_THR_IDLE" \
    --delay_idle_thr "$DELAY_IDLE_THR" --w_delay_idle "$W_DELAY_IDLE" \
    --out "$CKPT"

  $PY src/eval_gnn_multitarget.py \
    --mode nowcast \
    --npz "$OUTDIR/dataset_multi_qdelay_nomask.npz" \
    --ckpts "$CKPT" \
    --busy_thr "$BUSY_THR" \
    --calibrate_queue --calib_q "$CALIB_Q" \
    --calibrate_thr --thr_idle_thr "$THR_IDLE_THR" --thr_calib_q "$CALIB_Q" --thr_busy_thr 0.2 \
    --calibrate_delay --delay_idle_thr "$DELAY_IDLE_THR" --delay_calib_q "$CALIB_Q" --delay_busy_thr 5.0 \
    --out_csv "$OUTDIR/metrics/gnn_nowcast_qdelay_nomask_seed${S}_cal.csv"
done

# --- A5) ensemble eval nowcast
$PY src/eval_gnn_multitarget.py \
  --mode nowcast \
  --npz "$OUTDIR/dataset_multi_qdelay_nomask.npz" \
  --ckpts \
    "$OUTDIR/models/nowcast_qdelay_nomask_seed42.pt" \
    "$OUTDIR/models/nowcast_qdelay_nomask_seed123.pt" \
    "$OUTDIR/models/nowcast_qdelay_nomask_seed999.pt" \
  --busy_thr "$BUSY_THR" \
  --calibrate_queue --calib_q "$CALIB_Q" \
  --calibrate_thr --thr_idle_thr "$THR_IDLE_THR" --thr_calib_q "$CALIB_Q" --thr_busy_thr 0.2 \
  --calibrate_delay --delay_idle_thr "$DELAY_IDLE_THR" --delay_calib_q "$CALIB_Q" --delay_busy_thr 5.0 \
  --out_csv "$OUTDIR/metrics/gnn_nowcast_qdelay_nomask_ens_cal.csv"



# --- A6) train/eval lead-1 (seeds)
for S in "${TRAIN_SEEDS[@]}"; do
  CKPT="$OUTDIR/models/lead1_qdelay_nomask_seed${S}.pt"
  $PY src/train_lead1_multitarget.py \
    --data "$OUTDIR/dataset_multi_qdelay_nomask.npz" \
    --seed "$S" --busy_thr "$BUSY_THR" --temporal tcn --K 30 \
    --w_queue "$W_QUEUE" --w_queue_idle "$W_QUEUE_IDLE" \
    --w_thr "$W_THR" --w_util "$W_UTIL" --w_delay "$W_DELAY" \
    --thr_idle_thr "$THR_IDLE_THR" --w_thr_idle "$W_THR_IDLE" \
    --delay_idle_thr "$DELAY_IDLE_THR" --w_delay_idle "$W_DELAY_IDLE" \
    --out "$CKPT"

  $PY src/eval_gnn_multitarget.py \
    --mode lead1 --temporal tcn --K 30 \
    --npz "$OUTDIR/dataset_multi_qdelay_nomask.npz" \
    --ckpts "$CKPT" \
    --busy_thr "$BUSY_THR" \
    --calibrate_queue --calib_q "$CALIB_Q" \
    --calibrate_thr --thr_idle_thr "$THR_IDLE_THR" --thr_calib_q "$CALIB_Q" --thr_busy_thr 0.2 \
    --calibrate_delay --delay_idle_thr "$DELAY_IDLE_THR" --delay_calib_q "$CALIB_Q" --delay_busy_thr 5.0 \
    --out_csv "$OUTDIR/metrics/gnn_lead1_qdelay_nomask_seed${S}_cal.csv"
done

# --- A7) ensemble eval lead-1
$PY src/eval_gnn_multitarget.py \
  --mode lead1 --temporal tcn --K 30 \
  --npz "$OUTDIR/dataset_multi_qdelay_nomask.npz" \
  --ckpts \
    "$OUTDIR/models/lead1_qdelay_nomask_seed42.pt" \
    "$OUTDIR/models/lead1_qdelay_nomask_seed123.pt" \
    "$OUTDIR/models/lead1_qdelay_nomask_seed999.pt" \
  --busy_thr "$BUSY_THR" \
  --calibrate_queue --calib_q "$CALIB_Q" \
  --calibrate_thr --thr_idle_thr "$THR_IDLE_THR" --thr_calib_q "$CALIB_Q" --thr_busy_thr 0.2 \
  --calibrate_delay --delay_idle_thr "$DELAY_IDLE_THR" --delay_calib_q "$CALIB_Q" --delay_busy_thr 5.0 \
  --out_csv "$OUTDIR/metrics/gnn_lead1_qdelay_nomask_ens_cal.csv"


echo "[DONE] Key outputs in $OUTDIR/metrics/"
ls -1 "$OUTDIR/metrics/" | sed -n '1,200p'
