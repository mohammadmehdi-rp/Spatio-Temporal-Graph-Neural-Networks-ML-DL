#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
DO_CAPTURE=${DO_CAPTURE:-1}          # 1 = capture; 0 = reuse existing OUTDIR
CAPTURE_SEEDS=(${CAPTURE_SEEDS:-1})  # e.g. "1 2 3"
OUTROOT=${OUTROOT:-runs}

DOCKER_IMG=${DOCKER_IMG:-ndt/host:focal-nettools}
BW_BOTTLENECK=${BW_BOTTLENECK:-2}    # Mbps
BW_ACCESS=${BW_ACCESS:-1000}         # Mbps
UNDER_MBPS=${UNDER_MBPS:-1}
OVER_MBPS=${OVER_MBPS:-14}

TRAIN_SEEDS=(42 123 999)
BUSY_THR=${BUSY_THR:-50}
CALIB_Q=${CALIB_Q:-0.995}

# Option A (NO-MASK dataset)
W_QUEUE=${W_QUEUE:-1.0}
W_QUEUE_IDLE=${W_QUEUE_IDLE:-0.10}
W_THR=${W_THR:-1.0}
W_UTIL=${W_UTIL:-0.5}
W_RTT=${W_RTT:-0.0002}   # set 0 to disable RTT/delay training

TRAIN_LEAD1=${TRAIN_LEAD1:-0}
K_WINDOW=${K_WINDOW:-30}
TEMPORAL=${TEMPORAL:-tcn}

PY=${PY:-python3}

log() { echo "[$(date +'%F %T')] $*"; }

require_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing file: $f" >&2
    exit 1
  fi
}

ensure_sensors_links() {
  local outdir="$1"
  local processed="$2"

  # sensors.txt
  if [[ ! -f "$outdir/sensors.txt" ]]; then
    log "sensors.txt missing -> generating default sensors list"
    cat > "$outdir/sensors.txt" <<EOF
s1-eth1
s1-eth2
s2-eth1
s2-eth2
EOF
  fi

  # links.txt (build 21 undirected links -> 42 directed edges)
  if [[ ! -f "$outdir/links.txt" ]]; then
    log "links.txt missing -> generating from iface names in processed CSV"
    $PY - <<PY
import pandas as pd
from itertools import combinations
from pathlib import Path

outdir = Path("${outdir}")
processed = Path("${processed}")

df = pd.read_csv(processed)
if "iface" not in df.columns:
    raise SystemExit("processed CSV missing 'iface' column")

ifaces = sorted(set(df["iface"].astype(str).tolist()))
# Group by switch prefix (s1, s2) using part before '-'
groups = {}
for x in ifaces:
    sw = x.split("-", 1)[0]
    groups.setdefault(sw, []).append(x)

# Sort ports by eth index if present
def eth_key(iface):
    try:
        return int(iface.split("eth")[-1])
    except Exception:
        return 999999

for sw in groups:
    groups[sw] = sorted(groups[sw], key=eth_key)

links = []

# Complete graph within each switch group
for sw, ports in groups.items():
    for a, b in combinations(ports, 2):
        links.append((a, b))

# Add bottleneck inter-switch link (known from your capture: s1-eth1 <-> s2-eth1)
if "s1" in groups and "s2" in groups and "s1-eth1" in ifaces and "s2-eth1" in ifaces:
    links.append(("s1-eth1", "s2-eth1"))

# Write undirected edges (one per line)
with open(outdir / "links.txt", "w") as f:
    for a, b in links:
        f.write(f"{a} {b}\\n")

print(f"Wrote {outdir/'links.txt'} with {len(links)} undirected links")
PY
  fi
}

for CAPSEED in "${CAPTURE_SEEDS[@]}"; do
  OUTDIR="${OUTROOT}/dumbbell_seed${CAPSEED}_plus_A_nomask"
  mkdir -p "$OUTDIR"/{models,metrics}

  log "=== Option A (NO-MASK) | Capture seed=${CAPSEED} | OUTDIR=${OUTDIR} ==="

  if [[ "$DO_CAPTURE" -eq 1 ]]; then
    log "Running capture (requires sudo)..."
    sudo -E $PY src/run_dumbbell_capture_plus.py \
      --outdir "$OUTDIR" \
      --img "$DOCKER_IMG" \
      --bw_bottleneck "$BW_BOTTLENECK" \
      --under_mbps "$UNDER_MBPS" \
      --over_mbps "$OVER_MBPS" \
      --force_tbf --force_tbf_both \
      --ping

    log "Fixing ownership on OUTDIR after sudo capture..."
    sudo chown -R "$(id -un)":"$(id -gn)" "$OUTDIR"
    chmod -R u+rwX "$OUTDIR"
  else
    log "Skipping capture (DO_CAPTURE=0)."
  fi

  # Must exist
  require_file "$OUTDIR/data.csv"
  require_file "$OUTDIR/latency.csv"

  # 1) Process data (+ RTT cleanup is inside process_data_plus.py)
  log "Processing data -> processed_plus.csv"
  $PY src/process_data_plus.py "$OUTDIR/data.csv" \
    --out "$OUTDIR/processed_plus.csv" \
    --latency_csv "$OUTDIR/latency.csv"

  require_file "$OUTDIR/processed_plus.csv"

  # 2) Ensure sensors.txt and links.txt exist (autogenerate if missing)
  ensure_sensors_links "$OUTDIR" "$OUTDIR/processed_plus.csv"
  require_file "$OUTDIR/sensors.txt"
  require_file "$OUTDIR/links.txt"

  # 3) Build NO-MASK dataset
  log "Building dataset_multi_rttclean_nomask.npz (NO-MASK)"
  $PY src/gnn_prep_multitarget.py \
    --processed "$OUTDIR/processed_plus.csv" \
    --sensors_file "$OUTDIR/sensors.txt" \
    --links_file "$OUTDIR/links.txt" \
    --bw_access "$BW_ACCESS" \
    --bw_bottleneck "$BW_BOTTLENECK" \
    --out "$OUTDIR/dataset_multi_rttclean_nomask.npz"

  require_file "$OUTDIR/dataset_multi_rttclean_nomask.npz"

  # 4) Baselines (NO-MASK)
  log "Baselines NOWCAST (NO-MASK)"
  $PY src/eval_multitarget_baselines.py \
    --npz "$OUTDIR/dataset_multi_rttclean_nomask.npz" \
    --mode nowcast \
    --busy_thr "$BUSY_THR" \
    --out_csv "$OUTDIR/metrics/baselines_nowcast_rttclean_nomask.csv"

  log "Baselines LEAD1 (NO-MASK) + Kalman"
  $PY src/eval_multitarget_baselines.py \
    --npz "$OUTDIR/dataset_multi_rttclean_nomask.npz" \
    --mode lead1 \
    --busy_thr "$BUSY_THR" --kalman \
    --out_csv "$OUTDIR/metrics/baselines_lead1_rttclean_nomask.csv"

  # 5) Train/Eval NOWCAST GNNs
  for S in "${TRAIN_SEEDS[@]}"; do
    CKPT="$OUTDIR/models/nowcast_nomask_seed${S}.pt"
    log "Training NOWCAST (seed=${S}) -> ${CKPT}"
    $PY src/train_nowcast_multitarget.py \
      --data "$OUTDIR/dataset_multi_rttclean_nomask.npz" \
      --seed "$S" \
      --busy_thr "$BUSY_THR" \
      --w_queue "$W_QUEUE" --w_queue_idle "$W_QUEUE_IDLE" \
      --w_thr "$W_THR" --w_util "$W_UTIL" --w_rtt "$W_RTT" \
      --out "$CKPT"

    log "Eval NOWCAST (seed=${S}, calibrated queue)"
    $PY src/eval_gnn_multitarget.py \
      --mode nowcast \
      --npz "$OUTDIR/dataset_multi_rttclean_nomask.npz" \
      --ckpts "$CKPT" \
      --busy_thr "$BUSY_THR" \
      --calibrate_queue --calib_q "$CALIB_Q" \
      --out_csv "$OUTDIR/metrics/gnn_nowcast_nomask_seed${S}_cal.csv"
  done

  log "Eval NOWCAST ENSEMBLE (calibrated queue)"
  $PY src/eval_gnn_multitarget.py \
    --mode nowcast \
    --npz "$OUTDIR/dataset_multi_rttclean_nomask.npz" \
    --ckpts \
      "$OUTDIR/models/nowcast_nomask_seed42.pt" \
      "$OUTDIR/models/nowcast_nomask_seed123.pt" \
      "$OUTDIR/models/nowcast_nomask_seed999.pt" \
    --busy_thr "$BUSY_THR" \
    --calibrate_queue --calib_q "$CALIB_Q" \
    --out_csv "$OUTDIR/metrics/gnn_nowcast_nomask_ens_cal.csv"

  # Optional: lead-1 GNN
  if [[ "$TRAIN_LEAD1" -eq 1 ]]; then
    for S in "${TRAIN_SEEDS[@]}"; do
      CKPT="$OUTDIR/models/lead1_nomask_seed${S}.pt"
      log "Training LEAD1 (seed=${S}) -> ${CKPT}"
      $PY src/train_lead1_multitarget.py \
        --data "$OUTDIR/dataset_multi_rttclean_nomask.npz" \
        --K "$K_WINDOW" --temporal "$TEMPORAL" \
        --seed "$S" \
        --busy_thr "$BUSY_THR" \
        --w_queue "$W_QUEUE" --w_queue_idle "$W_QUEUE_IDLE" \
        --w_thr "$W_THR" --w_util "$W_UTIL" --w_rtt "$W_RTT" \
        --out "$CKPT"

      log "Eval LEAD1 (seed=${S}, calibrated queue)"
      $PY src/eval_gnn_multitarget.py \
        --mode lead1 --K "$K_WINDOW" --temporal "$TEMPORAL" \
        --npz "$OUTDIR/dataset_multi_rttclean_nomask.npz" \
        --ckpts "$CKPT" \
        --busy_thr "$BUSY_THR" \
        --calibrate_queue --calib_q "$CALIB_Q" \
        --out_csv "$OUTDIR/metrics/gnn_lead1_nomask_seed${S}_cal.csv"
    done

    log "Eval LEAD1 ENSEMBLE (calibrated queue)"
    $PY src/eval_gnn_multitarget.py \
      --mode lead1 --K "$K_WINDOW" --temporal "$TEMPORAL" \
      --npz "$OUTDIR/dataset_multi_rttclean_nomask.npz" \
      --ckpts \
        "$OUTDIR/models/lead1_nomask_seed42.pt" \
        "$OUTDIR/models/lead1_nomask_seed123.pt" \
        "$OUTDIR/models/lead1_nomask_seed999.pt" \
      --busy_thr "$BUSY_THR" \
      --calibrate_queue --calib_q "$CALIB_Q" \
      --out_csv "$OUTDIR/metrics/gnn_lead1_nomask_ens_cal.csv"
  fi

  log "DONE: OUTDIR=${OUTDIR}"
done

log "All done."
