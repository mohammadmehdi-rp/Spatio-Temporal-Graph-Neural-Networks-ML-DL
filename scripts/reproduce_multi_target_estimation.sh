#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# reproduce_results.sh
#
# End-to-end reproduction script for NDT multi-target lead-1 experiments:
#   queue_pkts, throughput_Mbps, utilization, qdelay_ms (residual-delay design)
#
# Pipeline:
#   (1) capture (optional) -> (2) process -> (3) build NPZ -> (4) train ensemble
#   -> (5) evaluate + calibrate -> (6) baselines -> (7) export tables
###############################################################################

###############################################################################
# EDIT ME (paths, env, and experiment identifiers)
###############################################################################
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Where raw captures/logs are written (from Mininet/ComNetsEmu)
RAW_ROOT="${REPO_ROOT}/data/raw"

# Where processed (cleaned/merged) time-series are written
PROC_ROOT="${REPO_ROOT}/data/processed"

# Where NPZ datasets are written
NPZ_ROOT="${REPO_ROOT}/data/npz"

# Where runs/checkpoints/metrics are written
RUNS_ROOT="${REPO_ROOT}/runs"

# A short tag used to name folders (keep stable to reproduce identical paths)
EXP_TAG="dumbbell_tbf_2mbps_pkt1500_altload_multitarget_lead1_residualDelay"

# If you use conda/venv, set this to your python interpreter
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Toggle capture (often not needed if you already have raw data)
DO_CAPTURE="${DO_CAPTURE:-0}"          # 1 to run capture, 0 to skip
DO_PROCESS="${DO_PROCESS:-1}"
DO_BUILD_NPZ="${DO_BUILD_NPZ:-1}"
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"
DO_BASELINES="${DO_BASELINES:-1}"
DO_TABLES="${DO_TABLES:-1}"

###############################################################################
# Experiment parameters (match the ones used for your reported results)
###############################################################################
BOTTLENECK_MBPS="${BOTTLENECK_MBPS:-2}"
PKT_BYTES="${PKT_BYTES:-1500}"
LEAD="${LEAD:-1}"

# Multi-targets (keep order stable)
TARGETS=("queue_pkts" "throughput_Mbps" "utilization" "qdelay_ms")

# Residual-delay design: predict Δdelay and reconstruct delay(t+1) = delay(t)+Δ̂
DELAY_MODE="${DELAY_MODE:-residual}"   # "residual" (final) or "direct"

# Ensemble seeds used for the final reported ensemble numbers
SEEDS=(0 1 2 3 4)

# Calibration targets used in your final run (queue + throughput; delay is residual)
CALIBRATE_TARGETS=("queue_pkts" "throughput_Mbps")

###############################################################################
# Helper utilities
###############################################################################
log() { echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run() {
  echo "+ $*"
  "$@"
}

# Pick the first existing file from a candidate list
pick_first_existing() {
  for f in "$@"; do
    if [[ -f "$f" ]]; then
      echo "$f"
      return 0
    fi
  done
  return 1
}

mkdir -p "$RAW_ROOT" "$PROC_ROOT" "$NPZ_ROOT" "$RUNS_ROOT"

log "Reproduction started"
log "Repo: $REPO_ROOT"
log "Git commit (if available): $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
log "EXP_TAG: $EXP_TAG"

###############################################################################
# (1) CAPTURE (optional)
###############################################################################
if [[ "$DO_CAPTURE" == "1" ]]; then
  log "Step (1) CAPTURE enabled"

  CAPTURE_SCRIPT="$(pick_first_existing \
    "${REPO_ROOT}/experiments/run_dumbbell.py" \
    "${REPO_ROOT}/experiments/dumbbell_tbf.py" \
    "${REPO_ROOT}/scripts/capture_dumbbell.sh" \
    "${REPO_ROOT}/scripts/run_capture.sh" \
  )" || {
    echo "ERROR: No capture script found. Set DO_CAPTURE=0 or add your script to candidate list." >&2
    exit 1
  }

  CAPTURE_OUT="${RAW_ROOT}/${EXP_TAG}"
  mkdir -p "$CAPTURE_OUT"

  case "$CAPTURE_SCRIPT" in
    *.py)
      run "$PYTHON_BIN" "$CAPTURE_SCRIPT" \
        --out "$CAPTURE_OUT" \
        --bottleneck_mbps "$BOTTLENECK_MBPS" \
        --pkt_bytes "$PKT_BYTES"
      ;;
    *.sh)
      run bash "$CAPTURE_SCRIPT" \
        "$CAPTURE_OUT" "$BOTTLENECK_MBPS" "$PKT_BYTES"
      ;;
    *)
      echo "ERROR: Unsupported capture script type: $CAPTURE_SCRIPT" >&2
      exit 1
      ;;
  esac
else
  log "Step (1) CAPTURE skipped (DO_CAPTURE=0)"
fi

RAW_IN="${RAW_ROOT}/${EXP_TAG}"
if [[ ! -d "$RAW_IN" ]]; then
  log "WARNING: RAW_IN does not exist: $RAW_IN"
  log "If you skipped capture, ensure your raw data folder matches EXP_TAG or set EXP_TAG accordingly."
fi

###############################################################################
# (2) PROCESS raw -> processed time-series
###############################################################################
if [[ "$DO_PROCESS" == "1" ]]; then
  log "Step (2) PROCESS enabled"

  PROCESS_SCRIPT="$(pick_first_existing \
    "${REPO_ROOT}/processing/process_dumbbell.py" \
    "${REPO_ROOT}/processing/build_timeseries.py" \
    "${REPO_ROOT}/scripts/process_data.sh" \
    "${REPO_ROOT}/scripts/run_processing.sh" \
  )" || {
    echo "ERROR: No processing script found. Add yours to candidate list." >&2
    exit 1
  }

  PROC_OUT="${PROC_ROOT}/${EXP_TAG}"
  mkdir -p "$PROC_OUT"

  case "$PROCESS_SCRIPT" in
    *.py)
      run "$PYTHON_BIN" "$PROCESS_SCRIPT" \
        --in "$RAW_IN" \
        --out "$PROC_OUT" \
        --targets "${TARGETS[@]}" \
        --lead "$LEAD" \
        --delay_mode "$DELAY_MODE"
      ;;
    *.sh)
      run bash "$PROCESS_SCRIPT" \
        "$RAW_IN" "$PROC_OUT" "$LEAD" "$DELAY_MODE"
      ;;
    *)
      echo "ERROR: Unsupported processing script type: $PROCESS_SCRIPT" >&2
      exit 1
      ;;
  esac
else
  log "Step (2) PROCESS skipped (DO_PROCESS=0)"
fi

PROC_IN="${PROC_ROOT}/${EXP_TAG}"

###############################################################################
# (3) BUILD NPZ dataset
###############################################################################
if [[ "$DO_BUILD_NPZ" == "1" ]]; then
  log "Step (3) BUILD_NPZ enabled"

  NPZ_SCRIPT="$(pick_first_existing \
    "${REPO_ROOT}/dataset/prepare_npz.py" \
    "${REPO_ROOT}/dataset/build_npz.py" \
    "${REPO_ROOT}/scripts/prepare_dataset.sh" \
    "${REPO_ROOT}/scripts/build_npz.sh" \
  )" || {
    echo "ERROR: No NPZ build script found. Add yours to candidate list." >&2
    exit 1
  }

  NPZ_OUT="${NPZ_ROOT}/${EXP_TAG}"
  mkdir -p "$NPZ_OUT"

  case "$NPZ_SCRIPT" in
    *.py)
      run "$PYTHON_BIN" "$NPZ_SCRIPT" \
        --in "$PROC_IN" \
        --out "$NPZ_OUT" \
        --targets "${TARGETS[@]}" \
        --lead "$LEAD"
      ;;
    *.sh)
      run bash "$NPZ_SCRIPT" \
        "$PROC_IN" "$NPZ_OUT" "$LEAD"
      ;;
    *)
      echo "ERROR: Unsupported NPZ script type: $NPZ_SCRIPT" >&2
      exit 1
      ;;
  esac
else
  log "Step (3) BUILD_NPZ skipped (DO_BUILD_NPZ=0)"
fi

NPZ_IN="${NPZ_ROOT}/${EXP_TAG}"

###############################################################################
# (4) TRAIN ensemble (spatio-temporal GNN lead-1 multi-target)
###############################################################################
if [[ "$DO_TRAIN" == "1" ]]; then
  log "Step (4) TRAIN enabled"

  TRAIN_SCRIPT="$(pick_first_existing \
    "${REPO_ROOT}/train/train_gnn.py" \
    "${REPO_ROOT}/train.py" \
    "${REPO_ROOT}/scripts/train.sh" \
    "${REPO_ROOT}/scripts/run_train.sh" \
  )" || {
    echo "ERROR: No training script found. Add yours to candidate list." >&2
    exit 1
  }

  for seed in "${SEEDS[@]}"; do
    RUN_DIR="${RUNS_ROOT}/${EXP_TAG}/seed_${seed}"
    mkdir -p "$RUN_DIR"

    case "$TRAIN_SCRIPT" in
      *.py)
        run "$PYTHON_BIN" "$TRAIN_SCRIPT" \
          --data "$NPZ_IN" \
          --run_dir "$RUN_DIR" \
          --seed "$seed" \
          --lead "$LEAD" \
          --targets "${TARGETS[@]}" \
          --delay_mode "$DELAY_MODE" \
          --model "graphsage+tcn" \
          --softplus_outputs 1 \
          --idle_aware_weighting 1
        ;;
      *.sh)
        run bash "$TRAIN_SCRIPT" \
          "$NPZ_IN" "$RUN_DIR" "$seed" "$LEAD" "$DELAY_MODE"
        ;;
      *)
        echo "ERROR: Unsupported training script type: $TRAIN_SCRIPT" >&2
        exit 1
        ;;
    esac
  done
else
  log "Step (4) TRAIN skipped (DO_TRAIN=0)"
fi

###############################################################################
# (5) EVAL ensemble + CALIBRATION
###############################################################################
if [[ "$DO_EVAL" == "1" ]]; then
  log "Step (5) EVAL enabled"

  EVAL_SCRIPT="$(pick_first_existing \
    "${REPO_ROOT}/eval/evaluate_ensemble.py" \
    "${REPO_ROOT}/evaluate.py" \
    "${REPO_ROOT}/scripts/eval.sh" \
    "${REPO_ROOT}/scripts/run_eval.sh" \
  )" || {
    echo "ERROR: No evaluation script found. Add yours to candidate list." >&2
    exit 1
  }

  OUT_DIR="${RUNS_ROOT}/${EXP_TAG}/ensemble_eval"
  mkdir -p "$OUT_DIR"

  # Collect run dirs
  RUN_DIRS=()
  for seed in "${SEEDS[@]}"; do
    RUN_DIRS+=("${RUNS_ROOT}/${EXP_TAG}/seed_${seed}")
  done

  case "$EVAL_SCRIPT" in
    *.py)
      run "$PYTHON_BIN" "$EVAL_SCRIPT" \
        --run_dirs "${RUN_DIRS[@]}" \
        --data "$NPZ_IN" \
        --out "$OUT_DIR" \
        --lead "$LEAD" \
        --targets "${TARGETS[@]}" \
        --delay_mode "$DELAY_MODE" \
        --calibrate_targets "${CALIBRATE_TARGETS[@]}"
      ;;
    *.sh)
      run bash "$EVAL_SCRIPT" "$NPZ_IN" "$OUT_DIR"
      ;;
    *)
      echo "ERROR: Unsupported eval script type: $EVAL_SCRIPT" >&2
      exit 1
      ;;
  esac
else
  log "Step (5) EVAL skipped (DO_EVAL=0)"
fi

###############################################################################
# (6) BASELINES (zero, persistence, AR(1), Kalman random-walk)
###############################################################################
if [[ "$DO_BASELINES" == "1" ]]; then
  log "Step (6) BASELINES enabled"

  BASELINE_SCRIPT="$(pick_first_existing \
    "${REPO_ROOT}/baselines/run_baselines.py" \
    "${REPO_ROOT}/eval/baselines.py" \
    "${REPO_ROOT}/scripts/baselines.sh" \
  )" || {
    echo "ERROR: No baseline script found. Add yours to candidate list." >&2
    exit 1
  }

  OUT_DIR="${RUNS_ROOT}/${EXP_TAG}/baselines"
  mkdir -p "$OUT_DIR"

  case "$BASELINE_SCRIPT" in
    *.py)
      run "$PYTHON_BIN" "$BASELINE_SCRIPT" \
        --data "$NPZ_IN" \
        --out "$OUT_DIR" \
        --lead "$LEAD" \
        --targets "${TARGETS[@]}" \
        --delay_mode "$DELAY_MODE" \
        --baselines "zero,persistence,ar1,kalman_rw"
      ;;
    *.sh)
      run bash "$BASELINE_SCRIPT" "$NPZ_IN" "$OUT_DIR"
      ;;
    *)
      echo "ERROR: Unsupported baseline script type: $BASELINE_SCRIPT" >&2
      exit 1
      ;;
  esac
else
  log "Step (6) BASELINES skipped (DO_BASELINES=0)"
fi

###############################################################################
# (7) TABLES / FIGURES export
###############################################################################
if [[ "$DO_TABLES" == "1" ]]; then
  log "Step (7) TABLES enabled"

  TABLES_SCRIPT="$(pick_first_existing \
    "${REPO_ROOT}/report/export_tables.py" \
    "${REPO_ROOT}/analysis/make_tables.py" \
    "${REPO_ROOT}/scripts/make_tables.sh" \
  )" || {
    echo "WARNING: No tables export script found. Skipping tables step."
    TABLES_SCRIPT=""
  }

  if [[ -n "$TABLES_SCRIPT" ]]; then
    OUT_DIR="${RUNS_ROOT}/${EXP_TAG}/tables"
    mkdir -p "$OUT_DIR"
    case "$TABLES_SCRIPT" in
      *.py)
        run "$PYTHON_BIN" "$TABLES_SCRIPT" \
          --exp_dir "${RUNS_ROOT}/${EXP_TAG}" \
          --out "$OUT_DIR"
        ;;
      *.sh)
        run bash "$TABLES_SCRIPT" "${RUNS_ROOT}/${EXP_TAG}" "$OUT_DIR"
        ;;
    esac
  fi
else
  log "Step (7) TABLES skipped (DO_TABLES=0)"
fi

log "Reproduction completed successfully"
log "Outputs:"
log "  RUNS: ${RUNS_ROOT}/${EXP_TAG}"
log "  NPZ:  ${NPZ_IN}"

