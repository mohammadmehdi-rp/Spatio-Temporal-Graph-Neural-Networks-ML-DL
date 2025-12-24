#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config (override via env vars)
# -----------------------------
IMG="${IMG:-ndt/host:focal-nettools}"
N_HOSTS="${N_HOSTS:-4}"

# Capture runs (3 runs, stored in runs/dumbbell_seed{1,2,3})
CAPTURE_RUNS=(${CAPTURE_RUNS:-1 2 3})

# Training seeds for GNN models
TRAIN_SEEDS=(${TRAIN_SEEDS:-42 123 999})

# Lead-1 model params
ENCODER="${ENCODER:-sage}"
TEMPORAL="${TEMPORAL:-tcn}"
K="${K:-30}"

# Busy threshold used in tables
BUSY_THR="${BUSY_THR:-50}"

# Dumbbell shaping + offered load (chosen to reliably create backlog)
BW_BOTTLENECK="${BW_BOTTLENECK:-2}"     # Mbit
UNDER_MBPS="${UNDER_MBPS:-1}"          # aggregate Mbps underload
OVER_MBPS="${OVER_MBPS:-14}"           # aggregate Mbps overload

# Idle-aware nowcast loss weight
IDLE_W="${IDLE_W:-2.0}"

PY="python3 -u"

# -----------------------------
# Helpers
# -----------------------------
cleanup_net() {
  # Clean Mininet/OVS and leftover docker hosts (avoid container name conflicts like /h1 already in use)
  sudo mn -c >/dev/null 2>&1 || true
  sudo ovs-vsctl --if-exists del-br s1 >/dev/null 2>&1 || true
  sudo ovs-vsctl --if-exists del-br s2 >/dev/null 2>&1 || true
  sudo docker ps -aq --filter "name=^/h[0-9]+$" | xargs -r sudo docker rm -f >/dev/null 2>&1 || true
  sudo pkill -f "iperf3 -s" >/dev/null 2>&1 || true
  sudo pkill -f "iperf3 -c" >/dev/null 2>&1 || true
}

need_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing file: $f"
    exit 2
  fi
}

need_file "src/run_dumbbell_capture.py"
need_file "src/train_nowcast_sparse.py"
need_file "src/train_lead1_sparse.py"
need_file "src/eval_nowcast_baselines.py"
need_file "src/calibrate_soft_then_scale.py"
need_file "src/models_gnn.py"

# -----------------------------
# Main loop
# -----------------------------
for rid in "${CAPTURE_RUNS[@]}"; do
  OUTDIR="runs/dumbbell_seed${rid}"
  NPZ="${OUTDIR}/dataset.npz"
  LOGDIR="${OUTDIR}/logs"
  MODELDIR="${OUTDIR}/models"
  METDIR="${OUTDIR}/metrics"

  mkdir -p "$LOGDIR" "$MODELDIR" "$METDIR"

  echo "============================================================"
  echo "[RUN ${rid}] Capture -> Train -> Eval  (outdir: $OUTDIR)"
  echo "============================================================"

  echo "[1/6] Cleanup old network state..."
  cleanup_net

  echo "[2/6] Capture dumbbell dataset (root required)..."
  sudo -E PYTHONUNBUFFERED=1 ${PY} src/run_dumbbell_capture.py \
    --outdir "$OUTDIR" \
    --n "$N_HOSTS" \
    --img "$IMG" \
    --bw_bottleneck "$BW_BOTTLENECK" \
    --under_mbps "$UNDER_MBPS" \
    --over_mbps "$OVER_MBPS" \
    --force_tbf --force_tbf_both \
    |& tee "${LOGDIR}/capture.log"

  # Fix ownership so later scripts can write (avoids PermissionError)
  sudo chown -R "${USER}:${USER}" "$OUTDIR"

  echo "[3/6] Nowcast baselines (defendable, no leakage) at thr=${BUSY_THR}..."
  ${PY} src/eval_nowcast_baselines.py \
    --npz "$NPZ" \
    --busy_thr "$BUSY_THR" \
    --out_csv "${METDIR}/nowcast_baselines_thr${BUSY_THR}.csv" \
    |& tee "${LOGDIR}/nowcast_baselines_thr${BUSY_THR}.log"

  echo "[4/6] Train NOWCAST (idle-aware + softplus) for seeds: ${TRAIN_SEEDS[*]}..."
  for s in "${TRAIN_SEEDS[@]}"; do
    ${PY} src/train_nowcast_sparse.py \
      --data "$NPZ" \
      --encoder "$ENCODER" \
      --softplus \
      --idle_w "$IDLE_W" \
      --seed "$s" \
      --out "${MODELDIR}/nowcast_${ENCODER}_idlefix_seed${s}.pt" \
      |& tee "${LOGDIR}/train_nowcast_${ENCODER}_idlefix_seed${s}.log"
  done

  echo "[5/6] Train LEAD-1 (${ENCODER}+${TEMPORAL}, K=${K}) for seeds: ${TRAIN_SEEDS[*]}..."
  for s in "${TRAIN_SEEDS[@]}"; do
    ${PY} src/train_lead1_sparse.py \
      --data "$NPZ" \
      --encoder "$ENCODER" \
      --temporal "$TEMPORAL" \
      --K "$K" \
      --seed "$s" \
      --out "${MODELDIR}/lead1_${ENCODER}_${TEMPORAL}K${K}_seed${s}.pt" \
      |& tee "${LOGDIR}/train_lead1_${ENCODER}_${TEMPORAL}K${K}_seed${s}.log"
  done

  echo "[6/6] Eval + Calibration + Table metrics (thr=${BUSY_THR})..."

  # --- calibrate lead-1 (soft+scale) and save the console output for tau/alpha parsing
  ${PY} src/calibrate_soft_then_scale.py \
    --data "$NPZ" \
    --ckpts \
      "${MODELDIR}/lead1_${ENCODER}_${TEMPORAL}K${K}_seed42.pt" \
      "${MODELDIR}/lead1_${ENCODER}_${TEMPORAL}K${K}_seed123.pt" \
      "${MODELDIR}/lead1_${ENCODER}_${TEMPORAL}K${K}_seed999.pt" \
    |& tee "${LOGDIR}/lead1_calibrate_soft_scale.log"

  # --- Compute all final metrics (GNN nowcast / lead1 raw+cal + KF lead1) into CSVs
  BUSY_THR="$BUSY_THR" ${PY} - <<'PY'
import os, re, csv, sys
import numpy as np
import torch
import torch.nn.functional as F

BUSY_THR = float(os.environ.get("BUSY_THR", "50"))
OUTDIR = os.environ.get("OUTDIR", "")
if not OUTDIR:
    # infer from cwd by searching known runs folder structure; fallback to sys.argv not used here
    # but we know script runs inside loop and OUTDIR isn't exported -> derive from log path
    # so just use current shell substitution by reading env not possible; we hardcode by scanning runs/
    # We'll instead detect newest runs/dumbbell_seed*/dataset.npz modified most recently.
    import glob, pathlib
    cand = sorted(glob.glob("runs/dumbbell_seed*/dataset.npz"), key=lambda p: pathlib.Path(p).stat().st_mtime)
    if not cand:
        raise RuntimeError("Cannot find runs/dumbbell_seed*/dataset.npz")
    npz_path = cand[-1]
    OUTDIR = str(pathlib.Path(npz_path).parent)
else:
    npz_path = os.path.join(OUTDIR, "dataset.npz")

LOGDIR = os.path.join(OUTDIR, "logs")
MODELDIR = os.path.join(OUTDIR, "models")
METDIR = os.path.join(OUTDIR, "metrics")

sys.path.insert(0, "src")
from models_gnn import GNNEncoder, NowcastHead, TCNHead, GRUHead

def softplus_np(x):
    x = x.astype(np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def rmse(a,b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a-b)**2))) if a.size else float("nan")

def split_metrics(y_true, y_pred, thr):
    idle = (y_true < thr)
    busy = ~idle
    out = {}
    out["global_rmse"] = rmse(y_pred.reshape(-1), y_true.reshape(-1))
    out["idle_rmse"]   = rmse(y_pred[idle], y_true[idle]) if idle.any() else float("nan")
    out["busy_rmse"]   = rmse(y_pred[busy], y_true[busy]) if busy.any() else float("nan")
    out["idle_fp"]     = float((y_pred[idle] >= thr).mean()) if idle.any() else float("nan")
    out["idle_mean"]   = float(y_pred[idle].mean()) if idle.any() else float("nan")
    out["n_idle"]      = int(idle.sum())
    out["n_busy"]      = int(busy.sum())
    return out

Z = np.load(npz_path, allow_pickle=True)
X = torch.from_numpy(Z["X"]).float()
Y = Z["Y"].astype(np.float64)
E = torch.from_numpy(Z["edges"]).long()
test_idx = Z["test_idx"].astype(int)
val_idx  = Z["val_idx"].astype(int)

Fin = X.shape[2]

# -------------------
# NOWCAST ensemble
# -------------------
now_ckpts = [
    os.path.join(MODELDIR, "nowcast_sage_idlefix_seed42.pt"),
    os.path.join(MODELDIR, "nowcast_sage_idlefix_seed123.pt"),
    os.path.join(MODELDIR, "nowcast_sage_idlefix_seed999.pt"),
]
now_preds = []

for p in now_ckpts:
    ck = torch.load(p, map_location="cpu")
    m = ck["meta"]
    enc = GNNEncoder(Fin, hid=m["hid"], layers=m["layers"], kind=m["encoder"], dropout=m["dropout"])
    head = NowcastHead(hid=m["hid"])
    enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
    enc.eval(); head.eval()
    P = []
    with torch.no_grad():
        for t in test_idx.tolist():
            h = enc(X[t], E)
            yhat = head(h).squeeze(-1)
            # nowcast checkpoints are trained with softplus -> enforce non-negativity
            if m.get("softplus", False):
                yhat = F.softplus(yhat)
            P.append(yhat.numpy())
    now_preds.append(np.stack(P, 0))

P_now = np.mean(np.stack(now_preds, 0), 0)    # [Ttest,N]
Y_now = Y[test_idx]                          # [Ttest,N]
m_now = split_metrics(Y_now, P_now, BUSY_THR)

# save nowcast metrics
with open(os.path.join(METDIR, f"gnn_nowcast_thr{int(BUSY_THR)}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method","global_rmse","idle_rmse","busy_rmse","idle_fp","idle_mean","n_idle","n_busy"])
    w.writerow(["ST-GNN(nowcast ensemble, idlefix+softplus)",
                m_now["global_rmse"], m_now["idle_rmse"], m_now["busy_rmse"],
                m_now["idle_fp"], m_now["idle_mean"], m_now["n_idle"], m_now["n_busy"]])

# -------------------
# LEAD-1 ensemble (aligned valid positions)
# -------------------
lead_ckpts = [
    os.path.join(MODELDIR, "lead1_sage_tcnK30_seed42.pt"),
    os.path.join(MODELDIR, "lead1_sage_tcnK30_seed123.pt"),
    os.path.join(MODELDIR, "lead1_sage_tcnK30_seed999.pt"),
]

# Determine intersection of valid test positions across ckpts (different K possible)
test_set = set(test_idx.tolist())
valid_lists = []
metas = []
for p in lead_ckpts:
    ck = torch.load(p, map_location="cpu")
    m = ck["meta"]; metas.append(m)
    K = int(m.get("K", 20))
    valid = [t for t in range(K, X.shape[0]-1) if t in test_set]
    valid_lists.append(set(valid))

valid = sorted(set.intersection(*valid_lists))
if not valid:
    raise RuntimeError("No valid lead-1 positions in test split after K alignment.")

def lead1_pred_one(ckpt_path, meta, valid_pos):
    K = int(meta.get("K", 20))
    enc = GNNEncoder(Fin, hid=meta["hid"], layers=meta["layers"], kind=meta["encoder"], dropout=meta["dropout"])
    if meta.get("temporal","tcn") == "tcn":
        head = TCNHead(hid=meta["hid"], K=K)
    else:
        head = GRUHead(hid=meta["hid"])
    ck = torch.load(ckpt_path, map_location="cpu")
    enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
    enc.eval(); head.eval()
    P = []
    with torch.no_grad():
        for t in valid_pos:
            H = [enc(X[s], E) for s in range(t-K+1, t+1)]
            yhat = head(torch.stack(H, 0)).numpy()  # raw
            P.append(yhat)
    return np.stack(P, 0)  # [Tvalid,N]

lead_preds = []
for p, m in zip(lead_ckpts, metas):
    lead_preds.append(lead1_pred_one(p, m, valid))

P_lead_raw = np.mean(np.stack(lead_preds, 0), 0)  # [Tvalid,N]
Y_lead = np.stack([Y[t+1] for t in valid], 0)     # [Tvalid,N]

# Posthoc non-negativity (what we used in the conversation)
P_lead_sp = softplus_np(P_lead_raw)

m_lead_raw = split_metrics(Y_lead, P_lead_sp, BUSY_THR)

# -------------------
# Parse tau/alpha from calibration log and compute calibrated metrics
# -------------------
log_path = os.path.join(LOGDIR, "lead1_calibrate_soft_scale.log")
txt = open(log_path, "r", encoding="utf-8", errors="ignore").read()
m = re.search(r"tau=([0-9.]+).*alpha=([0-9.]+)", txt)
if not m:
    raise RuntimeError("Could not parse tau/alpha from lead1_calibrate_soft_scale.log")
tau = float(m.group(1)); alpha = float(m.group(2))

P_lead_cal = alpha * np.maximum(0.0, P_lead_sp - tau)
m_lead_cal = split_metrics(Y_lead, P_lead_cal, BUSY_THR)

with open(os.path.join(METDIR, f"gnn_lead1_thr{int(BUSY_THR)}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method","global_rmse","idle_rmse","busy_rmse","idle_fp","idle_mean","n_idle","n_busy","tau","alpha"])
    w.writerow(["ST-GNN(lead1 ensemble, softplus posthoc)",
                m_lead_raw["global_rmse"], m_lead_raw["idle_rmse"], m_lead_raw["busy_rmse"],
                m_lead_raw["idle_fp"], m_lead_raw["idle_mean"], m_lead_raw["n_idle"], m_lead_raw["n_busy"], "", ""])
    w.writerow([f"ST-GNN(lead1 SOFT+SCALE; tau={tau:.3f}, alpha={alpha:.3f})",
                m_lead_cal["global_rmse"], m_lead_cal["idle_rmse"], m_lead_cal["busy_rmse"],
                m_lead_cal["idle_fp"], m_lead_cal["idle_mean"], m_lead_cal["n_idle"], m_lead_cal["n_busy"], tau, alpha])

# -------------------
# KF lead-1 metrics (same format)
# -------------------
Q = 50.0
R = 200.0
T, Np = Y.shape
x = Y[0].copy()
Pcov = np.ones(Np, dtype=np.float64)

xf = np.zeros((T, Np), dtype=np.float64)
xf[0] = x

for t in range(1, T):
    x_pred = x
    P_pred = Pcov + Q
    z = Y[t]
    Kk = P_pred / (P_pred + R)
    x = x_pred + Kk * (z - x_pred)
    Pcov = (1.0 - Kk) * P_pred
    xf[t] = x

# evaluate lead-1 on test_idx: predict Y[t+1] using xf[t]
test_set = set(test_idx.tolist())
eval_t = [t for t in range(T-1) if t in test_set]
Y_kf = np.stack([Y[t+1] for t in eval_t], 0)
P_kf = np.stack([xf[t]  for t in eval_t], 0)

m_kf = split_metrics(Y_kf, P_kf, BUSY_THR)

with open(os.path.join(METDIR, f"kf_lead1_thr{int(BUSY_THR)}.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method","global_rmse","idle_rmse","busy_rmse","idle_fp","idle_mean","n_idle","n_busy","Q","R"])
    w.writerow(["Kalman(RW lead1)", m_kf["global_rmse"], m_kf["idle_rmse"], m_kf["busy_rmse"],
                m_kf["idle_fp"], m_kf["idle_mean"], m_kf["n_idle"], m_kf["n_busy"], Q, R])

print(f"[OK] Wrote metrics into: {METDIR}")
print(f"  - nowcast_baselines_thr{int(BUSY_THR)}.csv")
print(f"  - gnn_nowcast_thr{int(BUSY_THR)}.csv")
print(f"  - gnn_lead1_thr{int(BUSY_THR)}.csv")
print(f"  - kf_lead1_thr{int(BUSY_THR)}.csv")
PY

  echo "[DONE] Run ${rid} complete. Outputs in: ${OUTDIR}/metrics"
done

echo
echo "All runs finished."
echo "Tip: main tables usually use runs/dumbbell_seed3/metrics/*thr${BUSY_THR}.csv"
