#!/usr/bin/env python3
"""
eval_gnn_multitarget.py

Evaluate multi-target GNN checkpoints on dataset_multi*.npz.

Supports:
  - mode: nowcast or lead1
  - ensemble over multiple ckpts
  - optional calibration (fit on VAL, apply on TEST)
  - optional residual delay reconstruction:
      * --delay_residual: delay_hat(t+1)=clamp(Y_delay(t)+delta_pred, 0, cap)
      * --delay_queue_residual: delay_hat(t+1)=clamp(queue_hat_pkts(t+1)*pkt_ms + delta_pred, 0, cap)

Assumptions:
  Y[...,0]=queue_pkts
  Y[...,1]=throughput_Mbps
  Y[...,2]=utilization
  Y[...,3]=delay_ms (qdelay_ms)
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models_gnn import GNNEncoder
from models_gnn_multitask import MultiNowcastHead, MultiTCNHead, MultiGRUHead


def load_npz(p: str) -> Dict[str, np.ndarray]:
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def breakdown(true: np.ndarray, pred: np.ndarray, thr: float, prefix: str) -> Dict[str, float]:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    idle = (true <= float(thr))
    busy = ~idle
    out = {
        f"{prefix}_global_rmse": rmse(pred.reshape(-1), true.reshape(-1)),
        f"{prefix}_idle_rmse": rmse(pred[idle], true[idle]) if idle.any() else float("nan"),
        f"{prefix}_busy_rmse": rmse(pred[busy], true[busy]) if busy.any() else float("nan"),
        f"{prefix}_idle_fp": float((pred[idle] > float(thr)).mean()) if idle.any() else float("nan"),
        f"{prefix}_idle_mean": float(pred[idle].mean()) if idle.any() else float("nan"),
    }
    return out


def apply_acts(y_raw: torch.Tensor, delay_linear: bool) -> torch.Tensor:
    """
    Apply target constraints:
      queue/thr: softplus
      util: sigmoid
      delay: softplus unless delay_linear True (signed delta)
    """
    y = y_raw.clone()
    if y.size(-1) >= 1:
        y[..., 0] = F.softplus(y_raw[..., 0])
    if y.size(-1) >= 2:
        y[..., 1] = F.softplus(y_raw[..., 1])
    if y.size(-1) >= 3:
        y[..., 2] = torch.sigmoid(y_raw[..., 2])
    if y.size(-1) >= 4:
        y[..., 3] = y_raw[..., 3] if delay_linear else F.softplus(y_raw[..., 3])
    return y


def fit_soft_then_scale(true: np.ndarray, pred: np.ndarray, idle_mask: np.ndarray, busy_mask: np.ndarray, q: float) -> Tuple[float, float]:
    """
    SOFT+SCALE:
      tau = quantile(pred[idle], q)
      adj = max(0, pred - tau)
      alpha = argmin ||alpha*adj - true|| on busy region
    """
    q = float(np.clip(q, 0.0, 1.0))
    if idle_mask.any():
        tau = float(np.quantile(pred[idle_mask], q))
    else:
        tau = 0.0
    adj = np.maximum(0.0, pred - tau)
    if busy_mask.any() and (adj[busy_mask] ** 2).sum() > 1e-12:
        alpha = float((adj[busy_mask] * true[busy_mask]).sum() / ((adj[busy_mask] ** 2).sum() + 1e-9))
    else:
        alpha = 1.0
    return tau, alpha


def apply_soft_then_scale(pred: np.ndarray, tau: float, alpha: float) -> np.ndarray:
    return float(alpha) * np.maximum(0.0, pred - float(tau))


def pkt_ms(pkt_bytes: float, bw_mbps: float) -> float:
    return float(pkt_bytes) * 8.0 / (float(bw_mbps) * 1e6) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["nowcast", "lead1"], default="nowcast")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)

    ap.add_argument("--temporal", choices=["tcn", "gru"], default="tcn")
    ap.add_argument("--K", type=int, default=30)

    ap.add_argument("--busy_thr", type=float, default=50.0)

    # Calibration
    ap.add_argument("--calibrate_queue", action="store_true")
    ap.add_argument("--calib_q", type=float, default=0.995)

    ap.add_argument("--calibrate_thr", action="store_true")
    ap.add_argument("--thr_idle_thr", type=float, default=0.05)
    ap.add_argument("--thr_busy_thr", type=float, default=0.20)
    ap.add_argument("--thr_calib_q", type=float, default=0.995)

    ap.add_argument("--calibrate_delay", action="store_true")
    ap.add_argument("--delay_idle_thr", type=float, default=1.0)
    ap.add_argument("--delay_busy_thr", type=float, default=5.0)
    ap.add_argument("--delay_calib_q", type=float, default=0.995)

    # Residual delay reconstruction
    ap.add_argument("--delay_residual", action="store_true",
                    help="Model delay head outputs signed delta; for lead1 reconstruct delay(t+1)=Y_delay(t)+delta.")
    ap.add_argument("--delay_queue_residual", action="store_true",
                    help="Reconstruct delay(t+1)=queue_hat_pkts(t+1)*pkt_ms + delta (lead1).")
    ap.add_argument("--bw_bottleneck", type=float, default=2.0)
    ap.add_argument("--pkt_bytes", type=float, default=1500.0)
    ap.add_argument("--delay_recon_cap", type=float, default=None)

    ap.add_argument("--out_csv", default="gnn_multitarget_metrics.csv")

    args = ap.parse_args()

    if args.delay_residual and args.delay_queue_residual:
        raise ValueError("Choose only one: --delay_residual OR --delay_queue_residual")

    Z = load_npz(args.npz)
    Xn = Z["X"].astype(np.float32)          # [T,N,F]
    Yn = Z["Y"].astype(np.float32)          # [T,N,Kt]
    edges_n = Z["edges"].astype(np.int64)

    train_idx = Z.get("train_idx", None)
    val_idx = Z.get("val_idx", None)
    test_idx = Z.get("test_idx", None)
    if test_idx is None or val_idx is None:
        raise RuntimeError("NPZ must contain val_idx and test_idx")

    T, N, Fin = Xn.shape
    Kt = Yn.shape[2]

    # label names if present
    lbl = Z.get("label_names", None)
    if lbl is not None:
        label_names = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in lbl]
    else:
        label_names = ["queue_pkts", "throughput_Mbps", "utilization", "qdelay_ms"][:Kt]

    # non-sensor mask if present
    non_sensor = None
    if "is_sensor" in Z:
        non_sensor = (~Z["is_sensor"].astype(bool))
    elif "sensor_mask" in Z:
        non_sensor = (~Z["sensor_mask"].astype(bool))

    # caps
    if args.delay_recon_cap is None and Kt >= 4:
        args.delay_recon_cap = float(np.max(Yn[..., 3]))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(Xn).to(dev)
    Y = torch.from_numpy(Yn).to(dev)
    edges = torch.from_numpy(edges_n).to(dev)

    # build models
    models = []
    for p in args.ckpts:
        ck = torch.load(p, map_location=dev)

        # infer hid from encoder weights (fallback 96)
        hid = None
        for k, v in ck["enc"].items():
            if k.endswith("lin.weight"):
                hid = int(v.shape[0])
                break
        if hid is None:
            hid = 96

        # infer number of encoder layers (must match training)
        # 1) prefer checkpoint meta if present
        layers = None
        meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
        if isinstance(meta, dict) and "layers" in meta:
            try:
                layers = int(meta["layers"])
            except Exception:
                layers = None

        # 2) otherwise infer from state_dict keys like: layers.0.*, layers.1.*, ...
        if layers is None:
            import re

            max_i = -1
            for k in ck["enc"].keys():
                m = re.match(r"^layers\.(\d+)\.", str(k))
                if m:
                    max_i = max(max_i, int(m.group(1)))
            layers = (max_i + 1) if max_i >= 0 else 2

        enc = GNNEncoder(in_dim=Fin, hid=hid, layers=layers, dropout=0.0).to(dev)
        enc.load_state_dict(ck["enc"], strict=True)

        if args.mode == "nowcast":
            head = MultiNowcastHead(hid=hid, out_dim=Kt).to(dev)
        else:
            if args.temporal == "tcn":
                # MultiTCNHead signature: (hid, K, out_dim)
                head = MultiTCNHead(hid=hid, K=int(args.K), out_dim=Kt).to(dev)
            else:
                head = MultiGRUHead(hid=hid, out_dim=Kt).to(dev)

        head.load_state_dict(ck["head"])
        enc.eval()
        head.eval()
        models.append((enc, head, hid))

    # prediction helper
    def predict_for_times(times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times, dtype=np.int64)

        if args.mode == "nowcast":
            tt = times[(times >= 0) & (times < T)]
            true = Yn[tt]  # [Tt,N,Kt]
            preds = []
            with torch.no_grad():
                for t0 in tt.tolist():
                    yh = []
                    for enc, head, _hid in models:
                        y_raw = head(enc(X[t0], edges))
                        y = apply_acts(y_raw, delay_linear=bool(args.delay_residual or args.delay_queue_residual))
                        # residual modes are intended for lead1; do not reconstruct in nowcast
                        yh.append(y)
                    preds.append(torch.stack(yh, 0).mean(0))
            pred = torch.stack(preds, 0).cpu().numpy()
            return true, pred

        # lead1
        Kwin = int(args.K)
        tt = times[(times >= Kwin - 1) & (times < T - 1)]
        true = Yn[tt + 1]  # [Tt,N,Kt]

        preds = []
        with torch.no_grad():
            for t0 in tt.tolist():
                yh = []
                for enc, head, _hid in models:
                    Hseq = torch.stack([enc(X[s], edges) for s in range(t0 - Kwin + 1, t0 + 1)], 0)  # [K,N,H]
                    y_raw = head(Hseq)  # [N,Kt]
                    y = apply_acts(y_raw, delay_linear=bool(args.delay_residual or args.delay_queue_residual))

                    # reconstruct absolute delay if residual mode enabled
                    if (args.delay_residual or args.delay_queue_residual) and Kt >= 4:
                        y = y.clone()
                        delta = y_raw[:, 3]  # signed
                        if args.delay_residual:
                            base = Y[t0, :, 3]  # current true delay
                        else:
                            base = y[:, 0] * float(pkt_ms(args.pkt_bytes, args.bw_bottleneck))  # queue baseline
                        d = base + delta
                        y[:, 3] = torch.clamp(d, min=0.0, max=float(args.delay_recon_cap))
                    yh.append(y)
                preds.append(torch.stack(yh, 0).mean(0))
        pred = torch.stack(preds, 0).cpu().numpy()
        return true, pred

    # select indices
    test_idx = np.asarray(test_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)

    if args.mode == "nowcast":
        eval_t = test_idx
        fit_t = val_idx
    else:
        eval_t = test_idx
        fit_t = val_idx

    # predict test and val
    true_test, pred_test = predict_for_times(eval_t)
    true_val, pred_val = predict_for_times(fit_t)

    # calibrations (fit on val, apply to test)
    # Queue calibration is always safe
    if args.calibrate_queue and Kt >= 1:
        thr = float(args.busy_thr)
        qt = true_val[..., 0]
        qp = pred_val[..., 0]
        idle = (qt < thr)
        busy = ~idle
        tau, alpha = fit_soft_then_scale(qt, qp, idle, busy, q=float(args.calib_q))
        pred_test[..., 0] = apply_soft_then_scale(pred_test[..., 0], tau, alpha)

    # Throughput calibration
    if args.calibrate_thr and Kt >= 2:
        tthr_i = float(args.thr_idle_thr)
        tthr_b = float(args.thr_busy_thr)
        tt = true_val[..., 1]
        tp = pred_val[..., 1]
        idle = (tt <= tthr_i)
        busy = (tt >= tthr_b)
        tau, alpha = fit_soft_then_scale(tt, tp, idle, busy, q=float(args.thr_calib_q))
        pred_test[..., 1] = apply_soft_then_scale(pred_test[..., 1], tau, alpha)

    # Delay calibration: only if NOT residual mode (residual delay can be signed internally)
    if args.calibrate_delay and Kt >= 4 and (not (args.delay_residual or args.delay_queue_residual)):
        dthr_i = float(args.delay_idle_thr)
        dthr_b = float(args.delay_busy_thr)
        dt = true_val[..., 3]
        dp = pred_val[..., 3]
        idle = (dt <= dthr_i)
        busy = (dt >= dthr_b)
        tau, alpha = fit_soft_then_scale(dt, dp, idle, busy, q=float(args.delay_calib_q))
        pred_test[..., 3] = apply_soft_then_scale(pred_test[..., 3], tau, alpha)

    # write metrics
    out_fields = [
        "mode", "target", "rmse_global", "rmse_nonSensor",
        "queue_global_rmse", "queue_idle_rmse", "queue_busy_rmse", "queue_idle_fp", "queue_idle_mean",
        "thr_global_rmse", "thr_idle_rmse", "thr_busy_rmse", "thr_idle_fp", "thr_idle_mean",
        "delay_global_rmse", "delay_idle_rmse", "delay_busy_rmse", "delay_idle_fp", "delay_idle_mean",
    ]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()

        for k, name in enumerate(label_names):
            Tk = true_test[..., k]
            Pk = pred_test[..., k]

            row = {
                "mode": args.mode,
                "target": name,
                "rmse_global": rmse(Pk.reshape(-1), Tk.reshape(-1)),
                "rmse_nonSensor": float("nan"),
                "queue_global_rmse": "", "queue_idle_rmse": "", "queue_busy_rmse": "", "queue_idle_fp": "", "queue_idle_mean": "",
                "thr_global_rmse": "", "thr_idle_rmse": "", "thr_busy_rmse": "", "thr_idle_fp": "", "thr_idle_mean": "",
                "delay_global_rmse": "", "delay_idle_rmse": "", "delay_busy_rmse": "", "delay_idle_fp": "", "delay_idle_mean": "",
            }

            if non_sensor is not None:
                row["rmse_nonSensor"] = rmse(Pk[:, non_sensor].reshape(-1), Tk[:, non_sensor].reshape(-1))

            if k == 0:
                row.update(breakdown(Tk, Pk, thr=float(args.busy_thr), prefix="queue"))
            if k == 1:
                row.update(breakdown(Tk, Pk, thr=float(args.thr_idle_thr), prefix="thr"))
            if k == 3:
                row.update(breakdown(Tk, Pk, thr=float(args.delay_idle_thr), prefix="delay"))

            w.writerow(row)

    print(f"OK: wrote {args.out_csv}")


if __name__ == "__main__":
    main()

