#!/usr/bin/env python3
"""eval_multitarget_baselines.py

Baselines for multi-target NPZ datasets.

Adds:
  - Creates parent directory for out_csv
  - Optional throughput and delay breakdown (idle/busy) similar to queue
"""

import argparse
import csv
from pathlib import Path
from typing import Dict

import numpy as np


def load_npz(p: str) -> Dict[str, np.ndarray]:
    D = np.load(p, allow_pickle=True)
    return {k: D[k] for k in D.files}


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else float("nan")


def fit_ar1(Y: np.ndarray, train_t: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    T, N = Y.shape
    t = np.asarray(train_t, int)
    t = t[(t >= 0) & (t < T - 1)]
    a = np.zeros(N, np.float64)
    for i in range(N):
        x = Y[t, i]
        z = Y[t + 1, i]
        m = (x != 0) | (z != 0)
        if m.sum() < 3:
            x2 = (x * x).sum() + lam
            xz = (x * z).sum()
        else:
            x2 = (x[m] * x[m]).sum() + lam
            xz = (x[m] * z[m]).sum()
        a[i] = xz / x2 if x2 > 0 else 0.0
    return a.astype(np.float64)


def kalman_random_walk_lead1(y: np.ndarray, Q: float = 50.0, R: float = 200.0) -> np.ndarray:
    T, N = y.shape
    x = np.zeros(N, np.float64)
    P = np.full(N, 1e3, np.float64)
    out = np.zeros((T, N), np.float64)
    for t in range(T):
        P = P + Q
        K = P / (P + R)
        x = x + K * (y[t] - x)
        P = (1 - K) * P
        out[t] = x
    return out


def breakdown(y_true: np.ndarray, y_pred: np.ndarray, thr: float, prefix: str) -> Dict[str, float]:
    idle = (y_true <= thr)
    busy = ~idle
    return {
        f"{prefix}_global_rmse": rmse(y_pred.reshape(-1), y_true.reshape(-1)),
        f"{prefix}_idle_rmse": rmse(y_pred[idle], y_true[idle]) if idle.any() else float("nan"),
        f"{prefix}_busy_rmse": rmse(y_pred[busy], y_true[busy]) if busy.any() else float("nan"),
        f"{prefix}_idle_fp": float((y_pred[idle] > thr).mean()) if idle.any() else float("nan"),
        f"{prefix}_idle_mean": float(y_pred[idle].mean()) if idle.any() else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="dataset_multi.npz")
    ap.add_argument("--mode", choices=["nowcast", "lead1"], default="lead1")
    ap.add_argument("--busy_thr", type=float, default=50.0)

    ap.add_argument("--thr_idle_thr", type=float, default=0.05)
    ap.add_argument("--delay_idle_thr", type=float, default=1.0)

    ap.add_argument("--out_csv", default="multitarget_baselines.csv")
    ap.add_argument("--kalman", action="store_true")
    ap.add_argument("--Q", type=float, default=50.0)
    ap.add_argument("--R", type=float, default=200.0)
    args = ap.parse_args()

    Z = load_npz(args.npz)
    Y = Z["Y"].astype(np.float64)
    is_sensor = Z.get("is_sensor", None)
    train_idx = Z["train_idx"].astype(int)
    test_idx = Z["test_idx"].astype(int)
    label_names = [str(x) for x in Z.get("label_names", ["queue_pkts", "throughput_Mbps", "utilization", "delay_ms"])]

    T, N, K = Y.shape
    non = (is_sensor.astype(np.uint8) == 0) if is_sensor is not None else None

    # evaluation times
    if args.mode == "nowcast":
        eval_t = test_idx
        Y_true = Y[eval_t]
        Y_prev = np.zeros_like(Y_true)
        for i, t in enumerate(eval_t.tolist()):
            if t - 1 >= 0:
                Y_prev[i] = Y[t - 1]
    else:
        eval_t = test_idx[(test_idx >= 0) & (test_idx < T - 1)]
        Y_true = Y[eval_t + 1]
        Y_prev = Y[eval_t]

    ar1_coef = None
    if args.mode == "lead1":
        ar1_coef = np.stack([fit_ar1(Y[:, :, k], train_idx, lam=1e-6) for k in range(K)], axis=1)

    kf_queue = None
    if args.mode == "lead1" and args.kalman:
        kf_queue = kalman_random_walk_lead1(Y[:, :, 0], Q=args.Q, R=args.R)[eval_t]

    fields = [
        "mode", "method", "target",
        "rmse_global", "rmse_nonSensor",
        "queue_global_rmse", "queue_idle_rmse", "queue_busy_rmse", "queue_idle_fp", "queue_idle_mean",
        "thr_global_rmse", "thr_idle_rmse", "thr_busy_rmse", "thr_idle_fp", "thr_idle_mean",
        "delay_global_rmse", "delay_idle_rmse", "delay_busy_rmse", "delay_idle_fp", "delay_idle_mean",
    ]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        methods = ["zero", "persistence"]
        if args.mode == "lead1":
            methods += ["ar1"]
            if args.kalman:
                methods += ["kalman_queue"]

        for method in methods:
            for k, name in enumerate(label_names):
                if method == "zero":
                    Pk = np.zeros_like(Y_true[:, :, k])
                elif method == "persistence":
                    Pk = Y_prev[:, :, k]
                elif method == "ar1":
                    a = ar1_coef[:, k]
                    Pk = Y_prev[:, :, k] * a[None, :]
                elif method == "kalman_queue":
                    if k != 0:
                        continue
                    Pk = kf_queue
                else:
                    continue

                Tk = Y_true[:, :, k]
                row = {
                    "mode": args.mode,
                    "method": method,
                    "target": name,
                    "rmse_global": rmse(Pk.reshape(-1), Tk.reshape(-1)),
                    "rmse_nonSensor": rmse(Pk[:, non].reshape(-1), Tk[:, non].reshape(-1)) if non is not None else float("nan"),
                    "queue_global_rmse": "", "queue_idle_rmse": "", "queue_busy_rmse": "", "queue_idle_fp": "", "queue_idle_mean": "",
                    "thr_global_rmse": "", "thr_idle_rmse": "", "thr_busy_rmse": "", "thr_idle_fp": "", "thr_idle_mean": "",
                    "delay_global_rmse": "", "delay_idle_rmse": "", "delay_busy_rmse": "", "delay_idle_fp": "", "delay_idle_mean": "",
                }

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
