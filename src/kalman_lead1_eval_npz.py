#!/usr/bin/env python3
"""
Per-port 1D Kalman Filter (random-walk) lead-1 baseline, evaluated on an NPZ dataset:
  x_{t+1} = x_t + w,  z_t = x_t + v,  w~N(0,Q), v~N(0,R)

Inputs:
  --data  dataset .npz (must contain Y [T,N] and test_idx)
Outputs:
  - all-frames micro/macro RMSE on test
  - busy-only  micro/macro RMSE on test (mask = Y>0)

Use this to compare apples-to-apples with your GNN ensemble.
"""
import argparse, numpy as np, math

def rmse(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else float("nan")

def macro_rmse_all(yhat, ytrue):
    # mean per-port RMSE across all test frames
    vals = []
    for j in range(ytrue.shape[1]):
        vals.append(rmse(yhat[:, j], ytrue[:, j]))
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")

def macro_rmse_busy(yhat, ytrue):
    # mean per-port RMSE using only frames where that port is busy
    vals = []
    for j in range(ytrue.shape[1]):
        m = ytrue[:, j] > 0
        if np.any(m):
            vals.append(rmse(yhat[m, j], ytrue[m, j]))
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")

def kf_lead1(Y, Q=50.0, R=200.0):
    """
    Run a scalar KF per port using z_t = Y[t] as measurements.
    Returns lead-1 predictions for all t (pred[t+1] = posterior at t).
    """
    T, N = Y.shape
    pred = np.zeros((T, N), dtype=float)
    # init with first observation
    x = Y[0].astype(float).copy()
    P = np.ones(N, dtype=float) * R
    for t in range(T - 1):
        # predict
        P_prior = P + Q
        # update with current measurement z_t
        z = Y[t].astype(float)
        K = P_prior / (P_prior + R)
        x = x + K * (z - x)
        P = (1.0 - K) * P_prior
        # lead-1 prediction for next step
        pred[t + 1] = x
    return pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset .npz (e.g., dataset_sparse_v4_fix.npz)")
    ap.add_argument("--Q", type=float, default=50.0, help="process noise variance")
    ap.add_argument("--R", type=float, default=200.0, help="measurement noise variance")
    args = ap.parse_args()

    Z = np.load(args.data, allow_pickle=True)
    Y = Z["Y"]              # [T,N]
    test_idx = Z["test_idx"]  # [Ttest]
    T, N = Y.shape

    pred = kf_lead1(Y, Q=args.Q, R=args.R)

    # Evaluate on valid test positions (t where we have pred for t+1)
    valid = [int(t) for t in test_idx if t < T - 1]
    Yhat = np.stack([pred[t + 1] for t in valid])   # [Ttest_valid, N]
    Ytrue = np.stack([Y[t + 1] for t in valid])     # [Ttest_valid, N]

    # All-frames
    micro_all = rmse(Yhat, Ytrue)
    macro_all = macro_rmse_all(Yhat, Ytrue)

    # Busy-only
    mask = Ytrue > 0
    micro_busy = rmse(Yhat[mask], Ytrue[mask]) if np.any(mask) else float("nan")
    macro_busy = macro_rmse_busy(Yhat, Ytrue)

    print(f"Test frames evaluated: {len(valid)} | Ports: {N}")
    print(f"KF lead-1 (Q={args.Q}, R={args.R}) → all-frames  micro {micro_all:.3f} | macro {macro_all:.3f}")
    print(f"KF lead-1 (Q={args.Q}, R={args.R}) → busy-only   micro {micro_busy:.3f} | macro {macro_busy:.3f}")

if __name__ == "__main__":
    main()
