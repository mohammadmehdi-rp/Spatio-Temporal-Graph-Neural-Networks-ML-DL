#!/usr/bin/env python3
import argparse
import numpy as np

def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="runs/.../dataset.npz")
    ap.add_argument("--Q", type=float, default=50.0)
    ap.add_argument("--R", type=float, default=200.0)
    ap.add_argument("--thr", type=float, default=50.0, help="busy threshold tau (idle if y < tau)")
    args = ap.parse_args()

    Z = np.load(args.data, allow_pickle=True)
    Y = Z["Y"].astype(np.float64)        # [T,N]
    test_idx = Z["test_idx"].astype(int)

    T, N = Y.shape

    # Kalman random-walk filtering per port, store x_{t|t}
    x = Y[0].copy()
    P = np.ones(N, dtype=np.float64)
    xf = np.zeros((T, N), dtype=np.float64)
    xf[0] = x

    Q, R = float(args.Q), float(args.R)

    for t in range(1, T):
        x_pred = x
        P_pred = P + Q
        z = Y[t]
        K = P_pred / (P_pred + R)
        x = x_pred + K * (z - x_pred)
        P = (1.0 - K) * P_pred
        xf[t] = x

    # lead-1: predict Y[t+1] using xf[t] for t in test_idx with t<=T-2
    tes = set(test_idx.tolist())
    eval_t = [t for t in range(T - 1) if t in tes]

    y_true = np.stack([Y[t + 1] for t in eval_t], axis=0)   # [Te,N]
    y_hat  = np.stack([xf[t]   for t in eval_t], axis=0)

    thr = float(args.thr)
    idle = (y_true < thr)
    busy = (y_true >= thr)

    global_rmse = rmse(y_hat.reshape(-1), y_true.reshape(-1))
    idle_rmse   = rmse(y_hat[idle], y_true[idle]) if np.any(idle) else float("nan")
    busy_rmse   = rmse(y_hat[busy], y_true[busy]) if np.any(busy) else float("nan")
    idle_fp     = float(np.mean(y_hat[idle] >= thr)) if np.any(idle) else float("nan")
    idle_mean   = float(np.mean(y_hat[idle])) if np.any(idle) else float("nan")

    print(f"KF lead-1 metrics (Q={Q:g}, R={R:g}, thr={thr:g})")
    print("Global RMSE:", global_rmse)
    print("Idle RMSE:", idle_rmse)
    print("Busy RMSE:", busy_rmse)
    print("Idle FP:", idle_fp)
    print("Idle mean yhat:", idle_mean)
    print("n_idle:", int(np.sum(idle)), "n_busy:", int(np.sum(busy)), "total:", int(y_true.size))

if __name__ == "__main__":
    main()
