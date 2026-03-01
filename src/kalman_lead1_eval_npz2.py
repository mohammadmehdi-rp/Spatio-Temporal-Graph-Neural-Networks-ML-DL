#!/usr/bin/env python3
import argparse
import numpy as np

def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else float("nan")

def micro_macro(yhat, y):
    # yhat,y: [T,N]
    micro = rmse(yhat.reshape(-1), y.reshape(-1))
    per = [rmse(yhat[:, i], y[:, i]) for i in range(y.shape[1])]
    macro = float(np.mean(per)) if per else micro
    return micro, macro

def micro_macro_masked(yhat, y, mask):
    # mask: [T,N] bool
    if not np.any(mask):
        return float("nan"), float("nan")
    micro = rmse(yhat[mask], y[mask])
    per = []
    for i in range(y.shape[1]):
        mi = mask[:, i]
        if np.any(mi):
            per.append(rmse(yhat[:, i][mi], y[:, i][mi]))
    macro = float(np.mean(per)) if per else micro
    return micro, macro

def kalman_rw_lead1(y, Q=50.0, R=200.0):
    """
    1D random-walk Kalman filter per port:
      x_t = x_{t-1} + w,  w~N(0,Q)
      z_t = x_t + v,      v~N(0,R)
    Returns lead-1 predictions for each time t: x_{t+1|t} which equals x_{t|t} under RW.
    """
    y = y.astype(np.float64)
    T, N = y.shape
    # state estimate and covariance per port
    x = y[0].copy()
    P = np.ones(N, dtype=np.float64) * 1.0

    # store filtered estimates x_{t|t}
    xf = np.zeros((T, N), dtype=np.float64)
    xf[0] = x

    for t in range(1, T):
        # predict
        x_pred = x
        P_pred = P + Q

        # update with observation y[t]
        z = y[t]
        K = P_pred / (P_pred + R)
        x = x_pred + K * (z - x_pred)
        P = (1.0 - K) * P_pred

        xf[t] = x

    # lead-1 prediction at time t is x_{t|t} (RW), used to predict y[t+1]
    return xf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to dataset.npz")
    ap.add_argument("--Q", type=float, default=50.0)
    ap.add_argument("--R", type=float, default=200.0)
    ap.add_argument("--busy_thr", type=float, default=0.0,
                    help="Busy threshold tau (busy if y >= tau). Default 0.")
    args = ap.parse_args()

    Z = np.load(args.data, allow_pickle=True)
    Y = Z["Y"].astype(np.float64)             # [T,N]
    test_idx = Z["test_idx"].astype(int)      # indices in [0..T-1]

    T, N = Y.shape

    # compute filtered estimates for all t
    xf = kalman_rw_lead1(Y, Q=args.Q, R=args.R)

    # build valid lead-1 evaluation positions:
    # predict Y[t+1] using xf[t], for t in test_idx with t <= T-2
    test_set = set(test_idx.tolist())
    eval_t = [t for t in range(T - 1) if t in test_set]  # t <= T-2

    y_true = np.stack([Y[t + 1] for t in eval_t], axis=0)   # [Te,N]
    y_hat  = np.stack([xf[t]   for t in eval_t], axis=0)    # [Te,N]

    print(f"Test frames evaluated: {len(eval_t)} | Ports: {N}")

    micro, macro = micro_macro(y_hat, y_true)
    print(f"KF lead-1 (Q={args.Q}, R={args.R}) → all-frames  micro {micro:.3f} | macro {macro:.3f}")

    # busy-only mask based on tau
    tau = float(args.busy_thr)
    busy_mask = (y_true >= tau)
    bm, bM = micro_macro_masked(y_hat, y_true, busy_mask)

    n_busy = int(np.sum(busy_mask))
    n_all = int(np.prod(y_true.shape))
    print(f"KF lead-1 (Q={args.Q}, R={args.R}) → busy-only (y>= {tau:g}) micro {bm:.3f} | macro {bM:.3f} | busy_pts {n_busy}/{n_all}")

if __name__ == "__main__":
    main()
