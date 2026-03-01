
#!/usr/bin/env python3
# Strict, leakage-free ARIMA(p,1,0) nowcast via walk-forward ridge on differences.
# For each test time t and each port j, fit AR(p) on diffs using only times < t from train+val, then predict y_t.

import argparse, numpy as np

def rmse(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(np.sqrt(np.mean((a-b)**2))) if a.size else float("nan")

def fit_ar_on_diff(y, p, lam, tr_times):
    # y: [T], d[t] = y[t]-y[t-1] defined for t>=1. We model d[t] ~ [d[t-1]..d[t-p]]
    T=len(y); d=y[1:]-y[:-1]
    # valid times for training: t in tr_times with t>=p+1 and t<T
    tr=[t for t in tr_times if (t>=p+1) and (t<T)]
    if len(tr) < max(5, p):  # need a few samples
        return None
    X=np.stack([d[np.array(tr)-1-k] for k in range(p)], axis=1)   # [n,p]
    z=d[np.array(tr)-1]                                          # [n]
    XtX = X.T @ X + lam * np.eye(p)
    w   = np.linalg.pinv(XtX) @ (X.T @ z)
    return w

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--l2", type=float, default=1000.0)
    args=ap.parse_args()

    Z   = np.load(args.data, allow_pickle=True)
    Y   = Z["Y"]                   # [T,N]
    tr0 = Z["train_idx"]
    v0  = Z["val_idx"]
    te0 = Z["test_idx"]
    T,N = Y.shape
    p   = args.p
    lam = args.l2

    # walk-forward over test times in ascending order
    test_times = te0.tolist()
    trainval   = np.sort(np.concatenate([tr0, v0])).tolist()

    all_preds  = []  # pooled predictions for micro
    all_truth  = []

    per_port_err = [[] for _ in range(N)]  # for macro

    for t in test_times:
        if t < p+1 or t >= T:
            continue
        # training times strictly before t
        tr_times = [s for s in trainval if s < t]
        if not tr_times:
            continue

        # build per-port predictions at this t
        yhat_t = np.full(N, np.nan, float)
        yref_t = Y[t].astype(float)

        d = Y[1:] - Y[:-1]  # precompute diffs for convenience

        for j in range(N):
            w = fit_ar_on_diff(Y[:,j], p, lam, tr_times)
            if w is None:
                continue
            # features for time t: [d[t-1], d[t-2], ..., d[t-p]]
            x = np.array([d[t-1-k, j] for k in range(p)], dtype=float)
            dhat = float(x @ w)
            yhat = float(Y[t-1, j] + dhat)
            yhat_t[j] = yhat

        # collect pooled errors for micro
        m = ~np.isnan(yhat_t)
        if np.any(m):
            all_preds.append(yhat_t[m])
            all_truth.append(yref_t[m])
            # per-port errs for macro
            for j in np.where(m)[0]:
                per_port_err[j].append((yhat_t[j] - yref_t[j])**2)

    if not all_preds:
        print("No valid test predictions (insufficient history for chosen p). Try smaller p.")
        return

    P = np.concatenate(all_preds)
    Tref = np.concatenate(all_truth)
    mic = float(np.sqrt(np.mean((P - Tref)**2)))

    # macro across ports (only over times where that port had a prediction)
    rms = []
    for j in range(N):
        if per_port_err[j]:
            rms.append(np.sqrt(np.mean(per_port_err[j])))
    mac = float(np.mean(rms)) if rms else float("nan")

    print(f"ARIMA({p},1,0) walk-forward ridge L2={lam} → all-frames micro {mic:.3f} | macro {mac:.3f} | test_used {len(all_preds)} | ports {N}")

if __name__ == "__main__":
    main()
