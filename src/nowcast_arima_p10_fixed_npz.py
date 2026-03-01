#!/usr/bin/env python3
# Leakage-free ARIMA(p,1,0) nowcast: fit AR(p) on diffs using train+val only.
# Predict y_t = y_{t-1} + \hat d_t with features [d_{t-1},...,d_{t-p}] where
# array indices are [t-2, t-3, ..., t-p-1] (NO use of d[t-1] which equals y_t - y_{t-1}).
import argparse, numpy as np

def rmse(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.sqrt(np.mean((a-b)**2)))
def macro_rmse(Yh, Yr): return float(np.nanmean([rmse(Yh[:,j], Yr[:,j]) for j in range(Yh.shape[1])]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--l2", type=float, default=1000.0)
    args=ap.parse_args()

    Z   = np.load(args.data, allow_pickle=True)
    Y   = Z["Y"]               # [T,N]
    train, val, test = Z["train_idx"], Z["val_idx"], Z["test_idx"]
    T,N = Y.shape
    p,lam = args.p, args.l2

    # Differences: d[idx] = y[idx+1] - y[idx]  (length T-1)
    D = Y[1:] - Y[:-1]         # [T-1, N]

    # Valid times for predicting y_t need indices t >= p+1
    t_test = [t for t in test.tolist() if t >= p+1 and t < T]

    # Train indices (no test leakage), also must satisfy t >= p+1
    tr_idx = np.sort(np.concatenate([train, val])).tolist()
    t_train = [t for t in tr_idx if t >= p+1 and t < T]

    # Build design for training: predict d_t using [d_{t-1},...,d_{t-p}]
    # In array terms: target D[t-1], features D[t-2],...,D[t-p-1]
    if not t_train:
        raise SystemExit("No train samples for chosen p. Reduce --p.")

    Yhat_list=[]; Yref_list=[]
    for j in range(N):
        Xtr = np.stack([ D[np.array(t_train)-1-k, j] for k in range(1,p+1) ], axis=1)   # [n_tr, p]
        ztr = D[np.array(t_train)-1, j]                                                 # d_t
        # Ridge
        XtX = Xtr.T @ Xtr + lam * np.eye(p)
        w   = np.linalg.pinv(XtX) @ (Xtr.T @ ztr)

        # Predict on test
        t_v = t_test
        Xte = np.stack([ D[np.array(t_v)-1-k, j] for k in range(1,p+1) ], axis=1)
        d_hat = (Xte @ w).reshape(-1)
        y_hat = Y[np.array(t_v)-1, j] + d_hat

        Yhat_list.append(y_hat)
        Yref_list.append(Y[t_v, j])

    Yhat = np.stack(Yhat_list, axis=1)   # [Ttest_valid, N]
    Yref = np.stack(Yref_list, axis=1)
    mic = rmse(Yhat, Yref); mac = macro_rmse(Yhat, Yref)
    print(f"ARIMA({p},1,0) ridge (fixed) L2={lam} → all-frames micro {mic:.3f} | macro {mac:.3f} | test {len(t_test)} | ports {N}")

if __name__=="__main__":
    main()
