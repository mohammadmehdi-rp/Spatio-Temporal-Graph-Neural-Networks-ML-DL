#!/usr/bin/env python3
import argparse, numpy as np, csv

def rmse(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(np.sqrt(np.mean((a-b)**2))) if a.size else float("nan")

def metrics(yhat, y, thr):
    yhat=np.asarray(yhat,float); y=np.asarray(y,float)
    idle = y <= thr
    busy = y > thr
    out = {}
    out["global_RMSE"] = rmse(yhat.reshape(-1), y.reshape(-1))
    out["idle_RMSE"]   = rmse(yhat[idle], y[idle]) if np.any(idle) else float("nan")
    out["busy_RMSE"]   = rmse(yhat[busy], y[busy]) if np.any(busy) else float("nan")
    out["idle_FP_rate"]= float(np.mean(yhat[idle] > thr)) if np.any(idle) else float("nan")
    out["idle_mean_pred"]= float(np.mean(yhat[idle])) if np.any(idle) else float("nan")
    out["n_idle"]= int(np.sum(idle))
    out["n_busy"]= int(np.sum(busy))
    return out

def fit_ar1_per_node(y_train):
    # y_train shape [Ttr, N]
    y0 = y_train[:-1]
    y1 = y_train[1:]
    N = y_train.shape[1]
    a = np.zeros(N, dtype=float)
    b = np.zeros(N, dtype=float)
    for i in range(N):
        x = y0[:, i]
        t = y1[:, i]
        # Fit t = a*x + b by least squares
        A = np.stack([x, np.ones_like(x)], axis=1)
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        a[i], b[i] = float(coef[0]), float(coef[1])
    return a, b

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="dataset_sparse_*.npz")
    ap.add_argument("--busy_thr", type=float, default=1.0)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--exclude_features", default="sensor_backlog,ar1_next",
                    help="Comma-separated feature names to exclude from Ridge baseline.")
    ap.add_argument("--ridge_alpha", type=float, default=1e-2)
    args=ap.parse_args()

    Z=np.load(args.npz, allow_pickle=True)
    X=Z["X"].astype(np.float32)          # [T,N,F]
    Y=Z["Y"].astype(np.float32)          # [T,N]
    tr=Z["train_idx"].astype(int)
    te=Z["test_idx"].astype(int)
    feat=[n.decode() if isinstance(n,bytes) else str(n) for n in Z["feat_names"]]

    y_tr = Y[tr]
    y_te = Y[te]

    rows=[]

    # 0) Zero baseline
    rows.append(("Zero", metrics(np.zeros_like(y_te), y_te, args.busy_thr)))

    # 1) Persistence: yhat[t] = y[t-1] (align using original time index)
    # For each test time t, use y at time t-1 if exists, else 0.
    yhat_p = np.zeros_like(y_te)
    for k, t in enumerate(te):
        if t-1 >= 0:
            yhat_p[k] = Y[t-1]
    rows.append(("Persistence (y[t-1])", metrics(yhat_p, y_te, args.busy_thr)))

    # 2) AR(1) per node trained on train split, predict one-step for each test time using y[t-1]
    a,b = fit_ar1_per_node(y_tr)
    yhat_ar1 = np.zeros_like(y_te)
    for k, t in enumerate(te):
        prev = Y[t-1] if (t-1 >= 0) else np.zeros(Y.shape[1], dtype=np.float32)
        yhat_ar1[k] = a*prev + b
    rows.append(("AR(1) fit on train", metrics(yhat_ar1, y_te, args.busy_thr)))

    # 3) Ridge regression baseline using contemporaneous features at time t (train->test), excluding leaky channels
    # Simple closed-form ridge per node: w = (X^T X + alpha I)^{-1} X^T y
    exclude=set([s.strip() for s in args.exclude_features.split(",") if s.strip()])
    keep_idx=[i for i,n in enumerate(feat) if n not in exclude]
    if len(keep_idx) >= 1:
        Xtr = X[tr][:,:,keep_idx]  # [Ttr,N,Fk]
        Xte = X[te][:,:,keep_idx]
        Fk = Xtr.shape[2]
        alpha=float(args.ridge_alpha)

        yhat_r = np.zeros_like(y_te)
        I = np.eye(Fk, dtype=np.float64)
        for i in range(Y.shape[1]):
            A = Xtr[:,i,:].astype(np.float64)
            t = y_tr[:,i].astype(np.float64)
            # solve (A^T A + alpha I) w = A^T t
            w = np.linalg.solve(A.T @ A + alpha*I, A.T @ t)
            yhat_r[:,i] = (Xte[:,i,:].astype(np.float64) @ w).astype(np.float32)
        rows.append((f"Ridge (alpha={alpha:g}, excl={','.join(sorted(exclude))})",
                     metrics(yhat_r, y_te, args.busy_thr)))

    # write CSV
    header=["method","global_RMSE","idle_RMSE","busy_RMSE","idle_FP_rate","idle_mean_pred","n_idle","n_busy"]
    with open(args.out_csv,"w",newline="") as f:
        w=csv.writer(f); w.writerow(header)
        for m,d in rows:
            w.writerow([m,d["global_RMSE"],d["idle_RMSE"],d["busy_RMSE"],d["idle_FP_rate"],
                        d["idle_mean_pred"],d["n_idle"],d["n_busy"]])
    print(f"[OK] wrote {args.out_csv} ({len(rows)} rows)")

if __name__=="__main__":
    main()
