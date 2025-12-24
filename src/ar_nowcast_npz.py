#!/usr/bin/env python3
import argparse, numpy as np

def rmse(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(np.sqrt(np.mean((a-b)**2)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--lags", type=int, default=1)
    ap.add_argument("--l2", type=float, default=200.0)
    ap.add_argument("--label", default="backlog_pkts")
    args=ap.parse_args()

    Z=np.load(args.data, allow_pickle=True)
    Y=Z["Y"]            # [T,N], already the target (e.g., backlog_pkts)
    test=Z["test_idx"]  # indices t for which we evaluate y_t (nowcast)

    # AR(p) ridge per-port: y_t ~ [y_{t-1}..y_{t-p}]
    p=args.lags; lam=args.l2
    T,N=Y.shape
    t_valid=[t for t in test.tolist() if t>=p]  # need p lags available
    Yhat=np.zeros((len(t_valid),N),float)
    Yref=np.stack([Y[t] for t in t_valid])      # truth at t

    for j in range(N):
        # Build design on train+val (everything except test) to avoid leakage
        tr_mask=np.ones(T, bool); tr_mask[t_valid]=False
        tr_idx=np.where(tr_mask)[0]
        tr_idx=[t for t in tr_idx if t>=p]
        if not tr_idx: continue
        X=np.stack([Y[np.array(tr_idx)-k, j] for k in range(1,p+1)], axis=1)  # [n,p]
        y=Y[np.array(tr_idx), j]
        # ridge: (X^T X + lam I) w = X^T y
        XtX=X.T@X + lam*np.eye(p)
        w=np.linalg.pinv(XtX)@(X.T@y)

        # predict on test-valid
        Xt=np.stack([Y[np.array(t_valid)-k, j] for k in range(1,p+1)], axis=1)
        Yhat[:,j]=Xt@w

    micro=rmse(Yhat, Yref)
    # macro (mean per-port RMSE)
    per=[]
    for j in range(N):
        per.append(rmse(Yhat[:,j], Yref[:,j]))
    macro=float(np.nanmean(per))
    print(f"AR({p}), L2={lam} → all-frames micro {micro:.3f} | macro {macro:.3f} | test {len(t_valid)} | ports {N}")

if __name__=="__main__":
    main()
