#!/usr/bin/env python3
# ARIMA(p,1,0) as AR(p) on differences; fit on train+val only; strict time alignment.
import argparse, numpy as np

def rmse(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.sqrt(np.mean((a-b)**2)))
def macro_rmse(Yh, Yr): return float(np.nanmean([rmse(Yh[:,j], Yr[:,j]) for j in range(Yh.shape[1])]))

def fit_ar_diff_per_port(y, p, lam, tr_idx):
    # y: [T], differences d[t]=y[t]-y[t-1] defined for t>=1
    T=len(y); d=y[1:]-y[:-1]                 # length T-1, index t maps to d[t-1]
    # valid train times: t where we predict d[t] using [d[t-1]..d[t-p]]
    tr=[t for t in tr_idx if (t>=p+1) and (t<T)]
    if not tr: return np.zeros(p)
    X=np.stack([d[np.array(tr)-1-k] for k in range(p)], 1)  # [n,p]
    z=d[np.array(tr)-1]                                     # d[t]
    XtX=X.T@X + lam*np.eye(p)
    w=np.linalg.pinv(XtX)@(X.T@z)
    return w

def predict_nowcast(y, w, p, test_idx):
    T=len(y); d=y[1:]-y[:-1]
    # nowcast at t uses d[t-1..t-p] and adds to y[t-1]
    t_valid=[t for t in test_idx if (t>=p+1) and (t<T)]
    if not t_valid: return np.array([]), np.array([])
    Xte=np.stack([d[np.array(t_valid)-1-k] for k in range(p)], 1)
    dhat=(Xte @ w).reshape(-1)
    yhat=y[np.array(t_valid)-1] + dhat
    return np.array(t_valid), yhat

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--l2", type=float, default=1000.0)
    args=ap.parse_args()

    Z=np.load(args.data, allow_pickle=True)
    Y=Z["Y"]; train=Z["train_idx"]; val=Z["val_idx"]; test=Z["test_idx"]
    tr_idx=np.sort(np.concatenate([train, val])).tolist()
    T,N=Y.shape; p=args.p; lam=args.l2

    t_common=None; YH=[]; YR=[]
    for j in range(N):
        w=fit_ar_diff_per_port(Y[:,j], p, lam, tr_idx)
        tv, yhat = predict_nowcast(Y[:,j], w, p, test.tolist())
        yref = Y[tv, j]
        if t_common is None: t_common=tv
        else:
            assert np.array_equal(t_common, tv), "inconsistent valid test times"
        YH.append(yhat); YR.append(yref)

    Yhat=np.stack(YH,1); Yref=np.stack(YR,1)
    mic=rmse(Yhat,Yref); mac=macro_rmse(Yhat,Yref)
    print(f"ARIMA({p},1,0) ridge L2={lam} → all-frames micro {mic:.3f} | macro {mac:.3f} | test {len(t_common)} | ports {N}")
    # quick guardrail
    if mic < 0.1:
        print("WARNING: RMSE < 0.1 — verify label scale & indexing; run sanity_check_npz.py.")

if __name__=="__main__":
    main()

