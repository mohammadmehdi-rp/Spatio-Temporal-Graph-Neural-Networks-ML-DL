#!/usr/bin/env python3
# Nowcast baseline: ARIMA(p,1,0) implemented as AR(p) on differenced series; predict y_t = y_{t-1} + \hat d_t
import argparse, numpy as np

def rmse(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.sqrt(np.mean((a-b)**2)))
def macro_rmse(yhat, yref):
    N=yref.shape[1]; vals=[]
    for j in range(N): vals.append(rmse(yhat[:,j], yref[:,j]))
    return float(np.nanmean(vals))

def fit_ar_on_diff(y, p, lam, tr_idx):
    # train indices must be >= p+1 (need p diffs + one more for diff itself)
    tr=[t for t in tr_idx if t>=p+1]
    if not tr: return np.zeros(p)
    d = y[1:] - y[:-1]              # length T-1
    # build X: d_t ~ [d_{t-1}..d_{t-p}]
    X = np.stack([d[np.array(tr)-1-k] for k in range(p)], axis=1)
    z = d[np.array(tr)-1]
    XtX = X.T@X + lam*np.eye(p)
    w = np.linalg.pinv(XtX)@(X.T@z)
    return w

def predict_nowcast_arima(y, w, p, test_idx):
    T=len(y); d = y[1:] - y[:-1]
    t_valid=[t for t in test_idx if t>=p+1 and t<T]
    yhat=[]
    for t in t_valid:
        # \hat d_t = w · [d_{t-1}..d_{t-p}]
        x = np.array([d[t-1-k] for k in range(p)])
        dhat = float(x @ w)
        yhat.append(y[t-1] + dhat)
    return np.array(t_valid), np.array(yhat)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--l2", type=float, default=1000.0)
    args=ap.parse_args()

    Z=np.load(args.data, allow_pickle=True)
    Y=Z["Y"]              # [T,N]
    train=Z["train_idx"]; val=Z["val_idx"]; test=Z["test_idx"]
    tr_idx = np.sort(np.concatenate([train, val])).tolist()
    T,N=Y.shape

    # fit per-port
    t_common=None
    Yhat_list=[]; Yref_list=[]
    for j in range(N):
        w = fit_ar_on_diff(Y[:,j], args.p, args.l2, tr_idx)
        t_v, yhat = predict_nowcast_arima(Y[:,j], w, args.p, test.tolist())
        yref = Y[t_v, j]
        if t_common is None: t_common = t_v
        else: assert np.all(t_common==t_v)
        Yhat_list.append(yhat); Yref_list.append(yref)

    Yhat = np.stack(Yhat_list, axis=1)   # [Ttest_valid, N]
    Yref = np.stack(Yref_list, axis=1)
    mic=rmse(Yhat, Yref); mac=macro_rmse(Yhat, Yref)
    print(f"ARIMA({args.p},1,0) ridge L2={args.l2} → all-frames micro {mic:.3f} | macro {mac:.3f} | test {len(t_common)} | ports {N}")

if __name__=="__main__": main()
