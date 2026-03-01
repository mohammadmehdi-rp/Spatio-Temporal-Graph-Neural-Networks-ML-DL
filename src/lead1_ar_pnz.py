#!/usr/bin/env python3
# Lead-1 baseline: AR(p) ridge on labels; target y_{t+1}
import argparse, numpy as np

def rmse(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.sqrt(np.mean((a-b)**2)))
def macro_rmse(yhat, yref):
    N=yref.shape[1]; vals=[]
    for j in range(N): vals.append(rmse(yhat[:,j], yref[:,j]))
    return float(np.nanmean(vals))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--l2", type=float, default=50.0)
    args=ap.parse_args()

    Z=np.load(args.data, allow_pickle=True)
    Y=Z["Y"]; train=Z["train_idx"]; val=Z["val_idx"]; test=Z["test_idx"]
    tr_idx = np.sort(np.concatenate([train, val]))
    T,N = Y.shape; p=args.lags; lam=args.l2

    # valid test t: have p lags and t+1 exists
    t_test=[t for t in test.tolist() if t>=p and t+1<T]
    # training times (no leakage): train+val where t>=p and t+1 exists
    t_tr=[t for t in tr_idx.tolist() if t>=p and t+1<T]

    # per-port ridge fit
    Yhat=[]; Yref=[]
    for j in range(N):
        # build design for training
        Xtr = np.stack([Y[np.array(t_tr)-k, j] for k in range(1,p+1)], axis=1)  # [n_tr, p]
        ytr = Y[np.array(t_tr)+1, j]                                           # y_{t+1}
        if Xtr.size==0:
            w = np.zeros(p)
        else:
            XtX = Xtr.T@Xtr + lam*np.eye(p)
            w = np.linalg.pinv(XtX)@(Xtr.T@ytr)

        # predict on test
        Xte = np.stack([Y[np.array(t_test)-k, j] for k in range(1,p+1)], axis=1)
        Yhat.append(Xte@w); Yref.append(Y[np.array(t_test)+1, j])

    Yhat=np.stack(Yhat, axis=1)   # [Ttest, N]
    Yref=np.stack(Yref, axis=1)

    mic=rmse(Yhat, Yref); mac=macro_rmse(Yhat, Yref)
    print(f"AR lead (lags={p}, L2={lam}) → all-frames micro {mic:.3f} | macro {mac:.3f} | test {len(t_test)} | ports {N}")

if __name__=="__main__": main()
