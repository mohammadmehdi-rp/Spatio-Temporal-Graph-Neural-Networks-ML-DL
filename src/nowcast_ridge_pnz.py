#!/usr/bin/env python3
# Nowcast baseline: linear ridge on features X_t (global, per-sample = (t,port))
import argparse, numpy as np

def rmse(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.sqrt(np.mean((a-b)**2)))
def macro_rmse(yhat, yref, tids, N):
    vals=[]
    for j in range(N):
        m = np.ones(len(tids), bool)  # all times for port j
        vals.append(rmse(yhat[:,j], yref[:,j]))
    return float(np.nanmean(vals))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="NPZ path (use full for Oracle, sparse for Masked)")
    ap.add_argument("--l2", type=float, default=1000.0)
    ap.add_argument("--standardize", action="store_true", help="z-score features per column using train+val stats")
    args=ap.parse_args()

    Z=np.load(args.data, allow_pickle=True)
    X=Z["X"]       # [T,N,F]
    Y=Z["Y"]       # [T,N]
    train=Z["train_idx"]; val=Z["val_idx"]; test=Z["test_idx"]
    tr = np.sort(np.concatenate([train, val]))
    T,N,F = X.shape

    # flatten samples: stack over time and node
    def make_split(tidx):
        t = tidx.tolist()
        Xt = X[t]             # [Tt,N,F]
        Yt = Y[t]             # [Tt,N]
        return Xt.reshape(-1, F), Yt.reshape(-1), t

    Xtr, ytr, _ = make_split(tr)
    Xte, yte, tte = make_split(test)

    # handle masked features: fill NaNs with 0 (same as “not observed”)
    Xtr = np.nan_to_num(Xtr, nan=0.0); Xte = np.nan_to_num(Xte, nan=0.0)

    # optional standardization
    if args.standardize:
        mu = Xtr.mean(axis=0); sigma = Xtr.std(axis=0); sigma[sigma==0]=1.0
        Xtr = (Xtr - mu)/sigma; Xte = (Xte - mu)/sigma

    # add bias
    Xtr_b = np.c_[Xtr, np.ones((Xtr.shape[0],1))]
    Xte_b = np.c_[Xte, np.ones((Xte.shape[0],1))]

    lam = args.l2
    I = np.eye(Xtr_b.shape[1]); I[-1,-1]=0.0  # don't regularize bias
    w = np.linalg.pinv(Xtr_b.T@Xtr_b + lam*I) @ (Xtr_b.T @ ytr)

    yhat = Xte_b @ w
    # reshape back to [Ttest, N] for macro
    Ttest=len(tte)
    Yhat = yhat.reshape(Ttest, N); Yref = Y[test][:]  # [Ttest,N]

    mic = rmse(Yhat, Yref)
    mac = float(np.mean([rmse(Yhat[:,j], Yref[:,j]) for j in range(N)]))
    print(f"Ridge nowcast (L2={lam}, data='{args.data}') → all-frames micro {mic:.3f} | macro {mac:.3f} | test {Ttest} | ports {N}")

if __name__=="__main__": main()
