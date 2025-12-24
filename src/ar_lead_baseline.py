#!/usr/bin/env python3
# ar_lead_baseline.py — predict next-sample backlog (lead-1) with AR(lags) Ridge; prints naive vs AR.
import argparse, pandas as pd, numpy as np, math
from numpy.linalg import inv

def rmse(a,b):
    e=(a-b).dropna(); 
    return float(np.sqrt((e**2).mean())) if len(e) else float("nan")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed", default="processed.csv")
    ap.add_argument("--label", default=None, choices=["backlog_pkts","backlog_bytes","qlen_pkts"])
    ap.add_argument("--lags", type=int, default=5)     # at 5 Hz, 5 lags ≈ 1s history
    ap.add_argument("--l2", type=float, default=50.0)
    args=ap.parse_args()

    df=pd.read_csv(args.processed, parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
    lab=args.label or ("backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts"))

    # target = next sample
    df["target"]=df.groupby("iface")[lab].shift(-1)
    for i in range(1,args.lags+1):
        df[f"{lab}_lag{i}"]=df.groupby("iface")[lab].shift(i)
    feats=[f"{lab}_lag{i}" for i in range(1,args.lags+1)]
    df=df.dropna(subset=feats+["target"]).sort_values(["timestamp","iface"])

    busy=df[df["target"]>0]
    uts=np.array(sorted((busy["timestamp"] if len(busy) else df["timestamp"]).unique()))
    split_t=uts[int(0.8*len(uts))] if len(uts)>1 else df["timestamp"].max()
    tr=df["timestamp"]<=split_t; te=df["timestamp"]>split_t

    # naive lead: yhat_{t+1} = y_t
    naive_rmse=rmse(df.loc[te,"target"], df.loc[te, f"{lab}_lag1"])

    Xtr=df.loc[tr,feats].to_numpy(float); ytr=df.loc[tr,"target"].to_numpy(float)
    Xte=df.loc[te,feats].to_numpy(float); yte=df.loc[te,"target"].to_numpy(float)
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    Xtr=np.hstack([Xtr,np.ones((Xtr.shape[0],1))]); Xte=np.hstack([Xte,np.ones((Xte.shape[0],1))])
    w=inv(Xtr.T@Xtr + args.l2*np.eye(Xtr.shape[1])) @ (Xtr.T@ytr)
    ar_rmse=rmse(pd.Series(yte), pd.Series(Xte@w))

    print(f"Lead-1 split: {split_t}")
    print(f"Naive lead (y_t→y_t+1) RMSE: {naive_rmse:.3f}")
    print(f"AR lead  (lags={args.lags}, L2={args.l2}) RMSE: {ar_rmse:.3f}")

if __name__=="__main__": main()
