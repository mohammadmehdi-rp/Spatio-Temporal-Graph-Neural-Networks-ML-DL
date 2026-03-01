#!/usr/bin/env python3
# ar_gridsearch.py — grid search AR(lags) Ridge over lags & L2, using processed.csv
import argparse, pandas as pd, numpy as np, math
from numpy.linalg import inv

def rmse(a,b): 
    e=(a-b).dropna(); 
    return float(np.sqrt((e**2).mean())) if len(e) else float("nan")

def fit_ar(df, lab, lags, l2):
    D=df.copy()
    for i in range(1,lags+1):
        D[f"{lab}_lag{i}"]=D.groupby("iface")[lab].shift(i)
    feats=[f"{lab}_lag{i}" for i in range(1,lags+1)]
    D=D.dropna(subset=feats+[lab]).sort_values(["timestamp","iface"])
    if D.empty: return (math.nan,0,0,None)
    busy=D[D[lab]>0]
    uts=np.array(sorted((busy["timestamp"] if len(busy) else D["timestamp"]).unique()))
    split_t=uts[int(0.8*len(uts))] if len(uts)>1 else D["timestamp"].max()
    tr=D["timestamp"]<=split_t; te=D["timestamp"]>split_t
    Xtr=D.loc[tr,feats].to_numpy(float); ytr=D.loc[tr,lab].to_numpy(float)
    Xte=D.loc[te,feats].to_numpy(float); yte=D.loc[te,lab].to_numpy(float)
    if len(ytr)==0 or len(yte)==0: return (math.nan,0,0,split_t)
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    Xtr=np.hstack([Xtr,np.ones((Xtr.shape[0],1))]); Xte=np.hstack([Xte,np.ones((Xte.shape[0],1))])
    w=inv(Xtr.T@Xtr + l2*np.eye(Xtr.shape[1])) @ (Xtr.T@ytr)
    yhat=Xte@w
    return (rmse(pd.Series(yte), pd.Series(yhat)), int(tr.sum()), int(te.sum()), split_t)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed", default="processed.csv")
    ap.add_argument("--label", default=None, choices=["backlog_pkts","backlog_bytes","qlen_pkts"])
    ap.add_argument("--lags", default="1,2,3,5,10,15")
    ap.add_argument("--l2",   default="1,10,50,200,1000")
    args=ap.parse_args()
    df=pd.read_csv(args.processed, parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
    lab=args.label or ("backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts"))
    best=(float("inf"),None,None,None,None)
    print(f"Grid over lags={args.lags}  L2={args.l2}  label={lab}")
    for L in [int(x) for x in args.lags.split(",") if x]:
        for l2 in [float(x) for x in args.l2.split(",") if x]:
            r,tr,te,split=fit_ar(df, lab, L, l2)
            print(f"AR lags={L:<2} L2={l2:<6} → RMSE {r:.3f} | train {tr} | test {te}")
            if r==r and r<best[0]: best=(r,L,l2,tr,te)
    print(f"\nBEST → RMSE {best[0]:.3f} with lags={best[1]} L2={best[2]} (train {best[3]} | test {best[4]})")

if __name__=="__main__": main()
