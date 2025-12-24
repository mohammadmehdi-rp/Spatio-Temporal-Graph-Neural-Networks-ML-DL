#!/usr/bin/env python3
"""
ARIMA(p,1,0) baseline without external libs:
- Difference the label: d_t = y_t - y_{t-1}
- Fit AR(p) with ridge on d_t per-interface (pooled for stability)
- Predict next diff, then integrate: ŷ_t = y_{t-1} + d̂_t
Reports micro/macro RMSE on test split (80/20 over busy timestamps).
"""
import argparse, numpy as np, pandas as pd, math
from numpy.linalg import inv

def rmse(a,b): e=(a-b).dropna(); return float(np.sqrt((e**2).mean())) if len(e) else float("nan")
def macro_rmse(df, y, yhat):
    vals=[]; 
    for _,g in df[[y,yhat,"iface"]].dropna().groupby("iface"):
        if len(g)>=2: vals.append(rmse(g[y], g[yhat]))
    return float(np.nanmean(vals)) if vals else float("nan")

def fit_eval(df, lab, p, l2):
    D=df.sort_values(["iface","timestamp"]).copy()
    D["diff"]=D.groupby("iface")[lab].diff()
    # build AR(p) on diff
    for k in range(1, p+1):
        D[f"diff_lag{k}"]=D.groupby("iface")["diff"].shift(k)
    # previous level for integration
    D["y_prev"]=D.groupby("iface")[lab].shift(1)
    feats=[f"diff_lag{k}" for k in range(1,p+1)]
    D=D.dropna(subset=feats+["diff","y_prev"])
    if D.empty: return (math.nan, math.nan, 0, 0, None)
    busy=D[D["diff"].abs()>0]
    uts=np.array(sorted((busy["timestamp"] if len(busy) else D["timestamp"]).unique()))
    split_t=uts[int(0.8*len(uts))] if len(uts)>1 else D["timestamp"].max()
    tr=D["timestamp"]<=split_t; te=D["timestamp"]>split_t

    Xtr=D.loc[tr,feats].to_numpy(float); ytr=D.loc[tr,"diff"].to_numpy(float)
    Xte=D.loc[te,feats].to_numpy(float); yte=D.loc[te,"diff"].to_numpy(float)
    if not len(ytr) or not len(yte): return (math.nan, math.nan, 0, 0, split_t)
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    Xtr=np.hstack([Xtr,np.ones((Xtr.shape[0],1))]); Xte=np.hstack([Xte,np.ones((Xte.shape[0],1))])
    w=inv(Xtr.T@Xtr + l2*np.eye(Xtr.shape[1])) @ (Xtr.T@ytr)
    d_hat = Xte@w
    yhat = D.loc[te,"y_prev"].to_numpy(float) + d_hat
    ytrue = D.loc[te,lab].to_numpy(float)

    micro = rmse(pd.Series(ytrue), pd.Series(yhat))
    tmp=D.loc[te,["iface"]].copy(); tmp["y"]=ytrue; tmp["yhat"]=yhat
    macro = macro_rmse(tmp.rename(columns={"y":lab}), lab, "yhat")
    return (micro, macro, int(tr.sum()), int(te.sum()), split_t)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed", default="processed.csv")
    ap.add_argument("--label", default=None, choices=["backlog_pkts","backlog_bytes","qlen_pkts"])
    ap.add_argument("--p_list", default="1,2,3,5,10")
    ap.add_argument("--l2_list", default="1,10,50,200,1000")
    args=ap.parse_args()

    df=pd.read_csv(args.processed, parse_dates=["timestamp"])
    lab=args.label or ("backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts"))

    best=(float("inf"),None,None)
    print(f"ARIMA(p,1,0) grid on {lab}: p={args.p_list}  L2={args.l2_list}")
    for p in [int(x) for x in args.p_list.split(",") if x]:
        for l2 in [float(x) for x in args.l2_list.split(",") if x]:
            micro, macro, tr, te, split_t = fit_eval(df, lab, p, l2)
            print(f" p={p:<2} L2={l2:<6} → micro {micro:.3f} | macro {macro:.3f} | train {tr} | test {te}")
            if micro==micro and micro<best[0]: best=(micro,p,l2)
    print(f"\nBEST ARIMA(p,1,0): micro {best[0]:.3f} with p={best[1]} L2={best[2]}")

if __name__=="__main__": main()
