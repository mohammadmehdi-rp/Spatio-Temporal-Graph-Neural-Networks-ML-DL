#!/usr/bin/env python3
# Ridge (with intercept) on GLOBAL features, with log1p(label) -> predict -> expm1 to RMSE in original units.
import argparse, pandas as pd, numpy as np
from numpy.linalg import inv
def rmse(a,b): 
    e=(a-b).dropna(); return float(np.sqrt((e**2).mean())) if len(e) else float("nan")
def macro_rmse(df,ycol,yhat):
    vals=[]; 
    for _,g in df[[ycol,yhat,"iface"]].dropna().groupby("iface"):
        if len(g)>=2: vals.append(rmse(g[ycol],g[yhat]))
    return float(np.nanmean(vals)) if vals else float("nan")
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="global_agg.csv")
    ap.add_argument("--l2", type=float, default=5000.0)
    ap.add_argument("--subset_ifaces", default="", help="comma list to eval on subset (optional)")
    args=ap.parse_args()
    df=pd.read_csv(args.data, parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
    label="backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts")
    feats=[c for c in df.columns if c not in ("timestamp","iface",label)]
    ifaces=[s.strip() for s in args.subset_ifaces.split(",") if s.strip()]
    if ifaces: df=df[df["iface"].isin(ifaces)].copy()
    busy=df[df[label]>0]; uts=np.array(sorted((busy["timestamp"] if len(busy) else df["timestamp"]).unique()))
    split_t=uts[int(0.8*len(uts))] if len(uts)>1 else df["timestamp"].max()
    tr=df["timestamp"]<=split_t; te=df["timestamp"]>split_t
    dtr=df.loc[tr,["iface",label]+feats].dropna(); dte=df.loc[te,["iface",label]+feats].dropna()
    if dtr.empty or dte.empty: raise SystemExit("Not enough train/test after split.")
    Xtr=dtr[feats].to_numpy(float); ytr=dtr[label].to_numpy(float)
    Xte=dte[feats].to_numpy(float); yte=dte[label].to_numpy(float)
    ytr_log=np.log1p(ytr)
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    Xtr=np.hstack([Xtr, np.ones((Xtr.shape[0],1))]); Xte=np.hstack([Xte, np.ones((Xte.shape[0],1))])
    w=inv(Xtr.T@Xtr + args.l2*np.eye(Xtr.shape[1])) @ (Xtr.T@ytr_log)
    yhat=np.expm1(Xte@w)
    micro=rmse(pd.Series(yte), pd.Series(yhat))
    tmp=dte.copy(); tmp["yhat"]=yhat
    macro=macro_rmse(tmp, label, "yhat")
    print(f"Label: {label}")
    print(f"Split time: {split_t}")
    print(f"Ridge GLOBAL (log) → micro {micro:.3f} | macro {macro:.3f} | train {len(dtr)} | test {len(dte)} | feats {len(feats)} | L2 {args.l2}")
if __name__=="__main__": main()
