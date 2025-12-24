#!/usr/bin/env python3
# bottleneck_only_eval.py — sanity: predict backlog on the bottleneck port only (e.g., s2-eth2)
import argparse, pandas as pd, numpy as np
from numpy.linalg import inv
ap=argparse.ArgumentParser(); ap.add_argument("--iface",default="s2-eth2"); ap.add_argument("--lags",type=int,default=10); ap.add_argument("--l2",type=float,default=50.0)
args=ap.parse_args()
df=pd.read_csv("processed.csv",parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
lab="backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts")
df=df[df["iface"]==args.iface].copy().sort_values("timestamp")
base=[c for c in df.columns if c.endswith("_per_s")] + (["throughput_Mbps"] if "throughput_Mbps" in df.columns else [])
for i in range(1,args.lags+1):
    for c in base: df[f"{c}_lag{i}"]=df.groupby("iface")[c].shift(i)
feats=[f"{c}_lag{i}" for i in range(1,args.lags+1) for c in base]
df=df.dropna(subset=feats+[lab])
uts=np.array(sorted(df["timestamp"].unique())); st=uts[int(0.8*len(uts))] if len(uts)>1 else df["timestamp"].max()
tr=df["timestamp"]<=st; te=df["timestamp"]>st
Xtr=df.loc[tr,feats].to_numpy(float); ytr=df.loc[tr,lab].to_numpy(float)
Xte=df.loc[te,feats].to_numpy(float); yte=df.loc[te,lab].to_numpy(float)
mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
Xtr=np.hstack([Xtr,np.ones((Xtr.shape[0],1))]); Xte=np.hstack([Xte,np.ones((Xte.shape[0],1))])
w=inv(Xtr.T@Xtr + args.l2*np.eye(Xtr.shape[1])) @ (Xtr.T@ytr)
import math; rmse=math.sqrt(((yte-(Xte@w))**2).mean()) if len(yte) else float("nan")
print(f"iface={args.iface}  RMSE={rmse:.3f}  train={tr.sum()}  test={te.sum()}  lags={args.lags}  l2={args.l2}")
