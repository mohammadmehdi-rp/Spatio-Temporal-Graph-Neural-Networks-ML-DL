#!/usr/bin/env python3
# Per-port AR(lags) baseline on the label; good sanity floor.

import argparse, pandas as pd, numpy as np
from numpy.linalg import inv
import math

def fit_ridge(X, y, l2):
    X = np.hstack([X, np.ones((X.shape[0],1))])  # intercept
    return inv(X.T@X + l2*np.eye(X.shape[1])) @ (X.T@y)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed", default="processed.csv")
    ap.add_argument("--label", default=None, choices=["backlog_pkts","backlog_bytes","qlen_pkts"])
    ap.add_argument("--lags", type=int, default=10)
    ap.add_argument("--l2", type=float, default=50.0)
    args=ap.parse_args()

    df = pd.read_csv(args.processed, parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
    lab = args.label or ("backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts"))

    # add label lags per iface
    for i in range(1, args.lags+1):
        df[f"{lab}_lag{i}"] = df.groupby("iface")[lab].shift(i)
    feats = [f"{lab}_lag{i}" for i in range(1, args.lags+1)]
    df = df.dropna(subset=feats+[lab])

    # time split on busy if possible
    busy = df[df[lab]>0]; uts = np.array(sorted((busy["timestamp"] if len(busy) else df["timestamp"]).unique()))
    split_t = uts[int(0.8*len(uts))] if len(uts)>1 else df["timestamp"].max()
    tr = df["timestamp"]<=split_t; te = df["timestamp"]>split_t

    # fit global AR across pooled ifaces (simple baseline)
    Xtr = df.loc[tr, feats].to_numpy(float); ytr = df.loc[tr, lab].to_numpy(float)
    Xte = df.loc[te, feats].to_numpy(float); yte = df.loc[te, lab].to_numpy(float)
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    w = fit_ridge(Xtr, ytr, args.l2)
    yhat = np.hstack([Xte, np.ones((Xte.shape[0],1))]) @ w
    rmse = math.sqrt(((yte - yhat)**2).mean()) if len(yte) else float("nan")
    print(f"AR(lags={args.lags}) Ridge → micro {rmse:.3f} | train {tr.sum()} | test {te.sum()} | split {split_t}")

if __name__=="__main__": main()
