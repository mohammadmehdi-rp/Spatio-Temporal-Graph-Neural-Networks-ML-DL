#!/usr/bin/env python3
# Select top-K global features by train-time correlation with the label.
import argparse, pandas as pd, numpy as np
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("global_csv")              # from global_features.py
    ap.add_argument("--k", type=int, default=80)
    ap.add_argument("--out", default="global_topk.csv")
    args=ap.parse_args()

    df=pd.read_csv(args.global_csv, parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
    label="backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts")

    # time split (same rule as baseline)
    busy=df[df[label]>0]
    uts=np.array(sorted((busy["timestamp"] if len(busy) else df["timestamp"]).unique()))
    split_t=uts[int(0.8*len(uts))] if len(uts)>1 else df["timestamp"].max()
    tr=df["timestamp"]<=split_t

    feat=[c for c in df.columns if "__" in c]  # global sensor features (+ lags)
    dtr=df.loc[tr, feat+[label]].dropna()
    if dtr.empty: raise SystemExit("Not enough train rows; collect longer run or reduce lags.")
    corr=dtr.corr(numeric_only=True)[label].abs().sort_values(ascending=False)
    keep=[c for c in corr.index if c in feat][:args.k]

    out=df[["timestamp","iface",label]+keep].copy()
    out.to_csv(args.out, index=False)
    print(f"OK: wrote {args.out} | K={len(keep)} | split={split_t}")
if __name__=="__main__": main()
