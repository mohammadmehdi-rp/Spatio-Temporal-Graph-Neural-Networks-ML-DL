#!/usr/bin/env python3
# global_features_agg.py — build compact GLOBAL features: per-sensor sums/means across bytes/packets + lags
import argparse, pandas as pd, numpy as np
ap=argparse.ArgumentParser(); ap.add_argument("processed"); ap.add_argument("sensors_txt"); ap.add_argument("--lags",type=int,default=5); ap.add_argument("--out",default="global_agg.csv")
args=ap.parse_args()
df=pd.read_csv(args.processed,parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
lab="backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts")
with open(args.sensors_txt) as f: sensors=[x.strip() for x in f if x.strip()]
feat=[c for c in df.columns if c.endswith("_per_s")] + (["throughput_Mbps"] if "throughput_Mbps" in df.columns else [])
# build per-timestamp aggregates over sensors
S=df[df["iface"].isin(sensors)].pivot_table(index="timestamp", columns="iface", values=feat, aggfunc="first").sort_index()
# simple reductions across sensors per feature
agg=pd.DataFrame(index=S.index)
for c in feat:
    sub=S[c]
    agg[f"sum__{c}"]=sub.sum(axis=1, min_count=1)
    agg[f"mean__{c}"]=sub.mean(axis=1)
    agg[f"max__{c}"]=sub.max(axis=1)
# add lags on these compact features
for i in range(1,args.lags+1):
    for c in list(agg.columns): agg[f"{c}_lag{i}"]=agg[c].shift(i)
Y=df[["timestamp","iface",lab]]
OUT=Y.merge(agg.reset_index(), on="timestamp", how="inner").sort_values(["timestamp","iface"])
OUT.to_csv(args.out, index=False); print(f"OK: {args.out}  rows={len(OUT)}  feats={OUT.shape[1]-3}  sensors={sensors}")
