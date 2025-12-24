#!/usr/bin/env python3
# expand_sensors.py — ensure sensors include bottleneck + neighbors, fill rest by variance
import argparse, pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument("--processed",default="processed.csv"); ap.add_argument("--include",default="s2-eth2,s2-eth1,s1-eth1"); ap.add_argument("--count",type=int,default=10); ap.add_argument("--out",default="sensors.txt")
args=ap.parse_args()
df=pd.read_csv(args.processed,parse_dates=["timestamp"])
lab="backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts")
must=[s.strip() for s in args.include.split(",") if s.strip()]
by=df.groupby("iface")[lab].std().sort_values(ascending=False)
sensors=[]; [sensors.append(m) for m in must if m in by.index and m not in sensors]
for iface in by.index:
    if len(sensors)>=args.count: break
    if iface not in sensors: sensors.append(iface)
open(args.out,"w").write("\n".join(sensors)); print("sensors=",sensors)
