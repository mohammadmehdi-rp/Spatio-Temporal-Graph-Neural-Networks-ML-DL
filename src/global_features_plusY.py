#!/usr/bin/env python3
# Build global sensor features (rates + sensors' own backlog) with lags,
# and make target = next-second backlog per iface.

import argparse, pandas as pd, numpy as np

def add_lags(df, cols, K):
    for i in range(1, K+1):
        for c in cols:
            df[f"{c}_lag{i}"] = df[c].shift(i)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", help="processed.csv from process_data.py")
    ap.add_argument("sensors_txt", help="sensors.txt (one iface per line)")
    ap.add_argument("--lags", type=int, default=10)
    ap.add_argument("--out", default="global_lead.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.processed, parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
    with open(args.sensors_txt) as f:
        sensors = [x.strip() for x in f if x.strip()]
    if not sensors:
        raise SystemExit("sensors.txt is empty")

    label = "backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts")
    feat_cols = [c for c in df.columns if c.endswith("_per_s")]
    if "throughput_Mbps" in df.columns:
        feat_cols.append("throughput_Mbps")

    # Global sensor RATE features
    parts = []
    for s in sensors:
        sub = df[df["iface"]==s][["timestamp"]+feat_cols].copy().set_index("timestamp")
        sub.columns = [f"{s}__{c}" for c in feat_cols]
        parts.append(sub)
    G = pd.concat(parts, axis=1).sort_index()

    # Global sensor LABEL (backlog) as features, too
    Yparts = []
    for s in sensors:
        ys = df[df["iface"]==s][["timestamp", label]].copy().set_index("timestamp")
        ys.columns = [f"{s}__Y"]
        Yparts.append(ys)
    YG = pd.concat(Yparts, axis=1).sort_index()

    G = pd.concat([G, YG], axis=1).sort_index()
    G = G.groupby(G.index).first().sort_index().fillna(method="ffill", limit=3)  # gentle ffill

    gcols = list(G.columns)
    if args.lags > 0:
        G = add_lags(G, gcols, args.lags)
    G = G.reset_index()

    # Target = next-second backlog per iface
    base = df[["timestamp","iface",label]].copy().sort_values(["iface","timestamp"])
    base["target_next"] = base.groupby("iface")[label].shift(-1)

    OUT = pd.merge(base[["timestamp","iface","target_next"]], G, on="timestamp", how="inner")
    OUT = OUT.dropna(subset=["target_next"]).sort_values(["timestamp","iface"])
    OUT.to_csv(args.out, index=False)
    print(f"OK: wrote {args.out} | rows={len(OUT)} | sensors={sensors} | feats={OUT.shape[1]-3}")

if __name__ == "__main__":
    main()
