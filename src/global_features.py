#!/usr/bin/env python3
# Build wide "global sensor" features per timestamp, then join to every iface row.
# Each feature is prefixed with "<sensor_iface>__".
import argparse, pandas as pd, numpy as np
def add_lags(df, cols, k):
    for i in range(1, k+1):
        for c in cols:
            df[f"{c}_lag{i}"] = df[c].shift(i)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", help="processed.csv from process_data.py")
    ap.add_argument("sensors_txt", help="sensors.txt (one iface per line)")
    ap.add_argument("--out", default="global.csv")
    ap.add_argument("--lags", type=int, default=10)
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

    # Build global (timestamp-indexed) matrix of sensor features
    G = (df[df["iface"].isin(sensors)]
         .set_index("timestamp"))
    parts = []
    for s in sensors:
        S = G[G["iface"]==s][feat_cols].copy()
        S.columns = [f"{s}__{c}" for c in feat_cols]
        parts.append(S)
    G = pd.concat(parts, axis=1).sort_index()
    # Forward-fill within short gaps; keep leading NaNs as-is
    G = G.groupby(G.index).first().sort_index().fillna(method="ffill", limit=3)

    # Add lags on GLOBAL features
    gcols = list(G.columns)
    if args.lags > 0:
        G = add_lags(G, gcols, args.lags)
    G = G.reset_index()

    # Join global features to every iface/label row at same timestamp
    Y = df[["timestamp","iface",label]].copy()
    OUT = pd.merge(Y, G, on="timestamp", how="inner").sort_values(["timestamp","iface"])

    OUT.to_csv(args.out, index=False)
    print(f"OK: wrote {args.out} | rows={len(OUT)} | sensors={sensors} | feats={len(OUT.columns)-3}")

if __name__ == "__main__":
    main()
