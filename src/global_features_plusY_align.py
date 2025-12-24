#!/usr/bin/env python3
# Build GLOBAL sensor features with proper 1s alignment + sensor backlog as features,
# and set target = next-second backlog per iface.

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
    ap.add_argument("--out", default="global_lead_align.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.processed, parse_dates=["timestamp"])
    df["ts"] = df["timestamp"].dt.floor("S")
    df = df.sort_values(["iface","ts"])

    with open(args.sensors_txt) as f:
        sensors = [x.strip() for x in f if x.strip()]
    if not sensors:
        raise SystemExit("sensors.txt is empty")

    label = "backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts")
    feat_cols = [c for c in df.columns if c.endswith("_per_s")]
    if "throughput_Mbps" in df.columns:
        feat_cols.append("throughput_Mbps")

    # Build a strict 1 Hz grid
    ts_all = pd.date_range(df["ts"].min(), df["ts"].max(), freq="S")
    grid = pd.DataFrame({"ts": ts_all})

    # Per-sensor rate features aligned to the grid
    parts = []
    for s in sensors:
        sub = df[df["iface"]==s][["ts"]+feat_cols].copy()
        sub = sub.drop_duplicates("ts").set_index("ts").reindex(ts_all).sort_index()
        sub = sub.fillna(method="ffill", limit=1)  # gentle ffill (max 1s)
        sub.columns = [f"{s}__{c}" for c in feat_cols]
        parts.append(sub)
    G_rate = pd.concat(parts, axis=1)

    # Per-sensor backlog as features (aligned)
    partsY = []
    for s in sensors:
        ys = df[df["iface"]==s][["ts", label]].copy()
        ys = ys.drop_duplicates("ts").set_index("ts").reindex(ts_all).sort_index()
        ys = ys.fillna(method="ffill", limit=1)
        ys.columns = [f"{s}__Y"]
        partsY.append(ys)
    G_y = pd.concat(partsY, axis=1)

    G = pd.concat([G_rate, G_y], axis=1)
    G = G.reset_index().rename(columns={"index":"ts"})

    # Add lags on GLOBAL features
    gcols = [c for c in G.columns if c!="ts"]
    if args.lags > 0:
        for i in range(1, args.lags+1):
            for c in gcols:
                G[f"{c}_lag{i}"] = G[c].shift(i)

    # Target = next-second backlog per iface on aligned grid
    base = (df[["ts","iface",label]]
            .drop_duplicates(["ts","iface"])
            .sort_values(["iface","ts"]))
    base["target_next"] = base.groupby("iface")[label].shift(-1)

    OUT = base.merge(G, on="ts", how="inner").dropna(subset=["target_next"])
    OUT = OUT.sort_values(["ts","iface"]).rename(columns={"ts":"timestamp"})
    OUT.to_csv(args.out, index=False)
    print(f"OK: wrote {args.out} | rows={len(OUT)} | sensors={sensors} | feats={OUT.shape[1]-3}")

if __name__ == "__main__":
    main()
