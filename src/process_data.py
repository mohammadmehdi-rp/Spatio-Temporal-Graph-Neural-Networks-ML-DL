# Reads data.csv → computes per-second features → chooses live queue label → writes processed.csv

import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="data.csv")
    ap.add_argument("--out", default="processed.csv")
    ap.add_argument("--label", choices=["backlog_pkts","backlog_bytes","qlen_pkts"], default=None,
                    help="Force a label; default: prefer backlog_pkts, then backlog_bytes, else qlen_pkts")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    need = {"timestamp","iface"}
    if not need.issubset(df.columns):
        raise SystemExit(f"Missing columns: {need - set(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["iface","timestamp"])
    df["dt"] = df.groupby("iface")["timestamp"].diff().dt.total_seconds()

    counter_cols = [c for c in ["rx_bytes","tx_bytes","rx_packets","tx_packets","rx_dropped","tx_dropped"] if c in df.columns]
    for c in counter_cols:
        d = df.groupby("iface")[c].diff()
        d = d.where(d >= 0)  # drop wraps/resets
        df[c+"_per_s"] = d / df["dt"]

    if {"rx_bytes_per_s","tx_bytes_per_s"}.issubset(df.columns):
        df["throughput_Mbps"] = (df["rx_bytes_per_s"] + df["tx_bytes_per_s"]) * 8 / 1e6

    if args.label:
        label = args.label
    else:
        order = ["backlog_pkts","backlog_bytes","qlen_pkts"]
        label = next((c for c in order if c in df.columns), None)
        if label is None:
            raise SystemExit("No queue label columns found (need one of: backlog_pkts, backlog_bytes, qlen_pkts).")

    df = df.dropna(subset=["dt", label])
    feat_cols = [c for c in df.columns if c.endswith("_per_s")]
    if "throughput_Mbps" in df.columns:
        feat_cols.append("throughput_Mbps")

    out = df[["timestamp","iface"] + feat_cols + [label]].copy()
    out.to_csv(args.out, index=False)

    print(f"OK: wrote {args.out}")
    print(f"Rows: {len(out)} | Ifaces: {out['iface'].nunique()} | Label: {label}")
    print("Feature cols:", len(feat_cols))

if __name__ == "__main__":
    main()
