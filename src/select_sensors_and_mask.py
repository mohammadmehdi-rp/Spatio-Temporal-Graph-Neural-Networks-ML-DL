# Picks sensors (include specific ports + top-variance), masks features on others → writes processed_masked.csv + sensors.txt

import argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("processed", nargs="?", default="processed.csv")
    ap.add_argument("--count", type=int, default=5, help="total number of sensors")
    ap.add_argument("--include", default="", help="comma-separated ifaces that MUST be sensors (e.g., s2-eth2,s3-eth2)")
    ap.add_argument("--out", default="processed_masked.csv")
    ap.add_argument("--sensors_txt", default="sensors.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.processed, parse_dates=["timestamp"]).sort_values(["timestamp","iface"])
    label = "backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts")
    if label not in df.columns:
        raise SystemExit("Label column not found in processed.csv")

    must = [s.strip() for s in args.include.split(",") if s.strip()]
    by_std = df.groupby("iface")[label].std().sort_values(ascending=False)
    sensors = []
    for m in must:
        if m in by_std.index and m not in sensors:
            sensors.append(m)
    for iface in by_std.index:
        if len(sensors) >= args.count: break
        if iface not in sensors:
            sensors.append(iface)
    sensors = sensors[:args.count]

    feat_cols = [c for c in df.columns if c.endswith("_per_s")]
    if "throughput_Mbps" in df.columns:
        feat_cols.append("throughput_Mbps")

    dfm = df.copy()
    dfm.loc[~dfm["iface"].isin(sensors), feat_cols] = float("nan")
    dfm.to_csv(args.out, index=False)
    with open(args.sensors_txt, "w") as f:
        f.write("\n".join(sensors))

    print("Sensors:", sensors)
    print(f"OK: wrote {args.out} and {args.sensors_txt}")

if __name__ == "__main__":
    main()
