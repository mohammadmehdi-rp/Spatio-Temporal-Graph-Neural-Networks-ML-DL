#!/usr/bin/env python3
"""process_data_plus.py

Extended preprocessing for multi-target experiments (queue + throughput + utilization + latency).

Inputs
------
- Raw 5 Hz interface telemetry CSV (collector_5hz.py output): must include timestamp, iface and counters.
- Optional latency CSV (ping-like RTT): must include timestamp and rtt_ms (or rtt).

Outputs
-------
processed_plus.csv with:
  timestamp, iface, <rate features>, throughput_Mbps, <queue label>, rtt_ms (optional), qdelay_ms (optional)

Key points
----------
1) Computes per-second rates from counters (bytes/s, packets/s, drops/s).
2) Derived throughput_Mbps = (rx_bytes_per_s + tx_bytes_per_s) * 8 / 1e6
3) If latency CSV is provided, merges by nearest timestamp.
4) Cleans RTT and (optionally) computes queueing delay:
      qdelay_ms = max(0, rtt_ms - rtt_base_ms)
   where rtt_base_ms is a low-quantile baseline (default 5th percentile).
   qdelay_ms is then optionally clipped at a high quantile (default 99th percentile).
"""

import argparse
from typing import Optional

import numpy as np
import pandas as pd


def _pick_label(df: pd.DataFrame, forced: Optional[str]) -> str:
    if forced:
        if forced not in df.columns:
            raise SystemExit(f"--label={forced} not found in input columns")
        return forced
    for c in ["backlog_pkts", "backlog_bytes", "qlen_pkts"]:
        if c in df.columns:
            return c
    raise SystemExit("No queue label columns found (need one of: backlog_pkts, backlog_bytes, qlen_pkts).")


def _load_latency_csv(path: str) -> pd.DataFrame:
    """Load latency trace CSV with columns timestamp,rtt_ms."""
    lat = pd.read_csv(path)
    if "timestamp" not in lat.columns:
        raise SystemExit("Latency CSV must contain a 'timestamp' column")
    if "rtt_ms" not in lat.columns:
        if "rtt" in lat.columns:
            lat = lat.rename(columns={"rtt": "rtt_ms"})
        else:
            raise SystemExit("Latency CSV must contain 'rtt_ms' or 'rtt' column")

    ts = lat["timestamp"]
    if np.issubdtype(ts.dtype, np.number):
        lat["timestamp"] = pd.to_datetime(lat["timestamp"], unit="s", utc=True, errors="coerce")
    else:
        lat["timestamp"] = pd.to_datetime(lat["timestamp"], utc=True, errors="coerce")

    lat = lat.dropna(subset=["timestamp"]).sort_values("timestamp")
    lat["rtt_ms"] = pd.to_numeric(lat["rtt_ms"], errors="coerce")
    return lat[["timestamp", "rtt_ms"]]


def _clean_rtt_series(rtt: pd.Series, treat_zeros_as_missing: bool, clip_q: float) -> pd.Series:
    """Clean RTT series:
    - Convert to numeric, optionally treat 0 as missing
    - Drop insane negatives
    - Forward/back-fill
    - Clip high tail (optional)
    """
    x = pd.to_numeric(rtt, errors="coerce")
    x = x.where(x >= 0)
    if treat_zeros_as_missing:
        x = x.where(x > 0)

    # Fill gaps
    x = x.ffill().bfill()

    # If still NaN, set to 0 (should be rare)
    x = x.fillna(0.0)

    # Optional clipping of extreme spikes
    clip_q = float(clip_q)
    if 0 < clip_q < 1:
        qv = float(np.nanquantile(x.to_numpy(dtype=float), clip_q))
        if np.isfinite(qv) and qv > 0:
            x = x.clip(upper=qv)

    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="data.csv")
    ap.add_argument("--out", default="processed_plus.csv")
    ap.add_argument(
        "--label",
        choices=["backlog_pkts", "backlog_bytes", "qlen_pkts"],
        default=None,
        help="Force a queue label; default: prefer backlog_pkts, then backlog_bytes, else qlen_pkts",
    )

    # Latency / qdelay options
    ap.add_argument("--latency_csv", default="", help="Optional latency CSV (timestamp,rtt_ms).")
    ap.add_argument("--latency_tolerance_ms", type=float, default=250.0, help="merge_asof tolerance.")
    ap.add_argument("--rtt_clip_q", type=float, default=0.99, help="Clip RTT at this quantile after filling.")
    ap.add_argument("--rtt_zeros_missing", action="store_true", help="Treat rtt_ms==0 as missing before filling.")

    ap.add_argument("--make_qdelay", action="store_true", help="If set, compute qdelay_ms = max(0, rtt_ms - base).")
    ap.add_argument("--qdelay_base_q", type=float, default=0.05, help="Quantile to estimate RTT baseline.")
    ap.add_argument("--qdelay_clip_q", type=float, default=0.99, help="Clip qdelay_ms at this quantile (0 disables).")

    args = ap.parse_args()

    df = pd.read_csv(args.input)
    need = {"timestamp", "iface"}
    if not need.issubset(df.columns):
        raise SystemExit(f"Missing columns: {need - set(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["iface", "timestamp"])

    # Compute dt per iface
    df["dt"] = df.groupby("iface")["timestamp"].diff().dt.total_seconds()

    counter_cols = [c for c in ["rx_bytes", "tx_bytes", "rx_packets", "tx_packets", "rx_dropped", "tx_dropped"] if c in df.columns]
    for c in counter_cols:
        d = df.groupby("iface")[c].diff()
        d = d.where(d >= 0)  # drop wraps/resets
        df[c + "_per_s"] = d / df["dt"]

    # Derived throughput (Mbps)
    if {"rx_bytes_per_s", "tx_bytes_per_s"}.issubset(df.columns):
        df["throughput_Mbps"] = (df["rx_bytes_per_s"] + df["tx_bytes_per_s"]) * 8 / 1e6

    label = _pick_label(df, args.label)

    # Drop rows without dt or label
    df = df.dropna(subset=["dt", label])

    rate_cols = [c for c in df.columns if c.endswith("_per_s")]
    feat_cols = list(rate_cols)
    if "throughput_Mbps" in df.columns:
        feat_cols.append("throughput_Mbps")

    out = df[["timestamp", "iface"] + feat_cols + [label]].copy()

    # Optional latency merge
    if args.latency_csv:
        lat = _load_latency_csv(args.latency_csv)
        out = out.sort_values("timestamp")
        tol = pd.Timedelta(milliseconds=float(args.latency_tolerance_ms))
        out = pd.merge_asof(out, lat, on="timestamp", direction="nearest", tolerance=tol)

        # Clean RTT (global per timestamp; replicated per iface)
        if "rtt_ms" in out.columns:
            out["rtt_ms"] = _clean_rtt_series(out["rtt_ms"], treat_zeros_as_missing=bool(args.rtt_zeros_missing), clip_q=float(args.rtt_clip_q))

            # Queueing delay (optional)
            if args.make_qdelay:
                base_q = float(args.qdelay_base_q)
                base_q = min(max(base_q, 0.0), 1.0)
                rtt_base = float(np.quantile(out["rtt_ms"].to_numpy(dtype=float), base_q))
                qd = (out["rtt_ms"] - rtt_base).clip(lower=0.0)

                clip_q = float(args.qdelay_clip_q)
                if 0 < clip_q < 1:
                    qv = float(np.quantile(qd.to_numpy(dtype=float), clip_q))
                    if np.isfinite(qv) and qv > 0:
                        qd = qd.clip(upper=qv)

                out["qdelay_ms"] = qd

    out.to_csv(args.out, index=False)

    has_rtt = "rtt_ms" in out.columns
    has_qd = "qdelay_ms" in out.columns
    print(f"OK: wrote {args.out}")
    print(f"Rows: {len(out)} | Ifaces: {out['iface'].nunique()} | Label: {label}")
    print(f"Rate cols: {len(rate_cols)} | Has throughput: {'throughput_Mbps' in out.columns} | Has RTT: {has_rtt} | Has qdelay: {has_qd}")


if __name__ == "__main__":
    main()
