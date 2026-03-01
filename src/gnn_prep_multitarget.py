#!/usr/bin/env python3
"""gnn_prep_multitarget.py

Build a multi-target NPZ dataset for state-estimation experiments (queue/throughput/util/delay),
with robust support for multi-probe RTT series.

Why this file exists
- Telemetry timestamps are on a grid (e.g., 5 Hz).
- Multi-probe RTT samples are periodic but not timestamp-identical (few ms jitter).
  Exact reindex() yields all-NaN → fillna(0) → all-zero RTT. This script avoids that by
  aligning RTT via nearest-time merge (merge_asof) with a tolerance.

CLI (kept compatible with run scripts):
  --processed
  --latency_multi_csv
  --links_file
  --bw_access
  --bw_bottleneck
  --out

Output NPZ contains (key fields):
  X: [T,N,F] float32  (normalised features)
  Y: [T,N,4] float32  (queue, throughput, util, delay)
  delay_series_ms: [T] or [T,K] float32 (graph-level delay series; multi-probe -> [T,K])
  probe_ids/src/dst: metadata for multi-probe RTT
  probe_prop_rtt_ms: per-probe propagation RTT estimate (5th percentile RTT)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Small helpers
# ----------------------------

def iface_group_key(iface: str) -> str:
    """Group interfaces belonging to the same device for clique edges."""
    s = str(iface)
    if "-eth" in s:
        return s.split("-eth", 1)[0]
    return s


def build_edges(nodes: List[str], links_file: str = "") -> np.ndarray:
    """Directed edge_index [2,E] over interface-nodes."""
    node_to_i = {n: i for i, n in enumerate(nodes)}
    src: List[int] = []
    dst: List[int] = []

    # 1) clique within each device
    buckets: Dict[str, List[int]] = {}
    for n, i in node_to_i.items():
        buckets.setdefault(iface_group_key(n), []).append(i)

    for _, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                src += [a, b]
                dst += [b, a]

    # 2) physical links
    if links_file and os.path.exists(links_file):
        with open(links_file, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) != 2:
                    continue
                a, b = p
                if a in node_to_i and b in node_to_i:
                    ia, ib = node_to_i[a], node_to_i[b]
                    src += [ia, ib]
                    dst += [ib, ia]

    return np.asarray([src, dst], dtype=np.int32)


def load_links_endpoints(links_file: str) -> List[str]:
    if not links_file or not os.path.exists(links_file):
        return []
    out: List[str] = []
    with open(links_file, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 2:
                out.extend(p)
    return sorted(set(out))


def fit_ar1_per_node(Y: np.ndarray, train_idx: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    """Per-node AR(1): y[t] ≈ a * y[t-1]."""
    T, N = Y.shape
    tr = np.asarray(train_idx, dtype=int)
    t_valid = tr[(tr >= 1) & (tr < T)]
    a = np.zeros(N, dtype=np.float32)
    for i in range(N):
        y = Y[:, i].astype(np.float64)
        x = y[t_valid - 1]
        z = y[t_valid]
        x2 = float(np.dot(x, x) + lam)
        xz = float(np.dot(x, z))
        ai = (xz / x2) if x2 > 0 else 0.0
        a[i] = np.float32(np.clip(ai, 0.0, 1.2))
    return a


# ----------------------------
# Multi-probe RTT alignment
# ----------------------------

def load_latency_multi(path: str) -> pd.DataFrame:
    """latency_multi.csv: timestamp, src, dst, rtt_ms (probe optional)."""
    lat = pd.read_csv(path)
    required = {"timestamp", "src", "dst"}
    missing = required - set(lat.columns)
    if missing:
        raise SystemExit(f"latency_multi_csv missing columns: {sorted(missing)}")
    if "rtt_ms" not in lat.columns:
        if "rtt" in lat.columns:
            lat = lat.rename(columns={"rtt": "rtt_ms"})
        else:
            raise SystemExit("latency_multi_csv must contain 'rtt_ms' (or 'rtt').")

    lat["timestamp"] = pd.to_datetime(lat["timestamp"], utc=True, errors="coerce")
    lat["src"] = pd.to_numeric(lat["src"], errors="coerce")
    lat["dst"] = pd.to_numeric(lat["dst"], errors="coerce")
    lat["rtt_ms"] = pd.to_numeric(lat["rtt_ms"], errors="coerce")

    lat = lat.dropna(subset=["timestamp", "src", "dst", "rtt_ms"]).copy()
    lat["src"] = lat["src"].astype(int)
    lat["dst"] = lat["dst"].astype(int)
    if "probe" not in lat.columns:
        lat["probe"] = lat["src"].astype(str) + "-" + lat["dst"].astype(str)

    lat = lat.sort_values(["probe", "timestamp"]).reset_index(drop=True)
    return lat[["timestamp", "src", "dst", "probe", "rtt_ms"]]


def align_latency_multi_to_ts(
    lat: pd.DataFrame,
    ts: List[pd.Timestamp],
    tol_ms: int = 180,
    max_missing: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align per-probe RTT samples to telemetry timestamps ts using merge_asof."""
    ts_utc = pd.to_datetime(pd.Series(ts), utc=True, errors="coerce")
    if ts_utc.isna().any():
        raise RuntimeError("Some telemetry timestamps could not be parsed as UTC datetimes.")
    ts_df = pd.DataFrame({"timestamp": ts_utc})

    probes = sorted(lat["probe"].unique().tolist())
    K = len(probes)
    T = len(ts_df)

    tol = pd.Timedelta(milliseconds=int(tol_ms))
    M = np.zeros((T, K), dtype=np.float64)
    prop = np.zeros((K,), dtype=np.float64)

    for k, pid in enumerate(probes):
        d = lat[lat["probe"] == pid][["timestamp", "rtt_ms"]].copy()
        d = d.dropna()
        d = d.groupby("timestamp", as_index=False)["rtt_ms"].mean()
        d = d.sort_values("timestamp")
        if len(d) == 0:
            raise RuntimeError(f"Empty RTT series for probe {pid}")

        prop[k] = float(np.percentile(d["rtt_ms"].to_numpy(dtype=float), 5.0))

        mrg = pd.merge_asof(
            ts_df,
            d,
            on="timestamp",
            direction="nearest",
            tolerance=tol,
        )
        series = mrg["rtt_ms"].to_numpy(dtype=np.float64)
        miss = float(np.isnan(series).mean())
        if miss > max_missing:
            raise RuntimeError(
                f"RTT alignment failed for probe={pid}: missing={miss:.1%}. "
                f"Increase --lat_tol_ms (currently {tol_ms}ms) or check timestamp units."
            )

        s = pd.Series(series).ffill().bfill()
        fill = float(s.median()) if np.isfinite(s.median()) else 0.0
        series = s.fillna(fill).to_numpy(dtype=np.float64)
        M[:, k] = series

    probe_src = np.array([int(x.split("-")[0]) for x in probes], dtype=np.int32)
    probe_dst = np.array([int(x.split("-")[1]) for x in probes], dtype=np.int32)
    probe_ids = np.array(probes, dtype=object)
    probe_prop_rtt_ms = prop.astype(np.float32)

    return M.astype(np.float32), probe_src, probe_dst, probe_ids, probe_prop_rtt_ms


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="processed_plus.csv")
    ap.add_argument("--latency_multi_csv", default="", help="Optional: multi-probe RTT CSV.")
    ap.add_argument("--lat_tol_ms", type=int, default=180, help="Nearest-join tolerance for RTT alignment (ms).")
    ap.add_argument("--links_file", default="links.txt")

    ap.add_argument("--bw_access", type=float, default=100.0)
    ap.add_argument("--bw_bottleneck", type=float, default=10.0)

    ap.add_argument("--fraction", type=float, default=0.4)
    ap.add_argument("--include", default="")
    ap.add_argument("--sensors_file", default="")
    ap.add_argument("--mask_rates_non_sensors", action="store_true")

    ap.add_argument("--out", default="dataset_multi.npz")
    args = ap.parse_args()

    # Load processed telemetry; force tz-aware UTC timestamps
    df = pd.read_csv(args.processed)
    if "timestamp" not in df.columns:
        raise SystemExit("processed CSV must contain a 'timestamp' column")
    if "iface" not in df.columns:
        raise SystemExit("processed CSV must contain an 'iface' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "iface"]).copy()
    df["iface"] = df["iface"].astype(str)
    df = df.sort_values(["timestamp", "iface"]).reset_index(drop=True)

    # Select label columns
    qcol = "backlog_pkts" if "backlog_pkts" in df.columns else (
        "backlog_bytes" if "backlog_bytes" in df.columns else (
            "qlen_pkts" if "qlen_pkts" in df.columns else ""
        )
    )
    if not qcol:
        raise SystemExit("processed CSV missing queue column: backlog_pkts/backlog_bytes/qlen_pkts")

    has_thr = "throughput_Mbps" in df.columns
    delay_col = "qdelay_ms" if "qdelay_ms" in df.columns else ("rtt_ms" if "rtt_ms" in df.columns else None)
    delay_name = delay_col if delay_col else "delay_ms"

    rate_feats = sorted([c for c in df.columns if c.endswith("_per_s")])

    nodes = sorted(df["iface"].unique().tolist())
    ts = sorted(df["timestamp"].unique().tolist())
    N, T = len(nodes), len(ts)
    node_to_i = {n: i for i, n in enumerate(nodes)}

    Xr = np.zeros((T, N, len(rate_feats)), dtype=np.float32)
    Yq = np.zeros((T, N), dtype=np.float32)
    Ythr = np.zeros((T, N), dtype=np.float32)
    Ydelay = np.zeros((T, N), dtype=np.float32)

    for n in nodes:
        ii = node_to_i[n]
        sub = df[df["iface"] == n].drop_duplicates("timestamp").set_index("timestamp").reindex(ts)

        for k, f in enumerate(rate_feats):
            Xr[:, ii, k] = pd.to_numeric(sub[f], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        Yq[:, ii] = pd.to_numeric(sub[qcol], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        if has_thr:
            Ythr[:, ii] = pd.to_numeric(sub["throughput_Mbps"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        if delay_col is not None:
            Ydelay[:, ii] = pd.to_numeric(sub[delay_col], errors="coerce").to_numpy(dtype=np.float32)

    if delay_col is not None:
        for i in range(N):
            Ydelay[:, i] = pd.Series(Ydelay[:, i]).ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)

    # Capacities
    bottleneck_endpoints = set(load_links_endpoints(args.links_file))
    cap = np.full((N,), float(args.bw_access), dtype=np.float32)
    for i, n in enumerate(nodes):
        if n in bottleneck_endpoints:
            cap[i] = float(args.bw_bottleneck)
    cap_ch = np.repeat(cap[None, :, None], T, axis=0)

    Yutil = (Ythr / (cap[None, :] + 1e-9)).astype(np.float32) if has_thr else np.zeros_like(Yq)

    # Sensors
    include = [s.strip() for s in str(args.include).split(",") if s.strip()]
    if args.sensors_file and os.path.exists(args.sensors_file):
        sensors = [s.strip() for s in open(args.sensors_file) if s.strip()]
        sensors = [s for s in sensors if s in node_to_i]
    else:
        must = [s for s in include if s in node_to_i]
        k = max(len(must), int(round(float(args.fraction) * N)))
        stds = np.std(Yq, axis=0)
        ranked = [nodes[i] for i in np.argsort(-stds)]
        sensors: List[str] = []
        for s in must:
            if s not in sensors:
                sensors.append(s)
        for s in ranked:
            if len(sensors) >= k:
                break
            if s not in sensors:
                sensors.append(s)
        sensors = sorted(sensors)

    is_sensor = np.zeros((N,), dtype=np.uint8)
    for s in sensors:
        is_sensor[node_to_i[s]] = 1

    if args.mask_rates_non_sensors and len(rate_feats) > 0:
        Xr = Xr * is_sensor.astype(np.float32)[None, :, None]

    # Sensor backlog channels + lags
    Smask = is_sensor[None, :].astype(np.float32)
    Sb = Yq * Smask
    Sb_l1 = np.vstack([np.zeros((1, N), np.float32), Sb[:-1]])
    Sb_l2 = np.vstack([np.zeros((2, N), np.float32), Sb[:-2]])
    Sb_l3 = np.vstack([np.zeros((3, N), np.float32), Sb[:-3]])

    # Split indices (busy-aware if possible)
    y_sum = Yq.sum(axis=1)
    busy = np.where(y_sum > 0.0)[0].astype(int)
    if len(busy) >= 10:
        b1 = int(round(0.70 * len(busy)))
        b2 = int(round(0.85 * len(busy)))
        b1 = max(1, min(len(busy) - 2, b1))
        b2 = max(b1 + 1, min(len(busy) - 1, b2))
        cut1 = int(busy[b1])
        cut2 = int(busy[b2])
        train_idx = np.arange(0, cut1 + 1, dtype=np.int64)
        val_idx = np.arange(cut1 + 1, cut2 + 1, dtype=np.int64)
        test_idx = np.arange(cut2 + 1, T, dtype=np.int64)
    else:
        cut1 = int(0.70 * T)
        cut2 = int(0.85 * T)
        train_idx = np.arange(0, cut1, dtype=np.int64)
        val_idx = np.arange(cut1, cut2, dtype=np.int64)
        test_idx = np.arange(cut2, T, dtype=np.int64)

    a = fit_ar1_per_node(Yq, train_idx)
    AR1 = (Yq * a[None, :])[..., None].astype(np.float32)

    cap_z = (cap - cap.mean()) / (cap.std() + 1e-6)
    static = np.repeat(cap_z[None, :, None].astype(np.float32), T, axis=0)

    is_sensor_ch = np.repeat(is_sensor[None, :, None].astype(np.float32), T, axis=0)

    X = np.concatenate(
        [
            Xr,
            cap_ch,
            Sb[..., None],
            Sb_l1[..., None],
            Sb_l2[..., None],
            Sb_l3[..., None],
            AR1,
            is_sensor_ch,
            static,
        ],
        axis=2,
    )

    feat_names = rate_feats + [
        "capacity_Mbps",
        "sensor_backlog",
        "sensor_backlog_lag1",
        "sensor_backlog_lag2",
        "sensor_backlog_lag3",
        "ar1_next",
        "is_sensor",
        "cap_Mbps_z",
    ]

    # Normalise using train only
    mu = np.zeros((len(feat_names),), dtype=np.float32)
    sd = np.ones((len(feat_names),), dtype=np.float32)
    Xtr = X[train_idx]
    for i, name in enumerate(feat_names):
        arr = Xtr[:, :, i]
        if name == "is_sensor":
            mu[i], sd[i] = 0.0, 1.0
        else:
            mu[i] = float(arr.mean())
            sd[i] = float(arr.std() + 1e-6)
    Xn = (X - mu[None, None, :]) / sd[None, None, :]

    edges = build_edges(nodes, args.links_file)

    # Labels stack: [T,N,4]
    Y = np.stack([Yq, Ythr, Yutil, Ydelay], axis=2).astype(np.float32)
    label_names = np.array(["queue", "throughput", "util", delay_name], dtype=object)

    # Graph-level delay series
    probe_src = np.zeros((0,), dtype=np.int32)
    probe_dst = np.zeros((0,), dtype=np.int32)
    probe_ids = np.zeros((0,), dtype=object)
    probe_prop_rtt_ms = np.zeros((0,), dtype=np.float32)

    if args.latency_multi_csv and os.path.exists(args.latency_multi_csv):
        lat = load_latency_multi(args.latency_multi_csv)
        delay_series_ms, probe_src, probe_dst, probe_ids, probe_prop_rtt_ms = align_latency_multi_to_ts(
            lat, ts, tol_ms=args.lat_tol_ms
        )
        delay_series_name = "rtt_ms"
    else:
        if delay_col is not None:
            delay_series_ms = np.mean(Ydelay, axis=1).astype(np.float32)  # [T]
        else:
            delay_series_ms = np.zeros((T,), dtype=np.float32)
        delay_series_name = delay_name

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(out_path),
        nodes=np.array(nodes, dtype=object),
        edges=edges,
        feat_names=np.array(feat_names, dtype=object),
        label_names=label_names,
        X=Xn.astype(np.float32),
        Y=Y,
        is_sensor=is_sensor,
        sensors=np.array(sensors, dtype=object),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        norm_mu=mu,
        norm_sd=sd,
        timestamps=np.array([pd.Timestamp(t).isoformat() for t in ts], dtype=object),
        capacities_Mbps=cap,
        bottleneck_endpoints=np.array(sorted(bottleneck_endpoints), dtype=object),
        ar1_coef=a,
        delay_series_ms=delay_series_ms,
        delay_series_name=np.array(delay_series_name, dtype=object),
        probe_src=probe_src,
        probe_dst=probe_dst,
        probe_ids=probe_ids,
        probe_prop_rtt_ms=probe_prop_rtt_ms,
    )

    info = {
        "out": str(out_path),
        "T": int(T),
        "N": int(N),
        "F": int(Xn.shape[2]),
        "E": int(edges.shape[1]),
        "delay_series_shape": list(np.asarray(delay_series_ms).shape),
        "num_probes": int(len(probe_ids)),
        "split": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "num_sensors": int(is_sensor.sum()),
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
