#!/usr/bin/env python3
import argparse, re, numpy as np, pandas as pd, os, json

def parse_sw_port(n):
    m = re.match(r"^s(\d+)-eth(\d+)$", str(n))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

def build_edges(nodes, links_file):
    sw_buckets = {}
    for i, n in enumerate(nodes):
        sw, _ = parse_sw_port(n)
        sw_buckets.setdefault(sw, []).append(i)
    src, dst = [], []
    for _, idxs in sw_buckets.items():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                src += [a, b]; dst += [b, a]
    if links_file and os.path.exists(links_file):
        with open(links_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] in nodes and parts[1] in nodes:
                    a = nodes.index(parts[0]); b = nodes.index(parts[1])
                    src += [a, b]; dst += [b, a]
    return np.asarray([src, dst], np.int32)

def pick_sensors(nodes, Y, include, frac):
    must = [m for m in include if m in nodes]
    k = max(len(must), int(round(frac * len(nodes))))
    stds = {n: float(np.std(Y[:, i])) for i, n in enumerate(nodes)}
    ranked = sorted(stds.items(), key=lambda kv: -kv[1])
    sensors = []
    for m in must:
        if m in nodes and m not in sensors: sensors.append(m)
    for n, _ in ranked:
        if len(sensors) >= k: break
        if n not in sensors: sensors.append(n)
    return sorted(sensors)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="processed.csv")
    ap.add_argument("--fraction", type=float, default=0.3)
    ap.add_argument("--include", default="s2-eth2,s2-eth1,s1-eth1,s1-eth2")
    ap.add_argument("--sensors_file", default="")
    ap.add_argument("--links_file", default="links.txt")
    ap.add_argument("--out", default="dataset_v2.npz")
    args = ap.parse_args()

    df = (pd.read_csv(args.processed, parse_dates=["timestamp"])
            .sort_values(["timestamp", "iface"]).reset_index(drop=True))
    label = "backlog_pkts" if "backlog_pkts" in df.columns else ("backlog_bytes" if "backlog_bytes" in df.columns else "qlen_pkts")
    rate_feats = [c for c in df.columns if c.endswith("_per_s")]
    if "throughput_Mbps" in df.columns: rate_feats.append("throughput_Mbps")

    nodes = sorted(df["iface"].dropna().unique().tolist()); idx = {n:i for i,n in enumerate(nodes)}
    ts = sorted(df["timestamp"].dropna().unique().tolist()); T, N = len(ts), len(nodes)

    Xr = np.zeros((T, N, len(rate_feats)), np.float32)
    Y  = np.zeros((T, N), np.float32)
    for n in nodes:
        ii = idx[n]
        sub = df[df["iface"]==n].drop_duplicates("timestamp").set_index("timestamp").reindex(ts)
        for k,f in enumerate(rate_feats): Xr[:, ii, k] = sub[f].astype(float).fillna(0.0).to_numpy()
        Y[:, ii] = sub[label].astype(float).fillna(0.0).to_numpy()

    include = [s.strip() for s in args.include.split(",") if s.strip()]
    if args.sensors_file and os.path.exists(args.sensors_file):
        sensors = [s.strip() for s in open(args.sensors_file) if s.strip()]
        sensors = [s for s in sensors if s in nodes]
    else:
        sensors = pick_sensors(nodes, Y, include, args.fraction)
    is_sensor = np.zeros((N,), np.uint8)
    for s in sensors:
        if s in idx: is_sensor[idx[s]] = 1

    # sensor backlog + lags (only for sensors; zeros elsewhere)
    Smask = is_sensor[None, :].astype(np.float32)          # [1,N]
    Sb    = (Y * Smask)                                    # [T,N]
    Sb_l1 = np.vstack([np.zeros((1,N),np.float32), Sb[:-1]])
    Sb_l2 = np.vstack([np.zeros((2,N),np.float32), Sb[:-2]])
    Sb_l3 = np.vstack([np.zeros((3,N),np.float32), Sb[:-3]])

    # static
    sw = np.array([parse_sw_port(n)[0] for n in nodes], np.float32)
    pt = np.array([parse_sw_port(n)[1] for n in nodes], np.float32)
    sw = (sw - sw.mean()) / (sw.std() + 1e-6)
    pt = (pt - pt.mean()) / (pt.std() + 1e-6)
    static = np.stack([sw, pt], 1).astype(np.float32)      # [N,2]
    static = np.repeat(static[None, :, :], T, axis=0)      # [T,N,2]

    # build feature tensor
    is_sensor_ch = np.repeat(Smask[:, :, None], T, axis=0).transpose(0,1,2)  # [T,N,1]
    X = np.concatenate([
        Xr,
        Sb[..., None],
        Sb_l1[..., None], Sb_l2[..., None], Sb_l3[..., None],
        is_sensor_ch,
        static
    ], axis=2)

    feat_names = rate_feats + [
        "sensor_backlog","sensor_backlog_lag1","sensor_backlog_lag2","sensor_backlog_lag3",
        "is_sensor","sw_id_z","port_id_z"
    ]

    # normalization (sensor_backlog* over sensors only)
    mu = np.zeros((len(feat_names),), np.float32)
    sd = np.ones_like(mu)
    for i,f in enumerate(feat_names):
        arr = X[:, :, i]
        if f == "is_sensor":
            mu[i] = 0.0; sd[i] = 1.0
        elif f.startswith("sensor_backlog"):
            mask = (is_sensor_ch[:, :, 0] == 1.0)
            vals = arr[mask]
            mu[i] = float(vals.mean()) if vals.size else 0.0
            sd[i] = float(vals.std() + 1e-6) if vals.size else 1.0
        else:
            mu[i] = float(arr.mean()); sd[i] = float(arr.std() + 1e-6)
    Xn = (X - mu[None, None, :]) / sd[None, None, :]

    edges = build_edges(nodes, args.links_file)

    # busy-aware split
    y_sum = Y.sum(1)
    busy  = np.where(y_sum > 0)[0]
    cut   = busy[int(0.8*len(busy))] if len(busy) > 1 else int(0.8 * T)
    train_idx = np.arange(0, cut + 1, dtype=np.int64)      # FIXED: dtype=
    test_idx  = np.arange(cut + 1, T, dtype=np.int64)      # FIXED: dtype=
    vstart    = int(max(0, len(train_idx) - 0.1 * len(train_idx)))
    val_idx   = train_idx[vstart:]
    train_idx = train_idx[:vstart]

    np.savez_compressed(args.out,
        nodes=np.array(nodes), edges=edges, feat_names=np.array(feat_names),
        label_name=np.array([label]), X=Xn.astype(np.float32), Y=Y.astype(np.float32),
        is_sensor=is_sensor, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        norm_mu=mu, norm_sd=sd, timestamps=np.array([pd.Timestamp(t).isoformat() for t in ts]),
        sensors=np.array(sensors)
    )
    print(json.dumps({
        "nodes": len(nodes), "edges": int(edges.shape[1]), "T": int(T),
        "features": len(feat_names), "sensors": sensors,
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    }, indent=2))

if __name__ == "__main__":
    main()
