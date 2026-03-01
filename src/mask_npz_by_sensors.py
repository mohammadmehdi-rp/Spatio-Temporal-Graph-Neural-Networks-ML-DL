#!/usr/bin/env python3
import argparse, numpy as np, sys
from pathlib import Path

def tolerant_ports(d, N):
    if "ports" in d.files:       return [str(x) for x in d["ports"]]
    if "port_names" in d.files:  return [str(x) for x in d["port_names"]]
    if "ifaces" in d.files:      return [str(x) for x in d["ifaces"]]
    return [f"port{j}" for j in range(N)]

def parse_sensors_txt(path):
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)        # e.g., dataset_sparse_v4_fix.npz
    ap.add_argument("--sensors", required=True)     # text file, one name per line
    ap.add_argument("--out", required=True)         # output npz
    ap.add_argument("--fill", choices=["zero","nan"], default="zero",
                    help="how to fill non-sensor features (zero=SAFE DEFAULT)")
    ap.add_argument("--add_obs_flag", action="store_true",
                    help="append a 1-channel binary flag: 1 if sensor node else 0")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X = d["X"].copy()     # [T,N,F]
    Y = d["Y"]            # [T,N]
    T, N, F = X.shape
    ports = tolerant_ports(d, N)
    name2idx = {nm:i for i,nm in enumerate(ports)}

    wanted = parse_sensors_txt(args.sensors)
    idx = sorted({name2idx[w] if w in name2idx else (int(w[4:]) if w.startswith("port") and w[4:].isdigit() else -1)
                  for w in wanted if w})
    idx = [i for i in idx if 0 <= i < N]
    if not idx:
        print("No valid sensors matched; check names vs NPZ ports list.", file=sys.stderr); sys.exit(1)

    mask_nodes = np.ones(N, dtype=bool)
    mask_nodes[idx] = False  # False = sensor (keep real values); True = non-sensor (mask)

    if args.fill == "nan":
        X[:, mask_nodes, :] = np.nan
    else:  # zero-fill
        X[:, mask_nodes, :] = 0.0

    if args.add_obs_flag:
        flag = np.zeros((T, N, 1), dtype=X.dtype)
        flag[:, idx, 0] = 1.0
        X = np.concatenate([X, flag], axis=2)

    out = {k: d[k] for k in d.files}
    out["X"] = X
    np.savez_compressed(args.out, **out)
    print(f"OK: wrote {args.out} | sensors kept: {len(idx)}/{N} | F_out={X.shape[2]}")
    print("Sensors:", ", ".join(ports[i] for i in idx))
    if np.isnan(X).any():
        print("WARNING: output contains NaNs (you used --fill nan).", file=sys.stderr)

if __name__ == "__main__":
    main()
