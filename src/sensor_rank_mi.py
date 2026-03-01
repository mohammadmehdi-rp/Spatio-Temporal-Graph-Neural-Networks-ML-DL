#!/usr/bin/env python3
import argparse, numpy as np, json, sys
from pathlib import Path

try:
    from sklearn.feature_selection import mutual_info_regression
except Exception as e:
    print("scikit-learn is required (pip install scikit-learn)", file=sys.stderr); raise

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    X = data["X"]              # [T, N, F]
    Y = data["Y"]              # [T, N]
    N = X.shape[1]
    if "ports" in data.files:
        ports = [str(p) for p in list(data["ports"])]
    elif "port_names" in data.files:
        ports = [str(p) for p in list(data["port_names"])]
    elif "ifaces" in data.files:
        ports = [str(p) for p in list(data["ifaces"])]
    else:
        ports = [f"port{j}" for j in range(N)]
    train_idx = data["train_idx"].astype(np.int64)
    val_idx   = data["val_idx"].astype(np.int64)
    return X, Y, ports, train_idx, val_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_sparse_v4_fix.npz")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out_txt", default="sensors_mi_k8.txt")
    ap.add_argument("--out_csv", default="sensor_mi_scores.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, Y, ports, train_idx, val_idx = load_npz(args.data)
    y_glob = Y.mean(axis=1)   # [T] global backlog per frame

    scores = []
    for j in range(X.shape[1]):
        Xj_tr = X[train_idx, j, :]    # [T_tr, F]
        y_tr  = y_glob[train_idx]     # [T_tr]
        mi = mutual_info_regression(Xj_tr, y_tr, discrete_features=False, random_state=args.seed)
        scores.append((j, ports[j], float(np.sum(mi))))

    scores.sort(key=lambda t: t[2], reverse=True)
    with open(args.out_csv, "w") as f:
        f.write("rank,idx,iface,mi_sum\n")
        for r,(j,name,sc) in enumerate(scores, start=1):
            f.write(f"{r},{j},{name},{sc:.6f}\n")
    top = [name for (_,name,_) in scores[:args.k]]
    Path(args.out_txt).write_text("\n".join(top))
    print("Top-K sensors:", ", ".join(top))
    print(f"wrote {args.out_txt} and {args.out_csv}")

if __name__ == "__main__":
    main()
