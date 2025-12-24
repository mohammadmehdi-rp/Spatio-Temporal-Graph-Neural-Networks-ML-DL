#!/usr/bin/env python3
import argparse, numpy as np, json, sys
from pathlib import Path

try:
    from sklearn.linear_model import Ridge
except Exception as e:
    print("scikit-learn is required (pip install scikit-learn)", file=sys.stderr); raise

def load_npz(path):
    data = np.load(path, allow_pickle=True)
    X = data["X"]              # [T, N, F]
    Y = data["Y"]              # [T, N]
    N = X.shape[1]
    # tolerant port names
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
    test_idx  = data["test_idx"].astype(np.int64) if "test_idx" in data.files else None
    assert Y.shape[1] == N, f"Y has {Y.shape[1]} ports but X has {N}"
    return X, Y, ports, train_idx, val_idx, test_idx

def zscore_fit(Xtr):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return mu, sd

def zscore_apply(X, mu, sd): return (X - mu) / sd

def flatten_features(X, sensors):
    Xs = X[:, sensors, :]              # [T, K, F]
    T, K, F = Xs.shape
    return Xs.reshape(T, K*F)          # [T, K*F]

def rmse(y_true, y_pred):
    e = (y_true - y_pred).reshape(-1)
    return float(np.sqrt(np.mean(e**2)))

def eval_subset_ridge(X, Y, sensors, train_idx, val_idx, l2, standardize):
    Xflat = flatten_features(X, sensors)     # [T, K*F]
    y = Y                                    # [T, N]
    Xtr, Xva = Xflat[train_idx], Xflat[val_idx]
    ytr, yva = y[train_idx], y[val_idx]
    if standardize:
        mu, sd = zscore_fit(Xtr)
        Xtr = zscore_apply(Xtr, mu, sd)
        Xva = zscore_apply(Xva, mu, sd)
    model = Ridge(alpha=l2, fit_intercept=True, random_state=42)
    model.fit(Xtr, ytr)                      # multi-output
    yhat = model.predict(Xva)
    mic = rmse(yva, yhat)
    per_port = [rmse(yva[:, j], yhat[:, j]) for j in range(y.shape[1])]
    mac = float(np.mean(per_port))
    return mic, mac

def greedy_select(X, Y, train_idx, val_idx, K, l2, standardize, ports):
    N = X.shape[1]
    remaining = set(range(N))
    chosen = []
    curve = []
    for step in range(1, K+1):
        best = (None, float("inf"), float("inf"))
        for j in list(remaining):
            cand = chosen + [j]
            mic, mac = eval_subset_ridge(X, Y, cand, train_idx, val_idx, l2, standardize)
            if mic < best[1]:
                best = (j, mic, mac)
        j, mic, mac = best
        chosen.append(j); remaining.remove(j)
        curve.append({"k": step, "rmse_micro": mic, "rmse_macro": mac,
                      "sensor_added_idx": int(j), "sensor_added_name": ports[j]})
        print(f"[greedy] k={step:2d} add={ports[j]:>10s} → micro {mic:.3f} | macro {mac:.3f}")
    return chosen, curve

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_sparse_v4_fix.npz")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--l2", type=float, default=1000.0)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--out_txt", default=None)
    ap.add_argument("--out_csv", default="sensor_greedy_curve.csv")
    ap.add_argument("--out_json", default="sensor_greedy_curve.json")
    args = ap.parse_args()

    X, Y, ports, train_idx, val_idx, _ = load_npz(args.data)
    chosen_idx, curve = greedy_select(X, Y, train_idx, val_idx, args.k, args.l2, args.standardize, ports)
    chosen_names = [ports[i] for i in chosen_idx]

    print("\nSelected sensors:", ", ".join(chosen_names))
    if args.out_txt:
        Path(args.out_txt).write_text("\n".join(chosen_names))
        print(f"wrote {args.out_txt}")
    with open(args.out_csv, "w") as f:
        f.write("k,rmse_micro,rmse_macro,added_idx,added_name\n")
        for r in curve:
            f.write(f"{r['k']},{r['rmse_micro']:.6f},{r['rmse_macro']:.6f},{r['sensor_added_idx']},{r['sensor_added_name']}\n")
    Path(args.out_json).write_text(json.dumps({"selected": chosen_names, "curve": curve}, indent=2))
    print(f"wrote {args.out_csv} and {args.out_json}")

if __name__ == "__main__":
    main()
