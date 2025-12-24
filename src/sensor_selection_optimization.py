#!/usr/bin/env python3
"""
sensor_selection_optimization.py

Optimize sensor placement given dataset_full_v4.npz.

Implements:
  - MI-like ranking (approx via per-feature correlation sum)
  - Greedy selection using a ridge baseline
  - ΔRMSE table vs the fixed 8-sensor set from dataset_sparse_v4_fix.npz

Usage:
  python sensor_selection_optimization.py \
      --npz_full dataset_full_v4.npz \
      --npz_sparse_fix dataset_sparse_v4_fix.npz \
      --budgets 4,6,8 \
      --l2 1e-2
"""

import argparse, numpy as np

def load_npz(path):
    Z = np.load(path, allow_pickle=True)
    return Z

def build_dataset(Z, sensor_idx, split_idx, use_mean=False):
    """
    Build (X_design, y) from full NPZ for given sensors and a time index split.

    X: (T, N, F), Y: (T, N)
    For each time t in split_idx and node j:
      x_(t,j) = concat features from sensors in sensor_idx
      y_(t,j) = Y[t, j]
    """
    X = Z["X"]      # (T, N, F)
    Y = Z["Y"]      # (T, N)
    T, N, F = X.shape
    split_idx = np.asarray(split_idx, dtype=int)

    # number of samples
    n_samples = len(split_idx) * N
    d = len(sensor_idx) * F

    Xd = np.empty((n_samples, d), dtype=np.float32)
    yd = np.empty(n_samples, dtype=np.float32)

    pos = 0
    for t in split_idx:
        # features for all nodes at time t: X[t] is (N, F)
        Xt = X[t]  # (N, F)
        Yt = Y[t]  # (N,)
        # build block of shape (N, len(sensor_idx)*F)
        sensor_feats = Xt[sensor_idx, :]  # (k, F)
        # same sensor_feats used for all target nodes
        block = np.tile(sensor_feats.flatten(), (N, 1))
        Xd[pos:pos+N, :] = block
        yd[pos:pos+N] = Yt
        pos += N

    return Xd, yd

def ridge_fit_predict(Xtr, ytr, Xte, l2):
    """Closed-form ridge regression with bias term."""
    # standardize features
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd

    # append bias
    Xtr_n = np.hstack([Xtr_n, np.ones((Xtr_n.shape[0], 1), dtype=Xtr_n.dtype)])
    Xte_n = np.hstack([Xte_n, np.ones((Xte_n.shape[0], 1), dtype=Xte_n.dtype)])

    d = Xtr_n.shape[1]
    I = np.eye(d, dtype=Xtr_n.dtype)
    w = np.linalg.inv(Xtr_n.T @ Xtr_n + l2 * I) @ (Xtr_n.T @ ytr)
    yhat = Xte_n @ w
    return yhat

def rmse(y_true, y_hat):
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)))

def eval_subset(Z, sensor_idx, l2=1e-2):
    """Evaluate a sensor subset: train on train_idx, validate on val_idx."""
    train_idx = Z["train_idx"]
    val_idx   = Z["val_idx"]

    Xtr, ytr = build_dataset(Z, sensor_idx, train_idx)
    Xva, yva = build_dataset(Z, sensor_idx, val_idx)

    yhat = ridge_fit_predict(Xtr, ytr, Xva, l2)
    return rmse(yva, yhat)

def corr_score_per_node(Z):
    """
    Approx MI-like score per node:
    For each node i, compute sum of |corr(feature_f(i), label(i))| over features.
    """
    X = Z["X"]  # (T, N, F)
    Y = Z["Y"]  # (T, N)
    T, N, F = X.shape

    scores = np.zeros(N, dtype=np.float64)
    for i in range(N):
        Xi = X[:, i, :]  # (T, F)
        yi = Y[:, i]     # (T,)
        # center
        Xi_c = Xi - Xi.mean(axis=0, keepdims=True)
        yi_c = yi - yi.mean()
        denom_y = np.sqrt((yi_c ** 2).sum()) + 1e-9

        for f in range(F):
            xf = Xi_c[:, f]
            denom_x = np.sqrt((xf ** 2).sum()) + 1e-9
            num = float((xf * yi_c).sum())
            corr = num / (denom_x * denom_y)
            scores[i] += abs(corr)

    return scores

def mi_ranking(Z):
    """Return node indices sorted by decreasing 'MI-like' score."""
    scores = corr_score_per_node(Z)
    return np.argsort(-scores), scores

def greedy_selection(Z, base_idx, k, l2=1e-2):
    """
    Greedy forward selection:
      start from base_idx (possibly empty),
      at each step add the node that yields lowest val RMSE.
    """
    N = Z["nodes"].shape[0]
    all_nodes = list(range(N))
    selected = list(base_idx)
    rmse_history = []

    while len(selected) < k:
        best_rmse = None
        best_node = None
        for cand in all_nodes:
            if cand in selected:
                continue
            trial = selected + [cand]
            val_rmse = eval_subset(Z, trial, l2=l2)
            if best_rmse is None or val_rmse < best_rmse:
                best_rmse = val_rmse
                best_node = cand
        selected.append(best_node)
        rmse_history.append(best_rmse)
        print(f"[greedy] |S|={len(selected)} best_node={Z['nodes'][best_node]} val_RMSE={best_rmse:.4f}")

    return selected, rmse_history

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_full", default="dataset_full_v4.npz")
    ap.add_argument("--npz_sparse_fix", default="dataset_sparse_v4_fix.npz")
    ap.add_argument("--budgets", default="4,6,8", help="comma-separated sensor counts")
    ap.add_argument("--l2", type=float, default=1e-2)
    args = ap.parse_args()

    Z_full = load_npz(args.npz_full)
    Z_fix  = load_npz(args.npz_sparse_fix)

    nodes = Z_full["nodes"]
    budgets = [int(x) for x in args.budgets.split(",") if x]

    # fixed baseline indices
    fixed_mask = Z_fix["is_sensor"].astype(bool)
    fixed_idx  = np.where(fixed_mask)[0]
    fixed_rmse = eval_subset(Z_full, fixed_idx, l2=args.l2)
    print("Fixed 8-sensor set:", nodes[fixed_idx])
    print(f"Fixed baseline val RMSE: {fixed_rmse:.4f}")

    # MI-like ranking (global)
    mi_order, mi_scores = mi_ranking(Z_full)
    print("\nTop 10 sensors by MI-like score:")
    for i in range(min(10, len(mi_order))):
        idx = mi_order[i]
        print(f"  {i+1}. {nodes[idx]} score={mi_scores[idx]:.4f}")

    print("\n=== Results table ===")
    print("method,k,val_RMSE,delta_vs_fixed,sensors")

    # evaluate MI top-k for each budget
    for k in budgets:
        mi_k = mi_order[:k]
        rmse_k = eval_subset(Z_full, mi_k, l2=args.l2)
        delta = rmse_k - fixed_rmse
        sens_names = ";".join(nodes[mi_k])
        print(f"mi,{k},{rmse_k:.4f},{delta:+.4f},{sens_names}")

    # greedy starting from empty set (or you can start from bottleneck + neighbors)
    for k in budgets:
        print(f"\nRunning greedy selection for k={k}")
        greedy_idx, hist = greedy_selection(Z_full, base_idx=[], k=k, l2=args.l2)
        rmse_k = hist[-1]
        delta = rmse_k - fixed_rmse
        sens_names = ";".join(nodes[greedy_idx])
        print(f"greedy,{k},{rmse_k:.4f},{delta:+.4f},{sens_names}")

if __name__ == "__main__":
    main()
