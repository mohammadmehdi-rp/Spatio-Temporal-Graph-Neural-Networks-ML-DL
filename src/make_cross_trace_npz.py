#!/usr/bin/env python3
import argparse
import numpy as np

def load_npz(p):
    D = np.load(p, allow_pickle=True)
    return {k: D[k] for k in D.files}

def save_npz(p, D):
    np.savez(p, **D)
    print(f"[OK] wrote {p}")

def renormalize_X(X_norm, mu, sd, train_idx):
    # reconstruct raw, then re-normalize using only training slice (no leakage)
    mu = mu.reshape(1,1,-1).astype(np.float32)
    sd = sd.reshape(1,1,-1).astype(np.float32)
    raw = X_norm.astype(np.float32) * sd + mu
    flat = raw[train_idx].reshape(-1, raw.shape[-1])
    mu2 = flat.mean(axis=0)
    sd2 = flat.std(axis=0)
    sd2 = np.where(sd2 < 1e-6, 1.0, sd2)
    X2 = (raw - mu2.reshape(1,1,-1)) / sd2.reshape(1,1,-1)
    return X2.astype(np.float32), mu2.astype(np.float32), sd2.astype(np.float32)

def build(src, split_frac=0.5, train_frac=0.85, direction="A2B", renorm=True):
    Z = load_npz(src)
    X = Z["X"]
    T = X.shape[0]
    split = int(T * split_frac)

    A = np.arange(0, split)
    B = np.arange(split, T)

    if direction == "A2B":
        src_idx, tgt_idx = A, B
        src_tag, tgt_tag = "A", "B"
    else:
        src_idx, tgt_idx = B, A
        src_tag, tgt_tag = "B", "A"

    n_src = len(src_idx)
    n_train = int(n_src * train_frac)
    train_idx = src_idx[:n_train]
    val_idx   = src_idx[n_train:]
    test_idx  = tgt_idx

    out = dict(Z)
    if renorm and ("norm_mu" in Z and "norm_sd" in Z):
        X2, mu2, sd2 = renormalize_X(Z["X"], Z["norm_mu"], Z["norm_sd"], train_idx)
        out["X"] = X2
        out["norm_mu"] = mu2
        out["norm_sd"] = sd2

    out["train_idx"] = train_idx.astype(np.int64)
    out["val_idx"]   = val_idx.astype(np.int64)
    out["test_idx"]  = test_idx.astype(np.int64)

    out["cross_source"] = np.array([src_tag], dtype=object)
    out["cross_target"] = np.array([tgt_tag], dtype=object)
    out["cross_split_idx"] = np.array([split], dtype=np.int64)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--split_frac", type=float, default=0.5)
    ap.add_argument("--train_frac", type=float, default=0.85)
    ap.add_argument("--no_renorm", action="store_true")
    ap.add_argument("--out_A2B", default="dataset_A2B.npz")
    ap.add_argument("--out_B2A", default="dataset_B2A.npz")
    args = ap.parse_args()

    A2B = build(args.in_npz, args.split_frac, args.train_frac, "A2B", renorm=(not args.no_renorm))
    B2A = build(args.in_npz, args.split_frac, args.train_frac, "B2A", renorm=(not args.no_renorm))

    save_npz(args.out_A2B, A2B)
    save_npz(args.out_B2A, B2A)

if __name__ == "__main__":
    main()

