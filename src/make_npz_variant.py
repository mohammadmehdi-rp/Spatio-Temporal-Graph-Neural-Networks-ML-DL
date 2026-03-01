#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path

def _decode_names(arr):
    out = []
    for n in arr:
        if isinstance(n, bytes):
            out.append(n.decode())
        else:
            out.append(str(n))
    return out

def main():
    ap = argparse.ArgumentParser(description="Create NPZ dataset variants by dropping or zeroing feature columns.")
    ap.add_argument("--in_npz", required=True, help="Input NPZ path")
    ap.add_argument("--out_npz", required=True, help="Output NPZ path")
    ap.add_argument("--drop_name", action="append", default=[], help="Exact feature name to drop (repeatable)")
    ap.add_argument("--drop_suffix", action="append", default=[], help="Drop any feature ending with this suffix (repeatable)")
    ap.add_argument("--drop_contains", action="append", default=[], help="Drop any feature containing this substring (repeatable)")
    ap.add_argument("--mode", choices=["drop", "zero"], default="drop",
                    help="drop=remove feature columns; zero=keep dims but set dropped columns to 0.")
    args = ap.parse_args()

    in_npz = Path(args.in_npz)
    out_npz = Path(args.out_npz)

    Z = np.load(in_npz, allow_pickle=True)
    D = {k: Z[k] for k in Z.files}

    if "X" not in D:
        raise RuntimeError("NPZ missing required key: X")
    if "feat_names" not in D:
        raise RuntimeError("NPZ missing required key: feat_names (needed to drop features safely)")

    X = D["X"]
    feat = _decode_names(D["feat_names"])
    F = X.shape[-1]
    drop_idx = set()

    # exact names
    for name in args.drop_name:
        if name in feat:
            drop_idx.add(feat.index(name))

    # suffix-based
    for suf in args.drop_suffix:
        for i, n in enumerate(feat):
            if n.endswith(suf):
                drop_idx.add(i)

    # contains-based
    for sub in args.drop_contains:
        for i, n in enumerate(feat):
            if sub in n:
                drop_idx.add(i)

    drop_idx = sorted(drop_idx)
    keep_idx = [i for i in range(F) if i not in drop_idx]

    print(f"[make_npz_variant] in : {in_npz}")
    print(f"[make_npz_variant] out: {out_npz}")
    print(f"[make_npz_variant] X shape: {tuple(X.shape)}")
    print(f"[make_npz_variant] drop {len(drop_idx)} / {F} features")

    if drop_idx:
        print("[make_npz_variant] dropped:")
        for i in drop_idx:
            print(f"  - {i:03d} {feat[i]}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "zero":
        X2 = X.copy()
        X2[..., drop_idx] = 0.0
        D["X"] = X2
        # feat_names unchanged
    else:
        D["X"] = X[..., keep_idx]
        D["feat_names"] = np.array([feat[i] for i in keep_idx], dtype=object)

    np.savez(out_npz, **D)
    print("[make_npz_variant] OK")

if __name__ == "__main__":
    main()
