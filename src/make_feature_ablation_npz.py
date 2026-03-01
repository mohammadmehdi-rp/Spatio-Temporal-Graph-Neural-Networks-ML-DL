#!/usr/bin/env python3
import argparse, numpy as np

def decode_names(arr):
    return [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--zero", default="", help="Comma-separated feature names to zero (exact match).")
    ap.add_argument("--zero_contains", default="", help="Comma-separated substrings; any feature containing one will be zeroed.")
    ap.add_argument("--print_names", action="store_true")
    args = ap.parse_args()

    Z = np.load(args.in_npz, allow_pickle=True)
    out = {k: Z[k] for k in Z.files}

    feat_names = out.get("feat_names", None)
    if feat_names is None:
        raise RuntimeError("NPZ has no feat_names; cannot do feature ablations safely.")

    names = decode_names(feat_names)
    if args.print_names:
        print("Feature names:")
        for i, n in enumerate(names):
            print(f"{i:02d}: {n}")
        return

    X = out["X"].copy()  # [T,N,F]
    F = X.shape[2]

    zero_exact = [s.strip() for s in args.zero.split(",") if s.strip()]
    zero_contains = [s.strip() for s in args.zero_contains.split(",") if s.strip()]

    idx_to_zero = set()

    for f in zero_exact:
        if f in names:
            idx_to_zero.add(names.index(f))
        else:
            print(f"[WARN] exact feature not found: {f}")

    for sub in zero_contains:
        for i, n in enumerate(names):
            if sub in n:
                idx_to_zero.add(i)

    idx_to_zero = sorted(list(idx_to_zero))
    if not idx_to_zero:
        print("[WARN] No features selected to zero. Writing identical NPZ.")
    else:
        print("Zeroing channels:")
        for i in idx_to_zero:
            print(f"  {i:02d}: {names[i]}")
        X[:, :, idx_to_zero] = 0.0

    out["X"] = X
    np.savez(args.out_npz, **out)
    print(f"[OK] wrote {args.out_npz}")

if __name__ == "__main__":
    main()
