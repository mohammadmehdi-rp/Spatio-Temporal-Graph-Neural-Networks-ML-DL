#!/usr/bin/env python3
import argparse, time, numpy as np, torch
from models_gnn import GNNEncoder, NowcastHead

def load_npz(p):
    D = np.load(p, allow_pickle=True)
    return {k: D[k] for k in D.files}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--encoder", choices=["sage","routenet"], required=True)
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--steps", type=int, default=200)     # timed steps
    ap.add_argument("--warmup", type=int, default=30)     # warmup steps
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    Z = load_npz(args.npz)
    X = torch.from_numpy(Z["X"]).float()    # [T,N,F]
    edges = torch.from_numpy(Z["edges"]).long()
    test_idx = torch.from_numpy(Z["test_idx"]).long()
    T, N, F = X.shape

    dev = torch.device(args.device)
    X = X.to(dev)
    edges = edges.to(dev)
    test_list = test_idx.tolist()

    enc = GNNEncoder(F, hid=args.hid, layers=args.layers, kind=args.encoder, dropout=args.dropout).to(dev)
    head = NowcastHead(hid=args.hid).to(dev)
    enc.eval(); head.eval()

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    # warmup
    with torch.no_grad():
        for i in range(min(args.warmup, len(test_list))):
            t = test_list[i]
            _ = head(enc(X[t], edges))
        sync()

    # timed
    n = min(args.steps, len(test_list))
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(n):
            t = test_list[i]
            _ = head(enc(X[t], edges))
        sync()
    t1 = time.perf_counter()

    ms_per_step = (t1 - t0) * 1000.0 / n
    print(f"encoder={args.encoder} device={args.device} steps={n}  ms/step={ms_per_step:.3f}")

if __name__ == "__main__":
    main()
