#!/usr/bin/env python3
import argparse, numpy as np, torch
from models_gnn import GNNEncoder, TCNHead, GRUHead

def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)          # full-sensor dataset_v4.npz
    ap.add_argument("--ckpt", required=True)          # teacher ckpt (lead1)
    ap.add_argument("--out", default="teacher_preds.npz")
    args=ap.parse_args()

    Z=load_npz(args.data)
    X=torch.from_numpy(Z["X"]); Y=torch.from_numpy(Z["Y"])
    edges=torch.from_numpy(Z["edges"]).long()
    Fin, T, N = X.shape[2], X.shape[0], X.shape[1]

    ckpt=torch.load(args.ckpt, map_location="cpu"); meta=ckpt["meta"]
    K=int(meta.get("K",20))
    enc=GNNEncoder(Fin, hid=meta["hid"], layers=meta["layers"], kind=meta["encoder"], dropout=meta["dropout"])
    head = TCNHead(hid=meta["hid"], K=K) if meta.get("temporal","tcn")=="tcn" else GRUHead(hid=meta["hid"])
    enc.load_state_dict(ckpt["enc"]); head.load_state_dict(ckpt["head"])
    enc.eval(); head.eval()

    Yhat=np.zeros((T,N), np.float32)
    with torch.no_grad():
        for t in range(K, T-1):
            H=[enc(X[s], edges) for s in range(t-K+1, t+1)]
            pred=head(torch.stack(H,0))              # predict y_{t+1}
            Yhat[t+1]=pred.numpy()
    np.savez_compressed(args.out, Yhat=Yhat, timestamps=Z["timestamps"], nodes=Z["nodes"])
    print(f"OK: wrote {args.out} | shape {Yhat.shape} | K={K}")

if __name__ == "__main__":
    main()
