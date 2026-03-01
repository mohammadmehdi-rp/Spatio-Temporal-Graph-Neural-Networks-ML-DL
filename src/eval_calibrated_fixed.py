#!/usr/bin/env python3
import argparse, numpy as np, torch, math
from models_gnn import GNNEncoder, TCNHead, GRUHead

def rmse(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(np.sqrt(np.mean((a-b)**2))) if a.size else float("nan")

def ensemble_preds(npz, ckpts, split_idx, task="lead1"):
    X=torch.from_numpy(npz["X"]); Y=torch.from_numpy(npz["Y"]); E=torch.from_numpy(npz["edges"]).long()
    Fin,N=X.shape[2], X.shape[1]
    # intersect valid t across ckpts (due to different K)
    valid_sets=[]
    metas=[]
    for p in ckpts:
        ck=torch.load(p,map_location="cpu"); m=ck["meta"]; metas.append(m)
        K=int(m.get("K",20))
        valid_sets.append([t for t in range(K, X.shape[0]-1) if t in set(split_idx.tolist())])
    base=set(valid_sets[0])
    for v in valid_sets[1:]: base &= set(v)
    valid=sorted(list(base))
    allp=[]
    for p,m in zip(ckpts,metas):
        K=int(m.get("K",20))
        enc=GNNEncoder(Fin, hid=m["hid"], layers=m["layers"], kind=m["encoder"], dropout=m["dropout"])
        head=TCNHead(hid=m["hid"], K=K) if m.get("temporal","tcn")=="tcn" else GRUHead(hid=m["hid"])
        ck=torch.load(p,map_location="cpu")
        enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
        enc.eval(); head.eval()
        yh=[]
        with torch.no_grad():
            for t in valid:
                H=[enc(X[s],E) for s in range(t-K+1, t+1)]
                yh.append(head(torch.stack(H,0)).numpy())
        allp.append(np.stack(yh))
    P=np.mean(np.stack(allp,0),0)                   # [Tvalid,N]
    Ytrue=np.stack([npz["Y"][t+1] for t in valid])  # [Tvalid,N]
    return P, Ytrue

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    args=ap.parse_args()

    Z=np.load(args.data, allow_pickle=True)
    Pte,Yte = ensemble_preds(Z, args.ckpts, Z["test_idx"], task="lead1")

    # Apply SOFT+SCALE: y' = alpha * max(0, y - tau)
    Pte_cal = args.alpha * np.maximum(0.0, Pte - args.tau)

    # Report
    all_rmse  = rmse(Pte_cal, Yte)
    busy_rmse = rmse(Pte_cal[Yte>0], Yte[Yte>0])
    print(f"Calibrated (tau={args.tau:.3f}, alpha={args.alpha:.3f}) → all-frames {all_rmse:.3f} | busy-only {busy_rmse:.3f}")

if __name__=="__main__":
    main()

