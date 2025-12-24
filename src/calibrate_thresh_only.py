#!/usr/bin/env python3
import argparse, numpy as np, torch
from models_gnn import GNNEncoder, TCNHead, GRUHead

def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}
def rmse(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.sqrt(np.mean((a-b)**2))) if a.size else float("nan")

def preds_ensemble(Z, ckpts, split_idx):
    X=torch.from_numpy(Z["X"]); Y=torch.from_numpy(Z["Y"]); E=torch.from_numpy(Z["edges"]).long()
    Fin,N=X.shape[2], X.shape[1]
    # intersect valid positions across ckpts (because K can differ)
    valids=[]
    metas=[]
    for p in ckpts:
        ck=torch.load(p,map_location="cpu"); m=ck["meta"]; metas.append(m)
        K=int(m.get("K",20))
        valids.append([t for t in range(K, X.shape[0]-1) if t in set(split_idx.tolist())])
    base=set(valids[0])
    for v in valids[1:]: base &= set(v)
    valid=sorted(base)
    # recompute aligned preds for each ckpt
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
                H=[enc(X[s],E) for s in range(t-K+1,t+1)]
                yh.append(head(torch.stack(H,0)).numpy())
        allp.append(np.stack(yh))
    P=np.mean(np.stack(allp,0),0)
    Ytrue=np.stack([Y[t+1].numpy() for t in valid])
    return P, Ytrue

def apply_thresh(P, tau):
    return np.where(P>=tau, P, 0.0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--grid_tau", nargs="*", default=["0","1","2","4","8","12","16"])
    ap.add_argument("--target", choices=["all","busy"], default="all")
    args=ap.parse_args()

    Z=load_npz(args.data)
    Pval,Yval = preds_ensemble(Z, args.ckpts, Z["val_idx"])
    Pte ,Yte  = preds_ensemble(Z, args.ckpts, Z["test_idx"])

    raw_all = rmse(Pte, Yte); raw_busy = rmse(Pte[Yte>0], Yte[Yte>0])
    print(f"RAW ensemble → all-frames {raw_all:.3f} | busy-only {raw_busy:.3f}")

    taus=[float(t) for t in args.grid_tau]
    best=(1e18,0.0)
    for tau in taus:
        Pv = apply_thresh(Pval, tau)
        score = rmse(Pv, Yval) if args.target=="all" else rmse(Pv[Yval>0], Yval[Yval>0])
        if score<best[0]: best=(score,tau)
    tau_star=best[1]

    Pt = apply_thresh(Pte, tau_star)
    all_rmse = rmse(Pt, Yte)
    busy_rmse = rmse(Pt[Yte>0], Yte[Yte>0])
    print(f"THRESH (tau={tau_star:.1f}) → all-frames {all_rmse:.3f} | busy-only {busy_rmse:.3f}")

if __name__=="__main__":
    main()
