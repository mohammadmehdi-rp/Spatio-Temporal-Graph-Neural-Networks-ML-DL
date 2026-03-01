#!/usr/bin/env python3
import argparse, numpy as np, torch
from models_gnn import GNNEncoder, TCNHead, GRUHead

def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}
def rmse(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.sqrt(np.mean((a-b)**2))) if a.size else float("nan")

def preds_ens(Z, ckpts, split_idx):
    X=torch.from_numpy(Z["X"]); Y=torch.from_numpy(Z["Y"]); E=torch.from_numpy(Z["edges"]).long()
    Fin,N=X.shape[2], X.shape[1]
    # intersect valid positions across ckpts (different K)
    valids=[]; metas=[]
    for p in ckpts:
        ck=torch.load(p,map_location="cpu"); m=ck["meta"]; metas.append(m)
        K=int(m.get("K",20))
        valids.append([t for t in range(K, X.shape[0]-1) if t in set(split_idx.tolist())])
    base=set(valids[0])
    for v in valids[1:]: base &= set(v)
    valid=sorted(base)
    # aligned predictions
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
    P=np.mean(np.stack(allp,0),0)                   # [Tvalid,N]
    Ytrue=np.stack([Z["Y"][t+1] for t in valid])    # [Tvalid,N]
    return P, Ytrue

def best_alpha_busy(P, Y):
    m = Y>0
    num = (P[m]*Y[m]).sum()
    den = (P[m]*P[m]).sum()
    a = 0.0 if den==0 else float(num/den)
    return max(0.0, min(2.0, a))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--qgrid", nargs="*", default=["0.80","0.90","0.95","0.98"])
    ap.add_argument("--target", choices=["all","busy"], default="all")
    args=ap.parse_args()

    Z=load_npz(args.data)
    Pval,Yval = preds_ens(Z, args.ckpts, Z["val_idx"])
    Pte ,Yte  = preds_ens(Z, args.ckpts, Z["test_idx"])

    raw_all = rmse(Pte, Yte)
    raw_busy = rmse(Pte[Yte>0], Yte[Yte>0])
    print(f"RAW ensemble → all-frames {raw_all:.3f} | busy-only {raw_busy:.3f}")

    # τ from val IDLE predictions only (Y==0)
    idle = (Yval==0)
    cand = [float(s) for s in args.qgrid if np.isfinite(float(s))]
    best = (1e18, None, None)  # score, tau, alpha
    for q in cand:
        if idle.any():
            tau = float(np.quantile(Pval[idle], q))
        else:
            tau = 0.0
        Pth = np.maximum(0.0, Pval - tau)       # soft-threshold on val
        a   = best_alpha_busy(Pth, Yval)        # fit α on BUSY frames (val)
        score = rmse(a*Pth, Yval) if args.target=="all" else rmse((a*Pth)[Yval>0], Yval[Yval>0])
        if score < best[0]:
            best = (score, tau, a)

    _, tau_star, a_star = best
    Pte_th = np.maximum(0.0, Pte - tau_star)
    Pte_cal = a_star * Pte_th

    print(f"SOFT+SCALE (tau={tau_star:.3f} from idle q, alpha={a_star:.3f} from busy) → "
          f"all-frames {rmse(Pte_cal,Yte):.3f} | busy-only {rmse(Pte_cal[Yte>0],Yte[Yte>0]):.3f}")

if __name__=="__main__":
    main()
