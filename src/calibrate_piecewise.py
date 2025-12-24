#!/usr/bin/env python3
import argparse, numpy as np, torch
from models_gnn import GNNEncoder, TCNHead, GRUHead

def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}
def rmse(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.sqrt(np.mean((a-b)**2))) if a.size else float("nan")

def preds_ens(Z, ckpts, split_idx):
    X=torch.from_numpy(Z["X"]); Y=torch.from_numpy(Z["Y"]); E=torch.from_numpy(Z["edges"]).long()
    Fin,N=X.shape[2], X.shape[1]
    # intersect valid indices
    valids=[]
    metas=[]
    for p in ckpts:
        ck=torch.load(p,map_location="cpu"); m=ck["meta"]; metas.append(m)
        K=int(m.get("K",20))
        valids.append([t for t in range(K, X.shape[0]-1) if t in set(split_idx.tolist())])
    base=set(valids[0])
    for v in valids[1:]: base&=set(v)
    valid=sorted(base)
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

def fit_piecewise(P, Y, tau, lam=1e-6):
    # Solve min || a0*(P<tau)*P - Y ||^2 + || a1*(P>=tau)*P - Y ||^2  (global a0,a1)
    M0=(P<tau).astype(float); M1=(P>=tau).astype(float)
    v0=(M0*P).reshape(-1); v1=(M1*P).reshape(-1); y=Y.reshape(-1)
    A=np.stack([v0,v1],1)  # [T*N, 2]
    R=np.diag([lam,lam])
    theta=np.linalg.pinv(A.T@A + R)@(A.T@y)
    a0,a1=float(theta[0]), float(theta[1])
    return a0,a1

def apply_piecewise(P, a0, a1, tau):
    M0=(P<tau).astype(float); M1=(P>=tau).astype(float)
    return a0*M0*P + a1*M1*P

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--taus", nargs="*", default=["0","1","2","4","8","12","16","24"])
    ap.add_argument("--target", choices=["all","busy"], default="all")
    ap.add_argument("--lam", type=float, default=1e-6)
    args=ap.parse_args()

    Z=load_npz(args.data)
    Pval,Yval = preds_ens(Z, args.ckpts, Z["val_idx"])
    Pte ,Yte  = preds_ens(Z, args.ckpts, Z["test_idx"])

    raw_all=rmse(Pte,Yte); raw_busy=rmse(Pte[Yte>0],Yte[Yte>0])
    print(f"RAW ensemble → all-frames {raw_all:.3f} | busy-only {raw_busy:.3f}")

    best=(1e18,0.0,0.0,0.0)
    for tau in [float(t) for t in args.taus]:
        a0,a1 = fit_piecewise(Pval, Yval, tau, lam=args.lam)
        Pv = apply_piecewise(Pval, a0, a1, tau)
        score = rmse(Pv, Yval) if args.target=="all" else rmse(Pv[Yval>0], Yval[Yval>0])
        if score<best[0]: best=(score, a0, a1, tau)
    _,a0,a1,tau = best
    Pt = apply_piecewise(Pte, a0, a1, tau)
    all_rmse = rmse(Pt, Yte); busy_rmse = rmse(Pt[Yte>0], Yte[Yte>0])
    print(f"PIECEWISE (a0={a0:.3f}, a1={a1:.3f}, tau={tau:.1f}) → all-frames {all_rmse:.3f} | busy-only {busy_rmse:.3f}")

if __name__=="__main__":
    main()
