#!/usr/bin/env python3
import argparse, numpy as np, torch, math
from models_gnn import GNNEncoder, TCNHead, GRUHead

def load_npz(p):
    D = np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}

def rmse(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.sqrt(np.mean((a-b)**2))) if a.size else float("nan")
def macro_all(yh,yt):
    vals=[rmse(yh[:,j], yt[:,j]) for j in range(yt.shape[1])]
    vals=[v for v in vals if math.isfinite(v)]; return float(np.mean(vals)) if vals else float("nan")
def macro_busy(yh,yt):
    vals=[]
    for j in range(yt.shape[1]):
        m=yt[:,j]>0
        if np.any(m): vals.append(rmse(yh[m,j], yt[m,j]))
    vals=[v for v in vals if math.isfinite(v)]; return float(np.mean(vals)) if vals else float("nan")

def get_preds(data, ckpts, split_idx, task="lead1"):
    X=torch.from_numpy(data["X"]); Y=torch.from_numpy(data["Y"]); edges=torch.from_numpy(data["edges"]).long()
    Fin,N=X.shape[2], X.shape[1]
    preds=[]; valid_sets=[]
    for path in ckpts:
        ck=torch.load(path, map_location="cpu"); m=ck["meta"]
        if task=="lead1":
            K=int(m.get("K",20))
            enc=GNNEncoder(Fin, hid=m["hid"], layers=m["layers"], kind=m["encoder"], dropout=m["dropout"])
            head=TCNHead(hid=m["hid"], K=K) if m.get("temporal","tcn")=="tcn" else GRUHead(hid=m["hid"])
            enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"]); enc.eval(); head.eval()
            valid=[t for t in range(K, X.shape[0]-1) if t in set(split_idx.tolist())]
            yh=[]
            with torch.no_grad():
                for t in valid:
                    H=[enc(X[s], edges) for s in range(t-K+1, t+1)]
                    yh.append(head(torch.stack(H,0)).numpy())
            preds.append(np.stack(yh)); valid_sets.append(valid)
        else:
            raise ValueError("task must be 'lead1'")
    # intersect valid positions across ckpts
    base=set(valid_sets[0])
    for v in valid_sets[1:]: base&=set(v)
    valid=sorted(list(base))
    # recompute aligned ensemble
    allp=[]
    for path in ckpts:
        ck=torch.load(path, map_location="cpu"); m=ck["meta"]; K=int(m.get("K",20))
        enc=GNNEncoder(Fin, hid=m["hid"], layers=m["layers"], kind=m["encoder"], dropout=m["dropout"])
        head=TCNHead(hid=m["hid"], K=K) if m.get("temporal","tcn")=="tcn" else GRUHead(hid=m["hid"])
        enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"]); enc.eval(); head.eval()
        yh=[]
        with torch.no_grad():
            for t in valid:
                H=[enc(X[s], edges) for s in range(t-K+1, t+1)]
                yh.append(head(torch.stack(H,0)).numpy())
        allp.append(np.stack(yh))
    P=np.mean(np.stack(allp,0),0)            # [Tvalid,N]
    Ytrue=np.stack([Y[t+1].numpy() for t in valid])  # [Tvalid,N]
    return P, Ytrue

def fit_affine(P, Y, lam=1e-6, mode="per-port"):
    T,N=P.shape
    if mode=="global":
        A=np.c_[P.reshape(-1), np.ones(T*N)]
        y=Y.reshape(-1)
        R = np.array([[lam,0],[0,0]], float)
        theta=np.linalg.pinv(A.T@A + R)@(A.T@y)
        a=np.full(N, theta[0]); b=np.full(N, theta[1])
    else:
        a=np.zeros(N); b=np.zeros(N)
        for j in range(N):
            pj=P[:,j]; yj=Y[:,j]
            A=np.c_[pj, np.ones_like(pj)]
            R=np.array([[lam,0],[0,0]], float)
            theta=np.linalg.pinv(A.T@A + R)@(A.T@yj)
            a[j], b[j] = float(theta[0]), float(theta[1])
    return a,b

def apply_affine(P, a, b, soft=0.0, nonneg=True):
    Yh = a[None,:]*P + b[None,:]
    if soft>0: Yh = Yh - soft
    if nonneg: Yh = np.maximum(0.0, Yh)
    return Yh

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--mode", choices=["per-port","global"], default="per-port")
    ap.add_argument("--soft", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=1e-6)
    args=ap.parse_args()

    Z=load_npz(args.data)
    Pval, Yval = get_preds(Z, args.ckpts, Z["val_idx"], task="lead1")
    Pte , Yte  = get_preds(Z, args.ckpts, Z["test_idx"], task="lead1")

    # pre-calibration scores
    print(f"RAW ensemble → all-frames RMSE: {rmse(Pte,Yte):.3f} | busy-only RMSE: {rmse(Pte[Yte>0],Yte[Yte>0]):.3f}")

    a,b = fit_affine(Pval, Yval, lam=args.lam, mode=args.mode)
    Pte_cal = apply_affine(Pte, a, b, soft=args.soft, nonneg=True)

    micro_all = rmse(Pte_cal, Yte)
    micro_busy = rmse(Pte_cal[Yte>0], Yte[Yte>0])
    macro_all_s = macro_all(Pte_cal, Yte)
    macro_busy_s = macro_busy(Pte_cal, Yte)
    print(f"CAL ensemble ({args.mode}, soft={args.soft}) → all-frames micro {micro_all:.3f} | macro {macro_all_s:.3f}")
    print(f"CAL ensemble ({args.mode}, soft={args.soft}) → busy-only  micro {micro_busy:.3f} | macro {macro_busy_s:.3f}")

if __name__=="__main__":
    main()
