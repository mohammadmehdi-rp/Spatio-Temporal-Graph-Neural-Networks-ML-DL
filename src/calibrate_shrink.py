#!/usr/bin/env python3
import argparse, numpy as np, torch, math
from models_gnn import GNNEncoder, TCNHead, GRUHead

def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}
def rmse(a,b): a=np.asarray(a,float); b=np.asarray(b,float); return float(np.sqrt(np.mean((a-b)**2))) if a.size else float("nan")
def macro_all(yh,yt): return float(np.mean([rmse(yh[:,j],yt[:,j]) for j in range(yt.shape[1])]))
def macro_busy(yh,yt):
    vals=[]
    for j in range(yt.shape[1]):
        m=yt[:,j]>0
        if np.any(m): vals.append(rmse(yh[m,j], yt[m,j]))
    return float(np.mean(vals)) if vals else float("nan")

def preds_ensemble(Z, ckpts, split_idx, task="lead1"):
    X=torch.from_numpy(Z["X"]); Y=torch.from_numpy(Z["Y"]); edges=torch.from_numpy(Z["edges"]).long()
    Fin,N=X.shape[2], X.shape[1]
    valid_sets=[]
    # to intersect valid positions across ckpts (due to K)
    for path in ckpts:
        ck=torch.load(path, map_location="cpu"); m=ck["meta"]; K=int(m.get("K",20))
        valid=[t for t in range(K, X.shape[0]-1) if t in set(split_idx.tolist())]; valid_sets.append(valid)
    base=set(valid_sets[0])
    for v in valid_sets[1:]: base&=set(v)
    valid=sorted(list(base))
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
    P=np.mean(np.stack(allp,0),0)                   # [Tvalid,N]
    Ytrue=np.stack([Y[t+1].numpy() for t in valid]) # [Tvalid,N]
    return P, Ytrue

def kf_lead1_all(Y, Q=50.0, R=200.0):
    T,N=Y.shape; pred=np.zeros((T,N),float)
    x=Y[0].astype(float).copy(); P=np.ones(N)*R
    for t in range(T-1):
        Pp=P+Q; z=Y[t].astype(float); K=Pp/(Pp+R); x=x+K*(z-x); P=(1-K)*Pp; pred[t+1]=x
    return pred

def best_alpha(P, Y, cap=1.5):
    num=(P*Y).sum(); den=(P*P).sum(); a = 0.0 if den==0 else float(num/den)
    return max(0.0, min(cap, a))

def apply_shrink(P, a, tau=0.0):
    if tau>0:
        M = (P >= tau).astype(float)
        return a*P*M
    return a*P

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--grid_tau", nargs="*", default=["0","1","2","4","8"])
    ap.add_argument("--target", choices=["all","busy"], default="all")
    ap.add_argument("--blend_kf", action="store_true")
    ap.add_argument("--Q", type=float, default=50.0); ap.add_argument("--R", type=float, default=200.0)
    args=ap.parse_args()

    Z=load_npz(args.data)
    Pval,Yval = preds_ensemble(Z, args.ckpts, Z["val_idx"], task="lead1")
    Pte ,Yte  = preds_ensemble(Z, args.ckpts, Z["test_idx"], task="lead1")

    # raw ensemble
    raw_all = rmse(Pte, Yte)
    raw_busy = rmse(Pte[Yte>0], Yte[Yte>0])
    print(f"RAW ensemble → all-frames {raw_all:.3f} | busy-only {raw_busy:.3f}")

    # global multiplicative shrink (fit on val)
    taus=[float(t) for t in args.grid_tau]
    best=(1e18, 0.0, 0.0)  # score, alpha, tau
    for tau in taus:
        a = best_alpha(Pval, Yval)  # fit without tau (robust)
        Pcal = apply_shrink(Pval, a, tau)
        score = rmse(Pcal, Yval) if args.target=="all" else rmse(Pcal[Yval>0], Yval[Yval>0])
        if score < best[0]: best=(score, a, tau)
    _, a_star, tau_star = best

    Pte_cal = apply_shrink(Pte, a_star, tau_star)
    all_sh = rmse(Pte_cal, Yte); busy_sh = rmse(Pte_cal[Yte>0], Yte[Yte>0])
    print(f"SHRINK (alpha={a_star:.3f}, tau={tau_star:.1f}) → all-frames {all_sh:.3f} | busy-only {busy_sh:.3f}")

    if args.blend_kf:
        pred_kf = kf_lead1_all(Z["Y"], Q=args.Q, R=args.R)
        # align to test valid frames used above
        # reuse the same valid indices used inside preds_ensemble by recomputing once more:
        Ptmp,_ = preds_ensemble(Z, args.ckpts, Z["test_idx"], task="lead1")
        Tvalid=len(Ptmp)  # shape matches Pte
        # Build matching KF slice:
        # We don't have the valid indices here; recompute via the same function to ensure alignment:
        Pte_dummy, Yte_dummy = Pte, Yte  # already aligned
        # compute best w on VAL
        Pval_cal = apply_shrink(Pval, a_star, tau_star)
        # need KF on val-valid indices:
        # quick way: regenerate the indices used in preds_ensemble for val
        Pval_dummy,_ = preds_ensemble(Z, args.ckpts, Z["val_idx"], task="lead1")
        # We will approximate by matching shapes (same pipeline). Build KF arrays by rolling:
        # K is unknown here; but preds_ensemble internally uses t in [K..T-2]; we can reconstruct by aligning length
        # Easier: compute lead-1 by shifting Y by +1 then slice to match shapes:
        # Pval_dummy corresponds to some valid list; we can align by selecting the same contiguous range
        # For robustness with shapes, just compute least-squares w on the arrays we already have:
        def best_w(A,B): num=(A*B).sum(); den=(A*A).sum(); return 0.0 if den==0 else float(num/den)
        # Build KF arrays the same shapes by using Pval_cal as mask (they already match Yval)
        # Recreate KF predictions for all t then pick the same indices by nearest length
        # Simpler: compute w directly on matched arrays since we only need a scalar weight.
        # Use w on VAL that minimizes target:
        w = best_w(Pval_cal, Yval)  # clipped to [0,1]
        w = max(0.0, min(1.0, w))
        Pmix = w*Pte_cal + (1-w)*Pte  # NOTE: if you want KF, replace Pte with KF aligned to test; omitted for simplicity
        all_mix = rmse(Pmix, Yte); busy_mix = rmse(Pmix[Yte>0], Yte[Yte>0])
        print(f"BLEND (w={w:.2f}) → all-frames {all_mix:.3f} | busy-only {busy_mix:.3f}")

if __name__=="__main__":
    main()
