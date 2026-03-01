#!/usr/bin/env python3
import argparse, numpy as np, torch, re
from models_gnn import GNNEncoder, NowcastHead, TCNHead, GRUHead

def rmse(a,b): return float(torch.sqrt(torch.mean((a-b)**2)).item())
def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}
def b2s(x): return x.decode() if isinstance(x, bytes) else str(x)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--nowcast_ckpt", default=None)
    ap.add_argument("--lead1_ckpt", default=None)
    args=ap.parse_args()

    Z=load_npz(args.data)
    X=torch.from_numpy(Z["X"]); Y=torch.from_numpy(Z["Y"])
    edges=torch.from_numpy(Z["edges"]).long()
    test_idx=torch.from_numpy(Z["test_idx"]).long()
    Fin=X.shape[2]

    # ---- NOWCAST (optional)
    if args.nowcast_ckpt:
        nc=torch.load(args.nowcast_ckpt, map_location="cpu"); meta=nc.get("meta",{})
        enc=GNNEncoder(Fin, hid=meta.get("hid",96), layers=meta.get("layers",3),
                       kind=meta.get("encoder","sage"), dropout=meta.get("dropout",0.2))
        head=NowcastHead(hid=meta.get("hid",96))
        enc.load_state_dict(nc["enc"]); head.load_state_dict(nc["head"])
        enc.eval(); head.eval()
        yh=[]; yt=[]
        with torch.no_grad():
            for t in test_idx.tolist():
                yy=Y[t]; m=(yy>0)
                if not m.any(): continue
                pred=head(enc(X[t], edges)); yh.append(pred[m]); yt.append(yy[m])
        if yh:
            print(f"Nowcast TEST busy-only RMSE: {rmse(torch.cat(yh), torch.cat(yt)):.3f}")
        else:
            print("Nowcast TEST busy-only RMSE: N/A (no busy)")

    # ---- LEAD-1 (optional)
    if args.lead1_ckpt:
        ld=torch.load(args.lead1_ckpt, map_location="cpu"); meta=ld.get("meta",{})
        temporal=meta.get("temporal","tcn")   # default to TCN if missing
        K=int(meta.get("K",20))
        enc2=GNNEncoder(Fin, hid=meta.get("hid",96), layers=meta.get("layers",2),
                        kind=meta.get("encoder","sage"), dropout=meta.get("dropout",0.2))
        head2=TCNHead(hid=meta.get("hid",96), K=K) if temporal=="tcn" else GRUHead(hid=meta.get("hid",96))
        enc2.load_state_dict(ld["enc"]); head2.load_state_dict(ld["head"])
        enc2.eval(); head2.eval()
        valid=[t for t in range(K, X.shape[0]-1) if t in set(test_idx.tolist())]
        yh=[]; yt=[]
        with torch.no_grad():
            for t in valid:
                yy=Y[t+1]; m=(yy>0)
                if not m.any(): continue
                H=[enc2(X[s], edges) for s in range(t-K+1, t+1)]
                pred=head2(torch.stack(H,0)); yh.append(pred[m]); yt.append(yy[m])
        if yh:
            print(f"Lead-1 TEST busy-only RMSE: {rmse(torch.cat(yh), torch.cat(yt)):.3f}")
        else:
            print("Lead-1 TEST busy-only RMSE: N/A (no busy)")

if __name__=="__main__":
    main()
