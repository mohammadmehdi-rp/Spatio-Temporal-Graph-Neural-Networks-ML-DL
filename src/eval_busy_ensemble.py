#!/usr/bin/env python3
import argparse, numpy as np, torch
from models_gnn import GNNEncoder, TCNHead, GRUHead

def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}
def rmse(a,b): return float(torch.sqrt(torch.mean((a-b)**2)).item())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    args=ap.parse_args()

    Z=load_npz(args.data)
    X=torch.from_numpy(Z["X"]); Y=torch.from_numpy(Z["Y"])
    edges=torch.from_numpy(Z["edges"]).long()
    test_idx=torch.from_numpy(Z["test_idx"]).long()
    Fin,N = X.shape[2], X.shape[1]

    preds=[]
    for path in args.ckpts:
        ck=torch.load(path, map_location="cpu"); m=ck["meta"]
        enc=GNNEncoder(Fin, hid=m["hid"], layers=m["layers"], kind=m["encoder"], dropout=m["dropout"])
        head=TCNHead(hid=m["hid"], K=m["K"]) if m.get("temporal","tcn")=="tcn" else GRUHead(hid=m["hid"])
        enc.load_state_dict(ck["enc"]); head.load_state_dict(ck["head"])
        enc.eval(); head.eval()
        # collect lead-1 preds over test busy-only frames
        valid=[t for t in range(m["K"], X.shape[0]-1) if t in set(test_idx.tolist())]
        yh=[]
        with torch.no_grad():
            for t in valid:
                H=[enc(X[s], edges) for s in range(t-m["K"]+1, t+1)]
                pred=head(torch.stack(H,0))        # y_{t+1}
                yh.append(pred)
        preds.append(torch.stack(yh))  # [Ttest,N]
    P=torch.mean(torch.stack(preds),0)
    # Compute busy-only RMSE on test
    valid=[t for t in range(m["K"], X.shape[0]-1) if t in set(test_idx.tolist())]
    ytrue=torch.stack([Y[t+1] for t in valid])
    mask=(ytrue>0)
    print(f"Lead-1 ENSEMBLE busy-only RMSE: {rmse(P[mask], ytrue[mask]):.3f}")

if __name__=="__main__":
    main()
