#!/usr/bin/env python3
import argparse, time, numpy as np, torch
from models_gnn import GNNEncoder, NowcastHead, TCNHead, GRUHead

def rmse(a,b): return float(torch.sqrt(torch.mean((a-b)**2)).item())
def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}

def micro_macro(yhat, y):
    micro = rmse(yhat.flatten(), y.flatten())
    N=y.shape[1]; vals=[]
    for i in range(N):
        if y[:,i].numel()>=2:
            vals.append(rmse(y[:,i], yhat[:,i]))
    macro=float(np.mean(vals)) if vals else micro
    return micro, macro

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.npz")
    ap.add_argument("--nowcast_ckpt", default="nowcast_ckpt.pt")
    ap.add_argument('--lead1_ckpt', default=None)
    args=ap.parse_args()

    Z=load_npz(args.data)
    X=torch.from_numpy(Z["X"]); Y=torch.from_numpy(Z["Y"])
    edges=torch.from_numpy(Z["edges"])
    test_idx=torch.from_numpy(Z["test_idx"])
    Fin=X.shape[2]

    # Nowcast
    nc=torch.load(args.nowcast_ckpt, map_location="cpu")
    enc=GNNEncoder(Fin, hid=nc["meta"]["hid"], layers=nc["meta"]["layers"], kind=nc["meta"]["encoder"], dropout=nc["meta"]["dropout"])
    head=NowcastHead(hid=nc["meta"]["hid"])
    enc.load_state_dict(nc["enc"]); head.load_state_dict(nc["head"])
    enc.eval(); head.eval()

    t0=time.time(); yh=[]; yt=[]
    with torch.no_grad():
        for t in test_idx.tolist():
            h=enc(X[t], edges); pred=head(h)
            yh.append(pred); yt.append(Y[t])
    yh=torch.stack(yh); yt=torch.stack(yt); dt=time.time()-t0
    micro, macro = micro_macro(yh, yt)
    print(f"Nowcast: micro {micro:.3f} | macro {macro:.3f} | latency {1000*dt/len(test_idx):.2f} ms/step")

    # Lead-1 (use K from checkpoint)
import os
if args.lead1_ckpt and os.path.exists(args.lead1_ckpt):
    ld=torch.load(args.lead1_ckpt, map_location='cpu')
    enc2.load_state_dict(ld['enc']); head2.load_state_dict(ld['head'])
    enc2.eval(); head2.eval()

    valid=[t for t in range(K, X.shape[0]-1) if t in set(test_idx.tolist())]
    t0=time.time(); yh=[]; yt=[]
    with torch.no_grad():
        for t in valid:
            Hseq=[enc2(X[s], edges) for s in range(t-K+1, t+1)]
            Hseq=torch.stack(Hseq,0)
            pred=head2(Hseq); yy=Y[t+1]
            yh.append(pred); yt.append(yy)
    yh=torch.stack(yh); yt=torch.stack(yt); dt=time.time()-t0
    micro, macro = micro_macro(yh, yt)
    print(f"Lead-1:  micro {micro:.3f} | macro {macro:.3f} | latency {1000*dt/len(valid):.2f} ms/step")

if __name__=="__main__":
    main()
