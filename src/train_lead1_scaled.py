#!/usr/bin/env python3
"""
Lead-1 with label scaling: loss on (y/scale). History K, GraphSAGE encoder + GRU/TCN head.
"""
import argparse, numpy as np, torch, torch.nn as nn
from models_gnn import GNNEncoder, TCNHead, GRUHead

def rmse(a,b): return float(torch.sqrt(torch.mean((a-b)**2)).item())
def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v3.npz")
    ap.add_argument("--encoder", choices=["sage","routenet"], default="sage")
    ap.add_argument("--temporal", choices=["gru","tcn"], default="gru")
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--y_cap", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lead1_scaled.pt")
    args=ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    Z=load_npz(args.data)
    X=torch.from_numpy(Z["X"]); Y=torch.from_numpy(Z["Y"])
    edges=torch.from_numpy(Z["edges"]).long()
    train_idx=torch.from_numpy(Z["train_idx"]).long()
    val_idx=torch.from_numpy(Z["val_idx"]).long()
    test_idx=torch.from_numpy(Z["test_idx"]).long()
    Fin, N, K = X.shape[2], X.shape[1], args.K

    scale = torch.full((N,), float(args.y_cap))

    enc=GNNEncoder(Fin, hid=args.hid, layers=args.layers, kind=args.encoder, dropout=args.dropout)
    head=GRUHead(hid=args.hid) if args.temporal=="gru" else TCNHead(hid=args.hid, K=K)
    model=torch.nn.Module(); model.enc=enc; model.head=head
    opt=torch.optim.Adam(model.parameters(), lr=args.lr)
    huber=nn.SmoothL1Loss(beta=args.delta)

    Ttot=X.shape[0]; valid=list(range(K, Ttot-1))
    tset=set(train_idx.tolist()); vset=set(val_idx.tolist()); teset=set(test_idx.tolist())
    train_pos=[t for t in valid if t in tset]
    val_pos=[t for t in valid if t in vset]
    test_pos=[t for t in valid if t in teset]

    best=(1e9,None); patience=12; bad=0
    for ep in range(1, args.epochs+1):
        model.train(); tot=0.0; steps=0
        for t in train_pos:
            yy=Y[t+1]; m=(yy>0)
            if not m.any(): continue
            Hseq=torch.stack([model.enc(X[s], edges) for s in range(t-K+1, t+1)], 0)  # [K,N,H]
            yhat=model.head(Hseq)                                 # raw pkts
            loss=huber((yhat/scale)[m], (yy/scale)[m])            # scaled loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot+=loss.item(); steps+=1

        # val on RAW RMSE (busy-only)
        model.eval()
        with torch.no_grad():
            yh=[]; yt=[]
            for t in val_pos:
                yy=Y[t+1]; m=(yy>0)
                if not m.any(): continue
                Hseq=torch.stack([model.enc(X[s], edges) for s in range(t-K+1, t+1)], 0)
                pred=model.head(Hseq); yh.append(pred[m]); yt.append(yy[m])
            val_rmse = rmse(torch.cat(yh), torch.cat(yt)) if yh else 1e9
        print(f"[lead1-scaled][ep {ep:03d}] train_loss={tot/max(1,steps):.5f} val_busy_RMSE={val_rmse:.3f}")

        if val_rmse < best[0]-1e-6:
            best=(val_rmse, {"enc":enc.state_dict(),"head":head.state_dict(),"meta":vars(args)})
            bad=0
        else:
            bad+=1
            if bad>=patience:
                print(f"Early stop at epoch {ep}."); break

    if best[1] is None:
        best=(val_rmse, {"enc":enc.state_dict(),"head":head.state_dict(),"meta":vars(args)})
    torch.save(best[1], args.out)

    # test (RAW micro)
    enc.load_state_dict(best[1]["enc"]); head.load_state_dict(best[1]["head"])
    with torch.no_grad():
        yh=[]; yt=[]
        for t in test_pos:
            Hseq=torch.stack([enc(X[s], edges) for s in range(t-K+1, t+1)], 0)
            pred=head(Hseq); yy=Y[t+1]
            yh.append(pred); yt.append(yy)
    yh=torch.stack(yh); yt=torch.stack(yt)
    print(f"TEST lead-1 micro RMSE: {rmse(yh.flatten(), yt.flatten()):.3f} | saved {args.out}")

if __name__=="__main__":
    main()
