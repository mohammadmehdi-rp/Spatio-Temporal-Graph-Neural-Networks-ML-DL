#!/usr/bin/env python3
"""
Nowcast with label scaling: loss on (y/scale) but predictions stay in raw pkts.
Scale default = 1000 (qdisc backlog_pkts cap). Busy-only training + early stop.
"""
import argparse, numpy as np, torch, torch.nn as nn
from models_gnn import GNNEncoder, NowcastHead

def rmse(a,b): return float(torch.sqrt(torch.mean((a-b)**2)).item())
def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v3.npz")
    ap.add_argument("--encoder", choices=["sage","routenet"], default="sage")
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--delta", type=float, default=0.1, help="Huber delta on scaled targets")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--y_cap", type=float, default=1000.0, help="per-node label scale (pkts)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="nowcast_scaled.pt")
    args=ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    Z=load_npz(args.data)
    X=torch.from_numpy(Z["X"])       # [T,N,F]
    Y=torch.from_numpy(Z["Y"])       # [T,N]
    edges=torch.from_numpy(Z["edges"]).long()
    train_idx=torch.from_numpy(Z["train_idx"]).long()
    val_idx=torch.from_numpy(Z["val_idx"]).long()
    test_idx=torch.from_numpy(Z["test_idx"]).long()
    Fin, N = X.shape[2], X.shape[1]

    # constant per-node scaling (pkts → fraction of cap)
    scale = torch.full((N,), float(args.y_cap))

    enc=GNNEncoder(Fin, hid=args.hid, layers=args.layers, kind=args.encoder, dropout=args.dropout)
    head=NowcastHead(hid=args.hid)
    model=torch.nn.Module(); model.enc=enc; model.head=head
    opt=torch.optim.Adam(model.parameters(), lr=args.lr)
    huber=nn.SmoothL1Loss(beta=args.delta)

    best=(1e9,None); patience=12; bad=0
    for ep in range(1, args.epochs+1):
        model.train(); tot=0.0; steps=0
        for t in train_idx.tolist():
            xt, yt = X[t], Y[t]        # [N,F], [N]
            m = (yt > 0)
            if not m.any(): continue
            yhat = model.head(model.enc(xt, edges))         # raw pkts
            loss = huber((yhat/scale)[m], (yt/scale)[m])    # scaled loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot+=loss.item(); steps+=1

        # val on RAW busy RMSE
        model.eval()
        with torch.no_grad():
            yh=[]; yt=[]
            for t in val_idx.tolist():
                yy=Y[t]; m=(yy>0)
                if not m.any(): continue
                pred=model.head(model.enc(X[t], edges))
                yh.append(pred[m]); yt.append(yy[m])
            val_rmse = rmse(torch.cat(yh), torch.cat(yt)) if yh else 1e9
        print(f"[nowcast-scaled][ep {ep:03d}] train_loss={tot/max(1,steps):.5f} val_busy_RMSE={val_rmse:.3f}")

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

    # test (RAW micro/macro)
    enc.load_state_dict(best[1]["enc"]); head.load_state_dict(best[1]["head"])
    with torch.no_grad():
        yh=[]; yt=[]
        for t in test_idx.tolist():
            yh.append(head(enc(X[t], edges))); yt.append(Y[t])
    yh=torch.stack(yh); yt=torch.stack(yt)
    N=yt.shape[1]
    micro = rmse(yh.flatten(), yt.flatten())
    macro = float(np.mean([rmse(yh[:,i], yt[:,i]) for i in range(N)]))
    print(f"TEST nowcast micro {micro:.3f} | macro {macro:.3f} | saved {args.out}")

if __name__=="__main__":
    main()
