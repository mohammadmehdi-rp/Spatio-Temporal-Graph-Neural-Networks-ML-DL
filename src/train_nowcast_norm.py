#!/usr/bin/env python3
"""
Nowcast with per-node label normalization (computed on TRAIN busy samples).
Loss is Huber on normalized targets; RMSE reported on raw scale.
"""
import argparse, numpy as np, torch, torch.nn as nn
from models_gnn import GNNEncoder, NowcastHead

def rmse(a,b): return float(torch.sqrt(torch.mean((a-b)**2)).item())
def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}

def compute_node_norm(Y, idx, busy_only=True):
    """Return per-node (mu, sd) as torch tensors from Y[idx]."""
    Yt=torch.from_numpy(Y[idx])  # [Ttr,N]
    if busy_only:
        m=(Yt>0).float()
        # avoid divide-by-zero
        cnt=m.sum(0).clamp(min=1.0)
        mu=(Yt*m).sum(0)/cnt
        sd=torch.sqrt((((Yt-mu)*m)**2).sum(0)/cnt).clamp(min=1e-3)
    else:
        mu=Yt.mean(0); sd=Yt.std(0).clamp(min=1e-3)
    return mu, sd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v2.npz")
    ap.add_argument("--encoder", choices=["sage","routenet"], default="sage")
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--delta", type=float, default=50.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="nowcast_v2_norm.pt")
    args=ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    Z=load_npz(args.data)
    X=torch.from_numpy(Z["X"])            # [T,N,F]
    Y=torch.from_numpy(Z["Y"])            # [T,N]
    edges=torch.from_numpy(Z["edges"]).long()
    train_idx=torch.from_numpy(Z["train_idx"]).long()
    val_idx=torch.from_numpy(Z["val_idx"]).long()
    test_idx=torch.from_numpy(Z["test_idx"]).long()
    Fin=X.shape[2]; N=X.shape[1]

    # per-node label normalization from TRAIN busy samples
    mu_node, sd_node = compute_node_norm(Z["Y"], Z["train_idx"])
    mu_node=mu_node.float(); sd_node=sd_node.float()

    enc=GNNEncoder(Fin, hid=args.hid, layers=args.layers, kind=args.encoder, dropout=args.dropout)
    head=NowcastHead(hid=args.hid)
    model=torch.nn.Module(); model.enc=enc; model.head=head
    opt=torch.optim.Adam(model.parameters(), lr=args.lr)
    huber=nn.SmoothL1Loss(beta=args.delta)

    best=(1e9,None); patience=10; bad=0
    for ep in range(1, args.epochs+1):
        model.train(); tot=0.0; steps=0
        for t in train_idx.tolist():
            xt=X[t]; yt=Y[t]  # [N]
            m=(yt>0)
            if not m.any(): continue  # skip fully idle steps
            yhat=model.head(model.enc(xt, edges))
            yn=(yt - mu_node)/sd_node
            yhatn=(yhat - mu_node)/sd_node
            loss=huber(yhatn[m], yn[m])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot+=loss.item(); steps+=1

        # val (busy-only) on RAW RMSE
        model.eval()
        with torch.no_grad():
            yh=[]; yt=[]
            for t in val_idx.tolist():
                xt=X[t]; yy=Y[t]; pred=model.head(model.enc(xt, edges))
                m=(yy>0)
                if m.any(): yh.append(pred[m]); yt.append(yy[m])
            val_rmse = rmse(torch.cat(yh), torch.cat(yt)) if yh else 1e9
        print(f"[nowcast-norm][ep {ep:03d}] train_loss={tot/max(1,steps):.5f} val_busy_RMSE={val_rmse:.3f}")

        if val_rmse<best[0]-1e-6:
            best=(val_rmse, {"enc":enc.state_dict(),"head":head.state_dict(),
                             "mu":mu_node, "sd":sd_node, "meta":vars(args)})
            bad=0
        else:
            bad+=1
            if bad>=patience:
                print(f"Early stop at epoch {ep}."); break

    if best[1] is None:
        best=(val_rmse, {"enc":enc.state_dict(),"head":head.state_dict(),"mu":mu_node,"sd":sd_node,"meta":vars(args)})
    torch.save(best[1], args.out)

    # test (report raw micro/macro)
    enc.load_state_dict(best[1]["enc"]); head.load_state_dict(best[1]["head"])
    with torch.no_grad():
        yh=[]; yt=[]
        for t in test_idx.tolist():
            pred=head(enc(X[t], edges)); yy=Y[t]
            yh.append(pred); yt.append(yy)
    yh=torch.stack(yh); yt=torch.stack(yt)

    def macro(yhat,y):
        vals=[]; 
        for i in range(N):
            a=yhat[:,i]; b=y[:,i]
            vals.append(rmse(a,b))
        return float(np.mean(vals))
    print(f"TEST nowcast micro {rmse(yh.flatten(), yt.flatten()):.3f} | macro {macro(yh,yt):.3f} | saved {args.out}")

if __name__=="__main__":
    main()
