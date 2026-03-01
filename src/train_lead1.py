#!/usr/bin/env python3
import argparse, numpy as np, torch, torch.nn as nn
from models_gnn import GNNEncoder, TCNHead, GRUHead

def rmse(a,b): return float(torch.sqrt(torch.mean((a-b)**2)).item())
def load_npz(path): D=np.load(path, allow_pickle=True); return {k:D[k] for k in D.files}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.npz")
    ap.add_argument("--encoder", choices=["sage","routenet"], default="sage")
    ap.add_argument("--temporal", choices=["tcn","gru"], default="tcn")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--delta", type=float, default=50.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lead1_ckpt.pt")
    args=ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    Z=load_npz(args.data)
    X=torch.from_numpy(Z["X"])       # [T,N,F]
    Y=torch.from_numpy(Z["Y"])       # [T,N]
    edges=torch.from_numpy(Z["edges"])
    train_idx=torch.from_numpy(Z["train_idx"]).long()
    val_idx=torch.from_numpy(Z["val_idx"]).long()
    test_idx=torch.from_numpy(Z["test_idx"]).long()
    Fin=X.shape[2]; K=args.K

    enc=GNNEncoder(Fin, hid=args.hid, layers=args.layers, kind=args.encoder, dropout=args.dropout)
    head=TCNHead(hid=args.hid, K=K) if args.temporal=="tcn" else GRUHead(hid=args.hid)
    model=torch.nn.Module(); model.enc=enc; model.head=head
    opt=torch.optim.Adam(model.parameters(), lr=args.lr)
    huber=nn.SmoothL1Loss(beta=args.delta)

    # valid time positions with full history and a lead target
    Ttot=X.shape[0]
    valid=list(range(K, Ttot-1))
    train_pos=[t for t in valid if t in set(train_idx.tolist())]
    val_pos  =[t for t in valid if t in set(val_idx.tolist())]
    test_pos =[t for t in valid if t in set(test_idx.tolist())]

    best=(1e9,None); patience=10; bad=0
    for ep in range(1, args.epochs+1):
        model.train(); tot=0.0; steps=0
        for t in train_pos:
            Hseq=[]
            for s in range(t-K+1, t+1):
                Hseq.append(model.enc(X[s], edges))
            Hseq=torch.stack(Hseq,0)        # [K,N,H]
            y_next=Y[t+1]                    # [N]
            yhat=model.head(Hseq)            # [N]
            mask=(y_next>0)
            loss=huber(yhat[mask], y_next[mask]) if mask.any() else huber(yhat, y_next)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot+=loss.item(); steps+=1

        # validation on busy-only
        model.eval()
        with torch.no_grad():
            yh=[]; yt=[]
            for t in val_pos:
                Hseq=[]
                for s in range(t-K+1, t+1):
                    Hseq.append(model.enc(X[s], edges))
                Hseq=torch.stack(Hseq,0)
                pred=model.head(Hseq); yy=Y[t+1]
                m=(yy>0)
                if m.any(): yh.append(pred[m]); yt.append(yy[m])
            val_rmse=rmse(torch.cat(yh), torch.cat(yt)) if yh else 1e9
        print(f"[lead1][ep {ep:03d}] train_loss={tot/max(1,steps):.5f} val_busy_RMSE={val_rmse:.3f}")

        if val_rmse<best[0]-1e-6:
            best=(val_rmse, {"enc":model.enc.state_dict(),"head":model.head.state_dict(),"meta":vars(args)})
            bad=0
        else:
            bad+=1
            if bad>=patience:
                print(f"Early stop at epoch {ep}."); break

    if best[1] is None:
        best=(val_rmse, {"enc":model.enc.state_dict(),"head":model.head.state_dict(),"meta":vars(args)})
    torch.save(best[1], args.out)

    # test
    model.enc.load_state_dict(best[1]["enc"]); model.head.load_state_dict(best[1]["head"])
    yh=[]; yt=[]
    with torch.no_grad():
        for t in test_pos:
            Hseq=[]
            for s in range(t-K+1, t+1):
                Hseq.append(model.enc(X[s], edges))
            Hseq=torch.stack(Hseq,0)
            pred=model.head(Hseq); yy=Y[t+1]
            yh.append(pred); yt.append(yy)
    yh=torch.stack(yh); yt=torch.stack(yt)
    print(f"TEST lead-1 micro RMSE: {rmse(yh.flatten(), yt.flatten()):.3f} | saved {args.out}")

if __name__=="__main__":
    main()
