#!/usr/bin/env python3
import argparse, numpy as np, torch, torch.nn as nn
from models_gnn import GNNEncoder, TCNHead, GRUHead

def rmse(a,b): return float(torch.sqrt(torch.mean((a-b)**2)).item())
def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}
def find_idx(names, key):
    names=[n.decode() if isinstance(n,bytes) else str(n) for n in names]
    for i,n in enumerate(names):
        if n==key: return i
    raise KeyError(key)

def focal_huber(pred_s, targ_s, w, delta):
    loss = nn.functional.smooth_l1_loss(pred_s, targ_s, reduction='none', beta=delta)
    return (loss * w).sum() / (w.sum() + 1e-9)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v4.npz")
    ap.add_argument("--encoder", choices=["sage","routenet"], default="sage")
    ap.add_argument("--temporal", choices=["gru","tcn"], default="tcn")
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--y_cap", type=float, default=1000.0)
    ap.add_argument("--sdrop", type=float, default=0.30)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="lead1_sparse.pt")
    args=ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    Z=load_npz(args.data)
    X=torch.from_numpy(Z["X"]); Y=torch.from_numpy(Z["Y"])
    edges=torch.from_numpy(Z["edges"]).long()
    train_idx=torch.from_numpy(Z["train_idx"]).long()
    val_idx=torch.from_numpy(Z["val_idx"]).long()
    test_idx=torch.from_numpy(Z["test_idx"]).long()
    feat_names=Z["feat_names"]
    Fin,N,K = X.shape[2], X.shape[1], args.K

    ch_back=[find_idx(feat_names,k) for k in ["sensor_backlog","sensor_backlog_lag1","sensor_backlog_lag2","sensor_backlog_lag3"]]
    ch_is_sensor = find_idx(feat_names,"is_sensor")

    enc=GNNEncoder(Fin, hid=args.hid, layers=args.layers, kind=args.encoder, dropout=args.dropout)
    head=TCNHead(hid=args.hid, K=K) if args.temporal=="tcn" else GRUHead(hid=args.hid)
    opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()), lr=args.lr)

    scale = torch.full((N,), float(args.y_cap))

    Ttot=X.shape[0]
    valid=list(range(K, Ttot-1))
    tset=set(train_idx.tolist()); vset=set(val_idx.tolist()); teset=set(test_idx.tolist())
    train_pos=[t for t in valid if t in tset]
    val_pos  =[t for t in valid if t in vset]
    test_pos =[t for t in valid if t in teset]

    best=(1e9,None); patience=12; bad=0
    for ep in range(1,args.epochs+1):
        enc.train(); head.train(); tot=0.0; steps=0
        for t in train_pos:
            yy = Y[t+1]; mbusy=(yy>0)
            if not mbusy.any(): continue

            # assemble history; apply same dropout set across the K frames (harder case)
            nodes_to_drop=None
            H=[]
            for s in range(t-K+1, t+1):
                xs = X[s].clone()
                sens = (xs[:, ch_is_sensor] > 0.5)
                if args.sdrop>0:
                    if nodes_to_drop is None:
                        nodes_to_drop = ((torch.rand_like(sens.float()) < args.sdrop) & sens)
                    if nodes_to_drop.any():
                        xs[nodes_to_drop][:, ch_back] = 0.0
                H.append(enc(xs, edges))
            Hseq = torch.stack(H, 0)     # [K,N,H]

            yhat = head(Hseq)            # raw pkts
            targ_s = (yy/scale); pred_s=(yhat/scale)
            w = (targ_s.clamp(0,1.5))**args.gamma
            loss = focal_huber(pred_s[mbusy], targ_s[mbusy], w[mbusy], args.delta)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(head.parameters()), 1.0)
            opt.step(); tot+=loss.item(); steps+=1

        # val busy-only RAW RMSE
        enc.eval(); head.eval()
        with torch.no_grad():
            yh=[]; yt=[]
            for t in val_pos:
                yy=Y[t+1]; m=(yy>0)
                if not m.any(): continue
                H=[enc(X[s], edges) for s in range(t-K+1, t+1)]
                pred=head(torch.stack(H,0)); yh.append(pred[m]); yt.append(yy[m])
            vrmse = rmse(torch.cat(yh), torch.cat(yt)) if yh else 1e9

        print(f"[lead1-sparse][ep {ep:03d}] train_loss={tot/max(1,steps):.5f} val_busy_RMSE={vrmse:.3f}")

        if vrmse<best[0]-1e-6:
            best=(vrmse, {"enc":enc.state_dict(),"head":head.state_dict(),"meta":vars(args)})
            bad=0
        else:
            bad+=1
            if bad>=patience:
                print(f"Early stop at epoch {ep}."); break

    if best[1] is None:
        best=(vrmse, {"enc":enc.state_dict(),"head":head.state_dict(),"meta":vars(args)})
    torch.save(best[1], args.out)

    # test RAW micro
    enc.load_state_dict(best[1]["enc"]); head.load_state_dict(best[1]["head"])
    with torch.no_grad():
        yh=[]; yt=[]
        for t in test_pos:
            H=[enc(X[s], edges) for s in range(t-K+1, t+1)]
            pred=head(torch.stack(H,0)); yy=Y[t+1]
            yh.append(pred); yt.append(yy)
    yh=torch.stack(yh); yt=torch.stack(yt)
    print(f"TEST lead-1 micro RMSE: {rmse(yh.flatten(), yt.flatten()):.3f} | saved {args.out}")

if __name__=="__main__":
    main()
