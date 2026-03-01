#!/usr/bin/env python3
import argparse, numpy as np, torch, torch.nn as nn
from models_gnn import GNNEncoder, NowcastHead

def rmse(a,b): return float(torch.sqrt(torch.mean((a-b)**2)).item())
def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}

def find_idx(names, key):
    names=[n.decode() if isinstance(n,bytes) else str(n) for n in names]
    for i,n in enumerate(names):
        if n==key: return i
    raise KeyError(key)

def focal_huber(pred_s, targ_s, w, delta):
    # SmoothL1 (Huber) with per-sample weights
    loss = nn.functional.smooth_l1_loss(pred_s, targ_s, reduction='none', beta=delta)
    return (loss * w).sum() / (w.sum() + 1e-9)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_v4.npz")
    ap.add_argument("--encoder", choices=["sage","routenet"], default="sage")
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--delta", type=float, default=0.1)        # Huber on scaled labels
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--y_cap", type=float, default=1000.0)     # scale (pkts)
    ap.add_argument("--sdrop", type=float, default=0.30)       # sensor dropout prob per node/step
    ap.add_argument("--gamma", type=float, default=1.5)        # focal exponent
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="nowcast_sparse.pt")
    args=ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    Z=load_npz(args.data)
    X=torch.from_numpy(Z["X"])       # [T,N,F] (normalized features)
    Y=torch.from_numpy(Z["Y"])       # [T,N]   (raw pkts)
    edges=torch.from_numpy(Z["edges"]).long()
    train_idx=torch.from_numpy(Z["train_idx"]).long()
    val_idx=torch.from_numpy(Z["val_idx"]).long()
    test_idx=torch.from_numpy(Z["test_idx"]).long()
    feat_names=Z["feat_names"]
    Fin,N = X.shape[2], X.shape[1]

    # feature channels we will occasionally blank on sensor nodes
    ch_back = [find_idx(feat_names,k) for k in ["sensor_backlog","sensor_backlog_lag1","sensor_backlog_lag2","sensor_backlog_lag3"]]
    ch_is_sensor = find_idx(feat_names,"is_sensor")

    enc=GNNEncoder(Fin, hid=args.hid, layers=args.layers, kind=args.encoder, dropout=args.dropout)
    head=NowcastHead(hid=args.hid)
    opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()), lr=args.lr)

    scale = torch.full((N,), float(args.y_cap))
    best=(1e9,None); patience=12; bad=0

    for ep in range(1,args.epochs+1):
        enc.train(); head.train(); tot=0.0; steps=0
        for t in train_idx.tolist():
            xt = X[t].clone()              # [N,F]
            yt = Y[t]                      # [N]
            mbusy = (yt>0)
            if not mbusy.any(): continue

            # sensor nodes mask from feature channel
            sens = (xt[:, ch_is_sensor] > 0.5)  # [N]
            if args.sdrop>0:
                drop = (torch.rand_like(sens.float()) < args.sdrop) & sens
                if drop.any():
                    xt[drop][:, ch_back] = 0.0  # zero the standardized backlog features for dropped sensors

            h = enc(xt, edges)
            yhat = head(h)                         # raw pkts

            # scaled targets + focal weights toward bursts
            targ_s = (yt/scale)
            pred_s = (yhat/scale)
            w = (targ_s.clamp(min=0, max=1.5))**args.gamma
            loss = focal_huber(pred_s[mbusy], targ_s[mbusy], w[mbusy], args.delta)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(head.parameters()), 1.0)
            opt.step(); tot+=loss.item(); steps+=1

        # val on busy-only RAW RMSE
        enc.eval(); head.eval()
        with torch.no_grad():
            yh=[]; yt=[]
            for t in val_idx.tolist():
                yy=Y[t]; m=(yy>0)
                if not m.any(): continue
                pred=head(enc(X[t], edges)); yh.append(pred[m]); yt.append(yy[m])
            vrmse = rmse(torch.cat(yh), torch.cat(yt)) if yh else 1e9

        print(f"[nowcast-sparse][ep {ep:03d}] train_loss={tot/max(1,steps):.5f} val_busy_RMSE={vrmse:.3f}")

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

    # test (RAW micro/macro)
    enc.load_state_dict(best[1]["enc"]); head.load_state_dict(best[1]["head"])
    with torch.no_grad():
        yh=[]; yt=[]
        for t in test_idx.tolist():
            yh.append(head(enc(X[t], edges))); yt.append(Y[t])
    yh=torch.stack(yh); yt=torch.stack(yt)
    micro=rmse(yh.flatten(), yt.flatten())
    macro=float(np.mean([rmse(yh[:,i], yt[:,i]) for i in range(N)]))
    print(f"TEST nowcast micro {micro:.3f} | macro {macro:.3f} | saved {args.out}")

if __name__=="__main__":
    main()
