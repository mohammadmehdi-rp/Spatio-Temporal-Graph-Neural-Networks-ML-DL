#!/usr/bin/env python3
import argparse, os, random, numpy as np, torch, torch.nn as nn
from models_gnn import GNNEncoder, TCNHead, GRUHead

def rmse(a,b): return float(torch.sqrt(torch.mean((a-b)**2)).item())
def load_npz(p): D=np.load(p, allow_pickle=True); return {k:D[k] for k in D.files}

# ----- Deterministic switches -----
def make_deterministic(seed=0):
    os.environ["PYTHONHASHSEED"]="0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

# ----- EMA of parameters -----
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        self.params = [p for p in model.parameters() if p.requires_grad]
    @torch.no_grad()
    def update(self):
        for s,p in zip(self.shadow, self.params): s.mul_(self.decay).add_(p, alpha=1-self.decay)
    @torch.no_grad()
    def apply(self):
        self.backup=[p.detach().clone() for p in self.params]
        for p,s in zip(self.params, self.shadow): p.data.copy_(s.data)
    @torch.no_grad()
    def restore(self):
        for p,b in zip(self.params, self.backup): p.data.copy_(b.data)

def focal_huber(pred_s, targ_s, w, delta):
    loss = nn.functional.smooth_l1_loss(pred_s, targ_s, reduction='none', beta=delta)
    return (loss * w).sum() / (w.sum() + 1e-9)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_student", required=True)
    ap.add_argument("--teacher_preds", required=True)
    ap.add_argument("--encoder", choices=["sage","routenet"], default="sage")
    ap.add_argument("--temporal", choices=["gru","tcn"], default="tcn")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--y_cap", type=float, default=1000.0)
    ap.add_argument("--sdrop", type=float, default=0.05)     # lighter dropout
    ap.add_argument("--gamma", type=float, default=2.0)      # stronger burst focus
    ap.add_argument("--alpha", type=float, default=0.7)      # truth vs teacher blend
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="lead1_distill_ema.pt")
    args=ap.parse_args()

    make_deterministic(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    S=load_npz(args.data_student)
    X=torch.from_numpy(S["X"]).to(device)
    Y=torch.from_numpy(S["Y"]).to(device)
    edges=torch.from_numpy(S["edges"]).long().to(device)
    is_sensor=torch.from_numpy(S["is_sensor"]).bool().to(device)
    train_idx=torch.from_numpy(S["train_idx"]).long()
    val_idx=torch.from_numpy(S["val_idx"]).long()
    test_idx=torch.from_numpy(S["test_idx"]).long()
    Fin,N,K = X.shape[2], X.shape[1], args.K

    T=load_npz(args.teacher_preds)
    Yt=torch.from_numpy(T["Yhat"]).float().to(device)

    enc=GNNEncoder(Fin, hid=args.hid, layers=args.layers, kind=args.encoder, dropout=args.dropout).to(device)
    head=TCNHead(hid=args.hid, K=K).to(device) if args.temporal=="tcn" else GRUHead(hid=args.hid).to(device)

    opt=torch.optim.AdamW(list(enc.parameters())+list(head.parameters()), lr=args.lr, weight_decay=args.wd)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.1)

    ema_e=EMA(enc, args.ema); ema_h=EMA(head, args.ema)

    # positions usable for lead-1
    Ttot=X.shape[0]
    valid=list(range(K, Ttot-1))
    tset=set(train_idx.tolist()); vset=set(val_idx.tolist()); teset=set(test_idx.tolist())
    train_pos=[t for t in valid if t in tset]
    val_pos  =[t for t in valid if t in vset]
    test_pos =[t for t in valid if t in teset]

    scale = torch.full((N,), float(args.y_cap), device=device)
    feat_names=[f.decode() if isinstance(f,bytes) else str(f) for f in S["feat_names"]]
    ch_is = feat_names.index("is_sensor")
    ch_back=[feat_names.index(k) for k in ["sensor_backlog","sensor_backlog_lag1","sensor_backlog_lag2","sensor_backlog_lag3"]]

    best=(1e9,None); patience=14; bad=0
    for ep in range(1, args.epochs+1):
        enc.train(); head.train(); tot=0.0; steps=0
        for t in train_pos:
            yy = Y[t+1]                  # ground truth
            tt = Yt[t+1].clamp(min=0)    # teacher
            H=[]; nodes_to_drop=None
            for s in range(t-K+1, t+1):
                xs = X[s].clone()
                sens = (xs[:, ch_is] > 0.5)
                if args.sdrop>0:
                    if nodes_to_drop is None:
                        nodes_to_drop = ((torch.rand(N, device=device) < args.sdrop) & sens)
                    if nodes_to_drop.any(): xs[nodes_to_drop][:, ch_back] = 0.0
                H.append(enc(xs, edges))
            Hseq=torch.stack(H,0)
            yhat=head(Hseq)

            # blend target: sensors -> truth; non-sensors -> alpha*truth + (1-alpha)*teacher
            targ = torch.where(is_sensor, yy, args.alpha*yy + (1-args.alpha)*tt)
            mbusy=(targ>0)
            targ_s=(targ/scale); pred_s=(yhat/scale)
            w = (targ_s.clamp(0,1.5))**args.gamma

            loss=focal_huber(pred_s[mbusy], targ_s[mbusy], w[mbusy], args.delta)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(head.parameters()), 1.0)
            opt.step(); ema_e.update(); ema_h.update(); tot+=loss.item(); steps+=1

        # EMA validation (busy-only, truth)
        enc.eval(); head.eval(); ema_e.apply(); ema_h.apply()
        with torch.no_grad():
            yh=[]; yt=[]
            for t in val_pos:
                yy=Y[t+1]; m=(yy>0)
                if not m.any(): continue
                H=[enc(X[s], edges) for s in range(t-K+1, t+1)]
                pred=head(torch.stack(H,0)); yh.append(pred[m]); yt.append(yy[m])
            vrmse = rmse(torch.cat(yh), torch.cat(yt)) if yh else 1e9
        ema_e.restore(); ema_h.restore()
        sched.step()

        print(f"[lead1-distill-ema][ep {ep:03d}] train_loss={tot/max(1,steps):.5f} val_busy_RMSE={vrmse:.3f}")

        if vrmse<best[0]-1e-6:
            ema_e.apply(); ema_h.apply()
            best=(vrmse, {"enc":enc.state_dict(),"head":head.state_dict(),
                          "meta":{"encoder":args.encoder,"temporal":args.temporal,"K":K,
                                  "hid":args.hid,"layers":args.layers,"dropout":args.dropout}})
            ema_e.restore(); ema_h.restore(); bad=0
        else:
            bad+=1
            if bad>=patience: print(f"Early stop at epoch {ep}."); break

    # Save best EMA weights
    torch.save(best[1], args.out)

    # Test with EMA
    enc.load_state_dict(best[1]["enc"]); head.load_state_dict(best[1]["head"])
    with torch.no_grad():
        yh=[]; yt=[]
        for t in test_pos:
            yy=Y[t+1]; m=(yy>0)
            if not m.any(): continue
            H=[enc(X[s], edges) for s in range(t-K+1, t+1)]
            pred=head(torch.stack(H,0)); yh.append(pred[m]); yt.append(yy[m])
    print(f"TEST lead-1 busy-only RMSE: {rmse(torch.cat(yh), torch.cat(yt)):.3f} | saved {args.out}")

if __name__=="__main__":
    main()
