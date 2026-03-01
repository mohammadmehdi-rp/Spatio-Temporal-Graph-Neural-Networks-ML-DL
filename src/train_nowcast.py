#!/usr/bin/env python3
"""
train_nowcast.py
GraphSAGE / RouteNet-lite encoder for per-step queue nowcasting (y_t).

- Trains on BUSY samples only (where y_t > 0) to stabilize learning.
- Uses Huber loss, early stopping on busy-only validation RMSE.
- Saves the best checkpoint and prints test micro/macro RMSE.

Inputs (from gnn_prep.py):
  dataset.npz with keys:
    X: [T, N, F] float32   (normalized inputs)
    Y: [T, N]    float32   (labels: backlog_* per node)
    edges: [2, E] int32    (directed edges: src→dst)
    train_idx, val_idx, test_idx: int64  (time indices)
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from models_gnn import GNNEncoder, NowcastHead

# --------------------- utils --------------------- #

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def rmse_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a - b) ** 2)).item())

def load_npz(path):
    D = np.load(path, allow_pickle=True)
    return {k: D[k] for k in D.files}

def micro_macro(yhat: torch.Tensor, y: torch.Tensor, busy_only: bool = False):
    """
    yhat, y: [T_test, N]
    Returns micro RMSE and macro RMSE (mean of per-node RMSE).
    If busy_only=True, compute only on y>0 positions.
    """
    if busy_only:
        mask = (y > 0)
    else:
        mask = torch.ones_like(y, dtype=torch.bool)

    a = yhat[mask]
    b = y[mask]
    micro = rmse_torch(a, b) if a.numel() else rmse_torch(yhat.flatten(), y.flatten())

    # macro by node
    N = y.shape[1]
    vals = []
    for i in range(N):
        m = mask[:, i]
        if int(m.sum()) >= 2:
            vals.append(rmse_torch(yhat[:, i][m], y[:, i][m]))
    macro = float(np.mean(vals)) if vals else micro
    return micro, macro

# --------------------- training --------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.npz", help="dataset from gnn_prep.py")
    ap.add_argument("--encoder", choices=["sage", "routenet"], default="sage")
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--delta", type=float, default=10.0, help="Huber delta")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="nowcast_ckpt.pt")
    args = ap.parse_args()

    set_seed(args.seed)

    Z = load_npz(args.data)
    X = torch.from_numpy(Z["X"])              # [T, N, F]
    Y = torch.from_numpy(Z["Y"])              # [T, N]
    edges = torch.from_numpy(Z["edges"]).long()  # [2, E] for index ops
    train_idx = torch.from_numpy(Z["train_idx"]).long()
    val_idx   = torch.from_numpy(Z["val_idx"]).long()
    test_idx  = torch.from_numpy(Z["test_idx"]).long()
    Fin = X.shape[2]

    device = torch.device("cpu")

    enc = GNNEncoder(Fin, hid=args.hid, layers=args.layers, kind=args.encoder, dropout=args.dropout)
    head = NowcastHead(hid=args.hid)
    model = torch.nn.Module()
    model.enc = enc
    model.head = head
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    huber = nn.SmoothL1Loss(beta=args.delta)

    best_val = float("inf")
    best_state = None
    patience = 10
    bad = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        # -------- train BUSY-ONLY -------- #
        for t in train_idx.tolist():
            xt = X[t].to(device)   # [N, F]
            yt = Y[t].to(device)   # [N]
            h = model.enc(xt, edges.to(device))
            yhat = model.head(h)   # [N]

            mask = (yt > 0)
            if mask.any():
                loss = huber(yhat[mask], yt[mask])
            else:
                # fallback if no busy samples at this t
                loss = huber(yhat, yt)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item())
            steps += 1

        # -------- validate BUSY-ONLY -------- #
        model.eval()
        with torch.no_grad():
            yh_list, y_list = [], []
            for t in val_idx.tolist():
                xt = X[t].to(device)
                yy = Y[t].to(device)
                pred = model.head(model.enc(xt, edges.to(device)))
                mask = (yy > 0)
                if mask.any():
                    yh_list.append(pred[mask])
                    y_list.append(yy[mask])
            if yh_list:
                yh = torch.cat(yh_list)
                yy = torch.cat(y_list)
                val_rmse = rmse_torch(yh, yy)
            else:
                # if val has no busy points, evaluate on all
                yh_all, yy_all = [], []
                for t in val_idx.tolist():
                    xt = X[t].to(device)
                    yy = Y[t].to(device)
                    yh_all.append(model.head(model.enc(xt, edges.to(device))))
                    yy_all.append(yy)
                val_rmse = rmse_torch(torch.stack(yh_all).flatten(), torch.stack(yy_all).flatten())

        print(f"[nowcast][ep {ep:03d}] train_loss={total_loss/max(1,steps):.6f} val_busy_RMSE={val_rmse:.3f}")

        # early stopping
        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_state = {
                "enc": model.enc.state_dict(),
                "head": model.head.state_dict(),
                "meta": vars(args)
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {ep} (no val improvement for {patience} epochs).")
                break

    # save best
    if best_state is None:
        best_state = {
            "enc": model.enc.state_dict(),
            "head": model.head.state_dict(),
            "meta": vars(args)
        }
    torch.save(best_state, args.out)

    # -------- test report (micro/macro, all steps) -------- #
    model.enc.load_state_dict(best_state["enc"])
    model.head.load_state_dict(best_state["head"])
    model.eval()

    preds, truths = [], []
    with torch.no_grad():
        for t in test_idx.tolist():
            xt = X[t].to(device)
            yy = Y[t].to(device)
            pred = model.head(model.enc(xt, edges.to(device)))
            preds.append(pred.cpu())
            truths.append(yy.cpu())

    yhat = torch.stack(preds)  # [T_test, N]
    y    = torch.stack(truths)

    micro_all, macro_all = micro_macro(yhat, y, busy_only=False)
    micro_busy, macro_busy = micro_macro(yhat, y, busy_only=True)

    print(f"TEST nowcast  (all)  → micro {micro_all:.3f} | macro {macro_all:.3f}")
    print(f"TEST nowcast (busy)  → micro {micro_busy:.3f} | macro {macro_busy:.3f}")
    print(f"Saved checkpoint → {args.out}")

if __name__ == "__main__":
    main()
