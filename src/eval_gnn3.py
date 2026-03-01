#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, time
from typing import Dict, List, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F

from models_gnn import GNNEncoder, NowcastHead, TCNHead, GRUHead


def load_npz(path: str) -> Dict[str, np.ndarray]:
    D = np.load(path, allow_pickle=True)
    return {k: D[k] for k in D.files}


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float(); b = b.float()
    return float(torch.sqrt(torch.mean((a - b) ** 2)).item())


def micro_macro(yhat: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    micro = rmse(yhat.flatten(), y.flatten())
    N = y.shape[1]
    per = [rmse(yhat[:, i], y[:, i]) for i in range(N)]
    macro = float(np.mean(per)) if per else micro
    return micro, macro


def micro_macro_masked(yhat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    if mask.sum().item() == 0:
        return None, None
    micro = rmse(yhat[mask], y[mask])
    N = y.shape[1]
    per = []
    for i in range(N):
        mi = mask[:, i]
        if mi.sum().item() == 0:
            continue
        per.append(rmse(yhat[:, i][mi], y[:, i][mi]))
    macro = float(np.mean(per)) if per else micro
    return micro, macro


def load_ckpt(path: str, device: torch.device) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "enc" not in ckpt or "head" not in ckpt or "meta" not in ckpt:
        raise ValueError(f"{path} does not look like a valid checkpoint (need enc/head/meta).")
    return ckpt


def postproc_from_meta(meta: Dict) -> Callable[[torch.Tensor], torch.Tensor]:
    # Match training-time output constraint
    if bool(meta.get("softplus", False)):
        return F.softplus
    return lambda x: x


def build_nowcast(ckpt: Dict, Fin: int, device: torch.device):
    meta = ckpt["meta"]
    enc = GNNEncoder(
        Fin, hid=int(meta["hid"]), layers=int(meta["layers"]),
        kind=meta["encoder"], dropout=float(meta["dropout"])
    ).to(device)
    head = NowcastHead(hid=int(meta["hid"])).to(device)
    enc.load_state_dict(ckpt["enc"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    enc.eval(); head.eval()
    post = postproc_from_meta(meta)
    return enc, head, post


def build_lead1(ckpt: Dict, Fin: int, device: torch.device):
    meta = ckpt["meta"]
    K = int(meta.get("K", 10))
    temporal = meta.get("temporal", "tcn")

    enc = GNNEncoder(
        Fin, hid=int(meta["hid"]), layers=int(meta["layers"]),
        kind=meta["encoder"], dropout=float(meta["dropout"])
    ).to(device)
    if temporal == "gru":
        head = GRUHead(hid=int(meta["hid"])).to(device)
    else:
        head = TCNHead(hid=int(meta["hid"]), K=K).to(device)

    enc.load_state_dict(ckpt["enc"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    enc.eval(); head.eval()
    post = postproc_from_meta(meta)
    return enc, head, post, K, temporal


@torch.no_grad()
def eval_nowcast(models, X, Y, edges, test_idx):
    preds, targs = [], []
    t0 = time.time()
    for t in test_idx.tolist():
        ph = None
        for enc, head, post in models:
            p = post(head(enc(X[t], edges)))
            ph = p if ph is None else (ph + p)
        ph = ph / float(len(models))
        preds.append(ph); targs.append(Y[t])
    dt = time.time() - t0
    preds = torch.stack(preds, 0)
    targs = torch.stack(targs, 0)
    return preds, targs, 1000.0 * dt / max(1, len(test_idx))


@torch.no_grad()
def eval_lead1(models, X, Y, edges, test_idx):
    Ks = {K for (_, _, _, K, _) in models}
    if len(Ks) != 1:
        raise ValueError(f"All lead-1 checkpoints must share the same K. Got {sorted(Ks)}")
    K = next(iter(Ks))

    T = X.shape[0]
    tes = set(test_idx.tolist())
    valid = [t for t in range(K, T - 1) if t in tes]  # predict y[t+1]
    if not valid:
        raise ValueError(f"No valid lead-1 test positions. Need some test_idx >= K and <= T-2. (K={K}, T={T})")

    # Cache encoder outputs per model for speed
    enc_cache = []
    for enc, _, _, _, _ in models:
        H_all = [enc(X[t], edges) for t in range(T)]
        enc_cache.append(torch.stack(H_all, 0))  # [T,N,H]

    preds, targs = [], []
    t0 = time.time()
    for t in valid:
        ph = None
        for (enc, head, post, K, _temporal), H_all in zip(models, enc_cache):
            Hseq = H_all[t - K + 1 : t + 1]   # [K,N,H]
            p = post(head(Hseq))
            ph = p if ph is None else (ph + p)
        ph = ph / float(len(models))
        preds.append(ph); targs.append(Y[t + 1])
    dt = time.time() - t0
    preds = torch.stack(preds, 0)
    targs = torch.stack(targs, 0)
    return preds, targs, 1000.0 * dt / max(1, len(valid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", "--data", dest="data", required=True)
    ap.add_argument("--task", choices=["nowcast", "lead1"], required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--report_busy", action="store_true")
    ap.add_argument("--busy_thr", type=float, default=0.0)
    args = ap.parse_args()

    device = torch.device("cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    Z = load_npz(args.data)
    X = torch.from_numpy(Z["X"]).float().to(device)      # [T,N,F]
    Y = torch.from_numpy(Z["Y"]).float().to(device)      # [T,N]
    edges = torch.from_numpy(Z["edges"]).long().to(device)
    test_idx = torch.from_numpy(Z["test_idx"]).long().to(device)

    Fin = X.shape[2]
    ckpts = [load_ckpt(p, device) for p in args.ckpt]

    if args.task == "nowcast":
        models = [build_nowcast(c, Fin, device) for c in ckpts]
        preds, targs, ms = eval_nowcast(models, X, Y, edges, test_idx)
        micro, macro = micro_macro(preds, targs)
        print(f"Nowcast TEST: frames={preds.shape[0]} ports={targs.shape[1]} | micro {micro:.3f} | macro {macro:.3f} | {ms:.2f} ms/step")
        if args.report_busy:
            mask = targs > args.busy_thr
            bm, bM = micro_macro_masked(preds, targs, mask)
            if bm is None:
                print("Nowcast busy-only: no busy samples in TEST split")
            else:
                print(f"Nowcast busy-only: micro {bm:.3f} | macro {bM:.3f} | thr>{args.busy_thr}")

            idle = (targs <= args.busy_thr)
            fp0 = float((preds[idle] > 0).float().mean().item()) if idle.any() else float("nan")
            fp1 = float((preds[idle] > 1).float().mean().item()) if idle.any() else float("nan")
            print(f"Nowcast idle FP: P(pred>0|idle)={fp0:.3f}  P(pred>1|idle)={fp1:.3f}")

    else:
        models = [build_lead1(c, Fin, device) for c in ckpts]
        K = models[0][3]; temporal = models[0][4]
        print(f"Lead-1 eval: ensemble={len(models)} | K={K} | temporal={temporal}")
        preds, targs, ms = eval_lead1(models, X, Y, edges, test_idx)
        micro, macro = micro_macro(preds, targs)
        print(f"Lead-1 TEST: frames={preds.shape[0]} ports={targs.shape[1]} | micro {micro:.3f} | macro {macro:.3f} | {ms:.2f} ms/step")
        if args.report_busy:
            mask = targs > args.busy_thr
            bm, bM = micro_macro_masked(preds, targs, mask)
            if bm is None:
                print("Lead-1 busy-only: no busy samples in TEST split")
            else:
                print(f"Lead-1 busy-only: micro {bm:.3f} | macro {bM:.3f} | thr>{args.busy_thr}")

if __name__ == "__main__":
    main()
