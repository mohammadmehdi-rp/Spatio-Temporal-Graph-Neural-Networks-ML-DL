#!/usr/bin/env python3
"""Evaluate trained GNN checkpoints on dataset_v4.npz.

Drop-in replacement for the repo's `src/eval_gnn.py`.

Supported checkpoints
---------------------
- Nowcast: produced by `train_nowcast*.py` (keys: enc, head, meta)
- Lead-1:  produced by `train_lead1*.py`  (keys: enc, head, meta)

Usage examples
--------------
# auto-detect nowcast vs lead1 from ckpt meta
python3 src/eval_gnn.py --npz runs/dumbbell_seed3/dataset.npz --ckpt nowcast.pt

python3 src/eval_gnn.py --npz runs/dumbbell_seed3/dataset.npz --ckpt lead1.pt --report_busy

# explicit
python3 src/eval_gnn.py --task nowcast --npz runs/dumbbell_seed3/dataset.npz --ckpt nowcast.pt
python3 src/eval_gnn.py --task lead1   --npz runs/dumbbell_seed3/dataset.npz --ckpt lead1.pt

# ensemble (average predictions)
python3 src/eval_gnn.py --npz runs/dumbbell_seed3/dataset.npz \
  --ckpt ckpt1.pt --ckpt ckpt2.pt --ckpt ckpt3.pt --report_busy
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from models_gnn import GNNEncoder, NowcastHead, TCNHead, GRUHead


def load_npz(path: str) -> Dict[str, np.ndarray]:
    D = np.load(path, allow_pickle=True)
    return {k: D[k] for k in D.files}


def _as_str_list(arr) -> List[str]:
    out = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return out


def rmse_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return float(torch.sqrt(torch.mean((a - b) ** 2)).item())


def micro_macro(yhat: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    """Compute micro (flatten) and macro (mean per-node) RMSE."""
    micro = rmse_torch(yhat.flatten(), y.flatten())
    N = y.shape[1]
    vals = [rmse_torch(yhat[:, i], y[:, i]) for i in range(N)]
    macro = float(np.mean(vals)) if vals else micro
    return micro, macro


def micro_macro_masked(
    yhat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
) -> Tuple[float, float] | Tuple[None, None]:
    """Masked micro/macro RMSE. Mask is boolean with same shape as y."""
    if mask.sum().item() == 0:
        return None, None

    micro = rmse_torch(yhat[mask], y[mask])
    N = y.shape[1]
    vals = []
    for i in range(N):
        mi = mask[:, i]
        if mi.sum().item() == 0:
            continue
        vals.append(rmse_torch(yhat[:, i][mi], y[:, i][mi]))
    macro = float(np.mean(vals)) if vals else micro
    return micro, macro


def _load_checkpoint(path: str, device: torch.device) -> Dict:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "meta" not in ckpt or "enc" not in ckpt or "head" not in ckpt:
        raise ValueError(f"Checkpoint {path} does not look like a repo checkpoint (expected keys: enc/head/meta).")
    return ckpt


def build_nowcast_model(ckpt: Dict, Fin: int, device: torch.device):
    meta = ckpt.get("meta", {})
    hid = int(meta.get("hid", 64))
    layers = int(meta.get("layers", 2))
    encoder_kind = meta.get("encoder", meta.get("kind", "sage"))
    dropout = float(meta.get("dropout", 0.1))

    enc = GNNEncoder(Fin, hid=hid, layers=layers, kind=encoder_kind, dropout=dropout).to(device)
    head = NowcastHead(hid=hid).to(device)
    enc.load_state_dict(ckpt["enc"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    enc.eval()
    head.eval()
    return enc, head


def build_lead1_model(ckpt: Dict, Fin: int, device: torch.device):
    meta = ckpt.get("meta", {})
    hid = int(meta.get("hid", 64))
    layers = int(meta.get("layers", 2))
    encoder_kind = meta.get("encoder", meta.get("kind", "sage"))
    dropout = float(meta.get("dropout", 0.1))
    temporal = meta.get("temporal", "tcn")
    K = int(meta.get("K", 10))

    enc = GNNEncoder(Fin, hid=hid, layers=layers, kind=encoder_kind, dropout=dropout).to(device)
    head = GRUHead(hid=hid).to(device) if temporal == "gru" else TCNHead(hid=hid, K=K).to(device)

    enc.load_state_dict(ckpt["enc"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    enc.eval()
    head.eval()
    return enc, head, K, temporal


@torch.no_grad()
def eval_nowcast(
    models: List[Tuple[torch.nn.Module, torch.nn.Module]],
    X: torch.Tensor,
    Y: torch.Tensor,
    edges: torch.Tensor,
    test_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    preds, targs = [], []
    t0 = time.time()
    for t in test_idx.tolist():
        ph = None
        for enc, head in models:
            p = head(enc(X[t], edges))
            ph = p if ph is None else (ph + p)
        ph = ph / float(len(models))
        preds.append(ph)
        targs.append(Y[t])
    dt = time.time() - t0
    preds = torch.stack(preds, dim=0)
    targs = torch.stack(targs, dim=0)
    ms_per_step = 1000.0 * dt / max(1, len(test_idx))
    return preds, targs, ms_per_step


@torch.no_grad()
def eval_lead1(
    models: List[Tuple[torch.nn.Module, torch.nn.Module, int]],
    X: torch.Tensor,
    Y: torch.Tensor,
    edges: torch.Tensor,
    test_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    Ks = {K for (_, _, K) in models}
    if len(Ks) != 1:
        raise ValueError(f"All lead-1 checkpoints must share the same K. Got: {sorted(Ks)}")
    K = next(iter(Ks))

    T = X.shape[0]
    tes = set(test_idx.tolist())
    test_pos = [t for t in range(K, T - 1) if t in tes]  # t up to T-2 inclusive
    if not test_pos:
        raise ValueError(f"No valid lead-1 test positions found. Need some test_idx >= K and <= T-2. (K={K}, T={T})")

    enc_caches: List[torch.Tensor] = []
    for enc, _, _ in models:
        H_all = [enc(X[t], edges) for t in range(T)]
        enc_caches.append(torch.stack(H_all, dim=0))  # [T,N,H]

    preds, targs = [], []
    t0 = time.time()
    for t in test_pos:
        ph = None
        for (enc, head, _), H_all in zip(models, enc_caches):
            Hseq = H_all[t - K + 1 : t + 1]  # [K,N,H]
            p = head(Hseq)
            ph = p if ph is None else (ph + p)
        ph = ph / float(len(models))
        preds.append(ph)
        targs.append(Y[t + 1])
    dt = time.time() - t0

    preds = torch.stack(preds, dim=0)
    targs = torch.stack(targs, dim=0)
    ms_per_step = 1000.0 * dt / max(1, len(test_pos))
    return preds, targs, ms_per_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        "--npz",
        dest="data",
        required=True,
        help="Path to dataset_v4.npz (or dataset.npz). `--npz` kept for backward compatibility.",
    )
    ap.add_argument(
        "--task",
        choices=["auto", "nowcast", "lead1"],
        default="auto",
        help="Which task to evaluate. 'auto' infers from checkpoint meta.",
    )
    ap.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help="Checkpoint path. Repeat --ckpt multiple times to ensemble-average predictions.",
    )
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--busy_thr", type=float, default=0.0, help="Busy mask threshold for Y (default: >0)")
    ap.add_argument("--report_busy", action="store_true", help="Also report busy-only micro/macro RMSE")
    ap.add_argument("--save_preds", default="", help="Optional output .npz to save preds/targets/test_idx.")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    Z = load_npz(args.data)
    X = torch.from_numpy(Z["X"]).to(device).float()          # [T,N,F]
    Y = torch.from_numpy(Z["Y"]).to(device).float()          # [T,N]
    edges = torch.from_numpy(Z["edges"]).to(device).long()    # [2,E]
    test_idx = torch.from_numpy(Z["test_idx"]).to(device).long()

    nodes = _as_str_list(Z.get("nodes", np.array([f"n{i}" for i in range(Y.shape[1])])))
    T, N, Fin = X.shape

    ckpts = []
    for p in args.ckpt:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        ckpts.append(_load_checkpoint(p, device))

    if args.task == "auto":
        meta0 = ckpts[0].get("meta", {})
        args.task = "lead1" if (("temporal" in meta0) or ("K" in meta0)) else "nowcast"
        print(f"[INFO] Auto task inference: {args.task}")

    if args.task == "nowcast":
        models = [build_nowcast_model(c, Fin, device) for c in ckpts]
        preds, targs, ms = eval_nowcast(models, X, Y, edges, test_idx)
        micro, macro = micro_macro(preds, targs)
        print(f"Nowcast TEST: frames={preds.shape[0]} ports={N} | micro {micro:.3f} | macro {macro:.3f} | {ms:.2f} ms/step")

        if args.report_busy:
            mask = targs > float(args.busy_thr)
            bm, bM = micro_macro_masked(preds, targs, mask)
            if bm is None:
                print("Nowcast busy-only: no busy samples in TEST split (mask empty)")
            else:
                print(f"Nowcast busy-only: micro {bm:.3f} | macro {bM:.3f} | thr>{args.busy_thr}")

        if args.save_preds:
            np.savez_compressed(
                args.save_preds,
                preds=preds.detach().cpu().numpy().astype(np.float32),
                targets=targs.detach().cpu().numpy().astype(np.float32),
                test_idx=test_idx.detach().cpu().numpy().astype(np.int64),
                nodes=np.array(nodes),
                task=np.array(["nowcast"]),
            )
            print(f"[OK] saved predictions to {args.save_preds}")

    else:
        built = [build_lead1_model(c, Fin, device) for c in ckpts]
        models = [(enc, head, K) for (enc, head, K, _temporal) in built]
        Kset = {K for (_, _, K, _) in built}
        temporal_set = {t for (_, _, _, t) in built}
        print(f"Lead-1 eval: ensemble={len(models)} | K={next(iter(Kset))} | temporal={','.join(sorted(temporal_set))}")

        preds, targs, ms = eval_lead1(models, X, Y, edges, test_idx)
        micro, macro = micro_macro(preds, targs)
        print(f"Lead-1 TEST: frames={preds.shape[0]} ports={N} | micro {micro:.3f} | macro {macro:.3f} | {ms:.2f} ms/step")

        if args.report_busy:
            mask = targs > float(args.busy_thr)
            bm, bM = micro_macro_masked(preds, targs, mask)
            if bm is None:
                print("Lead-1 busy-only: no busy samples in TEST split (mask empty)")
            else:
                print(f"Lead-1 busy-only: micro {bm:.3f} | macro {bM:.3f} | thr>{args.busy_thr}")

        if args.save_preds:
            np.savez_compressed(
                args.save_preds,
                preds=preds.detach().cpu().numpy().astype(np.float32),
                targets=targs.detach().cpu().numpy().astype(np.float32),
                test_idx=test_idx.detach().cpu().numpy().astype(np.int64),
                nodes=np.array(nodes),
                task=np.array(["lead1"]),
            )
            print(f"[OK] saved predictions to {args.save_preds}")


if __name__ == "__main__":
    main()
