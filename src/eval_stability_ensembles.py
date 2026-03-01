#!/usr/bin/env python3
import argparse, glob, os, random
import numpy as np
import torch
from models_gnn import GNNEncoder, NowcastHead

def rmse(y, p):
    y = y.astype(np.float64); p = p.astype(np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))

def macro_rmse(y, p):
    # average RMSE per node (port)
    y = y.astype(np.float64); p = p.astype(np.float64)
    rmses = []
    for n in range(y.shape[1]):
        rmses.append(np.sqrt(np.mean((y[:, n] - p[:, n])**2)))
    return float(np.mean(rmses))

def load_npz(path):
    D = np.load(path, allow_pickle=True)
    return {k: D[k] for k in D.files}

def load_ckpt_into(enc, head, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "enc" in ckpt and "head" in ckpt:
        enc.load_state_dict(ckpt["enc"], strict=False)
        head.load_state_dict(ckpt["head"], strict=False)
        return
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    if isinstance(ckpt, dict):
        sd_enc, sd_head = {}, {}
        for k, v in ckpt.items():
            if k.startswith("enc."):  sd_enc[k[4:]] = v
            elif k.startswith("head."): sd_head[k[5:]] = v
        if sd_enc and sd_head:
            enc.load_state_dict(sd_enc, strict=False)
            head.load_state_dict(sd_head, strict=False)
            return

        enc.load_state_dict(ckpt, strict=False)
        head.load_state_dict(ckpt, strict=False)
        return

    raise RuntimeError("Unrecognized checkpoint format")

@torch.no_grad()
def predict(enc, head, X, edges, idx):
    enc.eval(); head.eval()
    out = []
    for t in idx.tolist():
        h = enc(X[t], edges)
        yhat = head(h).squeeze(-1)
        out.append(yhat.detach().cpu().numpy())
    return np.stack(out, axis=0)  # [Tt,N]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt_glob", default="ens_models/nowcast_sage_seed*.pt")
    ap.add_argument("--encoder", choices=["sage","routenet"], default="sage")
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_subsets", type=int, default=30, help="random subsets per ensemble size")
    ap.add_argument("--out", default="stability_ensemble_results.npz")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    Z = load_npz(args.npz)
    X = torch.from_numpy(Z["X"]).float()
    Y = Z["Y"].astype(np.float32)
    edges = torch.from_numpy(Z["edges"]).long()
    test_idx = torch.from_numpy(Z["test_idx"]).long()

    device = torch.device(args.device)
    X = X.to(device); edges = edges.to(device)

    T, N, F = X.shape

    ckpts = sorted(glob.glob(args.ckpt_glob))
    if len(ckpts) == 0:
        raise RuntimeError(f"No checkpoints matched: {args.ckpt_glob}")
    M = len(ckpts)
    print(f"[OK] Found {M} checkpoints")

    # Load all predictions (so ensembling is cheap)
    preds = []
    indiv_micro = []
    indiv_macro = []

    for pth in ckpts:
        enc = GNNEncoder(F, hid=args.hid, layers=args.layers, kind=args.encoder, dropout=args.dropout).to(device)
        head = NowcastHead(hid=args.hid).to(device)
        load_ckpt_into(enc, head, pth, device)
        p = predict(enc, head, X, edges, test_idx)  # [Tt,N]
        preds.append(p)
        y_test = Y[test_idx.numpy()]
        indiv_micro.append(rmse(y_test, p))
        indiv_macro.append(macro_rmse(y_test, p))
        print(f"  {os.path.basename(pth)}  micro={indiv_micro[-1]:.3f} macro={indiv_macro[-1]:.3f}")

    preds = np.stack(preds, axis=0)  # [M,Tt,N]
    y_test = Y[test_idx.numpy()]

    indiv_micro = np.array(indiv_micro)
    indiv_macro = np.array(indiv_macro)

    print("\n=== Stability (single model) ===")
    print(f"micro RMSE: mean={indiv_micro.mean():.3f} std={indiv_micro.std():.3f}  (n={M})")
    print(f"macro RMSE: mean={indiv_macro.mean():.3f} std={indiv_macro.std():.3f}  (n={M})")

    # Ensemble-size curve: random subsets of size k
    ks = list(range(1, M+1))
    ens_micro_mean, ens_micro_std = [], []
    ens_macro_mean, ens_macro_std = [], []

    for k in ks:
        micros, macros = [], []
        for _ in range(args.n_subsets):
            idxs = np.random.choice(M, size=k, replace=False)
            p_ens = preds[idxs].mean(axis=0)  # [Tt,N]
            micros.append(rmse(y_test, p_ens))
            macros.append(macro_rmse(y_test, p_ens))
        ens_micro_mean.append(np.mean(micros)); ens_micro_std.append(np.std(micros))
        ens_macro_mean.append(np.mean(macros)); ens_macro_std.append(np.std(macros))
        print(f"[ens k={k:2d}] micro {ens_micro_mean[-1]:.3f} ± {ens_micro_std[-1]:.3f} | macro {ens_macro_mean[-1]:.3f} ± {ens_macro_std[-1]:.3f}")

    np.savez(
        args.out,
        ckpts=np.array(ckpts, dtype=object),
        indiv_micro=indiv_micro,
        indiv_macro=indiv_macro,
        ks=np.array(ks),
        ens_micro_mean=np.array(ens_micro_mean),
        ens_micro_std=np.array(ens_micro_std),
        ens_macro_mean=np.array(ens_macro_mean),
        ens_macro_std=np.array(ens_macro_std),
    )
    print(f"\n[OK] Saved {args.out}")

if __name__ == "__main__":
    main()
