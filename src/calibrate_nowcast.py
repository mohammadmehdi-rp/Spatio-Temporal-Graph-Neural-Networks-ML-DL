#!/usr/bin/env python3
import argparse, math
import numpy as np
import torch

from models_gnn import GNNEncoder, NowcastHead

def rmse(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def softplus_np(x):
    # numerically stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def load_npz(path):
    D = np.load(path, allow_pickle=True)
    return {k: D[k] for k in D.files}

def load_ckpt_into(enc, head, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    def try_load(sd_enc, sd_head):
        enc.load_state_dict(sd_enc, strict=False)
        head.load_state_dict(sd_head, strict=False)

    # Case 1: dict with obvious keys
    if isinstance(ckpt, dict):
        if "enc" in ckpt and "head" in ckpt and isinstance(ckpt["enc"], dict):
            try_load(ckpt["enc"], ckpt["head"]); return
        if "enc_state_dict" in ckpt and "head_state_dict" in ckpt:
            try_load(ckpt["enc_state_dict"], ckpt["head_state_dict"]); return
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            ckpt = ckpt["state_dict"]  # fallthrough

    # Case 2: combined state_dict (maybe with prefixes)
    if isinstance(ckpt, dict):
        sd_enc, sd_head = {}, {}
        for k, v in ckpt.items():
            if k.startswith("enc."):
                sd_enc[k[len("enc."):]] = v
            elif k.startswith("head."):
                sd_head[k[len("head."):]] = v

        if sd_enc and sd_head:
            try_load(sd_enc, sd_head); return

        # Case 3: maybe saved as plain encoder+head keys without prefixing
        # Try to load everything into both (non-strict)
        enc.load_state_dict(ckpt, strict=False)
        head.load_state_dict(ckpt, strict=False)
        return

    raise RuntimeError("Unrecognized checkpoint format.")

def infer(enc, head, X, edges, idx):
    # returns [len(idx), N] predictions
    preds = []
    enc.eval(); head.eval()
    with torch.no_grad():
        for t in idx.tolist():
            h = enc(X[t], edges)
            yhat = head(h).squeeze(-1)  # [N]
            preds.append(yhat.detach().cpu().numpy())
    return np.stack(preds, axis=0)

def fit_scale(y_true, y_pred):
    # minimize ||y - a p||^2
    denom = np.sum(y_pred * y_pred) + 1e-12
    a = float(np.sum(y_true * y_pred) / denom)
    return a

def fit_soft_scale(y_true, y_pred):
    """
    SOFT + SCALE:
      p_soft = s * softplus((p - tau)/s)
      y_cal  = a * p_soft
    We grid-search (tau, s) on val and fit a in closed form.
    """
    p = y_pred.reshape(-1)
    y = y_true.reshape(-1)

    # robust ranges based on prediction distribution
    ppos = p[p > 0]
    if len(ppos) < 10:
        # fallback: no positive preds; return identity
        return dict(tau=0.0, s=1.0, a=1.0)

    p50, p75, p90 = np.percentile(ppos, [50, 75, 90])
    std = float(np.std(ppos) + 1e-6)

    taus = [0.0, 0.25*p50, 0.5*p50, p50, p75, p90]
    ss   = [0.25*std, 0.5*std, std, 2.0*std]

    best = (1e30, 0.0, 1.0, 1.0)  # (rmse, tau, s, a)

    for tau in taus:
        for s in ss:
            z = (p - tau) / (s + 1e-12)
            p_soft = (s * softplus_np(z)).reshape(-1)

            a = fit_scale(y, p_soft)
            y_hat = a * p_soft
            r = rmse(y, y_hat)
            if r < best[0]:
                best = (r, float(tau), float(s), float(a))

    return dict(tau=best[1], s=best[2], a=best[3], val_rmse=best[0])

def apply_soft_scale(p, params):
    tau, s, a = params["tau"], params["s"], params["a"]
    z = (p - tau) / (s + 1e-12)
    p_soft = s * softplus_np(z)
    return a * p_soft

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="dataset_sparse_*.npz")
    ap.add_argument("--ckpt", required=True, help="nowcast_*.pt")
    ap.add_argument("--encoder", choices=["sage","routenet"], default="sage")
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--y_cap", type=float, default=1000.0)
    ap.add_argument("--busy_thr", type=float, default=50.0, help="Threshold (pkts) to split idle/busy using Y_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="nowcast_calibration_out.npz")
    args = ap.parse_args()

    Z = load_npz(args.npz)
    X = torch.from_numpy(Z["X"]).float()
    Y = Z["Y"].astype(np.float32)              # raw pkts
    edges = torch.from_numpy(Z["edges"]).long()
    val_idx = torch.from_numpy(Z["val_idx"]).long()
    test_idx = torch.from_numpy(Z["test_idx"]).long()

    T, N, F = X.shape
    device = torch.device(args.device)
    X = X.to(device)
    edges = edges.to(device)

    enc = GNNEncoder(F, hid=args.hid, layers=args.layers, kind=args.encoder, dropout=args.dropout).to(device)
    head = NowcastHead(hid=args.hid).to(device)
    load_ckpt_into(enc, head, args.ckpt, device)

    # Inference
    pred_val  = infer(enc, head, X, edges, val_idx)   # [Tv,N]
    pred_test = infer(enc, head, X, edges, test_idx)  # [Tt,N]
    y_val  = Y[val_idx.numpy()]   # [Tv,N]
    y_test = Y[test_idx.numpy()]  # [Tt,N]

    # Auto-scale: decide whether model outputs are normalized (0..1) or pkts
    rmse_raw_val = rmse(y_val, pred_val)
    rmse_cap_val = rmse(y_val, pred_val * args.y_cap)
    use_cap = rmse_cap_val < rmse_raw_val

    if use_cap:
        pred_val  = pred_val  * args.y_cap
        pred_test = pred_test * args.y_cap

    # --- Calibration fit on VAL ---
    a_scale = fit_scale(y_val.reshape(-1), pred_val.reshape(-1))
    pred_test_scale = a_scale * pred_test

    soft_params = fit_soft_scale(y_val, pred_val)  # returns tau,s,a
    pred_test_soft = apply_soft_scale(pred_test, soft_params)

    # --- Metrics on TEST ---
    def split_metrics(y_true, y_pred):
        thr = args.busy_thr
        idle = (y_true < thr)
        busy = ~idle
        out = {}
        out["global_rmse"] = rmse(y_true, y_pred)
        out["idle_rmse"]   = rmse(y_true[idle], y_pred[idle]) if idle.any() else float("nan")
        out["busy_rmse"]   = rmse(y_true[busy], y_pred[busy]) if busy.any() else float("nan")
        # idle "false positives": predicting > thr when true is idle
        out["idle_fp_rate"] = float(np.mean((y_pred[idle] > thr))) if idle.any() else float("nan")
        out["idle_mean_pred"] = float(np.mean(y_pred[idle])) if idle.any() else float("nan")
        return out

    m_raw   = split_metrics(y_test, pred_test)
    m_scale = split_metrics(y_test, pred_test_scale)
    m_soft  = split_metrics(y_test, pred_test_soft)

    # Print summary (copy-paste friendly)
    print("=== Nowcast calibration (TEST) ===")
    print(f"auto_scale_used_y_cap={use_cap}  (y_cap={args.y_cap})")
    print(f"busy_threshold={args.busy_thr} pkts\n")

    def pr(name, m):
        print(f"[{name}] global_RMSE={m['global_rmse']:.3f} | idle_RMSE={m['idle_rmse']:.3f} | busy_RMSE={m['busy_rmse']:.3f} "
              f"| idle_FP_rate={m['idle_fp_rate']:.3f} | idle_mean_pred={m['idle_mean_pred']:.3f}")

    pr("RAW",   m_raw)
    pr("SCALE", m_scale)
    pr("SOFT+SCALE", m_soft)

    print("\nCalibration parameters:")
    print(f"  SCALE: a={a_scale:.6f}")
    print(f"  SOFT+SCALE: tau={soft_params.get('tau',0.0):.6f}  s={soft_params.get('s',1.0):.6f}  a={soft_params.get('a',1.0):.6f}  (val_rmse={soft_params.get('val_rmse',float('nan')):.3f})")

    # Save outputs
    np.savez(
        args.out,
        y_test=y_test,
        pred_raw=pred_test,
        pred_scale=pred_test_scale,
        pred_soft=pred_test_soft,
        use_cap=np.array([use_cap]),
        y_cap=np.array([args.y_cap]),
        busy_thr=np.array([args.busy_thr]),
        a_scale=np.array([a_scale]),
        tau=np.array([soft_params.get("tau",0.0)]),
        s=np.array([soft_params.get("s",1.0)]),
        a_soft=np.array([soft_params.get("a",1.0)]),
    )
    print(f"\n[OK] Saved: {args.out}")

if __name__ == "__main__":
    main()
