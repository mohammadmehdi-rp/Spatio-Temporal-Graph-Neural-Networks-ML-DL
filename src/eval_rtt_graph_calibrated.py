#!/usr/bin/env python3
"""eval_rtt_graph.py

Evaluate the path-conditioned RTT graph model (Option-2) for horizon H.

This evaluator matches the training semantics in train_nowcast_rtt_graph.py:

  delta_hat[t, k] = Head( Encoder(X[t]) )
  y_hat_tfm[t, k] = y_base_tfm[t, k] + delta_hat[t, k]

where:
  - H == 0 (nowcast): y_base_tfm[t] = y_tfm[t-1]  (t >= 1)
  - H  > 0 (lead-H):  y_base_tfm[t] = y_tfm[t]

and y_tfm is the forward-transform of y_true_ms.

Multi-probe RTT is supported (delay_series_ms shape [T,K]). If checkpoints
store meta['probe_subset'] (e.g., from PROBE_SUBSET), we apply the same
subset (and ordering) to the dataset before computing metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from models_gnn import GNNEncoder


# ----------------------------
# Debug helpers
# ----------------------------


def _safe_corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation for flattened arrays; returns nan if undefined."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------
# IO + Metrics
# ----------------------------


def load_npz(p: str) -> Dict[str, np.ndarray]:
    Z = np.load(p, allow_pickle=True)
    return {k: Z[k] for k in Z.files}


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_alpha_per_probe(pred: np.ndarray, base: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Least-squares alpha per probe for shrinkage-to-base; clipped to [0,1]."""
    pred = np.asarray(pred, dtype=float)
    base = np.asarray(base, dtype=float)
    target = np.asarray(target, dtype=float)

    if pred.ndim == 1:
        pred = pred[:, None]
        base = base[:, None]
        target = target[:, None]

    d = pred - base
    t = target - base
    num = np.sum(d * t, axis=0)
    den = np.sum(d * d, axis=0) + eps
    a = num / den
    return np.clip(a, 0.0, 1.0).astype(np.float64)


def get_delay_matrix(Z: Dict[str, np.ndarray]) -> np.ndarray:
    """Return y_true as [T,K] float64."""
    if "delay_series_ms" in Z:
        y = Z["delay_series_ms"].astype(np.float64)
        if y.ndim == 1:
            return y[:, None]
        if y.ndim == 2:
            return y
        raise SystemExit(f"Unsupported delay_series_ms ndim={y.ndim}")

    if "Y" in Z and "label_names" in Z:
        names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in Z["label_names"]]
        k = names.index("rtt_ms") if "rtt_ms" in names else (len(names) - 1)
        y = Z["Y"].astype(np.float64)[:, :, k].mean(axis=1)
        return y[:, None]

    raise SystemExit("Dataset must contain delay_series_ms or (Y + label_names).")


def get_split_indices(Z: Dict[str, np.ndarray], T: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if all(k in Z for k in ("train_idx", "val_idx", "test_idx")):
        return (Z["train_idx"].astype(int), Z["val_idx"].astype(int), Z["test_idx"].astype(int))
    cut1 = int(0.70 * T)
    cut2 = int(0.85 * T)
    return (np.arange(0, cut1), np.arange(cut1, cut2), np.arange(cut2, T))


def valid_base_indices(idx: np.ndarray, H: int, T: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return base_idx and target_idx aligned for horizon H."""
    idx = np.asarray(idx, dtype=int)
    idx = idx[(idx >= 0) & (idx < T)]
    if H == 0:
        base = idx[idx >= 1]
        targ = base
    else:
        base = idx[(idx + H) < T]
        targ = base + H
    return base, targ


# ----------------------------
# Probe IDs + subset selection
# ----------------------------


def decode_str_list(arr: np.ndarray) -> List[str]:
    return [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]


def dataset_probe_ids(Z: Dict[str, np.ndarray], K: int) -> List[str]:
    if "probe_ids" in Z:
        ids = decode_str_list(Z["probe_ids"])
        if len(ids) == K:
            return ids
    if "probe_src" in Z and "probe_dst" in Z:
        ps = Z["probe_src"].astype(int).tolist()
        pd = Z["probe_dst"].astype(int).tolist()
        ids = [f"{s}-{d}" for (s, d) in zip(ps, pd)]
        if len(ids) == K:
            return ids
    return [f"probe{k}" for k in range(K)]


def normalize_id_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(v) for v in x if str(v).strip()]
    s = str(x).strip()
    if not s:
        return []
    # If stored as "a-b,c-d" in older ckpts.
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def apply_subset(y: np.ndarray, ids_full: List[str], subset_ids: List[str]) -> Tuple[np.ndarray, List[str], List[int]]:
    id2k = {pid: i for i, pid in enumerate(ids_full)}
    keep: List[int] = []
    for pid in subset_ids:
        if pid not in id2k:
            raise SystemExit(f"Requested probe {pid} not found in dataset probe_ids")
        keep.append(int(id2k[pid]))
    return y[:, keep], [ids_full[i] for i in keep], keep


# ----------------------------
# Transform helpers
# ----------------------------


def forward_transform(y_ms: np.ndarray, kind: str, prop_rtt_ms: np.ndarray) -> np.ndarray:
    y = np.asarray(y_ms, dtype=float)
    prop = np.asarray(prop_rtt_ms, dtype=float).reshape(1, -1)
    if kind == "raw":
        return y
    if kind == "residual_log1p":
        res = np.maximum(0.0, y - prop)
        return np.log1p(res)
    raise ValueError(f"Unknown transform: {kind}")


def inverse_transform(y_hat: np.ndarray, kind: str, prop_rtt_ms: np.ndarray) -> np.ndarray:
    y = np.asarray(y_hat, dtype=float)
    prop = np.asarray(prop_rtt_ms, dtype=float).reshape(1, -1)
    if kind == "raw":
        return y
    if kind == "residual_log1p":
        res = np.expm1(y)
        res = np.maximum(res, 0.0)
        return prop + res
    raise ValueError(f"Unknown transform: {kind}")


def compute_prop_row(Z: Dict[str, np.ndarray], K: int, keep_idx: Optional[List[int]], y_true: np.ndarray, train_base: np.ndarray) -> np.ndarray:
    """prop_rtt per probe: prefer dataset field; else estimate from RTT p5 on train."""
    if "probe_prop_rtt_ms" in Z:
        prop = np.asarray(Z["probe_prop_rtt_ms"], dtype=float).reshape(-1)
        if keep_idx is not None and len(keep_idx) > 0:
            if prop.size >= max(keep_idx) + 1:
                prop = prop[keep_idx]
        if prop.size == K:
            return prop.astype(np.float64)
    # estimate from train targets at aligned indices
    t = y_true[train_base]
    return np.percentile(t, 5.0, axis=0).astype(np.float64)


# ----------------------------
# Path-conditioned head (matches training)
# ----------------------------


def parse_sw_port_iface(n: str) -> Tuple[int, int]:
    """Parse interface name like 's3-eth2' -> (3,2)."""
    import re

    m = re.match(r"^s(\d+)-eth(\d+)$", str(n))
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def shortest_path_nodes_km_nsfnet(src: int, dst: int) -> Tuple[List[int], float]:
    """Return (path_nodes, total_km) for NSFNET shortest path."""
    from simple_nsfnet import NSFNET_LINKS_KM
    import heapq

    n = 14
    adj = {i: [] for i in range(n)}
    for u, v, km in NSFNET_LINKS_KM:
        adj[int(u)].append((int(v), float(km)))
        adj[int(v)].append((int(u), float(km)))

    dist = {i: float("inf") for i in range(n)}
    prev = {i: None for i in range(n)}
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == dst:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if not np.isfinite(dist[dst]):
        raise RuntimeError(f"NSFNET disconnected? no path {src}->{dst}")

    # reconstruct
    path = [dst]
    cur = dst
    while cur != src:
        p = prev[cur]
        if p is None:
            raise RuntimeError(f"Failed to reconstruct path {src}->{dst}")
        path.append(int(p))
        cur = int(p)
    path.reverse()
    return path, float(dist[dst])


def path_iface_indices_from_npz(nodes: List[str], edges_np: np.ndarray, src_sw: int, dst_sw: int) -> List[int]:
    """Compute interface-node indices along the NSFNET shortest path for pooling."""
    name2idx = {str(n): i for i, n in enumerate(nodes)}
    sw_of = np.array([parse_sw_port_iface(n)[0] for n in nodes], dtype=int)

    inter: Dict[Tuple[int, int], Tuple[str, str]] = {}
    src_e = edges_np[0].astype(int)
    dst_e = edges_np[1].astype(int)
    for a, b in zip(src_e, dst_e):
        sa, sb = int(sw_of[a]), int(sw_of[b])
        if sa < 0 or sb < 0 or sa == sb:
            continue
        key = (min(sa, sb), max(sa, sb))
        if key in inter:
            continue
        na, nb = str(nodes[a]), str(nodes[b])
        if sa < sb:
            inter[key] = (na, nb)
        else:
            inter[key] = (nb, na)

    sw_path, _km = shortest_path_nodes_km_nsfnet(src_sw, dst_sw)
    pick_names = set()
    pick_names.add(f"s{src_sw}-eth1")
    pick_names.add(f"s{dst_sw}-eth1")
    for u, v in zip(sw_path[:-1], sw_path[1:]):
        key = (min(int(u), int(v)), max(int(u), int(v)))
        if key not in inter:
            raise RuntimeError(f"Could not find inter-switch interface pair for hop {u}<->{v} in NPZ edges")
        a_name, b_name = inter[key]
        pick_names.add(a_name)
        pick_names.add(b_name)

    idxs = [name2idx[n] for n in pick_names if n in name2idx]
    idxs = sorted(set(int(i) for i in idxs))
    if len(idxs) < 2:
        raise RuntimeError(f"Path-conditioned pooling set too small (len={len(idxs)}).")
    return idxs


class _IndexBuffer(torch.nn.Module):
    def __init__(self, idxs: List[int]):
        super().__init__()
        self.register_buffer("idx", torch.tensor(list(idxs), dtype=torch.long))


class MultiProbeDelayHead(torch.nn.Module):
    """Shared MLP applied to mean-pooled path embeddings for each probe.

    forward(h_nodes) -> delta_hat[K] (in transform space)
    """

    def __init__(self, hid: int, path_idxs: List[List[int]], dropout: float = 0.1):
        super().__init__()
        self.path_idxs = torch.nn.ModuleList([_IndexBuffer(idxs) for idxs in path_idxs])
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hid, hid),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hid, 1),
        )

    def forward(self, h_nodes: torch.Tensor) -> torch.Tensor:
        pooled = []
        for b in self.path_idxs:
            g = h_nodes.index_select(0, b.idx).mean(dim=0)
            pooled.append(g)
        G = torch.stack(pooled, dim=0)  # [K,hid]
        return self.net(G).squeeze(-1)  # [K]


def probe_pairs_from_ids(ids: List[str]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for pid in ids:
        if "-" not in pid:
            raise SystemExit(f"Invalid probe id '{pid}' (expected a-b)")
        a, b = pid.split("-", 1)
        out.append((int(a), int(b)))
    return out


def predict_delta_ckpt(
    Z: Dict[str, np.ndarray],
    ckpt_path: str,
    device: torch.device,
    debug: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Return delta_hat in transform space as [T,Kpred], plus meta."""
    ck = torch.load(ckpt_path, map_location=device)
    meta = ck.get("meta", {})

    X = torch.from_numpy(Z["X"]).float().to(device)  # [T,N,F]
    edges = torch.from_numpy(Z["edges"]).long().to(device)
    T, _N, Fin = X.shape

    enc_hid = int(meta.get("hid", 96))
    enc_layers = int(meta.get("layers", 3))
    enc_kind = str(meta.get("encoder", "sage"))
    enc_dropout = float(meta.get("dropout", 0.2))

    ids = normalize_id_list(meta.get("probe_subset"))
    if not ids:
        ids = normalize_id_list(meta.get("probes")) or normalize_id_list(meta.get("probe_ids"))
    if not ids:
        # fall back to single-probe (ping_src/ping_dst)
        ids = [f"{int(meta.get('ping_src', 0))}-{int(meta.get('ping_dst', 13))}"]

    probes = probe_pairs_from_ids(ids)
    Kpred = len(probes)

    if "nodes" not in Z:
        raise SystemExit("dataset npz must contain 'nodes' for path-conditioned RTT evaluation")
    nodes = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in Z["nodes"]]
    path_idxs = [path_iface_indices_from_npz(nodes, Z["edges"], s, d) for (s, d) in probes]

    enc = GNNEncoder(Fin, hid=enc_hid, layers=enc_layers, kind=enc_kind, dropout=enc_dropout).to(device)
    head = MultiProbeDelayHead(hid=enc_hid, path_idxs=path_idxs, dropout=max(0.0, enc_dropout * 0.5)).to(device)

    enc.load_state_dict(ck["enc"])  # strict by default
    head_sd = ck.get("head")
    if not isinstance(head_sd, dict):
        raise SystemExit(f"Checkpoint missing 'head' state dict: {ckpt_path}")
    # Recompute path indices at eval-time; ignore stored buffers.
    head_sd = {k: v for k, v in head_sd.items() if not k.startswith("path_idxs.")}
    ik = head.load_state_dict(head_sd, strict=False)
    if debug and (len(ik.missing_keys) > 0 or len(ik.unexpected_keys) > 0):
        print(
            f"[load] {Path(ckpt_path).name}: head missing={ik.missing_keys} unexpected={ik.unexpected_keys}"
        )

    enc.eval(); head.eval()

    out = np.zeros((T, Kpred), dtype=np.float64)
    with torch.no_grad():
        for t in range(T):
            h = enc(X[t], edges)
            d = head(h).detach().cpu().numpy().astype(np.float64).reshape(-1)
            if d.size != Kpred:
                raise RuntimeError(f"delta size mismatch at t={t}: got {d.size} expected {Kpred}")
            out[t, :] = d

    return out, meta


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="dataset_multi.npz")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--out_csv", default="rtt_graph_metrics.csv")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    ap.add_argument("--clip", action="store_true", help="Clip preds to a high percentile of VAL targets.")
    ap.add_argument("--clip_q", type=float, default=99.9)
    ap.add_argument("--clip_mult", type=float, default=1.2)

    ap.add_argument("--shrink", action="store_true", help="Per-probe shrink-to-persistence using VAL.")

    ap.add_argument("--debug", action="store_true", help="Print debug diagnostics for transforms/deltas/loading.")
    ap.add_argument("--calibrate_alpha", action="store_true", help="Calibrate a shrinkage alpha on VAL (and fallback to persistence if worse).")
    ap.add_argument("--alpha_max", type=float, default=1.0, help="Max alpha when calibrating (default 1.0).")


    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    Z = load_npz(args.npz)

    y_true_full = np.maximum(get_delay_matrix(Z), 0.0)  # [T,Kfull]
    T, Kfull = y_true_full.shape
    ds_ids_full = dataset_probe_ids(Z, Kfull)

    # Determine probe subset (must be consistent across ckpts if present)
    subset_ids: Optional[List[str]] = None
    for p in args.ckpts:
        ck = torch.load(p, map_location="cpu")
        meta = ck.get("meta", {})
        ps = normalize_id_list(meta.get("probe_subset"))
        if ps:
            if subset_ids is None:
                subset_ids = ps
            elif subset_ids != ps:
                raise SystemExit(
                    "Inconsistent probe_subset across ckpts. "
                    f"First={subset_ids} vs {Path(p).name}={ps}"
                )

    keep_idx: Optional[List[int]] = None
    if subset_ids:
        y_true, _ids_sel, keep_idx = apply_subset(y_true_full, ds_ids_full, subset_ids)
        ds_ids = subset_ids
    else:
        y_true = y_true_full
        ds_ids = ds_ids_full

    T, K = y_true.shape
    train_idx, val_idx, test_idx = get_split_indices(Z, T)

    # Base/target alignment for horizon
    train_base, train_targ = valid_base_indices(train_idx, args.horizon, T)
    val_base, val_targ = valid_base_indices(val_idx, args.horizon, T)
    test_base, test_targ = valid_base_indices(test_idx, args.horizon, T)

    # Targets
    y_val_t = y_true[val_targ]
    y_test_t = y_true[test_targ]

    # Persistence baseline
    if args.horizon == 0:
        y_val_p = y_true[val_base - 1]
        y_test_p = y_true[test_base - 1]
    else:
        y_val_p = y_true[val_base]
        y_test_p = y_true[test_base]

    # Zero baseline
    y_test_z = np.zeros_like(y_test_t)

    rmse_persist_raw = rmse(y_test_p, y_test_t)
    if args.debug:
        print(f"[data] npz={args.npz} T={T} K={K} horizon={args.horizon} device={device.type}")
        print(f"[data] probe_ids={ds_ids}")
        print(f"[split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
        print(f"[align] base/target sizes: train={len(train_base)} val={len(val_base)} test={len(test_base)}")
        print(f"[baseline] persistence_rmse_raw={rmse_persist_raw:.6f}")

    # Propagation baseline for qdelay
    prop_row = compute_prop_row(Z, K=K, keep_idx=keep_idx, y_true=y_true, train_base=train_base)  # [K]

    def qdelay(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x - prop_row[None, :])

    rows: List[Dict] = []
    rows.append({
        "model": "baseline_zero",
        "horizon": int(args.horizon),
        "rmse_rtt_ms": rmse(y_test_z, y_test_t),
        "rmse_qdelay_ms": rmse(qdelay(y_test_z), qdelay(y_test_t)),
    })
    rows.append({
        "model": "baseline_persistence",
        "horizon": int(args.horizon),
        "rmse_rtt_ms": rmse(y_test_p, y_test_t),
        "rmse_qdelay_ms": rmse(qdelay(y_test_p), qdelay(y_test_t)),
    })

    # For clipping: per-probe max from validation targets
    if args.clip:
        clip_hi = np.percentile(y_val_t, args.clip_q, axis=0)
        clip_max = np.maximum(1.0, clip_hi * float(args.clip_mult))
    else:
        clip_max = None

    preds_all: List[np.ndarray] = []
    preds_all_sh: List[np.ndarray] = []
    val_rmses: List[float] = []

    for p in args.ckpts:
        delta_tr, meta = predict_delta_ckpt(Z, p, device=device, debug=args.debug)  # [T,Kpred] delta in tfm space

        # If dataset was subset, ckpt must already be trained/eval'd on that subset.
        if delta_tr.shape[1] != K:
            raise SystemExit(f"K mismatch: dataset K={K} but ckpt delta has K={delta_tr.shape[1]}: {p}")

        kind = str(meta.get("transform", "raw"))

        if args.debug:
            mh = meta.get("horizon", None)
            mo = meta.get("out_dim", None)
            print(f"\n[ckpt] {Path(p).name} meta.horizon={mh} meta.out_dim={mo} transform={kind}")

        # Prop for forward/inverse transform: prefer meta vector; else dataset vector
        prop_meta = meta.get("prop_rtt_ms", None)
        if isinstance(prop_meta, (list, tuple, np.ndarray)):
            prop_inv = np.asarray(prop_meta, dtype=float).reshape(-1)
            if prop_inv.size != K:
                prop_inv = prop_row
        else:
            prop_inv = prop_row

        # Build y_hat in transform space at the *base index* t
        y_tfm = forward_transform(y_true, kind=kind, prop_rtt_ms=prop_inv)

        base_tfm = np.zeros_like(y_tfm)
        if int(args.horizon) == 0:
            base_tfm[0, :] = y_tfm[0, :]
            base_tfm[1:, :] = y_tfm[:-1, :]
        else:
            base_tfm[:, :] = y_tfm

        if args.debug:
            # Sanity: persistence computed through the same transform/inverse path (delta=0)
            y_persist_modelpath = inverse_transform(base_tfm, kind=kind, prop_rtt_ms=prop_inv)
            y_persist_modelpath = np.maximum(y_persist_modelpath, 0.0)
            if clip_max is not None:
                y_persist_modelpath = np.clip(y_persist_modelpath, 0.0, clip_max[None, :])
            r_persist_modelpath = rmse(y_persist_modelpath[test_base], y_test_t)
            frac_below_prop = float(np.mean(y_true[test_base] < prop_inv[None, :]))
            print(
                f"[sanity] persist_rmse_raw={rmse_persist_raw:.6f} persist_rmse_modelpath={r_persist_modelpath:.6f} "
                f"(frac y<prop on TEST base={frac_below_prop:.3%})"
            )

            # Delta diagnostics in transform space
            if int(args.horizon) == 0:
                true_delta_test = y_tfm[test_base] - y_tfm[test_base - 1]
                true_delta_val = y_tfm[val_base] - y_tfm[val_base - 1]
            else:
                true_delta_test = y_tfm[test_targ] - y_tfm[test_base]
                true_delta_val = y_tfm[val_targ] - y_tfm[val_base]

            pred_delta_test = delta_tr[test_base]
            pred_delta_val = delta_tr[val_base]

            def _rms(x: np.ndarray) -> float:
                return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))

            print(
                f"[delta TEST tfm] rms_true={_rms(true_delta_test):.6f} rms_pred={_rms(pred_delta_test):.6f} "
                f"corr={_safe_corr_flat(true_delta_test, pred_delta_test):.4f} "
                f"pred_min={float(np.min(pred_delta_test)):.4f} pred_max={float(np.max(pred_delta_test)):.4f}"
            )
            print(
                f"[delta VAL  tfm] rms_true={_rms(true_delta_val):.6f} rms_pred={_rms(pred_delta_val):.6f} "
                f"corr={_safe_corr_flat(true_delta_val, pred_delta_val):.4f}"
            )
            # Per-probe RMS (useful when K is small)
            if K <= 12:
                for kk in range(K):
                    rt = _rms(true_delta_test[:, kk])
                    rp = _rms(pred_delta_test[:, kk])
                    cc = _safe_corr_flat(true_delta_test[:, kk], pred_delta_test[:, kk])
                    print(f"  [probe {ds_ids[kk]}] rms_true={rt:.6f} rms_pred={rp:.6f} corr={cc:.4f}")

        # Optional: calibrate shrinkage alpha on VAL in transform-space (helps prevent harming strong persistence)
        alpha = 1.0
        if args.calibrate_alpha:
            d_pred = delta_tr[val_base].reshape(-1)
            d_true = (y_tfm[val_targ] - base_tfm[val_base]).reshape(-1)
            denom = float(np.dot(d_pred, d_pred))
            if denom < 1e-12:
                alpha = 0.0
            else:
                alpha = float(np.dot(d_pred, d_true) / denom)
                alpha = float(np.clip(alpha, 0.0, float(args.alpha_max)))

            # Compare VAL RMSE in ms; if not better than persistence, fall back to alpha=0 (pure persistence)
            y_hat_val_tfm = base_tfm[val_base] + alpha * delta_tr[val_base]
            y_hat_val_ms = inverse_transform(y_hat_val_tfm, kind=kind, prop_rtt_ms=prop_inv)
            y_hat_val_ms = np.maximum(y_hat_val_ms, 0.0)
            val_rmse_ms = rmse(y_hat_val_ms, y_val_t)
            persist_val_rmse_ms = rmse(y_val_p, y_val_t)

            if val_rmse_ms >= persist_val_rmse_ms - 1e-12:
                alpha = 0.0

            if args.debug:
                print(f"[calib] alpha={alpha:.6f} (alpha_max={float(args.alpha_max):.3f}) val_rmse_ms={val_rmse_ms:.6f} persist_val_rmse_ms={persist_val_rmse_ms:.6f}")

        y_hat_tfm = base_tfm + alpha * delta_tr
        y_hat_ms = inverse_transform(y_hat_tfm, kind=kind, prop_rtt_ms=prop_inv)

        y_hat_ms = np.maximum(y_hat_ms, 0.0)

        # Clip
        if clip_max is not None:
            y_hat_ms = np.clip(y_hat_ms, 0.0, clip_max[None, :])

        # Evaluate on TEST base indices
        y_pred_test = y_hat_ms[test_base]
        r_rtt = rmse(y_pred_test, y_test_t)
        r_q = rmse(qdelay(y_pred_test), qdelay(y_test_t))

        rows.append({
            "model": f"rtt_graph_{Path(p).stem}",
            "horizon": int(args.horizon),
            "rmse_rtt_ms": float(r_rtt),
            "rmse_qdelay_ms": float(r_q),
        })

        preds_all.append(y_hat_ms)

        # VAL RMSE for weighting and shrink
        y_pred_val = y_hat_ms[val_base]
        vr = rmse(y_pred_val, y_val_t)
        val_rmses.append(float(vr))

        if args.shrink:
            alpha = fit_alpha_per_probe(y_pred_val, y_val_p, y_val_t)  # [K]
            y_pred_test_sh = y_test_p + (y_pred_test - y_test_p) * alpha[None, :]
            r_rtt_sh = rmse(y_pred_test_sh, y_test_t)
            r_q_sh = rmse(qdelay(y_pred_test_sh), qdelay(y_test_t))
            rows.append({
                "model": f"rtt_graph_{Path(p).stem}_shrunk",
                "horizon": int(args.horizon),
                "rmse_rtt_ms": float(r_rtt_sh),
                "rmse_qdelay_ms": float(r_q_sh),
            })
            y_sh = y_hat_ms.copy()
            y_sh[test_base] = y_pred_test_sh
            preds_all_sh.append(y_sh)

    # Ensembles
    if preds_all:
        ens = np.mean(np.stack(preds_all, axis=0), axis=0)
        y_pred_test = ens[test_base]
        rows.append({
            "model": "rtt_graph_ensemble",
            "horizon": int(args.horizon),
            "rmse_rtt_ms": float(rmse(y_pred_test, y_test_t)),
            "rmse_qdelay_ms": float(rmse(qdelay(y_pred_test), qdelay(y_test_t))),
        })

        # Weighted ensemble by inverse val RMSE^2
        if len(val_rmses) == len(preds_all):
            w = 1.0 / (np.square(np.asarray(val_rmses, dtype=float)) + 1e-12)
            w = w / np.sum(w)
            ensw = np.tensordot(w, np.stack(preds_all, axis=0), axes=(0, 0))
            y_pred_test = ensw[test_base]
            rows.append({
                "model": "rtt_graph_ensemble_w",
                "horizon": int(args.horizon),
                "rmse_rtt_ms": float(rmse(y_pred_test, y_test_t)),
                "rmse_qdelay_ms": float(rmse(qdelay(y_pred_test), qdelay(y_test_t))),
            })

    if args.shrink and preds_all_sh:
        ens_sh = np.mean(np.stack(preds_all_sh, axis=0), axis=0)
        y_pred_test = ens_sh[test_base]
        rows.append({
            "model": "rtt_graph_ensemble_shrunk",
            "horizon": int(args.horizon),
            "rmse_rtt_ms": float(rmse(y_pred_test, y_test_t)),
            "rmse_qdelay_ms": float(rmse(qdelay(y_pred_test), qdelay(y_test_t))),
        })

    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"OK: wrote {out}")
    if subset_ids:
        print(f"Probe subset: {','.join(subset_ids)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
