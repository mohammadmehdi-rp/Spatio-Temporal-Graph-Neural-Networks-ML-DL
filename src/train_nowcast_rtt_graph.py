#!/usr/bin/env python3
"""train_nowcast_rtt_graph.py

Path-conditioned NOWCAST training for end-to-end delay (RTT / qdelay).

Motivation
----------
In the multi-target edge-state dataset, RTT/qdelay is a *path-level* signal
that becomes replicated across all interfaces after merging by timestamp.
Learning it as an edge/node label is therefore a modelling mismatch.

This script trains a dedicated *path-conditioned* head:
  X[t, node, feat]  --(GNN encoder)--> h[t, node, hid]
  pool(h[t] over RTT path ports) --(MLP)--> \Delta y_hat[t]
  y_hat[t] = y_true[t-1] + \Delta y_hat[t]   (persistence-corrected nowcast)

Default target transform (recommended):
  y_residual = max(0, y_raw_ms - prop_rtt_ms)
  y_train = log1p(y_residual)
We evaluate in milliseconds after inverting the transform.

Works with dataset_multi.npz produced by gnn_prep_multitarget.py.
If delay_series_ms is missing, we fall back to mean(Y[...,delay_k]).
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models_gnn import GNNEncoder


def load_npz(p: str) -> Dict[str, np.ndarray]:
    Z = np.load(p, allow_pickle=True)
    return {k: Z[k] for k in Z.files}


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def parse_probe_list(s: str, n_hosts: int = 14) -> List[Tuple[int, int]]:
    """Parse probes like '0-13,1-12,2-11' into list[(src,dst)]."""
    out: List[Tuple[int, int]] = []
    if not s:
        return out
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' not in part:
            raise ValueError(f"Invalid probe '{part}' (expected a-b)")
        a, b = part.split('-', 1)
        src, dst = int(a), int(b)
        if not (0 <= src < n_hosts and 0 <= dst < n_hosts) or src == dst:
            raise ValueError(f"Invalid probe '{part}'")
        out.append((src, dst))
    return sorted(set(out))


def parse_probe_ids_csv(s: str, n_hosts: int = 14) -> List[str]:
    """Parse comma-separated probe ids like '0-13,1-12'.

    Preserves input order and removes duplicates.
    """
    out: List[str] = []
    seen = set()
    if not s:
        return out
    for part in str(s).split(','):
        pid = part.strip()
        if not pid or pid in seen:
            continue
        if '-' not in pid:
            raise ValueError(f"Invalid probe '{pid}' (expected a-b)")
        a, b = pid.split('-', 1)
        src, dst = int(a), int(b)
        if not (0 <= src < n_hosts and 0 <= dst < n_hosts) or src == dst:
            raise ValueError(f"Invalid probe '{pid}'")
        out.append(f"{src}-{dst}")
        seen.add(pid)
    return out


def decode_str_list(arr: np.ndarray) -> List[str]:
    return [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]


def parse_sw_port_iface(n: str) -> Tuple[int, int]:
    """Parse interface name like 's3-eth2' -> (3,2)."""
    import re

    m = re.match(r"^s(\d+)-eth(\d+)$", str(n))
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def shortest_path_km_nsfnet(src: int, dst: int) -> float:
    """Dijkstra on NSFNET_LINKS_KM (weights in km)."""
    from simple_nsfnet import NSFNET_LINKS_KM
    import heapq

    n = 14
    adj = {i: [] for i in range(n)}
    for u, v, km in NSFNET_LINKS_KM:
        adj[int(u)].append((int(v), float(km)))
        adj[int(v)].append((int(u), float(km)))

    dist = {i: float("inf") for i in range(n)}
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
                heapq.heappush(pq, (nd, v))
    if not np.isfinite(dist[dst]):
        raise RuntimeError(f"NSFNET disconnected? no path {src}->{dst}")
    return float(dist[dst])


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
    """Build a *path-conditioned* node index set for pooling.

    The NPZ graph uses nodes = switch interfaces (e.g., s0-eth1). Inter-switch
    edges exist only between the two interface endpoints of each core link.

    We:
      1) compute switch-level shortest path (NSFNET distances)
      2) map each hop (u,v) to its interface endpoints by scanning edges where
         parse_sw(u)!=parse_sw(v)
      3) include access ports s<src>-eth1 and s<dst>-eth1
    """

    # name->idx
    name2idx = {str(n): i for i, n in enumerate(nodes)}
    sw_of = np.array([parse_sw_port_iface(n)[0] for n in nodes], dtype=int)

    # Map undirected switch-pair -> (iface_name_at_min, iface_name_at_max)
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
        # store in key order
        if sa < sb:
            inter[key] = (na, nb)
        else:
            inter[key] = (nb, na)

    sw_path, _km = shortest_path_nodes_km_nsfnet(src_sw, dst_sw)

    pick_names = set()
    # access ports (host attachments)
    pick_names.add(f"s{src_sw}-eth1")
    pick_names.add(f"s{dst_sw}-eth1")

    # core hop interfaces
    for u, v in zip(sw_path[:-1], sw_path[1:]):
        key = (min(int(u), int(v)), max(int(u), int(v)))
        if key not in inter:
            raise RuntimeError(f"Could not find inter-switch interface pair for hop {u}<->{v} in NPZ edges")
        a_name, b_name = inter[key]
        pick_names.add(a_name)
        pick_names.add(b_name)

    # translate to indices
    idxs = [name2idx[n] for n in pick_names if n in name2idx]
    idxs = sorted(set(int(i) for i in idxs))
    if len(idxs) < 2:
        raise RuntimeError(f"Path-conditioned pooling set too small (len={len(idxs)}). Check nodes naming and ping_src/dst.")
    return idxs


def compute_prop_rtt_ms(src: int, dst: int, km_per_ms: float = 200.0, delay_scale: float = 1.0) -> float:
    """Propagation-only RTT (ms) for NSFNET shortest path."""
    km = shortest_path_km_nsfnet(src, dst)
    one_way_ms = (km / float(km_per_ms)) * float(delay_scale)
    return 2.0 * one_way_ms


@dataclass
class Transform:
    kind: str  # 'raw' | 'residual_log1p'
    prop_rtt_ms: np.ndarray

    def forward(self, y_ms: np.ndarray) -> np.ndarray:
        y = np.asarray(y_ms, dtype=float)
        prop = np.asarray(self.prop_rtt_ms, dtype=float)
        if self.kind == "raw":
            return y
        if self.kind == "residual_log1p":
            res = np.maximum(0.0, y - prop)
            return np.log1p(res)
        raise ValueError(f"Unknown transform: {self.kind}")

    def inverse(self, y_hat: np.ndarray) -> np.ndarray:
        y = np.asarray(y_hat, dtype=float)
        prop = np.asarray(self.prop_rtt_ms, dtype=float)
        if self.kind == "raw":
            return y
        if self.kind == "residual_log1p":
            res = np.expm1(y)
            res = np.maximum(res, 0.0)
            return prop + res
        raise ValueError(f"Unknown transform: {self.kind}")


class PathDelayHead(nn.Module):
    """Predict *delta* in transformed space from a path-conditioned pooled embedding."""

    def __init__(self, hid: int, path_idx: List[int], dropout: float = 0.1):
        super().__init__()
        self.register_buffer("path_idx", torch.tensor(path_idx, dtype=torch.long))
        self.net = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def forward(self, h_nodes: torch.Tensor) -> torch.Tensor:
        # Mean pooling over nodes along the probe path (interfaces only)
        g = h_nodes.index_select(0, self.path_idx).mean(dim=0)
        return self.net(g).squeeze(-1)


class MultiProbeDelayHead(nn.Module):
    """Shared MLP applied to mean-pooled path embeddings for each probe.

    forward(h_nodes) -> delta_hat[K] (in transform space)
    """

    def __init__(self, hid: int, path_idxs: List[List[int]], dropout: float = 0.1):
        super().__init__()
        self.path_idxs = nn.ModuleList([
            _IndexBuffer(idxs) for idxs in path_idxs
        ])
        self.net = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

    def forward(self, h_nodes: torch.Tensor) -> torch.Tensor:
        pooled = []
        for b in self.path_idxs:
            g = h_nodes.index_select(0, b.idx).mean(dim=0)
            pooled.append(g)
        G = torch.stack(pooled, dim=0)  # [K,hid]
        return self.net(G).squeeze(-1)  # [K]


class _IndexBuffer(nn.Module):
    def __init__(self, idxs: List[int]):
        super().__init__()
        self.register_buffer("idx", torch.tensor(list(idxs), dtype=torch.long))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_multi.npz")
    ap.add_argument("--out", default="runs/models/rtt_graph.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    ap.add_argument("--encoder", choices=["sage", "routenet"], default="sage")
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.20)

    ap.add_argument("--horizon", type=int, default=0, help="Forecast horizon in steps. 0 = nowcast (t-1->t), H>0 = lead-H (t->t+H).")

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--delta", type=float, default=0.10)

    # Optional regularization to keep residual corrections small (helps when persistence is strong)
    ap.add_argument("--delta_l2", type=float, default=0.0, help="L2 penalty on delta_hat in transform space (default 0).")

    # NSFNET propagation baseline
    ap.add_argument("--ping_src", type=int, default=0)
    ap.add_argument("--ping_dst", type=int, default=13)
    ap.add_argument("--probes", default="", help="Optional probe list '0-13,1-12' (overrides probes in dataset).")
    ap.add_argument("--km_per_ms", type=float, default=200.0)
    ap.add_argument("--delay_scale", type=float, default=1.0)

    ap.add_argument("--transform", choices=["raw", "residual_log1p"], default="residual_log1p")
    ap.add_argument("--use_qdelay", action="store_true", help="Prefer qdelay_ms series if present")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    Z = load_npz(args.data)
    X = torch.from_numpy(Z["X"]).float().to(device)  # [T,N,F]
    edges = torch.from_numpy(Z["edges"]).long().to(device)
    train_idx = Z.get("train_idx")
    val_idx = Z.get("val_idx")
    test_idx = Z.get("test_idx")
    if train_idx is None or val_idx is None or test_idx is None:
        raise SystemExit("dataset npz must contain train_idx/val_idx/test_idx")
    train_idx = torch.from_numpy(train_idx).long()
    val_idx = torch.from_numpy(val_idx).long()
    test_idx = torch.from_numpy(test_idx).long()

    T, N, Fin = X.shape

    # --- Load delay series (graph-level). Supports single-probe [T] and multi-probe [T,K].
    if "delay_series_ms" in Z:
        delay_raw = Z["delay_series_ms"].astype(np.float64)
    else:
        if "Y" not in Z:
            raise SystemExit("No delay_series_ms and no Y found in dataset")
        Y = Z["Y"].astype(np.float64)
        delay_raw = Y[:, :, -1].mean(axis=1)

    delay_raw = np.maximum(delay_raw, 0.0)
    if delay_raw.ndim == 1:
        delay_raw = delay_raw[:, None]

    # Delay series name (for logging only)
    delay_name = None
    if "delay_series_name" in Z:
        try:
            dn = Z["delay_series_name"]
            delay_name = str(dn.item() if hasattr(dn, "item") else dn)
        except Exception:
            delay_name = ""
    if delay_name is None:
        delay_name = "rtt_ms"

    # Probe ids + (src,dst) pairs (required for prop RTT + path pooling)
    probes: List[Tuple[int, int]] = []
    ds_probe_ids: List[str] = []

    if "probe_ids" in Z:
        ds_probe_ids = decode_str_list(Z["probe_ids"])
    elif "probe_src" in Z and "probe_dst" in Z and delay_raw.shape[1] > 1:
        ps = Z["probe_src"].astype(int).tolist()
        pd = Z["probe_dst"].astype(int).tolist()
        ds_probe_ids = [f"{s}-{d}" for (s, d) in zip(ps, pd)]

    if ds_probe_ids:
        # derive (src,dst) from ids if needed
        if "probe_src" in Z and "probe_dst" in Z and len(ds_probe_ids) == len(Z["probe_src"]):
            ps = Z["probe_src"].astype(int).tolist()
            pd = Z["probe_dst"].astype(int).tolist()
            probes = list(zip(ps, pd))
        else:
            # parse ids like "1-10" -> (1,10)
            probes = [(int(a), int(b)) for (a, b) in (pid.split("-", 1) for pid in ds_probe_ids)]

    # Fallback for single-probe datasets
    if not ds_probe_ids:
        ds_probe_ids = [f"{int(args.ping_src)}-{int(args.ping_dst)}"]
    if not probes:
        probes = [(int(args.ping_src), int(args.ping_dst))]

    if delay_raw.shape[1] != len(ds_probe_ids):
        # Allow the common single-series case where dataset omits probe ids.
        if not (delay_raw.shape[1] == 1 and len(ds_probe_ids) == 1):
            raise SystemExit(
                f"Probe mismatch: delay_series_ms has K={delay_raw.shape[1]} but dataset probe_ids has K={len(ds_probe_ids)}"
            )

    # Optional probe subset selection:
    # - CLI: --probes "a-b,c-d"  (highest priority)
    # - Env: PROBE_SUBSET="a-b,c-d"  (used by scripts)
    subset_cli = str(args.probes).strip()
    subset_env = str(os.getenv("PROBE_SUBSET", "")).strip()
    subset_spec = subset_cli or subset_env
    subset_source = "cli" if subset_cli else ("env" if subset_env else "")

    probe_subset: Optional[List[str]] = None
    if subset_spec:
        probe_subset = parse_probe_ids_csv(subset_spec, n_hosts=14)
        if delay_raw.shape[1] == 1 and len(probe_subset) > 1:
            raise SystemExit("dataset contains a single delay series but multiple probes were requested")
        id2k = {pid: k for k, pid in enumerate(ds_probe_ids)}
        keep = []
        for pid in probe_subset:
            if pid not in id2k:
                raise SystemExit(f"Requested probe {pid} not found in dataset probe_ids")
            keep.append(int(id2k[pid]))
        # Apply in *requested order*
        delay_raw = delay_raw[:, keep]
        probes = [probes[k] for k in keep]
        ds_probe_ids = [ds_probe_ids[k] for k in keep]
        probe_subset = ds_probe_ids.copy()
        print(f"[rtt-graph] Applied probe subset ({subset_source}): {','.join(probe_subset)}")

    K = int(delay_raw.shape[1])

    # Propagation RTT baseline per probe
    prop_rtt_ms = np.array(
        [compute_prop_rtt_ms(s, d, km_per_ms=args.km_per_ms, delay_scale=args.delay_scale) for (s, d) in probes],
        dtype=np.float64,
    )
    tfm = Transform(kind=args.transform, prop_rtt_ms=prop_rtt_ms)
    y_train = tfm.forward(delay_raw)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)  # [T,K]

    # Build path-conditioned pooling sets (interface nodes along each probe path)
    if "nodes" not in Z:
        raise SystemExit("dataset npz must contain 'nodes' array for path-conditioned delay training")
    nodes = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in Z["nodes"]]
    path_idxs = [path_iface_indices_from_npz(nodes, Z["edges"], s, d) for (s, d) in probes]
    path_ifaces = [[nodes[i] for i in idxs] for idxs in path_idxs]

    enc = GNNEncoder(Fin, hid=args.hid, layers=args.layers, kind=args.encoder, dropout=args.dropout).to(device)
    # Head predicts delta (in transform space) to be added to y[t-1] for each probe
    head = MultiProbeDelayHead(hid=args.hid, path_idxs=path_idxs, dropout=max(0.0, args.dropout * 0.5)).to(device)

    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=args.lr)
    # --- Training (epoch-level early stopping) ---------------------------------
    best = float("inf")
    best_state = None
    bad = 0

    # helper: compute val/test RMSE in ms (flattened across probes)
    def eval_rmse(idx: torch.Tensor) -> Tuple[float, float]:
        enc.eval(); head.eval()
        with torch.no_grad():
            preds = []  # list of [K]
            idx_np = idx.cpu().numpy().astype(int)

            if int(args.horizon) == 0:
                # nowcast: predict y[t] from y[t-1] + delta_hat(X[t])
                idx_np = idx_np[idx_np >= 1]
                for t in idx_np.tolist():
                    h = enc(X[t], edges)
                    delta_hat = head(h)  # [K]
                    base = y_train_t[t - 1]  # [K]
                    y_hat_t = base + delta_hat
                    preds.append(y_hat_t.detach().cpu().numpy())
                if len(preds) == 0:
                    return float("inf"), float("inf")
                y_hat_tfm = np.stack(preds, axis=0)  # [S,K]
                y_hat_ms = tfm.inverse(y_hat_tfm)
                y_true_ms = delay_raw[idx_np, :]
            else:
                # lead-H: predict y[t+H] from y[t] + delta_hat(X[t])
                H = int(args.horizon)
                idx_np = idx_np[(idx_np >= 0) & (idx_np + H < T)]
                for t in idx_np.tolist():
                    h = enc(X[t], edges)
                    delta_hat = head(h)  # [K]
                    base = y_train_t[t]  # [K]  (persistence for lead-H)
                    y_hat_t = base + delta_hat
                    preds.append(y_hat_t.detach().cpu().numpy())
                if len(preds) == 0:
                    return float("inf"), float("inf")
                y_hat_tfm = np.stack(preds, axis=0)  # [S,K]
                y_hat_ms = tfm.inverse(y_hat_tfm)
                y_true_ms = delay_raw[idx_np + H, :]

        prop = prop_rtt_ms.reshape(1, -1)
        return (
            rmse(y_hat_ms, y_true_ms),
            rmse(np.maximum(0.0, y_hat_ms - prop), np.maximum(0.0, y_true_ms - prop)),
        )

    # Init: start close to persistence by zero-initialising the final layer of the head MLP.
    # This makes delta_hat ~= 0 at start, i.e., y_hat ~= base.
    try:
        last = head.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
    except Exception:
        pass

    for ep in range(1, args.epochs + 1):
        enc.train(); head.train()
        losses = []

        H = int(args.horizon)
        # Shuffle time indices each epoch for SGD stability
        t_list = train_idx.cpu().numpy().astype(int).tolist()
        np.random.shuffle(t_list)

        for t in t_list:
            # nowcast uses t>=1; lead-H uses t+H < T
            if H == 0:
                if t < 1:
                    continue
                t_base = t - 1
                t_targ = t
            else:
                if t < 0 or t + H >= T:
                    continue
                t_base = t
                t_targ = t + H

            opt.zero_grad(set_to_none=True)
            h = enc(X[t], edges)
            delta_hat = head(h)  # [K]

            base = y_train_t[t_base].detach()
            pred = base + delta_hat
            targ = y_train_t[t_targ]

            loss = nn.functional.smooth_l1_loss(pred, targ, beta=float(args.delta))

            # Optional: keep deltas small when persistence is already strong.
            if float(args.delta_l2) > 0:
                loss = loss + float(args.delta_l2) * (delta_hat ** 2).mean()

            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        # --- end epoch: evaluate once ----------------------------------------
        tr_loss = float(np.mean(losses)) if losses else float("inf")
        val_rmse_ms, val_res_rmse = eval_rmse(val_idx)

        prop_mean = float(np.mean(prop_rtt_ms))
        print(
            f"[rtt-graph][ep {ep:03d}] H={int(args.horizon)} train_loss={tr_loss:.5f} "
            f"val_RTT_RMSE_ms={val_rmse_ms:.3f} val_Qdelay_RMSE_ms={val_res_rmse:.3f} "
            f"(K={K}, transform={tfm.kind}, prop_mean={prop_mean:.3f}ms, delay_name={delay_name})"
        )

        if val_rmse_ms < best - 1e-6:
            best = val_rmse_ms
            best_state = {
                "enc": enc.state_dict(),
                "head": head.state_dict(),
                "meta": {
                    "seed": int(args.seed),
                    "encoder": str(args.encoder),
                    "hid": int(args.hid),
                    "layers": int(args.layers),
                    "horizon": int(args.horizon),
                    "dropout": float(args.dropout),
                    "K": int(K),
                    "out_dim": int(K),
                    "transform": str(tfm.kind),
                    "prop_rtt_ms": [float(x) for x in prop_rtt_ms.tolist()],
                    "probe_ids": [str(pid) for pid in ds_probe_ids],
                    "probes": [f"{s}-{d}" for (s, d) in probes],
                    "probe_subset": [str(pid) for pid in (probe_subset or [])],
                    "ping_src": int(probes[0][0]),
                    "ping_dst": int(probes[0][1]),
                    "km_per_ms": float(args.km_per_ms),
                    "delay_scale": float(args.delay_scale),
                    "delay_name": str(delay_name) if delay_name is not None else "",
                    "pool": "path_ifaces_mean",
                    "delta_mode": f"persistence_additive_tfm_H{int(args.horizon)}",
                    "path_ifaces": path_ifaces,
                },
            }
            bad = 0
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"Early stop at epoch {ep} (no val improvement for {bad} epochs).")
                break

    if best_state is None:
        raise SystemExit("Training failed (no best_state)")

    # Restore best and save once
    enc.load_state_dict(best_state["enc"])
    head.load_state_dict(best_state["head"])

    test_rmse_ms, test_res_rmse = eval_rmse(test_idx)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_state["metrics"] = {"test_rmse_ms": float(test_rmse_ms), "test_qdelay_rmse_ms": float(test_res_rmse)}
    torch.save(best_state, str(out_path))
    print(f"TEST rtt-graph RMSE (rtt_ms, qdelay_ms) = {test_rmse_ms:.3f}, {test_res_rmse:.3f} | saved {out_path}")

if __name__ == "__main__":
    main()
