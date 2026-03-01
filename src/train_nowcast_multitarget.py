#!/usr/bin/env python3
"""
train_nowcast_multitarget.py

Multi-target NOWCAST training for NDT.
Targets (assumed Y[...,k] order):
  k=0 queue_pkts
  k=1 throughput_Mbps
  k=2 utilization
  k=3 delay_ms (qdelay_ms preferred, rtt_ms fallback)

Key features:
- Sensor node loss reweighting: --sensor_w
- Throughput scaling uses --thr_cap (NOT capacities_Mbps)
- Idle-bias control for throughput/delay + minimum weights
- Correct in-place sensor dropout for backlog/rate features
- Early stopping with --patience (default 6)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from models_gnn import GNNEncoder
from models_gnn_multitask import MultiNowcastHead, apply_target_activations


def load_npz(p: str) -> Dict[str, np.ndarray]:
    D = np.load(p, allow_pickle=True)
    return {k: D[k] for k in D.files}


def rmse_t(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a - b) ** 2)).item())


@dataclass
class Scales:
    queue_pkts: float
    throughput_Mbps: float
    utilization: float
    delay_ms: float


def huber_scaled(
    pred: torch.Tensor,
    targ: torch.Tensor,
    scale: float,
    delta: float,
    w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pred_s = pred / float(scale)
    targ_s = targ / float(scale)
    # PyTorch smooth_l1_loss: beta parameter exists in modern torch; safe for 1.13+.
    loss = nn.functional.smooth_l1_loss(pred_s, targ_s, reduction="none", beta=float(delta))
    if w is not None:
        loss = loss * w
        return loss.sum() / (w.sum() + 1e-9)
    return loss.mean()


def zero_channels_inplace(x: torch.Tensor, node_mask: torch.Tensor, ch_list: Sequence[int]) -> None:
    """In-place x[nodes, channels] = 0 with correct advanced indexing."""
    if not ch_list:
        return
    if node_mask is None or not torch.any(node_mask):
        return
    idx = torch.nonzero(node_mask, as_tuple=True)[0]
    if idx.numel() == 0:
        return
    ch = torch.as_tensor(list(ch_list), dtype=torch.long, device=x.device)
    x[idx[:, None], ch[None, :]] = 0.0


def main() -> None:
    ap = argparse.ArgumentParser()

    # IO / model
    ap.add_argument("--data", default="dataset_multi.npz")
    ap.add_argument("--encoder", choices=["sage", "routenet"], default="sage")
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/models/nowcast_multitarget.pt")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    # thresholds
    ap.add_argument("--busy_thr", type=float, default=50.0)  # queue busy threshold

    # NEW: sensor weighting
    ap.add_argument(
        "--sensor_w",
        type=float,
        default=20.0,
        help="Multiply per-node loss weight for sensor nodes by this factor (non-sensors keep weight 1).",
    )

    # scales (normalization, not hard caps)
    ap.add_argument("--queue_cap", type=float, default=1000.0)
    ap.add_argument("--thr_cap", type=float, default=20.0, help="Throughput normalization scale (Mbps).")
    ap.add_argument("--delay_cap", type=float, default=600.0, help="Delay normalization scale (ms).")

    # backward-compatible aliases (kept separate to avoid argparse dest collisions)
    ap.add_argument("--rtt_cap", type=float, default=None, help="Alias for --delay_cap")
    ap.add_argument("--w_rtt", type=float, default=None, help="Alias for --w_delay")

    # loss weights
    ap.add_argument("--w_queue", type=float, default=2.0)
    ap.add_argument("--w_queue_idle", type=float, default=0.10)

    ap.add_argument("--w_thr", type=float, default=2.0)
    ap.add_argument("--w_util", type=float, default=0.50)
    ap.add_argument("--w_delay", type=float, default=0.40)

    # throughput idle bias / weighting
    ap.add_argument("--thr_idle_thr", type=float, default=0.05)
    ap.add_argument("--w_thr_idle", type=float, default=6.0)
    ap.add_argument("--thr_min_w", type=float, default=0.20, help="Minimum per-node weight for throughput main loss.")

    # delay idle bias / weighting
    ap.add_argument("--delay_idle_thr", type=float, default=1.0)
    ap.add_argument("--w_delay_idle", type=float, default=0.20)
    ap.add_argument("--delay_min_w", type=float, default=0.05, help="Minimum per-node weight for delay main loss.")

    # busy emphasis + sensor dropout
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--sdrop", type=float, default=0.10)
    ap.add_argument("--drop_rates", action="store_true")

    args = ap.parse_args()

    # alias mapping
    if args.rtt_cap is not None:
        args.delay_cap = float(args.rtt_cap)
    if args.w_rtt is not None:
        args.w_delay = float(args.w_rtt)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    Z = load_npz(args.data)
    X = torch.from_numpy(Z["X"]).float().to(device)  # [T,N,F]
    Y = torch.from_numpy(Z["Y"]).float().to(device)  # [T,N,K]
    edges = torch.from_numpy(Z["edges"]).long().to(device)
    train_idx = torch.from_numpy(Z["train_idx"]).long()
    val_idx = torch.from_numpy(Z["val_idx"]).long()
    test_idx = torch.from_numpy(Z["test_idx"]).long()
    feat_names = Z["feat_names"]

    T, N, Fin = X.shape
    out_dim = Y.shape[2]

    names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in feat_names]

    def find_idx(key: str) -> int:
        for i, n in enumerate(names):
            if n == key:
                return i
        raise KeyError(f"Feature '{key}' not found in feat_names")

    ch_is_sensor = find_idx("is_sensor")
    ch_back = [find_idx(k) for k in ["sensor_backlog", "sensor_backlog_lag1", "sensor_backlog_lag2", "sensor_backlog_lag3"]]
    ch_rates = [i for i, n in enumerate(names) if n.endswith("_per_s")]

    # node sensor mask
    if "is_sensor" in Z:
        is_sensor = torch.from_numpy(Z["is_sensor"].astype(np.float32)).to(device).view(-1)  # [N]
    else:
        is_sensor = (X[0, :, ch_is_sensor] > 0.5).float().view(-1)

    non_sensor_mask = (is_sensor < 0.5)

    # per-node weights
    node_w = 1.0 + (float(args.sensor_w) - 1.0) * is_sensor  # [N]

    scales = Scales(
        queue_pkts=float(args.queue_cap),
        throughput_Mbps=float(args.thr_cap),  # IMPORTANT FIX
        utilization=1.0,
        delay_ms=float(args.delay_cap),
    )

    enc = GNNEncoder(Fin, hid=args.hid, layers=args.layers, kind=args.encoder, dropout=args.dropout).to(device)
    head = MultiNowcastHead(hid=args.hid, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=args.lr)

    best_score = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, args.epochs + 1):
        enc.train()
        head.train()
        tot = 0.0
        steps = 0

        for t in train_idx.tolist():
            xt = X[t].clone()
            yt = Y[t]  # [N,K]

            # sensor dropout mask (only on sensor nodes)
            sens = (xt[:, ch_is_sensor] > 0.5)
            if args.sdrop > 0:
                drop = (torch.rand(N, device=device) < float(args.sdrop)) & sens
                if drop.any():
                    zero_channels_inplace(xt, drop, ch_back)
                    if args.drop_rates and len(ch_rates) > 0:
                        zero_channels_inplace(xt, drop, ch_rates)

            pred = apply_target_activations(head(enc(xt, edges)))  # [N,K]

            # queue
            yq = yt[:, 0]
            m_busy_q = (yq >= float(args.busy_thr))
            m_idle_q = ~m_busy_q

            if m_busy_q.any():
                wq_base = (torch.clamp(yq / scales.queue_pkts, 0.0, 1.5) ** float(args.gamma)).detach()
                wq = (wq_base * node_w).detach()
                lq_busy = huber_scaled(pred[m_busy_q, 0], yq[m_busy_q], scales.queue_pkts, args.delta, w=wq[m_busy_q])
            else:
                lq_busy = torch.zeros((), device=device)

            lq_idle = (
                huber_scaled(pred[m_idle_q, 0], yq[m_idle_q], scales.queue_pkts, args.delta, w=node_w[m_idle_q])
                if m_idle_q.any()
                else torch.zeros((), device=device)
            )
            lq = lq_busy + float(args.w_queue_idle) * lq_idle

            # throughput
            ythr = yt[:, 1]
            m_idle_thr = (ythr <= float(args.thr_idle_thr))
            wthr_base = (
                float(args.thr_min_w) + (torch.clamp(ythr / scales.throughput_Mbps, 0.0, 1.5) ** float(args.gamma))
            ).detach()
            wthr = (wthr_base * node_w).detach()
            lthr_main = huber_scaled(pred[:, 1], ythr, scales.throughput_Mbps, args.delta, w=wthr)
            lthr_idle = (
                huber_scaled(pred[m_idle_thr, 1], ythr[m_idle_thr], scales.throughput_Mbps, args.delta, w=node_w[m_idle_thr])
                if m_idle_thr.any()
                else torch.zeros((), device=device)
            )
            lthr = lthr_main + float(args.w_thr_idle) * lthr_idle

            # utilization
            yutil = yt[:, 2]
            lutil = huber_scaled(pred[:, 2], yutil, scales.utilization, args.delta, w=node_w)

            # delay (qdelay/rtt)
            ydel = yt[:, 3]
            m_idle_del = (ydel <= float(args.delay_idle_thr))
            wdel_base = (
                float(args.delay_min_w) + (torch.clamp(ydel / scales.delay_ms, 0.0, 1.5) ** float(args.gamma))
            ).detach()
            wdel = (wdel_base * node_w).detach()
            ldel_main = huber_scaled(pred[:, 3], ydel, scales.delay_ms, args.delta, w=wdel)
            ldel_idle = (
                huber_scaled(pred[m_idle_del, 3], ydel[m_idle_del], scales.delay_ms, args.delta, w=node_w[m_idle_del])
                if m_idle_del.any()
                else torch.zeros((), device=device)
            )
            ldel = ldel_main + float(args.w_delay_idle) * ldel_idle

            loss = float(args.w_queue) * lq + float(args.w_thr) * lthr + float(args.w_util) * lutil + float(args.w_delay) * ldel

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step()

            tot += float(loss.item())
            steps += 1

        # --- validation score: queue busy RMSE + throughput non-sensor RMSE
        enc.eval()
        head.eval()
        with torch.no_grad():
            qh, qt = [], []
            th, tt = [], []
            for t in val_idx.tolist():
                p = apply_target_activations(head(enc(X[t], edges)))
                yt = Y[t]

                yq = yt[:, 0]
                m = (yq >= float(args.busy_thr))
                if m.any():
                    qh.append(p[m, 0])
                    qt.append(yq[m])

                if non_sensor_mask.any():
                    th.append(p[non_sensor_mask, 1])
                    tt.append(yt[non_sensor_mask, 1])

            vq = rmse_t(torch.cat(qh), torch.cat(qt)) if qh else 1e9
            vthr_non = rmse_t(torch.cat(th), torch.cat(tt)) if th else 1e9

        vscore = vq + 0.1 * vthr_non
        print(
            f"[nowcast-multi][ep {ep:03d}] "
            f"train_loss={tot/max(1,steps):.5f} "
            f"val_queue_busy_RMSE={vq:.3f} "
            f"val_thr_nonSensor_RMSE={vthr_non:.3f} "
            f"(sensor_w={float(args.sensor_w):.1f}, thr_cap={float(args.thr_cap):.1f})"
        )

        if vscore < best_score - 1e-6:
            best_score = vscore
            best_state = {
                "enc": enc.state_dict(),
                "head": head.state_dict(),
                "meta": {
                    **vars(args),
                    "out_dim": int(out_dim),
                    "label_names": [str(x) for x in Z.get("label_names", [])],
                },
            }
            bad = 0
        else:
            bad += 1
            if bad >= int(args.patience):
                print(f"Early stop at epoch {ep}.")
                break

    if best_state is None:
        best_state = {"enc": enc.state_dict(), "head": head.state_dict(), "meta": vars(args)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.out)

    # --- test summary
    enc.load_state_dict(best_state["enc"])
    head.load_state_dict(best_state["head"])
    enc.eval()
    head.eval()

    with torch.no_grad():
        P, G = [], []
        for t in test_idx.tolist():
            P.append(apply_target_activations(head(enc(X[t], edges))))
            G.append(Y[t])
        P = torch.stack(P)
        G = torch.stack(G)

    rmses = [rmse_t(P[..., k].flatten(), G[..., k].flatten()) for k in range(out_dim)]
    print(
        "TEST nowcast multi RMSE (queue,thr,util,delay) = "
        + ", ".join(f"{x:.3f}" for x in rmses)
        + f" | saved {args.out}"
    )


if __name__ == "__main__":
    main()
