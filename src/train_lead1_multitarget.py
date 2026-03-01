#!/usr/bin/env python3
"""
train_lead1_multitarget.py

Multi-target LEAD-1 training for NDT using temporal head (TCN or GRU).
Predict Y[t+1] from window X[t-K+1 : t].

This version adds *delay residual modes* to make qdelay competitive:

  1) --delay_residual
       Predict signed delta-delay and reconstruct:
         delay_hat(t+1) = clamp( delay_true(t) + delta_pred, 0, cap )

  2) --delay_queue_residual
       Use queue-based baseline (Little's law style) plus signed residual:
         base_ms = queue_hat_pkts(t+1) * pkt_ms
         delay_hat(t+1) = clamp( base_ms + delta_pred, 0, cap )

Notes:
- delta_pred is taken from the *RAW* delay head output (no Softplus).
- Queue/Throughput use Softplus; Utilization uses Sigmoid (as before).
- Early stopping remains based on queue busy RMSE + 0.1*throughput non-sensor RMSE
  (same as your original script), so behaviour stays stable.

Compatible with your repo's:
  models_gnn.py (GNNEncoder)
  models_gnn_multitask.py (MultiTCNHead, MultiGRUHead, apply_target_activations)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from models_gnn import GNNEncoder
from models_gnn_multitask import MultiTCNHead, MultiGRUHead, apply_target_activations


def load_npz(p: str) -> Dict[str, np.ndarray]:
    D = np.load(p, allow_pickle=True)
    return {k: D[k] for k in D.files}


@dataclass
class Scales:
    queue_pkts: float
    throughput_Mbps: float
    utilization: float
    delay_ms: float


def rmse_t(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a - b) ** 2)).item())


def huber_scaled(
    pred: torch.Tensor,
    targ: torch.Tensor,
    scale: float,
    delta: float,
    w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pred_s = pred / float(scale)
    targ_s = targ / float(scale)
    loss = nn.functional.smooth_l1_loss(pred_s, targ_s, reduction="none", beta=float(delta))
    if w is not None:
        loss = loss * w
        return loss.sum() / (w.sum() + 1e-9)
    return loss.mean()


def zero_channels_inplace(x: torch.Tensor, node_mask: torch.Tensor, ch_list: Sequence[int]) -> None:
    if not ch_list:
        return
    if node_mask is None or not torch.any(node_mask):
        return
    idx = torch.nonzero(node_mask, as_tuple=True)[0]
    if idx.numel() == 0:
        return
    ch = torch.as_tensor(list(ch_list), dtype=torch.long, device=x.device)
    x[idx[:, None], ch[None, :]] = 0.0


def _pkt_ms(pkt_bytes: float, bw_mbps: float) -> float:
    # time to transmit one packet on bottleneck in milliseconds
    # pkt_ms = (pkt_bytes*8 bits) / (bw_mbps*1e6 bits/s) * 1e3 ms/s
    return float(pkt_bytes) * 8.0 / (float(bw_mbps) * 1e6) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()

    # IO / model
    ap.add_argument("--data", default="dataset_multi.npz")
    ap.add_argument("--encoder", choices=["sage", "routenet"], default="sage")
    ap.add_argument("--temporal", choices=["tcn", "gru"], default="tcn")
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--hid", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/models/lead1_multitarget.pt")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    # thresholds
    ap.add_argument("--busy_thr", type=float, default=50.0)

    # sensor weighting
    ap.add_argument("--sensor_w", type=float, default=20.0)

    # scales
    ap.add_argument("--queue_cap", type=float, default=1000.0)
    ap.add_argument("--thr_cap", type=float, default=20.0, help="Throughput normalization scale (Mbps).")
    ap.add_argument("--delay_cap", type=float, default=600.0)

    # aliases
    ap.add_argument("--rtt_cap", type=float, default=None, help="Alias for --delay_cap")
    ap.add_argument("--w_rtt", type=float, default=None, help="Alias for --w_delay")

    # loss weights
    ap.add_argument("--w_queue", type=float, default=2.0)
    ap.add_argument("--w_queue_idle", type=float, default=0.10)
    ap.add_argument("--w_thr", type=float, default=2.0)
    ap.add_argument("--w_util", type=float, default=0.50)
    ap.add_argument("--w_delay", type=float, default=0.40)

    # throughput idle bias
    ap.add_argument("--thr_idle_thr", type=float, default=0.05)
    ap.add_argument("--w_thr_idle", type=float, default=6.0)
    ap.add_argument("--thr_min_w", type=float, default=0.20)

    # delay idle bias
    ap.add_argument("--delay_idle_thr", type=float, default=1.0)
    ap.add_argument("--w_delay_idle", type=float, default=0.20)
    ap.add_argument("--delay_min_w", type=float, default=0.05)

    # busy emphasis + dropout
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--sdrop", type=float, default=0.10)
    ap.add_argument("--drop_rates", action="store_true")

    # --- NEW: delay residual modes
    ap.add_argument(
        "--delay_residual",
        action="store_true",
        help="Train delay head as signed delta and reconstruct delay(t+1)=delay(t)+delta_pred (clamped).",
    )
    ap.add_argument(
        "--delay_queue_residual",
        action="store_true",
        help="Train delay as queue-based baseline + signed residual: delay_hat = queue_hat_pkts*pkt_ms + delta_pred (clamped).",
    )
    ap.add_argument("--bw_bottleneck", type=float, default=2.0, help="Bottleneck bandwidth (Mbps) for queue->delay baseline.")
    ap.add_argument("--pkt_bytes", type=float, default=1500.0, help="Packet size (bytes) for queue->delay baseline.")
    ap.add_argument(
        "--delay_recon_cap",
        type=float,
        default=None,
        help="Optional clamp max for reconstructed delay (ms). If unset, no max clamp is applied.",
    )

    args = ap.parse_args()

    if args.rtt_cap is not None:
        args.delay_cap = float(args.rtt_cap)
    if args.w_rtt is not None:
        args.w_delay = float(args.w_rtt)

    # validate residual flags
    if args.delay_residual and args.delay_queue_residual:
        raise ValueError("Choose only one: --delay_residual OR --delay_queue_residual (not both).")
    if args.delay_queue_residual and float(args.bw_bottleneck) <= 0:
        raise ValueError("--bw_bottleneck must be > 0 for --delay_queue_residual.")

    pkt_ms = _pkt_ms(args.pkt_bytes, args.bw_bottleneck) if args.delay_queue_residual else 0.0

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

    feat_names = Z.get("feat_names", None)
    names = []
    if feat_names is not None:
        names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in feat_names]

    Ttot, N, Fin = X.shape
    out_dim = int(Y.shape[2])
    Kwin = int(args.K)

    if out_dim < 4 and (args.delay_residual or args.delay_queue_residual):
        raise ValueError("Residual delay modes require at least 4 targets (queue,thr,util,delay).")

    # feature channel indices (best-effort)
    def find_idx(key: str) -> Optional[int]:
        for i, n in enumerate(names):
            if n == key:
                return i
        return None

    ch_is_sensor = find_idx("is_sensor")
    ch_backlog = find_idx("backlog_pkts")

    ch_rates: Sequence[int] = []
    if names:
        tmp = []
        for k in ["tx_bps", "rx_bps", "tx_pps", "rx_pps", "tx_bytes", "rx_bytes"]:
            j = find_idx(k)
            if j is not None:
                tmp.append(int(j))
        ch_rates = tmp

    # non-sensor mask (for throughput validation)
    if "is_sensor" in Z:
        non_sensor_mask = torch.from_numpy(~Z["is_sensor"].astype(bool)).to(device)
    else:
        if ch_is_sensor is not None:
            # derive from any time slice (static)
            non_sensor_mask = ~(X[0, :, ch_is_sensor] > 0.5)
        else:
            non_sensor_mask = torch.ones((N,), dtype=torch.bool, device=device)

    # valid positions where window exists and t+1 exists
    train_pos = train_idx[(train_idx >= Kwin - 1) & (train_idx < Ttot - 1)].to(device)
    val_pos = val_idx[(val_idx >= Kwin - 1) & (val_idx < Ttot - 1)].to(device)
    test_pos = test_idx[(test_idx >= Kwin - 1) & (test_idx < Ttot - 1)].to(device)

    scales = Scales(
        queue_pkts=float(args.queue_cap),
        throughput_Mbps=float(args.thr_cap),
        utilization=1.0,
        delay_ms=float(args.delay_cap),
    )

    enc = GNNEncoder(in_dim=Fin, hid=args.hid, layers=args.layers, dropout=args.dropout).to(device)
    if args.temporal == "tcn":
        # IMPORTANT: MultiTCNHead signature is (hid, K, out_dim)
        head = MultiTCNHead(hid=args.hid, K=Kwin, out_dim=out_dim).to(device)
    else:
        head = MultiGRUHead(hid=args.hid, out_dim=out_dim).to(device)

    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=float(args.lr))

    # helper: build prediction with optional residual reconstruction
    def forward_pred(Hseq: torch.Tensor, t: int) -> torch.Tensor:
        """
        Returns pred [N,out_dim] in *absolute target space*.
        For residual delay modes, pred[:,3] is reconstructed delay(t+1).
        """
        y_raw = head(Hseq)                      # [N,K]
        pred = apply_target_activations(y_raw)  # queue/thr/util/delay with nonneg constraints

        if (args.delay_residual or args.delay_queue_residual) and pred.size(-1) >= 4:
            pred = pred.clone()
            delta = y_raw[:, 3]  # signed
            if args.delay_residual:
                base = Y[t, :, 3]
            else:
                # queue-based baseline (use predicted queue at t+1)
                base = pred[:, 0] * float(pkt_ms)
            d = base + delta
            if args.delay_recon_cap is not None:
                pred[:, 3] = torch.clamp(d, min=0.0, max=float(args.delay_recon_cap))
            else:
                pred[:, 3] = torch.clamp(d, min=0.0)

        return pred

    best_score = 1e18
    best_state: Optional[Dict[str, object]] = None
    bad = 0

    for ep in range(1, int(args.epochs) + 1):
        enc.train()
        head.train()

        tot = 0.0
        steps = 0

        perm = torch.randperm(train_pos.numel(), device=device)
        for idx in perm.tolist():
            t = int(train_pos[idx].item())
            y_next = Y[t + 1]  # [N,K]
            y_curr = Y[t]      # [N,K] (for residual baseline if used)

            # queue masks for "busy" emphasis
            yq = y_next[:, 0]
            m_busy_q = (yq >= float(args.busy_thr))
            m_idle_q = ~m_busy_q

            # node weights (sensor weighting)
            if "is_sensor" in Z:
                is_sensor = torch.from_numpy(Z["is_sensor"].astype(bool)).to(device)
            elif ch_is_sensor is not None:
                is_sensor = (X[t, :, ch_is_sensor] > 0.5)
            else:
                is_sensor = torch.zeros((N,), dtype=torch.bool, device=device)

            node_w = torch.ones((N,), dtype=torch.float32, device=device)
            node_w[is_sensor] = float(args.sensor_w)

            # build window embeddings with optional in-place sensor dropout
            H_list = []
            nodes_to_drop = None
            for s in range(t - Kwin + 1, t + 1):
                xs = X[s]
                if float(args.sdrop) > 0 and (ch_is_sensor is not None) and (ch_backlog is not None):
                    xs = xs.clone()
                    sens = (xs[:, ch_is_sensor] > 0.5)
                    if nodes_to_drop is None:
                        nodes_to_drop = (torch.rand_like(sens.float()) < float(args.sdrop)) & sens
                    if torch.any(nodes_to_drop):
                        # drop backlog
                        xs[nodes_to_drop, ch_backlog] = 0.0
                        # optionally drop rates
                        if args.drop_rates and len(ch_rates) > 0:
                            zero_channels_inplace(xs, nodes_to_drop, ch_rates)
                H_list.append(enc(xs, edges))

            Hseq = torch.stack(H_list, 0)  # [K,N,H]
            pred = forward_pred(Hseq, t)   # [N,K] (absolute)

            # --- queue loss
            if m_busy_q.any():
                wq_base = (
                    torch.clamp(yq / scales.queue_pkts, 0.0, 1.5) ** float(args.gamma)
                )
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

            # --- throughput loss + idle emphasis
            ythr = y_next[:, 1]
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

            # --- utilization loss
            yutil = y_next[:, 2]
            lutil = huber_scaled(pred[:, 2], yutil, scales.utilization, args.delta, w=node_w)

            # --- delay loss (absolute delay, after residual reconstruction if enabled)
            ydel = y_next[:, 3]
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

            loss = (
                float(args.w_queue) * lq
                + float(args.w_thr) * lthr
                + float(args.w_util) * lutil
                + float(args.w_delay) * ldel
            )

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
            dh, dt = [], []

            DELAY_BUSY_THR = 50.0  # ms (fixed busy threshold for qdelay validation)

            for t0 in val_pos.tolist():
                t0 = int(t0)
                H = torch.stack([enc(X[s], edges) for s in range(t0 - Kwin + 1, t0 + 1)], 0)
                p = forward_pred(H, t0)
                y_next = Y[t0 + 1]

                # queue busy RMSE
                yq = y_next[:, 0]
                m_q = (yq >= float(args.busy_thr))
                if m_q.any():
                    qh.append(p[m_q, 0])
                    qt.append(yq[m_q])

                # throughput non-sensor RMSE (as before)
                if non_sensor_mask.any():
                    th.append(p[non_sensor_mask, 1])
                    tt.append(y_next[non_sensor_mask, 1])

                # delay busy RMSE (critical!)
                if y_next.size(1) >= 4:
                    yd = y_next[:, 3]
                    m_d = (yd >= float(DELAY_BUSY_THR))
                    if m_d.any():
                        dh.append(p[m_d, 3])
                        dt.append(yd[m_d])

            vq = rmse_t(torch.cat(qh), torch.cat(qt)) if qh else 1e9
            vthr_non = rmse_t(torch.cat(th), torch.cat(tt)) if th else 1e9
            vdel_busy = rmse_t(torch.cat(dh), torch.cat(dt)) if dh else 1e9

        # Weighting: delay RMSE is in ms, so scale it down
        vscore = vq + 0.1 * vthr_non + 0.02 * vdel_busy

        print(
            f"[lead1-multi][ep {ep:03d}] "
            f"train_loss={tot/max(1,steps):.5f} "
            f"val_queue_busy_RMSE={vq:.3f} "
            f"val_thr_nonSensor_RMSE={vthr_non:.3f} "
            f"val_delay_busy_RMSE={vdel_busy:.3f} "
            f"(sensor_w={float(args.sensor_w):.1f})"
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
                    "pkt_ms": float(pkt_ms),
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
        for t0 in test_pos.tolist():
            t0 = int(t0)
            H = torch.stack([enc(X[s], edges) for s in range(t0 - Kwin + 1, t0 + 1)], 0)
            P.append(forward_pred(H, t0))
            G.append(Y[t0 + 1])
        P = torch.stack(P)
        G = torch.stack(G)

    rmses = [rmse_t(P[..., k].flatten(), G[..., k].flatten()) for k in range(out_dim)]
    print(
        "TEST lead-1 multi RMSE (queue,thr,util,delay) = "
        + ", ".join(f"{x:.3f}" for x in rmses)
        + f" | saved {args.out}"
    )


if __name__ == "__main__":
    main()
