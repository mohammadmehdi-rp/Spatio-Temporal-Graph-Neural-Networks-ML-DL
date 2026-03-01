#!/usr/bin/env python3
"""models_gnn_multitask.py

Multi-task output heads for the existing Graph encoder in src/models_gnn.py.

We keep the encoder unchanged and only provide heads that emit K targets per
node, e.g. queue backlog, throughput, utilization, latency.
"""

import torch
import torch.nn as nn


class MultiNowcastHead(nn.Module):
    """Per-node multi-target MLP head: h[n, H] -> y[n, K]."""

    def __init__(self, hid: int = 64, out_dim: int = 4):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)  # [N,K]


class MultiTCNHead(nn.Module):
    """Temporal head for lead-1: sequence of node embeddings -> y[n,K]."""

    def __init__(self, hid: int = 64, K: int = 30, out_dim: int = 4):
        super().__init__()
        C = hid
        self.K = K
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(C, C, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
        )
        self.out = nn.Linear(C, out_dim)

    def forward(self, Hseq: torch.Tensor) -> torch.Tensor:
        # Hseq: [K,N,H]
        x = Hseq.permute(1, 2, 0)  # [N,H,K]
        z = self.net(x)  # [N,H,K]
        last = z[:, :, -1]  # [N,H]
        return self.out(last)  # [N,K]


class MultiGRUHead(nn.Module):
    def __init__(self, hid: int = 64, out_dim: int = 4):
        super().__init__()
        self.gru = nn.GRU(input_size=hid, hidden_size=hid, num_layers=1, batch_first=False)
        self.out = nn.Linear(hid, out_dim)

    def forward(self, Hseq: torch.Tensor) -> torch.Tensor:
        # Hseq: [K,N,H]
        Y, _ = self.gru(Hseq)
        last = Y[-1]
        return self.out(last)  # [N,K]


def apply_target_activations(y_raw: torch.Tensor, util_index: int = 2) -> torch.Tensor:
    """Apply simple non-negativity constraints.

    Assumptions (default label order from gnn_prep_multitarget.py):
      0: queue_pkts          -> Softplus (>=0)
      1: throughput_Mbps     -> Softplus (>=0)
      2: utilization         -> Sigmoid  (0..1)
      3: rtt_ms              -> Softplus (>=0)
    """
    y = y_raw
    if y.size(-1) >= 1:
        y0 = nn.functional.softplus(y[..., 0])
        y = torch.cat([y0.unsqueeze(-1), y[..., 1:]], dim=-1)
    if y.size(-1) >= 2:
        y1 = nn.functional.softplus(y[..., 1])
        y = torch.cat([y[..., 0:1], y1.unsqueeze(-1), y[..., 2:]], dim=-1)
    if y.size(-1) >= util_index + 1:
        u = torch.sigmoid(y[..., util_index])
        parts = []
        parts.append(y[..., :util_index])
        parts.append(u.unsqueeze(-1))
        parts.append(y[..., util_index + 1 :])
        y = torch.cat(parts, dim=-1)
    if y.size(-1) >= 4:
        y3 = nn.functional.softplus(y[..., 3])
        y = torch.cat([y[..., 0:3], y3.unsqueeze(-1), y[..., 4:]], dim=-1) if y.size(-1) > 4 else torch.cat([y[..., 0:3], y3.unsqueeze(-1)], dim=-1)
    return y
