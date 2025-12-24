#!/usr/bin/env python3
import torch, torch.nn as nn
import math

def mean_agg(h, edges, N):
    src, dst = edges
    agg = torch.zeros_like(h)
    agg.index_add_(0, dst, h[src])
    deg = torch.bincount(dst, minlength=N).clamp(min=1).unsqueeze(1).to(h.dtype)
    return agg / deg

class GraphSageLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.ReLU(), dropout=0.0):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.act = act
        self.drop = nn.Dropout(dropout)
        self.bn = nn.Identity()  # BN → Identity (more stable on small graphs)

    def forward(self, h, edges):
        N = h.size(0)
        neigh = mean_agg(h, edges, N)
        out = self.lin_self(h) + self.lin_neigh(neigh)
        out = self.act(out)
        out = self.drop(out)
        out = self.bn(out)
        return out

class RouteNetLiteLayer(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.bn = nn.Identity()

    def forward(self, h, edges):
        src, dst = edges
        q = self.q(h); k = self.k(h); v = self.v(h)

        # Scaled dot-product scores (reduces explosion)
        d = h.size(1)
        scores = (q[dst] * k[src]).sum(dim=1) / math.sqrt(d)  # [E]
        N = h.size(0)

        # TRUE per-destination max (stable softmax). Requires torch>=1.12-ish.
        max_per_dst = torch.full((N,), -torch.inf, device=h.device, dtype=scores.dtype)

        if hasattr(max_per_dst, "scatter_reduce_"):
            max_per_dst.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
        else:
            # Fallback (correct but slower): loop over N (OK for N~24, but slower overall)
            for i in range(N):
                m = scores[dst == i]
                if m.numel() > 0:
                    max_per_dst[i] = m.max()

        exp = torch.exp((scores - max_per_dst[dst]).clamp(min=-50.0, max=50.0))  # [E]
        sum_per_dst = torch.zeros((N,), device=h.device, dtype=scores.dtype)
        sum_per_dst.index_add_(0, dst, exp)
        alpha = exp / (sum_per_dst[dst] + 1e-9)  # [E]

        # Safety (should be unnecessary after stable softmax, but cheap insurance)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

        msg = v[src] * alpha.unsqueeze(1)  # [E,H]
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)

        out = self.proj(agg) + h
        out = self.act(out)
        out = self.drop(out)
        out = self.bn(out)
        return out
  

class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hid=64, layers=2, kind="sage", dropout=0.1):
        super().__init__()
        self.in_lin = nn.Linear(in_dim, hid)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            if kind=="sage":
                self.layers.append(GraphSageLayer(hid, hid, dropout=dropout))
            elif kind=="routenet":
                self.layers.append(RouteNetLiteLayer(hid, dropout=dropout))
            else:
                raise ValueError("kind must be 'sage' or 'routenet'")
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edges):
        h = self.act(self.in_lin(x))
        for l in self.layers:
            h = l(h, edges)
        return h

class NowcastHead(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )
    def forward(self, h):
        return self.mlp(h).squeeze(-1)

class TCNHead(nn.Module):
    def __init__(self, hid=64, K=10):
        super().__init__()
        C = hid
        self.net = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(C, C, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
        )
        self.out = nn.Linear(C, 1)

    def forward(self, Hseq):  # [K,N,H]
        x = Hseq.permute(1, 2, 0)   # [N,H,K]
        z = self.net(x)             # [N,H,K]
        last = z[:, :, -1]          # [N,H]
        return self.out(last).squeeze(-1)

class GRUHead(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.gru = nn.GRU(input_size=hid, hidden_size=hid, num_layers=1, batch_first=False)
        self.out = nn.Linear(hid, 1)
    def forward(self, Hseq):  # [K,N,H]
        Y, _ = self.gru(Hseq)
        last = Y[-1]
        return self.out(last).squeeze(-1)
