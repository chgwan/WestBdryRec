# src/ml/models.py
# -*- coding: utf-8 -*-
"""M1: ResMLP snapshot model. M2: Transformer over the actuator series."""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Pre-norm residual MLP block: x + W2 drop(relu(W1 LN(x)))."""

    def __init__(self, d, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.lin1 = nn.Linear(d, d)
        self.lin2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = torch.relu(self.lin1(h))
        h = self.drop(h)
        return x + self.lin2(h)


class ResMLP(nn.Module):
    """M1: actuator snapshot (n_in,) -> 32 rho, joint across angles."""

    def __init__(self, n_in, n_out=32, hidden=256, depth=4, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_in, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x):                            # (B, n_in)
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h)
        return self.out(self.norm(h))                # (B, n_out)


class ActSeqTransformer(nn.Module):
    """M2: actuator time series (B, L, n_act) + mask -> 32 rho per step.

    ``mask`` (B, L) is bool with True = keep; converted internally to the
    ``src_key_padding_mask`` convention (True = ignore).
    """

    def __init__(self, n_act, n_out=32, d=64, heads=4, depth=3, dropout=0.1, max_len=512):
        super().__init__()
        self.in_proj = nn.Linear(n_act, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_out)

    def forward(self, x, mask):                      # x (B,L,n_act), mask (B,L) bool True=keep
        h = self.in_proj(x) + self.pos[:, : x.size(1)]
        h = self.tr(h, src_key_padding_mask=~mask)
        return self.out(self.norm(h))                # (B, L, n_out)


class ActSeqGRU(nn.Module):
    """M2 (DCS): actuator series (B, L, n_act) + mask -> 32 rho per step.

    Linear-time GRU encoder (handles the full ~40-58k-step DCS series).
    ``mask`` (B, L) bool is accepted for API parity with the predict path; the
    GRU runs over all steps and the loss masks invalid steps externally.
    """

    def __init__(self, n_act, n_out=32, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_act, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x, mask):                       # x (B,L,n_act), mask (B,L) bool
        h = self.in_proj(x)
        h, _ = self.gru(h)
        return self.out(self.norm(h))                 # (B, L, n_out)
