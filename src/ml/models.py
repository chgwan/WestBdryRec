# src/ml/models.py
# -*- coding: utf-8 -*-
"""M1: ResMLP snapshot model. M2: Transformer over the actuator series."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos_encoding import VARIANTS, apply_rope, inv_freqs


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


class _Attn(nn.Module):
    """Causal multi-head self-attention, optionally rotary on Q/K."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        if d % heads:
            raise ValueError(f"d_model {d} must be divisible by heads {heads}")
        self.h, self.dh, self.p = heads, d // heads, dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, rope_pos, rope_freqs):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (z.view(B, L, self.h, self.dh).transpose(1, 2) for z in (q, k, v))
        if rope_pos is not None:
            q = apply_rope(q, rope_pos, rope_freqs)
            k = apply_rope(k, rope_pos, rope_freqs)
        # is_causal=True and NO attn_mask, deliberately: a mask forces the math
        # backend, which materialises (B, heads, L, L) -- 8.18 GiB vs 0.26 GiB
        # measured at batch 16 / L 2048 / fp32, per layer. See spec 3.1 and
        # tests/test_ml_attn_causal.py.
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, L, D))


class ActSeqAttn(nn.Module):
    """M3: windowed causal transformer. (B,L,n_act) + positions -> (B,L,34).

    ``pe`` selects one of :data:`~src.ml.pos_encoding.VARIANTS`. ``rope_*`` rotates
    Q/K by window-relative offsets; ``upe_*`` adds an absolute table to the projected
    input. Exactly one mechanism is ever active, so the encoding is a clean 5-level
    factor rather than a stack.

    Right-padding needs no attention mask: under causality a real position never
    attends rightward, so padded steps cannot reach it, and their outputs are dropped
    by the loss mask. ``tests/test_ml_attn_causal.py`` asserts this for all 5 variants
    rather than trusting it.
    """

    def __init__(self, n_act, n_out=34, d=256, heads=8, depth=6, ffn=1024,
                 dropout=0.1, pe="rope_idx"):
        super().__init__()
        if pe not in VARIANTS:
            raise ValueError(f"pe must be one of {VARIANTS}, got {pe!r}")
        self.pe = pe
        self.in_proj = nn.Linear(n_act, d)
        self.attn = nn.ModuleList([_Attn(d, heads, dropout) for _ in range(depth)])
        self.ln_a = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.ln_f = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.ff = nn.ModuleList([
            nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(ffn, d)) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_out)
        self.register_buffer(
            "rope_freqs", torch.from_numpy(inv_freqs(d // heads)).float(),
            persistent=False)

    def forward(self, x, p):
        """``x`` (B,L,n_act); ``p`` is (B,L) rope offsets or the (B,L,d) upe table.

        Which one is decided by ``self.pe`` and produced by ``DCSWindowDataset``; the
        rank is asserted so a variant/dataset mismatch fails loudly instead of
        broadcasting into silence.
        """
        h = self.in_proj(x)
        if self.pe.startswith("upe"):
            assert p.dim() == 3, \
                f"{self.pe} needs a (B,L,d) table, got {tuple(p.shape)}"
            h = h + p
            rope_pos = None
        else:
            assert p.dim() == 2, \
                f"{self.pe} needs (B,L) offsets, got {tuple(p.shape)}"
            rope_pos = p
        for at, na, ff, nf in zip(self.attn, self.ln_a, self.ff, self.ln_f):
            h = h + at(na(h), rope_pos, self.rope_freqs)
            h = h + ff(nf(h))
        return self.out(self.norm(h))
