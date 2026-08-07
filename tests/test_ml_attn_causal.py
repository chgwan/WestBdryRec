# -*- coding: utf-8 -*-
"""The two invariants that make the windowed attention model correct at all.

T2 -- right-padding needs NO attention mask. Under causality a real position never
attends rightward, so padded steps cannot reach it. If this is false, every number
the 2x5 matrix produces is contaminated by padding.

T4 -- the mem-efficient SDPA kernel must be selected, never math. Passing an
attn_mask silently drops to math, which materialises (B,8,2048,2048): measured
8.18 GiB vs 0.26 GiB at batch 16 fp32, per layer.

Spec: docs/superpowers/specs/2026-08-07-windowed-attention-pe-matrix-design.md 3.1
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.models import ActSeqAttn  # noqa: E402
from src.ml.pos_encoding import VARIANTS, upe_table  # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _model(pe, d=32, heads=4, depth=2, n_act=5):
    torch.manual_seed(0)
    m = ActSeqAttn(n_act=n_act, n_out=34, d=d, heads=heads, depth=depth,
                   ffn=4 * d, dropout=0.0, pe=pe)
    return m.eval()


def _payload(pe, L, d=32, start=0):
    """The (L,) rope offsets or (L,d) upe table a window of length L would carry."""
    ipos = np.arange(start, start + L, dtype=np.float64)
    if pe.startswith("rope"):
        return torch.from_numpy((ipos - ipos[0]).astype(np.float32))
    return torch.from_numpy(upe_table(pe, ipos, ipos, d))


# --------------------------------------------------------------------- T2 ---
@pytest.mark.parametrize("pe", VARIANTS)
def test_right_padding_cannot_change_real_positions(pe):
    """THE justification for using no attention mask. Must hold for every variant."""
    m, L, pad = _model(pe), 12, 7
    torch.manual_seed(1)
    x = torch.randn(1, L, 5)
    p = _payload(pe, L)
    with torch.no_grad():
        short = m(x, p[None])

    xp = torch.cat([x, torch.randn(1, pad, 5)], dim=1)          # garbage in the pad
    pp = torch.cat([p, torch.zeros_like(p[:pad])], dim=0)
    with torch.no_grad():
        long = m(xp, pp[None])

    assert torch.allclose(short, long[:, :L], atol=1e-5), (
        f"{pe}: padding leaked into real positions -- an attention mask would be "
        "required and the memory argument in spec 3.1 collapses")


def test_a_later_step_cannot_influence_an_earlier_one():
    """Plain causality, independent of padding."""
    m = _model("rope_idx")
    torch.manual_seed(2)
    x = torch.randn(1, 10, 5)
    p = _payload("rope_idx", 10)[None]
    with torch.no_grad():
        a = m(x, p)
    x2 = x.clone()
    x2[0, 7:] += 100.0                                  # perturb the future only
    with torch.no_grad():
        b = m(x2, p)
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-5)
    assert not torch.allclose(a[:, 7:], b[:, 7:], atol=1e-3)


# --------------------------------------------------------------------- T4 ---
@cuda_only
def test_forward_runs_without_the_math_backend():
    """Restrict SDPA to flash/mem-efficient; an attn_mask would raise here."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    m = ActSeqAttn(n_act=5, n_out=34, d=64, heads=4, depth=2, ffn=256,
                   dropout=0.0, pe="rope_idx").cuda().eval()
    x = torch.randn(2, 2048, 5, device="cuda")
    p = torch.arange(2048.0, device="cuda").expand(2, 2048)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        with torch.no_grad():
            out = m(x, p)
    assert out.shape == (2, 2048, 34)


@cuda_only
def test_peak_memory_stays_far_below_the_math_backend():
    """Math would need GiB here; mem-efficient needs a fraction of one."""
    m = ActSeqAttn(n_act=5, n_out=34, d=64, heads=4, depth=2, ffn=256,
                   dropout=0.0, pe="rope_idx").cuda().eval()
    x = torch.randn(4, 2048, 5, device="cuda")
    p = torch.arange(2048.0, device="cuda").expand(4, 2048)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        m(x, p)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    assert peak < 1.0, f"peak {peak:.2f} GiB -- looks like the math backend"


# ------------------------------------------------------------------ shapes ---
@pytest.mark.parametrize("pe", VARIANTS)
def test_output_shape_and_param_count(pe):
    m = _model(pe, d=256, heads=8, depth=6, n_act=22)
    x = torch.randn(2, 64, 22)
    p = _payload(pe, 64, d=256)
    if p.dim() == 1:
        p = p[None].expand(2, 64)
    else:
        p = p[None].expand(2, 64, 256)
    with torch.no_grad():
        out = m(x, p)
    assert out.shape == (2, 64, 34)
    n = sum(q.numel() for q in m.parameters())
    assert 4.0e6 < n < 6.0e6, f"expected ~4.8M params at d256/8h/6L, got {n:,}"


def test_wrong_payload_rank_fails_loudly():
    m = _model("upe_idx")
    with pytest.raises(AssertionError, match="table"):
        m(torch.randn(1, 8, 5), torch.zeros(1, 8))
    m2 = _model("rope_idx")
    with pytest.raises(AssertionError, match="offsets"):
        m2(torch.randn(1, 8, 5), torch.zeros(1, 8, 32))


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="pe must be one of"):
        ActSeqAttn(n_act=5, pe="sinusoid")
