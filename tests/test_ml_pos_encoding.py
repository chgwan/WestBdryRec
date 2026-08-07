# -*- coding: utf-8 -*-
"""The positional encodings, and the two properties the whole experiment rests on.

The 2x5 matrix asks "does real time beat step count". That question is only
meaningful if the time and index encodings are identical on a perfectly uniform
axis -- otherwise a measured difference could just be a different choice of
wavelengths. These tests pin that equivalence, plus the two numerical traps
(a dead lambda=2 channel, and float32 phase loss at large absolute positions).

Spec: docs/superpowers/specs/2026-08-07-windowed-attention-pe-matrix-design.md 3.2
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.pos_encoding import (  # noqa: E402
    ROPE_BASE, VARIANTS, apply_rope, inv_freqs, modal_cadence, upe_table, wavelengths,
)


def _uniform(nt=512, c=2.048e-3, t0=0.0533):
    """A perfectly uniform axis, with the nonzero time[0] every real shot has."""
    n0 = round(t0 / c)
    t = n0 * c + c * np.arange(nt, dtype=np.float64)   # t0 snapped to integer*c
    ipos = n0 + np.arange(nt, dtype=np.float64)         # integer-counting path
    tpos = t / c                                         # time-division path
    return t, ipos, tpos, c


# ------------------------------------------------------------------ cadence ---
def test_modal_cadence_ignores_float32_jitter_and_rare_gaps():
    c = 2.048e-3
    dt = np.full(1000, c)
    dt[:50] += 1e-6                     # storage jitter
    dt[500] = 6 * c                     # one real dropout
    t = np.concatenate([[0.0], np.cumsum(dt)])
    assert modal_cadence(t) == pytest.approx(c, abs=1e-9)


def test_modal_cadence_needs_two_samples():
    with pytest.raises(ValueError, match="cadence"):
        modal_cadence(np.array([1.0]))


# -------------------------------------------------------------- wavelengths ---
def test_wavelength_range_matches_the_spec_table():
    """lambda in [2pi, 2pi*base**((D-2)/D)] -- the top index is D-2, not D."""
    lam = wavelengths(32, ROPE_BASE)                  # rope head_dim
    assert lam.size == 16
    assert lam[0] == pytest.approx(2 * np.pi, rel=1e-9)
    assert lam[-1] == pytest.approx(2 * np.pi * ROPE_BASE ** (30 / 32), rel=1e-9)
    assert lam[-1] == pytest.approx(35332.9, rel=1e-4)
    lam256 = wavelengths(256, ROPE_BASE)              # upe_idx / upe_time
    assert lam256.size == 128
    assert lam256[-1] == pytest.approx(58469.6, rel=1e-4)
    lam128 = wavelengths(128, ROPE_BASE)              # upe_both, per half
    assert lam128.size == 64
    assert lam128[-1] == pytest.approx(54410.1, rel=1e-4)


def test_shortest_wavelength_is_2pi_not_2_so_no_channel_is_dead():
    """At lambda = 2 the sine channel is sin(pi*i) == 0 for every integer i."""
    lam = wavelengths(64, ROPE_BASE)
    assert lam.min() > 6.0, "a lambda near 2 would make one sin channel identically zero"
    tbl = upe_table("upe_idx", np.arange(64, dtype=np.float64),
                    np.arange(64, dtype=np.float64), 64)
    assert np.abs(tbl).max(axis=0).min() > 1e-3, "no channel may be identically zero"


def test_inv_freqs_rejects_odd_dim():
    with pytest.raises(ValueError, match="even"):
        inv_freqs(31)


# ---------------------------------------- THE equivalence the matrix rests on ---
def test_upe_time_equals_upe_idx_on_a_uniform_axis():
    """Spec T3: identical by construction when t == c*i, so V2 is a real null."""
    _t, ipos, tpos, _c = _uniform()
    a = upe_table("upe_time", ipos, tpos, 256)
    b = upe_table("upe_idx", ipos, tpos, 256)
    assert np.allclose(a, b, atol=1e-5), "V2's upe pair must be numerically identical"


def test_upe_both_halves_are_the_two_single_variants():
    _t, ipos, tpos, _c = _uniform()
    both = upe_table("upe_both", ipos, tpos, 256)
    assert both.shape == (ipos.size, 256)
    assert np.allclose(both[:, :128], upe_table("upe_time", ipos, tpos, 128), atol=1e-5)
    assert np.allclose(both[:, 128:], upe_table("upe_idx", ipos, tpos, 128), atol=1e-5)


def test_rope_offsets_agree_on_a_uniform_axis():
    """rope is relative, so both variants shift by the window start; then they match."""
    _t, ipos, tpos, _c = _uniform()
    assert np.allclose(ipos - ipos[0], tpos - tpos[0], atol=1e-6)


def test_a_real_gap_makes_time_and_index_disagree():
    """The signal the matrix hunts for: only the irregularity separates them."""
    c = 2.048e-3
    dt = np.full(400, c)
    dt[200] = 7 * c                                   # a 6-sample dropout
    t = np.concatenate([[0.0], np.cumsum(dt)])
    ipos = np.arange(t.size, dtype=np.float64)
    tpos = t / c
    drift = (tpos - tpos[0]) - (ipos - ipos[0])
    assert drift[-1] == pytest.approx(6.0, abs=1e-6), "6 skipped samples of drift"


# ---------------------------------------------------------------- numerics ---
def test_upe_table_is_phase_accurate_at_large_absolute_positions():
    """float32 accumulation would lose the phase: pi*i ~ 1.6e5 at i = 50616."""
    i = np.arange(50_000, 50_064, dtype=np.float64)
    tbl = upe_table("upe_idx", i, i, 64)
    ref = np.sin(i * inv_freqs(64, ROPE_BASE)[0])     # fastest channel, float64
    assert np.abs(tbl[:, 0].astype(np.float64) - ref).max() < 1e-3
    assert np.isfinite(tbl).all()
    assert tbl.dtype == np.float32


def test_upe_table_rejects_a_rope_variant():
    i = np.arange(8, dtype=np.float64)
    with pytest.raises(ValueError, match="not a upe variant"):
        upe_table("rope_idx", i, i, 64)


# -------------------------------------------------------------- apply_rope ---
def test_apply_rope_preserves_norm_and_is_a_pure_rotation():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16, 32)
    pos = torch.arange(16.0).expand(2, 16)
    fr = torch.from_numpy(inv_freqs(32)).float()
    y = apply_rope(x, pos, fr)
    assert y.shape == x.shape
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_apply_rope_at_position_zero_is_the_identity():
    x = torch.randn(1, 2, 5, 8)
    fr = torch.from_numpy(inv_freqs(8)).float()
    y = apply_rope(x, torch.zeros(1, 5), fr)
    assert torch.allclose(x, y, atol=1e-6)


def test_apply_rope_dot_product_depends_only_on_the_offset():
    """The defining property of a RELATIVE encoding -- why the window shift is safe."""
    torch.manual_seed(1)
    q, k = torch.randn(1, 1, 1, 32), torch.randn(1, 1, 1, 32)
    fr = torch.from_numpy(inv_freqs(32)).float()
    def dot(pi, pj):
        a = apply_rope(q, torch.tensor([[float(pi)]]), fr)
        b = apply_rope(k, torch.tensor([[float(pj)]]), fr)
        return float((a * b).sum())
    assert dot(10, 3) == pytest.approx(dot(1010, 1003), abs=1e-3)


def test_variants_tuple_is_exactly_the_five_spec_names():
    assert VARIANTS == ("rope_idx", "rope_time", "upe_idx", "upe_time", "upe_both")
