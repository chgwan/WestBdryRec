# src/ml/pos_encoding.py
# -*- coding: utf-8 -*-
"""Positional encodings for the M3 windowed-attention tier.

Five variants spanning {relative, absolute} x {step index, real time}::

    rope_idx    rotary on Q/K, p = window-relative step offset
    rope_time   rotary on Q/K, p = window-relative elapsed time, in cadence units
    upe_idx     additive absolute table, p = sample count since ignitron
    upe_time    additive absolute table, p = real seconds since ignitron / c_modal
    upe_both    additive absolute table, half the channels each

Every position is expressed in *cadence units* -- divided by ``c_modal``, the
dataset's own modal sample spacing -- and anchored at the ignitron. On a perfectly
uniform axis ``t == c_modal * i``, so the time and index variants become identical
**by construction**. That is what makes an index-vs-time result mean "the axis is
irregular" rather than "different wavelengths were picked", and it is why the
uniform NpzUni500 arms serve as a measured seed-noise floor.

Spec: docs/superpowers/specs/2026-08-07-windowed-attention-pe-matrix-design.md 3.2
"""
import numpy as np
import torch

VARIANTS = ("rope_idx", "rope_time", "upe_idx", "upe_time", "upe_both")
ROPE_BASE = 10000.0
UPE_BASE = 10000.0


def modal_cadence(time):
    """The modal sample spacing in seconds, from one shot's ``time`` array.

    Rounded to nanoseconds before the mode is taken: float32 storage splits one
    physical cadence across many bins (only 58 % of NpzUni500's intervals are
    *exactly* 2.000 ms, though 100 % are within 10 us).
    """
    dt = np.diff(np.asarray(time, dtype=np.float64))
    if dt.size == 0:
        raise ValueError("need at least 2 samples to infer a cadence")
    u, c = np.unique(np.round(dt, 9), return_counts=True)
    return float(u[int(np.argmax(c))])


def inv_freqs(dim, base=ROPE_BASE):
    """``theta_k = base ** (-2k/dim)`` for ``k = 0 .. dim/2-1`` (float64).

    Wavelength is ``2*pi/theta_k``, so the range is ``[2*pi, 2*pi*base**((dim-2)/dim)]``
    -- the top index is ``dim-2``, not ``dim``. The ``2*pi`` floor matters: a
    wavelength of 2 would make that sine channel ``sin(pi*i) == 0`` for all integer
    ``i``, i.e. a dead channel.
    """
    if dim % 2:
        raise ValueError(f"dim must be even, got {dim}")
    return base ** (-(np.arange(0, dim, 2, dtype=np.float64) / dim))


def wavelengths(dim, base=ROPE_BASE):
    """``2*pi/theta_k``, in units of ``p``. For asserting the documented ranges."""
    return 2.0 * np.pi / inv_freqs(dim, base)


def apply_rope(x, pos, freqs):
    """Rotate ``x`` (B,H,L,D) by ``pos``*``freqs``. ``pos`` (B,L), ``freqs`` (D//2,).

    Rotates the (even, odd) channel pairs, so the Q.K dot product depends only on
    ``pos_i - pos_j`` -- which is why the caller may shift ``pos`` to the window
    start without changing anything.
    """
    ang = pos[:, None, :, None] * freqs[None, None, None, :]      # (B,1,L,D/2)
    cos, sin = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def _sincos(p, dim):
    """(L, dim) float32 interleaved sin/cos of ``p`` at the canonical wavelengths.

    Accumulated in float64 and cast at the end: these positions are ABSOLUTE, and
    the fastest channel's angle reaches ~1.6e5 rad at i = 50,616. float32 carries
    ~7 significant digits, which would leave ~2 digits of phase.
    """
    p = np.asarray(p, dtype=np.float64)
    ang = p[:, None] * inv_freqs(dim, UPE_BASE)[None, :]
    out = np.empty((p.size, dim), dtype=np.float64)
    out[:, 0::2] = np.sin(ang)
    out[:, 1::2] = np.cos(ang)
    return out.astype(np.float32)


def upe_table(variant, ipos, tpos, d_model):
    """Additive absolute ``(L, d_model)`` float32 table for a ``upe_*`` variant.

    ``ipos`` and ``tpos`` are already ignitron-anchored and in cadence units (see
    :class:`~src.ml.dataset.DCSWindowDataset`); they are deliberately NOT shifted to
    the window -- these encodings are absolute, which is the point of the variant.
    """
    if variant == "upe_idx":
        return _sincos(ipos, d_model)
    if variant == "upe_time":
        return _sincos(tpos, d_model)
    if variant == "upe_both":
        half = d_model // 2
        return np.concatenate([_sincos(tpos, half), _sincos(ipos, half)], axis=1)
    raise ValueError(f"not a upe variant: {variant!r}; expected one of "
                     f"{[v for v in VARIANTS if v.startswith('upe')]}")
