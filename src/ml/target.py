# -*- coding: utf-8 -*-
"""The DCS model target: r(theta)@32 plus the absolute polar centre.

34 columns: ``[0:32]`` is ``r(theta)`` about that slice's own centre, ``[32]`` is
``Rgeom`` and ``[33]`` is ``Zgeom``, all in absolute metres. There is deliberately no
fixed reference origin -- theta and r are already measured about (Rgeom, Zgeom), so an
offset against some constant point would leave two competing notions of origin in one
target vector.

Because the centre is absolute, ``Rgeom``'s ~2.44 m mean is out of scale with the ~0.53 m
radii even though the *spreads* match (0.060 / 0.028 / 0.056). Raw MSE would therefore let
one column of 34 dominate the early gradient into the shared trunk of the neural models,
so callers standardize with :func:`target_mean_std` (train shots only) and invert with
:func:`destandardize` before anything downstream sees the numbers.

Rationale and measurements: docs/superpowers/specs/2026-08-05-newtrain-rebuild-design.md §7.
"""
import pathlib

import numpy as np

N_RHO = 32                      # r(theta) columns
N_CENTER = 2                    # (Rgeom, Zgeom)
N_OUT = N_RHO + N_CENTER        # 34


def load_target(npz_path):
    """``(T (nt, 34) float64, finite (nt,) bool)`` for one NpzGeom shot.

    ``finite`` is True only where every column is finite. Slices rejected by the quality
    filters carry ``Y = NaN`` by design, so this is the caller's signal, not an error.
    """
    d = np.load(npz_path)
    if "center" not in d.files:
        raise ValueError(
            f"{npz_path}: missing 'center' array — the 34-column target "
            "(r(theta)@32 + absolute Rgeom, Zgeom) requires an NpzGeom-style dataset")
    Y = d["Y"].astype(float)
    C = d["center"].astype(float)
    if Y.ndim != 2 or Y.shape[1] != N_RHO:
        raise ValueError(f"{npz_path}: expected {N_RHO} rho columns, got {Y.shape}")
    if C.shape != (Y.shape[0], N_CENTER):
        raise ValueError(f"{npz_path}: center must be (nt, {N_CENTER}), got {C.shape}")
    T = np.concatenate([Y, C], axis=1)
    return T, np.isfinite(T).all(axis=1)


def target_mean_std(npz_dir, shots, eps=1e-6):
    """Per-column mean/std over ``shots``' finite **and** valid rows.

    Pass train shots only: statistics taken over val or test rows leak.
    """
    acc = []
    for s in shots:
        p = pathlib.Path(npz_dir) / f"{int(s)}.npz"
        if not p.exists():
            continue
        T, finite = load_target(p)
        v = finite & np.load(p)["valid"].astype(bool)
        if v.any():
            acc.append(T[v])
    if not acc:
        raise ValueError(f"no finite valid target rows in {npz_dir} for {len(shots)} shots")
    A = np.concatenate(acc)
    return A.mean(axis=0), np.maximum(A.std(axis=0), eps)


def standardize(T, mean, std):
    """``(T - mean) / std`` with float64 arithmetic."""
    return (np.asarray(T, float) - np.asarray(mean, float)) / np.asarray(std, float)


def destandardize(P, mean, std):
    """Inverse of :func:`standardize`; returns metres."""
    return np.asarray(P, float) * np.asarray(std, float) + np.asarray(mean, float)


def split_outputs(P):
    """``(rho (..., 32), centre (..., 2))`` from a 34-column array in metres."""
    P = np.asarray(P, float)
    if P.shape[-1] != N_OUT:
        raise ValueError(f"expected {N_OUT} columns, got {P.shape[-1]}")
    return P[..., :N_RHO], P[..., N_RHO:]


def uniform_theta(n_rho):
    """Uniform angle grid [0, 2π) with ``n_rho`` samples (endpoint excluded)."""
    n = int(n_rho)
    if n < 4:
        raise ValueError(f"n_rho must be >= 4, got {n}")
    return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)


def n_out_for(n_rho):
    """Target width for an angle count: ``n_rho`` radii + 2 centre columns."""
    return int(n_rho) + N_CENTER


def radii_from_polyline(bnd, center, theta):
    """Ray-cast radii of a closed polyline about ``center`` at each angle.

    For every angle the ray ``center + t·d`` is intersected with every
    polyline edge ``a + u·e`` (2x2 solve via cross products); the smallest
    positive ``t`` with ``u`` in ``[-1e-9, 1+1e-9]`` (vertex slack) is the
    radius. Intersection points
    lie exactly on the polyline (linear on the hit edge — no smoothing).
    A non-finite input, or any angle with no hit (non-star-shaped slice),
    returns an all-NaN row, matching the representation's star-shape
    contract.
    """
    bnd = np.asarray(bnd, float)
    c = np.asarray(center, float)
    th = np.asarray(theta, float)
    if not (np.isfinite(bnd).all() and np.isfinite(c).all()
            and np.isfinite(th).all()):
        return np.full(th.shape, np.nan)
    d = np.stack([np.cos(th), np.sin(th)], axis=1)          # (K, 2)
    dx, dy = d[:, None, 0], d[:, None, 1]
    a = bnd[None, :, :]
    e = np.roll(bnd, -1, axis=0) - bnd                      # edge vectors
    ex, ey = e[..., 0], e[..., 1]                           # (1, N)
    ax, ay = a[..., 0], a[..., 1]
    vx, vy = ax - c[0], ay - c[1]
    denom = dx * ey - dy * ex                               # cross(d, e)
    t_num = vx * ey - vy * ex                               # cross(v, e)
    u_num = vx * dy - vy * dx                               # cross(v, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(denom) > 1e-12, t_num / denom, np.inf)
        u = np.where(np.abs(denom) > 1e-12, u_num / denom, np.inf)
    # A ray through a vertex lands on u = 0 of one edge and u = 1 of the next;
    # rounding decides the sign of the ~1e-17 remainder, so admit a slack
    # window instead of rejecting the hit outright (slack ~1e-9 edge lengths).
    hit = np.isfinite(t) & (t > 0) & (u >= -1e-9) & (u < 1 + 1e-9)
    t = np.where(hit, t, np.inf)
    radii = t.min(axis=1)
    if not np.isfinite(radii).all():
        return np.full(th.shape, np.nan)
    return radii
