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
