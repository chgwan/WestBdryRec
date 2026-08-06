# -*- coding: utf-8 -*-
"""Unit tests for the 34-column DCS target (r(theta)@32 + absolute centre)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.target import (  # noqa: E402
    N_CENTER, N_OUT, N_RHO, destandardize, load_target, split_outputs,
    standardize, target_mean_std,
)


def _write(path, nt=6, nan_rows=(), valid=None):
    """A minimal NpzGeom-shaped file: Y, center, valid, time."""
    Y = np.tile(np.linspace(0.4, 0.6, N_RHO), (nt, 1))
    C = np.column_stack([np.full(nt, 2.44), np.full(nt, -0.02)])
    for j in nan_rows:
        Y[j] = np.nan
    v = np.ones(nt, bool) if valid is None else np.asarray(valid, bool)
    np.savez(path, Y=Y.astype(np.float32), center=C.astype(np.float32),
             valid=v, time=np.arange(nt, dtype=np.float32) * 0.01)
    return path


def test_load_target_concatenates_rho_then_centre(tmp_path):
    p = _write(tmp_path / "1.npz")
    T, finite = load_target(p)
    assert T.shape == (6, N_OUT) and N_OUT == N_RHO + N_CENTER
    assert np.allclose(T[:, :N_RHO], np.linspace(0.4, 0.6, N_RHO))
    assert np.allclose(T[:, N_RHO], 2.44)      # absolute Rgeom, not an offset
    assert np.allclose(T[:, N_RHO + 1], -0.02)
    assert finite.all()


def test_load_target_flags_nan_rows(tmp_path):
    p = _write(tmp_path / "1.npz", nan_rows=(2, 4))
    T, finite = load_target(p)
    assert not finite[2] and not finite[4] and finite.sum() == 4


def test_load_target_rejects_wrong_shapes(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, Y=np.zeros((5, 31), np.float32),
             center=np.zeros((5, 2), np.float32), valid=np.ones(5, bool))
    with pytest.raises(ValueError, match="rho columns"):
        load_target(p)
    np.savez(p, Y=np.zeros((5, N_RHO), np.float32),
             center=np.zeros((5, 3), np.float32), valid=np.ones(5, bool))
    with pytest.raises(ValueError, match=r"center must be"):
        load_target(p)


def test_target_mean_std_uses_only_finite_valid_rows(tmp_path):
    # row 0 invalid, row 1 NaN -> neither may influence the statistics
    _write(tmp_path / "1.npz", nt=4, nan_rows=(1,),
           valid=[False, True, True, True])
    mean, std = target_mean_std(tmp_path, [1])
    assert mean.shape == (N_OUT,) and std.shape == (N_OUT,)
    assert np.allclose(mean[N_RHO], 2.44)
    assert (std > 0).all(), "std must be floored so division is safe"


def test_standardize_roundtrip(tmp_path):
    p = _write(tmp_path / "1.npz")
    T, _ = load_target(p)
    mean = T.mean(0)
    std = np.maximum(T.std(0), 1e-6)
    Z = standardize(T, mean, std)
    assert np.allclose(destandardize(Z, mean, std), T, atol=1e-9)


def test_split_outputs():
    P = np.arange(2 * N_OUT, dtype=float).reshape(2, N_OUT)
    rho, centre = split_outputs(P)
    assert rho.shape == (2, N_RHO) and centre.shape == (2, N_CENTER)
    assert np.array_equal(centre[:, 0], P[:, N_RHO])
    with pytest.raises(ValueError, match="columns"):
        split_outputs(np.zeros((2, N_RHO)))
