# -*- coding: utf-8 -*-
"""The dropout mechanism for the rope_idx-vs-rope_time dose-response.

The load-bearing rule: the drop mask is deterministic per shot and INDEPENDENT of the
training seed. rope_idx/rope_time must drop identical slices (only the PE differs), and
the 3 training seeds must drop identical slices (seed variance is pure training
stochasticity). A mask that drifted with the training seed would confound the comparison.

Spec: docs/superpowers/specs/2026-08-13-dropout-rope-idx-vs-time-design.md §4, §7
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.dataset import DCSWindowDataset, drop_keep  # noqa: E402


# ----------------------------------------------------------------- drop_keep ---
def test_drop_zero_is_all_keep():
    k = drop_keep(1000, shot=57295, drop_frac=0.0, drop_seed=0)
    assert k.dtype == bool and k.all()


def test_drop_deterministic_per_shot():
    """Same (shot, drop_seed, f) -> same mask, regardless of global RNG state."""
    a = drop_keep(2000, 57295, 0.5, 0)
    np.random.seed(12345); torch.manual_seed(999)          # perturb global state
    b = drop_keep(2000, 57295, 0.5, 0)
    assert np.array_equal(a, b), "the mask must not depend on global RNG state"


def test_drop_independent_of_training_seed():
    """THE load-bearing rule: training-seed perturbation must not move the mask."""
    a = drop_keep(2000, 57295, 0.5, drop_seed=0)
    for tseed in (0, 1, 2):
        torch.manual_seed(tseed); np.random.seed(tseed)
        b = drop_keep(2000, 57295, 0.5, drop_seed=0)
        assert np.array_equal(a, b), f"training seed {tseed} moved the drop mask"


def test_drop_fraction_and_seed_vary_the_mask():
    a = drop_keep(4000, 57295, 0.5, 0)
    assert 0.4 < a.mean() < 0.6                                    # ~half kept
    assert not np.array_equal(a, drop_keep(4000, 57295, 0.5, 1))   # seed changes it
    assert not np.array_equal(a, drop_keep(4000, 57295, 0.3, 0))   # fraction changes it


def test_drop_differs_between_shots():
    """Per-shot streams: two shots at the same f must not share one mask."""
    a = drop_keep(4000, 57295, 0.5, 0)
    b = drop_keep(4000, 57296, 0.5, 0)
    assert not np.array_equal(a, b)


def test_drop_never_empties_a_shot():
    """Even at f=0.99 the floor guard keeps the shot non-degenerate."""
    k = drop_keep(500, 57295, 0.99, 0)
    assert int(k.sum()) >= 100


# ---------------------------------------------------- DCSWindowDataset dropout ---
def _fixture(tmp_path, nt=4000, n_act=3, monkeypatch=None):
    """One synthetic NpzGeom-shaped shot (7.npz) on a uniform 2.048 ms axis."""
    from src.ml import dataset as D
    Y = np.tile(np.linspace(0.4, 0.6, 32), (nt, 1)).astype(np.float32)
    valid = np.ones(nt, bool)
    t = (0.0533 + 2.048e-3 * np.arange(nt)).astype(np.float32)
    np.savez(tmp_path / "7.npz", Y=Y,
             center=np.tile([2.4, 0.05], (nt, 1)).astype(np.float32),
             valid=valid, time=t)
    A = np.ones((nt, n_act), np.float32)
    monkeypatch.setattr(D, "read_series", lambda p, cfg, ncm: (A.copy(), valid.copy()))
    return n_act


def _ds(tmp_path, pe="rope_idx", **kw):
    return DCSWindowDataset(tmp_path, [7], cfg={}, ncm={}, mean=np.zeros(3),
                            std=np.ones(3), pe=pe, **kw)


def test_dataset_f0_reproduces_no_dropout(tmp_path, monkeypatch):
    """f=0 must keep every step and produce the same windows as today."""
    _fixture(tmp_path, monkeypatch=monkeypatch)
    base = _ds(tmp_path)
    f0 = _ds(tmp_path, drop_frac=0.0, drop_seed=0)
    assert len(base.index) == len(f0.index)
    for j in range(len(base.index)):
        assert f0.index[j] == base.index[j], "f=0 windows must match exactly"


def test_dataset_f_positive_shortens_and_keeps_subset(tmp_path, monkeypatch):
    """window_blocks partitions [0, nt) exactly, so the last block end == n_retained."""
    _fixture(tmp_path, monkeypatch=monkeypatch)
    base = _ds(tmp_path)
    ds = _ds(tmp_path, drop_frac=0.5, drop_seed=0)
    assert base.index[-1][4] == 4000                  # no dropout: all 4000 steps
    n_keep = ds.index[-1][4]
    assert 1800 < n_keep < 2200, f"f=0.5 should retain ~half, got {n_keep}"
    assert len(ds.index) < len(base.index)            # fewer windows to tile


def test_dataset_same_drop_seed_drops_same_slices_across_pe(tmp_path, monkeypatch):
    """rope_idx and rope_time must drop IDENTICAL slices."""
    _fixture(tmp_path, monkeypatch=monkeypatch)
    a = _ds(tmp_path, pe="rope_idx", drop_frac=0.5, drop_seed=0)
    b = _ds(tmp_path, pe="rope_time", drop_frac=0.5, drop_seed=0)
    assert len(a.index) == len(b.index)
    for j in range(len(a.index)):
        assert a.index[j] == b.index[j], "rope_idx and rope_time must retain the same steps"


def test_dataset_rope_positions_diverge_under_dropout(tmp_path, monkeypatch):
    """After dropout rope_idx is consecutive; rope_time carries the real gaps."""
    _fixture(tmp_path, nt=3000, monkeypatch=monkeypatch)
    ri = _ds(tmp_path, pe="rope_idx", drop_frac=0.6, drop_seed=0)
    rt = _ds(tmp_path, pe="rope_time", drop_frac=0.6, drop_seed=0)
    _a, _y, lm_i, p_i = ri[0]
    _a2, _y2, lm_t, p_t = rt[0]
    scored_i = p_i[lm_i].numpy()
    assert np.allclose(np.diff(scored_i[:20]), 1.0), "rope_idx must be consecutive"
    scored_t = p_t[lm_t].numpy()
    assert (np.diff(scored_t) > 1.5).any(), "rope_time must show real gaps under dropout"
