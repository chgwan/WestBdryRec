# -*- coding: utf-8 -*-
"""The masked sequence loss must survive rejected slices.

NpzGeom stores Y = NaN where the quality filters rejected a slice. M2 masks by
multiplication, and NaN * 0 = NaN -- which poisons the loss and every gradient. The
dataset must therefore zero-fill non-finite targets while keeping them masked out.
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.target import N_OUT  # noqa: E402


def test_nan_times_zero_poisons_a_multiplicative_mask():
    """Pin the failure mode itself, so nobody 'simplifies' the zero-fill away."""
    Y = torch.tensor([[1.0, float("nan"), 3.0]])
    pred = torch.zeros_like(Y, requires_grad=True)
    w = torch.tensor([[1.0, 0.0, 1.0]])
    loss = (((pred - Y) ** 2) * w).sum() / w.sum()
    assert torch.isnan(loss), "if this ever passes, the hazard is gone and so is the fix"
    loss.backward()
    assert torch.isnan(pred.grad).any(), "NaN target contaminates gradients"


def test_seq_dataset_zero_fills_rejected_targets(tmp_path, monkeypatch):
    from src.ml import dataset as D
    from src.ml import train as T  # DCSSeqDataset is re-exported here

    nt, n_rho, n_act = 8, 32, 3
    Y = np.tile(np.linspace(0.4, 0.6, n_rho), (nt, 1)).astype(np.float32)
    Y[3] = np.nan                                    # a filter-rejected slice
    A = np.ones((nt, n_act), np.float32)
    mask = np.ones(nt, bool)

    monkeypatch.setattr(D, "read_series", lambda p, cfg, ncm: (A.copy(), mask.copy()))
    p = tmp_path / "1.npz"
    np.savez(p, Y=Y, center=np.zeros((nt, 2), np.float32), valid=mask,
             time=np.arange(nt, dtype=np.float32))

    ds = T.DCSSeqDataset(tmp_path, [1], cfg={}, ncm={},
                         mean=np.zeros(n_act), std=np.ones(n_act))
    a, y, m = ds[0]
    assert torch.isfinite(y).all(), "targets handed to the loss must be finite"
    assert not bool(m[3]), "the rejected step must still be masked out"
    assert bool(m[0]) and bool(m[7])

    # the trainer's exact objective must now be finite, with zero gradient at the mask
    pred = torch.zeros_like(y, requires_grad=True)
    w = m.unsqueeze(-1).float()
    loss = (((pred - y) ** 2) * w).sum() / w.sum().clamp(min=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.allclose(pred.grad[3], torch.zeros(N_OUT))
