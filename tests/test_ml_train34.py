# -*- coding: utf-8 -*-
"""The three DCS models must emit 34 outputs and ship target standardization stats."""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.models import ActSeqGRU, ResMLP  # noqa: E402
from src.ml.target import N_OUT  # noqa: E402


def test_models_accept_34_outputs():
    m1 = ResMLP(n_in=7, n_out=N_OUT, hidden=16, depth=1, dropout=0.0)
    assert m1(torch.zeros(4, 7)).shape == (4, N_OUT)
    m2 = ActSeqGRU(n_act=5, n_out=N_OUT, hidden=8, layers=1, dropout=0.0)
    out = m2(torch.zeros(1, 6, 5), torch.ones(1, 6, dtype=torch.bool))
    assert out.shape == (1, 6, N_OUT)


def test_snapshot_dataset_yields_standardized_34_targets(tmp_path, monkeypatch):
    from src.ml import dataset as D
    from src.ml import train as T  # DCSSnapshotDataset is re-exported here

    nt, n_feat = 10, 6
    feats = np.random.default_rng(0).normal(size=(nt, n_feat)).astype(np.float32)
    mask = np.ones(nt, bool)
    monkeypatch.setattr(D, "read_snapshot", lambda p, cfg, ncm: (feats.copy(), mask.copy()))

    Y = np.tile(np.linspace(0.4, 0.6, 32), (nt, 1)).astype(np.float32)
    C = np.column_stack([np.full(nt, 2.44), np.full(nt, -0.02)]).astype(np.float32)
    np.savez(tmp_path / "1.npz", Y=Y, center=C, valid=mask,
             time=np.arange(nt, dtype=np.float32))

    from src.ml.target import target_mean_std
    tm, ts = target_mean_std(tmp_path, [1])
    ds = T.DCSSnapshotDataset(tmp_path, [1], cfg={}, ncm={},
                              mean=np.zeros(n_feat), std=np.ones(n_feat),
                              tgt_mean=tm, tgt_std=ts)
    _x, y = ds[0]
    assert y.shape == (N_OUT,)
    assert torch.isfinite(y).all()
    # constant columns standardize to ~0 given the floored std
    assert abs(float(y[32])) < 1e-3
