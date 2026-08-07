# -*- coding: utf-8 -*-
"""Spec T5: m3 must predict exactly the rows _pred_m2 does, in time order.

bench.score_predictions only WARNS and skips a shot whose prediction count
disagrees with the truth; score_dcs34 raises. Either way a tiling bug would
corrupt the 2x5 comparison rather than stop it, so the row count is asserted here
against the mask expression copied from scripts/train_dcs.py:_pred_m2.
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.dataset import DCSWindowDataset  # noqa: E402
from src.ml.models import ActSeqAttn  # noqa: E402
from src.ml.pos_encoding import VARIANTS  # noqa: E402
from src.ml.target import N_OUT  # noqa: E402


def _fixture(tmp_path, monkeypatch, nt=4000, n_act=3, bad=(17, 998, 3999)):
    from src.ml import dataset as D
    Y = np.tile(np.linspace(0.4, 0.6, 32), (nt, 1)).astype(np.float32)
    valid = np.ones(nt, bool)
    for b in bad:
        Y[b] = np.nan
        valid[b] = False
    np.savez(tmp_path / "7.npz", Y=Y,
             center=np.tile([2.4, 0.05], (nt, 1)).astype(np.float32),
             valid=valid, time=(0.0533 + 2.048e-3 * np.arange(nt)).astype(np.float32))
    A = np.random.default_rng(0).normal(size=(nt, n_act)).astype(np.float32)
    monkeypatch.setattr(D, "read_series",
                        lambda p, cfg, ncm: (A.copy(), valid.copy()))
    return n_act, valid


@pytest.mark.parametrize("pe", VARIANTS)
def test_gathered_rows_equal_the_predict_mask_count(tmp_path, monkeypatch, pe):
    n_act, valid = _fixture(tmp_path, monkeypatch)
    ds = DCSWindowDataset(tmp_path, [7], cfg={}, ncm={},
                          mean=np.zeros(n_act), std=np.ones(n_act),
                          pe=pe, d_model=32)
    model = ActSeqAttn(n_act=n_act, n_out=N_OUT, d=32, heads=4, depth=2, ffn=64,
                       dropout=0.0, pe=pe).eval()
    rows = []
    with torch.no_grad():
        for j in range(len(ds)):
            a, _y, lm, p = ds[j]
            z = model(a[None], p[None])[0].numpy()
            rows.append(z[lm.numpy()])
    out = np.concatenate(rows)
    # v = read_series mask & target finite -- _pred_m2's expression verbatim
    v_count = int((valid & np.isfinite(np.load(tmp_path / "7.npz")["Y"]).all(1)).sum())
    assert out.shape == (v_count, N_OUT), \
        f"{pe}: gathered {out.shape[0]} rows, predict mask has {v_count}"


def test_gathered_rows_are_in_time_order(tmp_path, monkeypatch):
    """Concatenating windows in index order must reproduce v's ordering."""
    n_act, valid = _fixture(tmp_path, monkeypatch, nt=4000, bad=())
    ds = DCSWindowDataset(tmp_path, [7], cfg={}, ncm={},
                          mean=np.zeros(n_act), std=np.ones(n_act),
                          pe="rope_idx")
    seen = []
    for j in range(len(ds)):
        _k, ws, _we, bs, be = ds.index[j]
        _a, _y, lm, _p = ds[j]
        seen.append(np.arange(ws, _we)[lm.numpy()])
    order = np.concatenate(seen)
    assert (order == np.arange(4000)).all(), "predictions must come out time-ordered"
