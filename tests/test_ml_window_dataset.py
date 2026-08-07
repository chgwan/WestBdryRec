# -*- coding: utf-8 -*-
"""The window tiling contract -- the thing that makes M3's numbers comparable at all.

Scored blocks must be disjoint and cover [0, nt) exactly. Two things break silently
otherwise:

  training -- a slice counted twice is silently reweighted in the loss;
  eval     -- score_dcs34 raises on a row-count mismatch, but bench.score_predictions
              only WARNS and skips the shot, so a partial tiling could quietly drop
              shots from a comparison.

The loss mask must also equal _pred_m2's v (read_series's mask AND the target's
finite mask), so m3 keeps or drops exactly the slices m2 does.

Spec: docs/superpowers/specs/2026-08-07-windowed-attention-pe-matrix-design.md 4
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.dataset import (  # noqa: E402
    CTX_DEFAULT, W_DEFAULT, DCSWindowDataset, pad_collate, window_blocks,
)
from src.ml.pos_encoding import VARIANTS  # noqa: E402
from src.ml.target import N_OUT  # noqa: E402


# -------------------------------------------------------------------- T1 ---
@pytest.mark.parametrize("nt", [1, 7, 512, 1535, 1536, 1537, 2047, 2048, 2049,
                                3072, 6047, 9999, 49426])
def test_scored_blocks_partition_the_shot_exactly(nt):
    blocks = window_blocks(nt, W_DEFAULT, CTX_DEFAULT)
    covered = np.zeros(nt, dtype=int)
    for ws, we, bs, be in blocks:
        assert 0 <= ws <= bs < be <= we <= nt
        assert we - ws <= W_DEFAULT, "a window may never exceed w"
        assert bs - ws <= CTX_DEFAULT, "the context prefix may never exceed ctx"
        covered[bs:be] += 1
    assert (covered == 1).all(), (
        f"nt={nt}: every step must be scored exactly once, got counts "
        f"{sorted(set(covered.tolist()))}")


def test_stride_is_w_minus_ctx_and_context_is_full_after_the_first_block():
    blocks = window_blocks(6047)
    assert blocks[0] == (0, 1536, 0, 1536), "no context exists before t=0"
    assert blocks[1] == (1024, 3072, 1536, 3072)
    assert blocks[-1][3] == 6047, "the tiling must reach the end of the shot"
    for ws, we, bs, be in blocks[1:]:
        assert bs - ws == CTX_DEFAULT
        assert we - ws <= W_DEFAULT


def test_short_shot_is_one_window():
    assert window_blocks(1000) == [(0, 1000, 0, 1000)]


def test_ctx_must_be_smaller_than_w():
    with pytest.raises(ValueError, match="must be <"):
        window_blocks(100, w=512, ctx=512)


# ------------------------------------------------------------- the dataset ---
def _fixture(tmp_path, nt=4000, n_act=3, bad=(17,), monkeypatch=None):
    """One synthetic shot: a NpzGeom-shaped npz plus a patched read_series."""
    from src.ml import dataset as D
    Y = np.tile(np.linspace(0.4, 0.6, 32), (nt, 1)).astype(np.float32)
    for b in bad:
        Y[b] = np.nan                                # a filter-rejected slice
    center = np.tile([2.4, 0.05], (nt, 1)).astype(np.float32)
    valid = np.ones(nt, bool)
    for b in bad:
        valid[b] = False
    t = (0.0533 + 2.048e-3 * np.arange(nt)).astype(np.float32)
    p = tmp_path / "7.npz"
    np.savez(p, Y=Y, center=center, valid=valid, time=t)
    A = np.ones((nt, n_act), np.float32)
    monkeypatch.setattr(D, "read_series",
                        lambda path, cfg, ncm: (A.copy(), valid.copy()))
    return p, nt, n_act, valid


@pytest.mark.parametrize("pe", VARIANTS)
def test_every_valid_slice_is_scored_exactly_once(tmp_path, monkeypatch, pe):
    _p, nt, n_act, valid = _fixture(tmp_path, monkeypatch=monkeypatch)
    ds = DCSWindowDataset(tmp_path, [7], cfg={}, ncm={},
                          mean=np.zeros(n_act), std=np.ones(n_act), pe=pe, d_model=32)
    counted = np.zeros(nt, dtype=int)
    for j in range(len(ds)):
        (_ws, _we, bs, be) = ds.index[j][1:]
        _a, _y, lm, _p = ds[j]
        counted[bs:be] += lm.numpy()[bs - _ws:be - _ws]
    assert (counted == valid.astype(int)).all(), \
        "the loss must see each valid slice once and each invalid slice never"


def test_targets_are_finite_where_the_mask_is_set(tmp_path, monkeypatch):
    """NaN * 0 = NaN would poison the loss -- the trap DCSSeqDataset documents."""
    _p, _nt, n_act, _v = _fixture(tmp_path, monkeypatch=monkeypatch)
    ds = DCSWindowDataset(tmp_path, [7], cfg={}, ncm={},
                          mean=np.zeros(n_act), std=np.ones(n_act))
    for j in range(len(ds)):
        _a, y, lm, _p = ds[j]
        assert torch.isfinite(y).all(), "every target handed to the loss must be finite"
        pred = torch.zeros_like(y, requires_grad=True)
        w = lm.unsqueeze(-1).float()
        loss = (((pred - y) ** 2) * w).sum() / w.sum().clamp(min=1.0)
        assert torch.isfinite(loss)


def test_context_prefix_is_never_scored(tmp_path, monkeypatch):
    _p, _nt, n_act, _v = _fixture(tmp_path, monkeypatch=monkeypatch)
    ds = DCSWindowDataset(tmp_path, [7], cfg={}, ncm={},
                          mean=np.zeros(n_act), std=np.ones(n_act))
    for j in range(1, len(ds)):
        ws, _we, bs, _be = ds.index[j][1:]
        _a, _y, lm, _p = ds[j]
        assert not lm[: bs - ws].any(), "context steps are attended, never scored"


@pytest.mark.parametrize("pe", VARIANTS)
def test_payload_rank_matches_the_variant(tmp_path, monkeypatch, pe):
    _p, _nt, n_act, _v = _fixture(tmp_path, monkeypatch=monkeypatch)
    ds = DCSWindowDataset(tmp_path, [7], cfg={}, ncm={},
                          mean=np.zeros(n_act), std=np.ones(n_act), pe=pe, d_model=32)
    _a, _y, _lm, p = ds[0]
    assert p.dim() == (1 if pe.startswith("rope") else 2)
    if p.dim() == 2:
        assert p.shape[1] == 32


def test_rope_offsets_start_at_zero_in_every_window(tmp_path, monkeypatch):
    """Absolute t/c would reach ~49,300 and lose float32 phase precision."""
    _p, _nt, n_act, _v = _fixture(tmp_path, monkeypatch=monkeypatch)
    for pe in ("rope_idx", "rope_time"):
        ds = DCSWindowDataset(tmp_path, [7], cfg={}, ncm={},
                              mean=np.zeros(n_act), std=np.ones(n_act), pe=pe)
        for j in range(len(ds)):
            _a, _y, _lm, p = ds[j]
            assert abs(float(p[0])) < 1e-4, f"{pe}: window offsets must start at 0"
            assert float(p.max()) < W_DEFAULT + 1


def test_upe_positions_are_absolute_and_differ_between_windows(tmp_path, monkeypatch):
    _p, _nt, n_act, _v = _fixture(tmp_path, monkeypatch=monkeypatch)
    ds = DCSWindowDataset(tmp_path, [7], cfg={}, ncm={},
                          mean=np.zeros(n_act), std=np.ones(n_act),
                          pe="upe_idx", d_model=32)
    # windows have different lengths (the first is w-ctx, later ones are w); compare
    # the two absolute tables over their common row count so allclose can run.
    p0, p1 = ds[0][3], ds[1][3]
    n = min(p0.shape[0], p1.shape[0])
    assert not torch.allclose(p0[:n], p1[:n]), \
        "an absolute encoding must distinguish where the window sits"


def test_unknown_variant_is_rejected(tmp_path, monkeypatch):
    _p, _nt, n_act, _v = _fixture(tmp_path, monkeypatch=monkeypatch)
    with pytest.raises(ValueError, match="pe must be one of"):
        DCSWindowDataset(tmp_path, [7], cfg={}, ncm={},
                         mean=np.zeros(n_act), std=np.ones(n_act), pe="nope")


# ------------------------------------------------------------- pad_collate ---
def test_pad_collate_right_pads_and_masks_the_padding(tmp_path, monkeypatch):
    _p, _nt, n_act, _v = _fixture(tmp_path, monkeypatch=monkeypatch)
    ds = DCSWindowDataset(tmp_path, [7], cfg={}, ncm={},
                          mean=np.zeros(n_act), std=np.ones(n_act))
    batch = [ds[0], ds[len(ds) - 1]]                 # 1536-long and a ragged tail
    A, Y, M, P = pad_collate(batch, w=W_DEFAULT)
    assert A.shape == (2, W_DEFAULT, n_act)
    assert Y.shape == (2, W_DEFAULT, N_OUT)
    assert M.shape == (2, W_DEFAULT) and M.dtype == torch.bool
    assert P.shape == (2, W_DEFAULT)
    for i, (a, _y, _lm, _p) in enumerate(batch):
        assert not M[i, a.shape[0]:].any(), "padded steps must never be scored"


def test_pad_collate_handles_a_upe_table(tmp_path, monkeypatch):
    _p, _nt, n_act, _v = _fixture(tmp_path, monkeypatch=monkeypatch)
    ds = DCSWindowDataset(tmp_path, [7], cfg={}, ncm={},
                          mean=np.zeros(n_act), std=np.ones(n_act),
                          pe="upe_both", d_model=32)
    A, _Y, _M, P = pad_collate([ds[0], ds[1]], w=W_DEFAULT)
    assert P.shape == (2, W_DEFAULT, 32)
