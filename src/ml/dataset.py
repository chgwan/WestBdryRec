# src/ml/dataset.py
# -*- coding: utf-8 -*-
"""Feature engineering + torch datasets for the actuator-predictor models.

``engineer(h5_path)`` builds the M0 snapshot (raw actuators + derived features).
All torch datasets live here: ``SnapshotDataset`` / ``SeqDataset`` (IMAS T0 H5 +
NPZ target) and ``DCSSnapshotDataset`` / ``DCSSeqDataset`` (strict-actuator NPZ,
34 outputs) for M1/M2.

The raw actuator names (RAW) and derived feature config are read from
configs/m0_model.yml (single source of truth, no hardcoded names).
"""
import json
import pathlib

import h5py
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset

from ..proj_config import get_proj_config
from ..data.imas_flat_top import flat_top_window
from .dcs_features import read_snapshot, read_series
from .pos_encoding import VARIANTS, modal_cadence, upe_table
from .target import load_target, standardize


# ── Feature engineering (M0 snapshot from raw actuators) ─────────────

def _load_model_inputs():
    """Read M0 model config. input_names: primary = NPZ meta.json (self-describing
    built data); fallback = m0_model.yml. derived + field roles: from m0_model.yml.
    Returns ``(raw_names, n_pf, derived, b0_name, lh_name, ic_name)``."""
    cfg = get_proj_config()
    config_path = cfg.base_dir / "configs" / "m0_model.yml"
    with open(config_path) as f:
        mcfg = yaml.safe_load(f)
    names = None
    meta_path = cfg.imas_npz_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            names = json.load(f).get("input_names")
    if not names:
        names = mcfg["inputs"]
    n_pf = sum(1 for n in names if n.startswith("pf_"))
    return (names, n_pf, mcfg["derived"],
            mcfg["b0_field"], mcfg["lh_field"], mcfg["ic_field"])


RAW, N_PF, DERIVED, _B0_NAME, _LH_NAME, _IC_NAME = _load_model_inputs()
FEATURE_ORDER = RAW + DERIVED
_B0_IDX = RAW.index(_B0_NAME)
_LH_IDX = RAW.index(_LH_NAME)
_IC_IDX = RAW.index(_IC_NAME)


def _read(h, name, nt, fill=0.0):
    if name not in h:
        return np.full(nt, fill)
    ds = h[name]
    if getattr(ds, "shape", None) is None:
        return np.full(nt, fill)
    return np.asarray(ds, float).reshape(-1)


def engineer(h5_path):
    """Build the M0 snapshot: raw actuators + derived features + flat-top valid mask.

    Returns ``(feats (nt, len(FEATURE_ORDER)), valid (nt,))``."""
    h5_path = pathlib.Path(h5_path)
    with h5py.File(h5_path, "r") as h:
        t = np.asarray(h["time"], float)
        nt = t.size
        raw = np.column_stack([_read(h, n, nt, fill=0.0) for n in RAW])
        pf = raw[:, :N_PF]
        b0 = raw[:, _B0_IDX]
        lh = raw[:, _LH_IDX]; ic = raw[:, _IC_IDX]
        ip = _read(h, "ip", nt, fill=np.nan)
        # derived (non-circular, per-slice)
        pf_norm = np.linalg.norm(pf, axis=1) / np.maximum(np.abs(b0), 1e-9)
        heat = lh + ic
        win = flat_top_window(t, ip) if np.isfinite(ip).all() else None
        t0 = t[win[0]] if win is not None else t[0]
        time_in_flat = (t - t0)
        dt = np.gradient(t)
        cum_heat = np.concatenate([[0.0], np.cumsum(heat[:-1] * dt[1:])])
        feats = np.column_stack([raw, pf_norm, heat, time_in_flat, cum_heat]).astype(np.float32)
        valid = np.isfinite(feats).all(axis=1)
        if win is not None:
            m = np.zeros(nt, bool); m[win[0]:win[1] + 1] = True
            valid &= m
    return feats, valid


# ── IMAS T0 torch datasets (M1 snapshot, M2 sequence) ────────────────

def keep_mask(std):
    """Boolean column mask selecting std > 0 (drops constant features)."""
    return np.asarray(std, float) > 0


class SnapshotDataset(Dataset):
    """Per-slice (engineered-actuator snapshot -> rho) over a shot set.

    Uses :func:`engineer` to build features. Inputs are standardized with TRAIN
    ``(mean, std)``; constant columns are dropped via ``keep = std > 0``.
    """

    def __init__(self, h5_dir, npz_dir, shots, mean, std, max_per_shot=None):
        self.rows = []
        rng = np.random.default_rng(0)
        self.keep = keep_mask(std)
        for s in shots:
            f = pathlib.Path(h5_dir) / f"{int(s)}.h5"
            if not f.exists():
                continue
            X, vfeat = engineer(f)
            d = np.load(pathlib.Path(npz_dir) / f"{int(s)}.npz")
            Y = d["Y"].astype(np.float32)
            v = (vfeat & d["valid"].astype(bool)
                 & np.isfinite(Y).all(1) & np.isfinite(X).all(1))
            idx = np.where(v)[0]
            if max_per_shot and idx.size > max_per_shot:
                idx = np.sort(rng.choice(idx, max_per_shot, replace=False))
            Xk = X[:, self.keep]
            for i in idx:
                self.rows.append((Xk[i], Y[i]))
        self.mean = np.asarray(mean, float)[self.keep]
        self.std = np.asarray(std, float)[self.keep]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y = self.rows[i]
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)


class SeqDataset(Dataset):
    """One whole shot per item: actuator series (L, n_act) + rho (L,32) + mask (L,).

    Used by M2. Missing columns are filled with 0 (e.g. IC when absent).
    """

    def __init__(self, h5_dir, npz_dir, shots, raw_names, mean, std):
        self.shots = []
        self.mean = np.asarray(mean, float)
        self.std = np.maximum(np.asarray(std, float), 1e-6)
        for s in shots:
            f = pathlib.Path(h5_dir) / f"{int(s)}.h5"
            if not f.exists():
                continue
            with h5py.File(f, "r") as h:
                t = np.asarray(h["time"], float)
                cols = []
                for nm in raw_names:
                    if nm in h and getattr(h[nm], "shape", None) is not None:
                        cols.append(np.asarray(h[nm], float).reshape(-1))
                    else:
                        cols.append(np.zeros(t.size))
                A = np.column_stack(cols).astype(np.float32)
            d = np.load(pathlib.Path(npz_dir) / f"{int(s)}.npz")
            Y = d["Y"].astype(np.float32)
            valid = d["valid"].astype(bool)
            m = np.isfinite(A).all(1) & np.isfinite(Y).all(1) & valid
            self.shots.append((A, Y, valid, m))

    def __len__(self):
        return len(self.shots)

    def __getitem__(self, i):
        A, Y, _, m = self.shots[i]
        A = (A - self.mean) / self.std
        return (torch.from_numpy(A), torch.from_numpy(Y),
                torch.from_numpy(m.astype(np.float32)))


# ── DCS strict-actuator torch datasets (M1 snapshot, M2 sequence) ─────

class DCSSnapshotDataset(Dataset):
    """Per-slice (engineered strict-actuator snapshot -> rho) over a shot set."""

    def __init__(self, npz_dir, shots, cfg, ncm, mean, std, max_per_shot=None,
                 tgt_mean=None, tgt_std=None):
        self.keep = np.asarray(std, float) > 0
        self.mean = np.asarray(mean, float)[self.keep]
        self.std = np.asarray(std, float)[self.keep]
        self.tgt_mean = None if tgt_mean is None else np.asarray(tgt_mean, float)
        self.tgt_std = None if tgt_std is None else np.asarray(tgt_std, float)
        self.rows = []
        rng = np.random.default_rng(0)
        mps = max_per_shot if max_per_shot is not None else cfg.get("max_per_shot")
        for s in shots:
            p = pathlib.Path(npz_dir) / f"{int(s)}.npz"
            if not p.exists():
                continue
            feats, mask = read_snapshot(p, cfg, ncm)
            T_all, finite = load_target(p)
            if self.tgt_mean is not None:
                T_all = standardize(T_all, self.tgt_mean, self.tgt_std)
            v = mask & finite & np.isfinite(feats).all(1)
            idx = np.where(v)[0]
            if mps and idx.size > mps:
                idx = np.sort(rng.choice(idx, mps, replace=False))
            Xk = feats[:, self.keep]
            for i in idx:
                self.rows.append((Xk[i], T_all[i].astype(np.float32)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y = self.rows[i]
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)


class DCSSeqDataset(Dataset):
    """Per-shot full actuator series (n_act -> 34 outputs (rho + centre)) over a shot set.

    Each item is one whole shot: ``(A(L, n_act), Y(L, 34), mask(L,))``. The
    series is not subsampled (the GRU is linear-time). Normalization uses the
    per-channel mean/std over valid steps; invalid steps are kept (inputs zeroed by
    ``read_series``, targets zeroed here) and masked out of the loss by the trainer.
    """

    def __init__(self, npz_dir, shots, cfg, ncm, mean, std,
                 tgt_mean=None, tgt_std=None):
        self.mean = np.asarray(mean, float)
        self.std = np.maximum(np.asarray(std, float), 1e-6)
        self.tgt_mean = None if tgt_mean is None else np.asarray(tgt_mean, float)
        self.tgt_std = None if tgt_std is None else np.asarray(tgt_std, float)
        self.rows = []
        n_act = None
        for s in shots:
            p = pathlib.Path(npz_dir) / f"{int(s)}.npz"
            if not p.exists():
                continue
            A, mask = read_series(p, cfg, ncm)
            T_all, finite = load_target(p)
            if self.tgt_mean is not None:
                T_all = standardize(T_all, self.tgt_mean, self.tgt_std)
            v = mask & finite & np.isfinite(A).all(1)
            # zero-fill AFTER masking: the trainer multiplies by the mask and
            # NaN * 0 = NaN, which would make the loss and every gradient NaN.
            T_all = np.nan_to_num(T_all, nan=0.0, posinf=0.0, neginf=0.0)
            self.rows.append((A, T_all.astype(np.float32), v))
            n_act = A.shape[1]
        self.n_act = int(n_act) if n_act is not None else 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        A, Y, m = self.rows[i]
        A = ((A - self.mean) / self.std).astype(np.float32)
        return (torch.from_numpy(A), torch.from_numpy(Y),
                torch.from_numpy(m))


# ── DCS windowed torch dataset (M3 attention) ─────────────────────────

W_DEFAULT = 2048        # window length in steps (4.19 s on the 488 Hz axis)
CTX_DEFAULT = 512       # attended-but-unscored prefix (1.05 s)


def window_blocks(nt, w=W_DEFAULT, ctx=CTX_DEFAULT):
    """``[(win_start, win_end, blk_start, blk_end)]`` tiling ``[0, nt)``.

    Scored blocks are **disjoint and cover ``[0, nt)`` exactly**, at stride
    ``w - ctx``; each is preceded by up to ``ctx`` steps that are attended but never
    scored, so every scored step has left context. The first block has none (nothing
    exists before ``t = 0``) and the last window may be short.

    The partition property is what guarantees each valid slice enters the loss once
    and that eval reproduces the predict mask exactly -- asserted in
    ``tests/test_ml_window_dataset.py``.
    """
    stride = w - ctx
    if stride <= 0:
        raise ValueError(f"ctx {ctx} must be < w {w}")
    out, b = [], 0
    while b < nt:
        be = min(b + stride, nt)
        out.append((max(0, b - ctx), be, b, be))
        b = be
    return out


def pad_collate(batch, w=W_DEFAULT):
    """Right-pad variable-length windows to ``w``. Returns ``(A, Y, M, P)``.

    No attention mask is produced or needed: the model is causal, so a real position
    never attends rightward into the padding, and padded steps carry ``M = False``.
    """
    a0, y0, _m0, p0 = batch[0]
    B = len(batch)
    A = torch.zeros(B, w, a0.shape[1], dtype=torch.float32)
    Y = torch.zeros(B, w, y0.shape[1], dtype=torch.float32)
    M = torch.zeros(B, w, dtype=torch.bool)
    P = (torch.zeros(B, w, dtype=torch.float32) if p0.dim() == 1
         else torch.zeros(B, w, p0.shape[1], dtype=torch.float32))
    for i, (a, y, m, p) in enumerate(batch):
        n = a.shape[0]
        A[i, :n], Y[i, :n], M[i, :n], P[i, :n] = a, y, m, p
    return A, Y, M, P


def drop_keep(nt, shot, drop_frac, drop_seed, floor=100):
    """Deterministic per-shot retention mask for the dropout dose-response.

    Independent of the training seed (uses ``drop_seed + shot``), so ``rope_idx`` and
    ``rope_time`` drop identical slices and the 3 training seeds drop identical slices.
    At ``drop_frac <= 0`` it is a no-op (all-keep). A floor guard keeps every shot
    non-degenerate so windowing never sees an empty sequence.
    """
    if drop_frac <= 0:
        return np.ones(nt, dtype=bool)
    keep = np.random.default_rng(drop_seed + int(shot)).random(nt) >= drop_frac
    if int(keep.sum()) < floor:
        keep = np.ones(nt, dtype=bool)
    return keep


class DCSWindowDataset(Dataset):
    """Per-window actuator series for M3. One item is one window of one shot.

    ``(A(L, n_act), Y(L, 34), loss_mask(L,), P)`` where ``P`` is the positional
    payload the ``pe`` variant needs: ``(L,)`` window-relative offsets for ``rope_*``,
    or the ``(L, d_model)`` absolute additive table for ``upe_*``. ``L`` is the actual
    window length; :func:`pad_collate` pads to ``w``.

    ``loss_mask`` is ``v & in_scored_block``, with ``v`` **byte-for-byte** the mask
    ``scripts/train_dcs.py:_pred_m2`` uses -- ``read_series``'s mask AND the target's
    finite mask. Do not "improve" it: m3 must keep and drop exactly the slices m2
    does, or the arms are not comparable.

    Positions are ignitron-anchored and in cadence units. ``time[0]`` is nonzero in
    100 % of shots (median 0.053 s = 26 samples), so an index taken as the NPZ row
    number would sit a per-shot constant away from ``t / c`` -- the pair would then
    measure span-anchored-vs-ignitron-anchored, a confound, instead of index-vs-time.
    Hence ``n0 = round(time[0] / c_modal)``.
    """

    def __init__(self, npz_dir, shots, cfg, ncm, mean, std, pe="rope_idx",
                 d_model=256, w=W_DEFAULT, ctx=CTX_DEFAULT,
                 tgt_mean=None, tgt_std=None, drop_frac=0.0, drop_seed=0):
        if pe not in VARIANTS:
            raise ValueError(f"pe must be one of {VARIANTS}, got {pe!r}")
        self.pe, self.d_model, self.w, self.ctx = pe, int(d_model), int(w), int(ctx)
        self.drop_frac, self.drop_seed = float(drop_frac), int(drop_seed)
        self.mean = np.asarray(mean, float)
        self.std = np.maximum(np.asarray(std, float), 1e-6)
        self.tgt_mean = None if tgt_mean is None else np.asarray(tgt_mean, float)
        self.tgt_std = None if tgt_std is None else np.asarray(tgt_std, float)
        self.shots, self.index = [], []
        n_act = None
        for s in shots:
            p = pathlib.Path(npz_dir) / f"{int(s)}.npz"
            if not p.exists():
                continue
            A, mask = read_series(p, cfg, ncm)
            T, finite = load_target(p)
            v = mask & finite                     # == _pred_m2's v, exactly
            if not v.any():
                continue
            if self.tgt_mean is not None:
                T = standardize(T, self.tgt_mean, self.tgt_std)
            # zero-fill AFTER masking: NaN * 0 = NaN poisons the loss and every
            # gradient (see DCSSeqDataset and tests/test_ml_seq_mask.py).
            T = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            t = np.load(p)["time"].astype(np.float64)
            keep = drop_keep(t.size, int(s), self.drop_frac, self.drop_seed)
            if not keep.all():
                A, T, v, t = A[keep], T[keep], v[keep], t[keep]
            c = modal_cadence(t)
            n0 = float(round(float(t[0]) / c))
            ipos = n0 + np.arange(t.size, dtype=np.float64)
            tpos = t / c
            k = len(self.shots)
            self.shots.append((A, T, v, ipos, tpos))
            self.index.extend((k, *blk) for blk in window_blocks(t.size, w, ctx))
            n_act = A.shape[1]
        self.n_act = int(n_act) if n_act is not None else 0

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        k, ws, we, bs, _be = self.index[j]
        A, T, v, ipos, tpos = self.shots[k]
        a = ((A[ws:we] - self.mean) / self.std).astype(np.float32)
        lm = v[ws:we].copy()
        lm[: bs - ws] = False                      # the context prefix is not scored
        return (torch.from_numpy(a), torch.from_numpy(T[ws:we]),
                torch.from_numpy(lm), torch.from_numpy(self._pos(ipos[ws:we],
                                                                tpos[ws:we])))

    def _pos(self, ipos, tpos):
        """This window's positional payload: rope offsets, or the absolute upe table.

        ``rope_*`` shifts to the window start -- RoPE only uses differences, and an
        absolute ``t/c`` would reach ~49,300 on the longest shot, where float32's ~7
        significant digits leave ~2 digits of phase. ``upe_*`` must NOT be shifted:
        it is absolute by definition.
        """
        if self.pe == "rope_idx":
            return (ipos - ipos[0]).astype(np.float32)
        if self.pe == "rope_time":
            return (tpos - tpos[0]).astype(np.float32)
        return upe_table(self.pe, ipos, tpos, self.d_model)
