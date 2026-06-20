# src/ml/dataset.py
# -*- coding: utf-8 -*-
"""Feature engineering + torch datasets for the actuator-predictor models.

``engineer(h5_path)`` builds the M0 snapshot (raw actuators + derived features).
``SnapshotDataset`` / ``SeqDataset`` wrap shots into torch datasets for M1/M2.

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


# ── Torch datasets (M1 snapshot, M2 sequence) ────────────────────────

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
