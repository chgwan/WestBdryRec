# src/actuator_predictor/dataset.py
# -*- coding: utf-8 -*-
"""Torch datasets: snapshot (M1) and actuator-sequence (M2).

Both datasets standardize inputs with a TRAIN-derived (mean, std). Constant
columns (std == 0) are dropped via the ``keep`` boolean mask so the model's
``n_in`` is stable between training and prediction.
"""
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset


def keep_mask(std):
    """Boolean column mask selecting std > 0 (drops constant features)."""
    return np.asarray(std, float) > 0


class SnapshotDataset(Dataset):
    """Per-slice (engineered-actuator snapshot -> rho) over a shot set.

    ``feature_fn`` is e.g. :func:`features.engineer` returning ``(X, valid)``.
    Inputs are standardized with TRAIN ``(mean, std)``; constant columns are
    dropped via ``keep = std > 0`` so every emitted ``x`` has ``keep.sum()`` dims.
    """

    def __init__(self, h5_dir, npz_dir, shots, feature_fn, mean, std, max_per_shot=None):
        self.rows = []                               # list of (x, y) arrays (post-keep)
        rng = np.random.default_rng(0)
        self.keep = keep_mask(std)
        for s in shots:
            f = pathlib.Path(h5_dir) / f"{int(s)}.h5"
            if not f.exists():
                continue
            X, vfeat = feature_fn(f)
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

    Used by M2. ``raw_names`` selects the raw actuator columns read from the H5
    file; missing columns are filled with 0 (e.g. IC when absent).
    """

    def __init__(self, h5_dir, npz_dir, shots, raw_names, mean, std):
        import h5py
        self.shots = []
        self.mean = np.asarray(mean, float)
        self.std = np.asarray(std, float)
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
                        cols.append(np.zeros(t.size))     # IC -> 0 etc.
                A = np.column_stack(cols).astype(np.float32)   # (L, n_act)
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
