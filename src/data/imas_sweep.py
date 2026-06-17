# src/data/imas_sweep.py
# -*- coding: utf-8 -*-
"""IMAS-native input-observability sweep core: select tier columns from the IMAS NPZ,
pool valid slices, fit a HistGBT per boundary angle, score cross-shot."""
import json
import pathlib

import numpy as np


def load_imas_meta(npz_dir):
    return json.loads(pathlib.Path(npz_dir).joinpath("meta.json").read_text())


def group_cols(meta, groups):
    """Ascending X-column indices for the given group names."""
    want = set(groups)
    cols = []
    for item in meta["inputs"]:
        if item["group"] in want:
            cols.extend(range(item["cols"][0], item["cols"][1]))
    return cols


def tier_groups(meta, tier):
    """Group names in a tier (e.g. T0 -> [pf_currents, tf, lh_power])."""
    return [item["group"] for item in meta["inputs"] if item["tier"] == tier]


def pool(npz_dir, shots, cols, max_slices=None, seed=0):
    """Stack valid ``(X[:, cols], Y)`` rows over shots, optional per-shot subsample."""
    rng = np.random.default_rng(seed)
    npz_dir = pathlib.Path(npz_dir)
    Xs, Ys = [], []
    for sh in shots:
        f = npz_dir / f"{int(sh)}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        X = d["X"][:, cols].astype(np.float64)
        Y = d["Y"].astype(np.float64)
        m = d["valid"].astype(bool) & np.isfinite(X).all(1) & np.isfinite(Y).all(1)
        idx = np.where(m)[0]
        if idx.size == 0:
            continue
        if max_slices and idx.size > max_slices:
            idx = np.sort(rng.choice(idx, max_slices, replace=False))
        Xs.append(X[idx]); Ys.append(Y[idx])
    if not Xs:
        raise ValueError("no valid pooled rows")
    return np.concatenate(Xs), np.concatenate(Ys)


def _drop_constant(Xtr, Xte):
    keep = Xtr.std(axis=0) > 0
    return Xtr[:, keep], Xte[:, keep], int((~keep).sum())


def run_point(npz_dir, train_shots, test_shots, groups, max_slices=None,
              max_iter=150, seed=0):
    """One sweep point: pool train/test, fit HistGBT per angle, score cross-shot."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from ..ml.metrics import boundary_metrics

    meta = load_imas_meta(npz_dir)
    cols = group_cols(meta, groups)
    Xtr, Ytr = pool(npz_dir, train_shots, cols, max_slices, seed)
    Xte, Yte = pool(npz_dir, test_shots, cols, max_slices, seed)
    Xtr, Xte, n_dropped = _drop_constant(Xtr, Xte)
    pred = np.empty_like(Yte)
    for a in range(Ytr.shape[1]):
        reg = HistGradientBoostingRegressor(max_iter=max_iter, max_depth=6)
        reg.fit(Xtr, Ytr[:, a])
        pred[:, a] = reg.predict(Xte)
    m = boundary_metrics(pred, Yte, y_train_mean=Ytr.mean(axis=0))
    m.update({"n_features": int(Xtr.shape[1]), "n_dropped": int(n_dropped),
              "n_test": int(len(Yte))})
    return m
