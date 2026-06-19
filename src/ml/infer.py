# src/ml/infer.py
# -*- coding: utf-8 -*-
"""M0 inference: train+save the M0 shape model + axis model, load, and predict the
LCFS (shape lcfs_rho + absolute (R,Z) on the predicted axis) from a shot's strict-
actuator inputs. Backs scripts/train_m0.py and scripts/infer_m0.py."""
import pathlib

import h5py
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from . import bench, features as F
from .axis_frame import reconstruct_absolute

BEST_HP = dict(max_iter=200, max_depth=8, learning_rate=0.1, l2_regularization=1.0)


def _shot_xy(h5_dir, npz_dir, shot):
    """(X[valid], Y[valid], axis_r[valid], axis_z[valid]) for one shot, or None."""
    f = pathlib.Path(h5_dir) / f"{int(shot)}.h5"
    if not f.exists():
        return None
    X, vf = F.engineer(f)
    d = np.load(pathlib.Path(npz_dir) / f"{int(shot)}.npz")
    Y = d["Y"].astype(float)
    with h5py.File(f, "r") as h:
        ar = np.asarray(h["magnetic_axis_r"], float)
        az = np.asarray(h["magnetic_axis_z"], float)
    v = (vf & d["valid"].astype(bool) & np.isfinite(X).all(1) & np.isfinite(Y).all(1))
    return X[v], Y[v], ar[v], az[v]


def train_save(h5_dir, npz_dir, out_path, hp=BEST_HP):
    """Fit M0 (32 per-angle HistGBTs) + axis model (2 HistGBTs) on the train shots;
    save joblib {m0, axis_r, axis_z, keep}; return a meta dict."""
    train, _val, _test = bench.load_filtered_split(npz_dir)
    Xs, Ys, Ar, Az = [], [], [], []
    for s in train:
        g = _shot_xy(h5_dir, npz_dir, s)
        if g is None:
            continue
        Xs.append(g[0]); Ys.append(g[1]); Ar.append(g[2]); Az.append(g[3])
    Xtr, Ytr = np.concatenate(Xs), np.concatenate(Ys)
    ar, az = np.concatenate(Ar), np.concatenate(Az)
    keep = Xtr.std(0) > 0
    Xk = Xtr[:, keep]
    m0 = [HistGradientBoostingRegressor(**hp).fit(Xk, Ytr[:, a]) for a in range(Ytr.shape[1])]
    axis_r = HistGradientBoostingRegressor(**hp).fit(Xk, ar)
    axis_z = HistGradientBoostingRegressor(**hp).fit(Xk, az)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"m0": m0, "axis_r": axis_r, "axis_z": axis_z, "keep": keep}, out_path)
    pred = np.column_stack([m.predict(Xk) for m in m0])
    ss_res = float(((Ytr - pred) ** 2).sum())
    ss_tot = float(((Ytr - Ytr.mean(0)) ** 2).sum())
    train_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"hp": hp, "train_r2": train_r2, "n_train": int(len(Ytr)),
            "n_angles": int(Ytr.shape[1]), "n_features": int(keep.sum()),
            "kept_features": [n for n, k in zip(F.FEATURE_ORDER, keep) if k]}


def load(out_path):
    return joblib.load(out_path)


def predict_shot(shot, artifact, h5_dir):
    """Predict the LCFS for a shot over its flat-top valid slices: shape lcfs_rho +
    absolute (R,Z) on the predicted axis. Returns the prediction dict, or None."""
    f = pathlib.Path(h5_dir) / f"{int(shot)}.h5"
    if not f.exists():
        return None
    with h5py.File(f, "r") as h:
        th = np.asarray(h["lcfs_theta"], float)
        time = np.asarray(h["time"], float)
    X, vf = F.engineer(f)
    keep = artifact["keep"]
    v = vf & np.isfinite(X).all(1)
    idx = np.where(v)[0]
    if idx.size == 0:
        return None
    Xv = X[idx][:, keep]
    rho = np.column_stack([m.predict(Xv) for m in artifact["m0"]])     # (n,32)
    ar = artifact["axis_r"].predict(Xv)                                # (n,)
    az = artifact["axis_z"].predict(Xv)
    R, Z = reconstruct_absolute(ar, az, rho, th)                       # (n,32)
    return {"lcfs_rho": rho.astype(np.float32),
            "axis": np.column_stack([ar, az]).astype(np.float32),
            "R": R.astype(np.float32), "Z": Z.astype(np.float32),
            "time": time[idx].astype(np.float32), "theta": th.astype(np.float32),
            "valid_idx": idx.astype(np.int64)}
