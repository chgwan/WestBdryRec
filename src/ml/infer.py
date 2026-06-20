# src/ml/infer.py
# -*- coding: utf-8 -*-
"""M0 inference: load the artifact and predict the LCFS (shape lcfs_rho + absolute
(R,Z) on the predicted axis) from a shot's strict-actuator inputs.
Backs scripts/infer_m0.py. Training lives in train.py (train_save)."""
import pathlib

import h5py
import joblib
import numpy as np

from . import features as F
from .axis_frame import reconstruct_absolute


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
