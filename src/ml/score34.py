# -*- coding: utf-8 -*-
"""Score 34-column DCS predictions: r(theta) plus the absolute polar centre.

The rho block is scored with exactly the metric the 32-column baseline used, so those
numbers stay comparable in kind. The centre is reported on its own scale in millimetres,
and the absolute-boundary RMSE combines both -- it is the only number that reflects the
error a downstream consumer of the reconstructed (R,Z) actually sees.

Width is validated up front and a raise is preferred to a skip: ``bench.score_predictions``
silently drops shots whose shapes disagree and returns ``n_shots = 0`` while still writing
a row, which looks like a scored run and is not one.
"""
import pathlib

import numpy as np

from .axis_frame import reconstruct_absolute
from .metrics import boundary_metrics, ccc
from .predictions import load_predictions
from .target import N_OUT, N_RHO, load_target, split_outputs


def _truth(npz_dir, shot, t_min=0.0):
    """``T (n,34)`` over the rows the predict paths kept."""
    p = pathlib.Path(npz_dir) / f"{int(shot)}.npz"
    d = np.load(p)
    T, finite = load_target(p)
    v = finite & d["valid"].astype(bool) & (d["time"].astype(float) >= t_min)
    return T[v]


def score_dcs34(pred_path, npz_dir, train_shots, test_shots, theta):
    """Pooled metrics over ``test_shots``. ``theta`` is the (32,) angle grid in radians."""
    preds = load_predictions(pred_path)
    for s, yp in preds.items():
        if np.asarray(yp).shape[-1] != N_OUT:
            raise ValueError(f"shot {s}: predictions are {np.asarray(yp).shape[-1]} wide, "
                             f"expected {N_OUT}")
    rho_tr = np.concatenate([_truth(npz_dir, s)[:, :N_RHO] for s in train_shots])
    y_train_mean = rho_tr.mean(axis=0)

    rp, rt, cp, ct, ap, at = [], [], [], [], [], []
    per_shot_ccc = []
    for s in test_shots:
        s = int(s)
        if s not in preds:
            continue
        T = _truth(npz_dir, s)
        P = np.asarray(preds[s], float)
        if P.shape[0] != T.shape[0]:
            raise ValueError(f"shot {s}: {P.shape[0]} predicted rows vs {T.shape[0]} truth "
                             "rows -- the predict mask and the scoring mask disagree")
        pr, pc = split_outputs(P)
        tr_, tc = split_outputs(T)
        rp.append(pr); rt.append(tr_)
        cp.append(pc); ct.append(tc)
        per_shot_ccc.append(ccc(pr, tr_))
        # Absolute boundary: rebuild BOTH pred and truth on the uniform theta grid.
        # Raw bnd_RZ lives at native (irregular) vertex angles, so pairing a
        # uniform-grid prediction against it is a category error -- the model
        # predicts resampled rho and cannot reproduce native-angle detail. The
        # honest metric compares like-for-like reconstructions.
        Rp, Zp = reconstruct_absolute(pc[:, 0], pc[:, 1], pr, theta)
        Rt, Zt = reconstruct_absolute(tc[:, 0], tc[:, 1], tr_, theta)
        ap.append(np.stack([Rp, Zp], axis=-1)); at.append(np.stack([Rt, Zt], axis=-1))
    if not rp:
        raise ValueError("no test shot had predictions -- nothing was scored")

    m = boundary_metrics(np.concatenate(rp), np.concatenate(rt),
                         y_train_mean=y_train_mean)
    C_p, C_t = np.concatenate(cp), np.concatenate(ct)
    err = (C_p - C_t) * 1000.0                                  # mm
    m["rgeom_mae_mm"] = float(np.abs(err[:, 0]).mean())
    m["zgeom_mae_mm"] = float(np.abs(err[:, 1]).mean())
    m["centre_rmse_mm"] = float(np.sqrt((err ** 2).sum(axis=1).mean()))
    A_p, A_t = np.concatenate(ap), np.concatenate(at)
    m["abs_bnd_rmse_mm"] = float(np.sqrt((((A_p - A_t) * 1000.0) ** 2).sum(axis=-1).mean()))
    m["per_shot_ccc"] = per_shot_ccc
    m["ccc_p90"] = float(np.percentile(per_shot_ccc, 90)) if per_shot_ccc else float("nan")
    m["n_shots"] = len(per_shot_ccc)
    return m
