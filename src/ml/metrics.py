# -*- coding: utf-8 -*-
"""Cross-shot r(theta) boundary metrics: G1 cosine similarity, G2 R^2, RMSE, floor."""
import numpy as np


def boundary_metrics(y_pred, y_true, mask=None, y_train_mean=None):
    """Boundary metrics (meters). ``mask`` (N,) bool selects rows; ``y_train_mean``
    is the per-angle train mean used as the cross-shot floor baseline.

    Returns {similarity, r2, rmse_cm, floor_cm, n}.
    """
    yp = np.asarray(y_pred, float)
    yt = np.asarray(y_true, float)
    if mask is not None:
        m = np.asarray(mask, bool)
        yp, yt = yp[m], yt[m]
    ybar = np.asarray(y_train_mean, float) if y_train_mean is not None else yt.mean(axis=0)
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - ybar) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    num = (yp * yt).sum(axis=1)
    den = np.linalg.norm(yp, axis=1) * np.linalg.norm(yt, axis=1)
    sim = float((num / np.clip(den, 1e-12, None)).mean())
    rmse = float(np.sqrt(((yt - yp) ** 2).mean()))
    floor = float(np.sqrt(((yt - ybar) ** 2).mean()))
    return {"similarity": sim, "r2": float(r2),
            "rmse_cm": rmse * 100, "floor_cm": floor * 100, "n": int(yt.shape[0])}