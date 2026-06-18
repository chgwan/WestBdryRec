# src/actuator_predictor/axis_frame.py
# -*- coding: utf-8 -*-
"""Pure helpers for the axis-frame test: absolute-LCFS reconstruction from the
axis + rho(theta), bias-aware axis metrics, and absolute (R,Z) LCFS metrics."""
import numpy as np


def reconstruct_absolute(axis_r, axis_z, rho, theta):
    """Rebuild the LCFS in absolute (R,Z) from the axis + rho(theta) + theta.

    ``axis_r/axis_z``: scalar or (nt,); ``rho``: (nt,32) or (32,); ``theta``: (32,).
    The axis broadcasts over the angle axis. Returns ``R, Z`` shaped like ``rho``."""
    ar = np.asarray(axis_r, float)
    az = np.asarray(axis_z, float)
    th = np.asarray(theta, float).reshape(-1)
    rho = np.asarray(rho, float)
    R = ar[..., None] + rho * np.cos(th)
    Z = az[..., None] + rho * np.sin(th)
    return R, Z


def debiased_axis_metrics(pred, true, train_mean):
    """Bias-aware scalar R^2 for one axis coordinate (R or Z), in metres.

    R^2 is computed on the deviation from the train mean (the trivial ~2.5 m bias
    for R), so the bias cannot be read as skill. ``ss_res`` uses the raw error
    (true - pred), ``ss_tot`` uses (true - train_mean)."""
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    b = float(train_mean)
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - b) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"bias": b,
            "signal_std": float(true.std()),
            "residual_bias": float((pred - true).mean()),
            "rmse": float(np.sqrt(((pred - true) ** 2).mean())),
            "r2": r2}


def absolute_rz_metrics(R_pred, Z_pred, R_true, Z_true, train_mean_R, train_mean_Z):
    """Absolute (R,Z) LCFS metrics for the end-to-end comparison (shape + axis
    error folded in). R_pred/R_true: (N,32) metres; train_mean_*: (32,) per-angle
    train means (the floor). Returns R^2 and RMSE in cm."""
    def _r2_rmse(p, t, m):
        p = np.asarray(p, float); t = np.asarray(t, float)
        ss_res = float(((t - p) ** 2).sum())
        ss_tot = float(((t - m) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return r2, float(np.sqrt(((t - p) ** 2).mean()))

    r2_R, rmse_R = _r2_rmse(R_pred, R_true, train_mean_R)
    r2_Z, rmse_Z = _r2_rmse(Z_pred, Z_true, train_mean_Z)
    Rp, Rt, Zp, Zt = (np.asarray(a, float) for a in (R_pred, R_true, Z_pred, Z_true))
    boundary = float(np.sqrt(((Rp - Rt) ** 2 + (Zp - Zt) ** 2).mean()))
    return {"r2_R": r2_R, "r2_Z": r2_Z,
            "rmse_R_cm": rmse_R * 100, "rmse_Z_cm": rmse_Z * 100,
            "boundary_rmse_cm": boundary * 100}
