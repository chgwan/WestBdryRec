# src/actuator_predictor/features.py
# -*- coding: utf-8 -*-
"""Engineered per-slice actuator features for the M0 snapshot model.

Raw actuators (per slice) + per-slice derived features (ratios, heating sums,
time-in-flat-top, cumulative heating integral). All non-circular."""
import pathlib
import numpy as np
import h5py

from ..data.imas_flat_top import flat_top_window

RAW = ["pf_A", "pf_Bb", "pf_Bh", "pf_Db", "pf_Dh",
       "pf_Divertor_bottom1_HFS", "pf_Divertor_bottom1_LFS",
       "pf_Divertor_bottom2_HFS", "pf_Divertor_bottom2_LFS",
       "pf_Divertor_top1_HFS", "pf_Divertor_top1_LFS",
       "pf_Divertor_top2_HFS", "pf_Divertor_top2_LFS",
       "pf_Eb", "pf_Eh", "pf_Fb", "pf_Fh", "b0",
       "lh_power_launched_total", "ic_power_launched_total"]

FEATURE_ORDER = RAW + ["pf_norm", "lh_plus_ic", "time_in_flat", "cum_heat"]


def _read(h, name, nt, fill=0.0):
    if name not in h:
        return np.full(nt, fill)
    ds = h[name]
    if getattr(ds, "shape", None) is None:
        return np.full(nt, fill)
    return np.asarray(ds, float).reshape(-1)


def engineer(h5_path):
    h5_path = pathlib.Path(h5_path)
    with h5py.File(h5_path, "r") as h:
        t = np.asarray(h["time"], float)
        nt = t.size
        raw = np.column_stack([_read(h, n, nt, fill=0.0) for n in RAW])  # (nt, 20)
        pf = raw[:, :17]
        b0 = raw[:, 17]
        lh = raw[:, 18]; ic = raw[:, 19]
        ip = _read(h, "ip", nt, fill=np.nan)
        # derived (non-circular, per-slice)
        pf_norm = np.linalg.norm(pf, axis=1) / np.maximum(np.abs(b0), 1e-9)
        heat = lh + ic
        # flat-top window for time-in-flat + cumulative integral
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
