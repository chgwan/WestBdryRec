# src/ml/features.py
# -*- coding: utf-8 -*-
"""Engineered per-slice actuator features for the M0 snapshot model.

Raw actuators (per slice) + per-slice derived features (ratios, heating sums,
time-in-flat-top, cumulative heating integral). All non-circular.

The raw actuator names (RAW) are read from configs/m0_model.yml (inputs list)
— the single source of truth for what M0 reads."""
import pathlib

import h5py
import numpy as np
import yaml

from ..proj_config import get_proj_config
from ..data.imas_flat_top import flat_top_window


def _load_model_inputs():
    """Read the M0 input list from configs/m0_model.yml (single source of truth
    for what M0 reads). Returns ``(raw_names, n_pf)``."""
    cfg = get_proj_config()
    config_path = cfg.base_dir / "configs" / "m0_model.yml"
    with open(config_path) as f:
        yml = yaml.safe_load(f)
    names = yml["inputs"]
    n_pf = sum(1 for n in names if n.startswith("pf_"))
    return names, n_pf


RAW, N_PF = _load_model_inputs()
FEATURE_ORDER = RAW + ["pf_norm", "lh_plus_ic", "time_in_flat", "cum_heat"]

# Named column indices (derived from config order, no magic numbers)
_B0_IDX = RAW.index("b0")
_LH_IDX = RAW.index("lh_power_launched_total")
_IC_IDX = RAW.index("ic_power_launched_total")


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
        raw = np.column_stack([_read(h, n, nt, fill=0.0) for n in RAW])
        pf = raw[:, :N_PF]
        b0 = raw[:, _B0_IDX]
        lh = raw[:, _LH_IDX]; ic = raw[:, _IC_IDX]
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
