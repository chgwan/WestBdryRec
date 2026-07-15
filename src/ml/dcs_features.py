# -*- coding: utf-8 -*-
"""DCS strict-actuator feature layer for MergedNpz.

Reads the 18 strict-actuator columns from each NPZ (heating NaN->0, essential
NaN->drop), adds engineered snapshot features, and yields the actuator series
for the temporal model. Config-driven via configs/dcs_model.yml.
"""
import json
import pathlib

import numpy as np
import yaml

from ..proj_config import get_proj_config


def load_dcs_config():
    cfg = get_proj_config()
    with open(cfg.base_dir / "configs" / "dcs_model.yml") as f:
        return yaml.safe_load(f)


def load_meta(npz_dir):
    return json.loads((pathlib.Path(npz_dir) / "meta.json").read_text())


def node_col_map(meta):
    """{node_name: X column index} from the meta inputs layout."""
    out = {}
    for g in meta["inputs"]:
        for i, node in enumerate(g["nodes"]):
            out[node] = int(g["cols"][0]) + i
    return out


def strict_channels(cfg):
    """[(node, kind, nan_policy)] in config order."""
    return [(c["node"], c["kind"], c["nan"]) for c in cfg["channels"]]


_DERIVED = 4  # pf_norm, lh_plus_ic, cum_heat, time_rel


def _apply_nan(col, policy, essential_mask):
    """Return a finite column and the updated essential mask.

    policy "zero": NaN -> 0.0, mask unchanged. policy "drop": NaN -> 0.0 AND the
    slice is marked invalid in the mask (consumer drops it via the mask). Zeroing
    dropped NaNs keeps derived features finite on masked-out rows.
    """
    if policy == "zero":
        return np.where(np.isfinite(col), col, 0.0), essential_mask
    finite = np.isfinite(col)
    return np.where(finite, col, 0.0), essential_mask & finite


def read_snapshot(npz_path, cfg, ncm):
    """(feats(nt, n_chan+4), mask(nt,)). Heating NaN->0; essential NaN->drop slice."""
    d = np.load(npz_path)
    X = d["X"].astype(float); valid = d["valid"].astype(bool); t = d["time"].astype(float)
    chans = strict_channels(cfg)
    nt = X.shape[0]
    raw = np.zeros((nt, len(chans)))
    essential = np.ones(nt, bool)
    kinds = []
    for k, (node, kind, policy) in enumerate(chans):
        col, essential = _apply_nan(X[:, ncm[node]], policy, essential)
        raw[:, k] = col
        kinds.append(kind)
    pf = raw[:, [k for k, kd in enumerate(kinds) if kd == "pf"]]
    heat = raw[:, [k for k, kd in enumerate(kinds) if kd in ("lh", "ic")]].sum(axis=1)
    dt = np.gradient(t)
    cum = np.concatenate([[0.0], np.cumsum(heat[:-1] * dt[1:])])
    derived = np.column_stack([np.linalg.norm(pf, axis=1), heat, t - t[0], cum])
    feats = np.column_stack([raw, derived]).astype(np.float32)
    return feats, valid & essential


def read_series(npz_path, cfg, ncm):
    """(A(nt, n_chan), mask(nt,)) — raw actuator series for the temporal model."""
    d = np.load(npz_path)
    X = d["X"].astype(float); valid = d["valid"].astype(bool)
    chans = strict_channels(cfg)
    nt = X.shape[0]
    A = np.zeros((nt, len(chans)), np.float32)
    essential = np.ones(nt, bool)
    for k, (node, _kind, policy) in enumerate(chans):
        col, essential = _apply_nan(X[:, ncm[node]], policy, essential)
        A[:, k] = col
    return A, valid & essential
