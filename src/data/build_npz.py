# -*- coding: utf-8 -*-
"""Build per-shot NPZ arrays for polar-r(theta) LCFS prediction.

Reads each time-aligned ``MergedH5/<shot>.h5`` and writes ``ProjDB/datasets/MergedNpz/<shot>.npz``
with the input feature matrix ``X``, the polar target ``Y = r(theta)``, the raw
boundary ``bnd_RZ`` (R,Z), the ``time`` vector and a ``valid`` mask, trimmed to the
span of slices that survive the quality filters. A shared ``meta.json`` records the
input layout, dimensions, per-shot slice counts, the filter criteria and thresholds
in force, and dataset-wide normalization stats (per-feature X mean/std, per-angle Y
mean/std).

**The polar origin is per slice**, namely that slice's plasma geometric center
``(Rgeom, Zgeom)`` from ``inputs/GMAG_GEOM``, stored in the ``center`` array -- not
the fixed ``(2.5, 0)`` this module used previously. The filters in
:mod:`src.data.filter` establish enclosure and star-shapedness about that same
center, which is exactly the precondition ``radii_on_grid`` needs; a fixed origin
cannot satisfy it for boundaries that do not contain the origin. Consequence:
``Y`` alone no longer determines the absolute boundary -- reconstructing (R,Z)
needs ``center`` too (see ``src/ml/axis_frame.reconstruct_absolute``).

Per-slice filter verdicts travel with the data: ``fail`` (first failing criterion,
0 = kept), ``only`` (criterion that rejects it alone) and ``flags`` (order-free
per-criterion pass bits).

Inputs come from ``configs/base.yml`` (``data.input_list``): the actuator/
diagnostic scope traces (PF/CS coils, heating power+phase, Ip reference, line
density, measured Ip), read from ``dcs/<scope>/<actual|ref>`` with an
``inputs/<node>`` fallback for MDS+ signals (e.g. SMAG_IP). Target is always
``targets/GMAG_BND``.

Importable as a package module; driven by ``scripts/run_data_pre.py``.
"""
import json
import pathlib
import shutil

import h5py
import numpy as np
import yaml

from ..proj_config import get_proj_config
from ..utils import pmap
from .filter import DEFAULTS as FILTER_DEFAULTS
from .filter import REGISTRY as FILTER_REGISTRY
from .filter import SliceQuality

N_POINTS = 32          # native LCFS vertices per slice
N_ANGLES = 32          # fixed uniform angle grid size
# There is deliberately no fixed polar origin here. Y is projected about each slice's own
# (Rgeom, Zgeom); a module-level constant origin would only invite a caller to reuse it and
# reintroduce the multi-valued-r(theta) bug it used to cause. See docs/lcfs_filters.md.

# Equilibrium scalar LABELS for the guided model: (name, h5 node, channel, scale).
# Used ONLY as training targets -- never an inference input. Channels/scale per
# Task A0 verification: GMAG_SHAF[1]=beta+li/2 (/1000), GMAG_BELI[5]=lidia (/1000).
# (q95/GMAG_Q is all-zero in the dataset -> not used.)
SCALAR_NODES = [
    ("beli", "GMAG_SHAF", 1, 1e-3),
    ("li",   "GMAG_BELI", 5, 1e-3),
]
N_SCALARS = len(SCALAR_NODES)

# A grid point whose bracketing native reconstructions are farther apart than this
# was interpolated across a dropout rather than within the normal 2.048 ms cadence.
# 1.5x the modal step: wide enough not to flag ordinary jitter, tight enough to
# catch every real gap. Only a census threshold -- validity is S0-S5's alone.
FAB_GAP_MS = 3.072


def _read_scalar(hf, node, ch, i0, i1):
    """Channel ``ch`` of a multi-channel GMAG signal, windowed ``[i0:i1+1]``.

    Reconstruction signals are stored ``(n_channels, nt)`` (like ``targets/GMAG_BND``),
    resolved under ``inputs/<node>``. Returns a ``(nt,)`` float array or ``None``.
    """
    path = f"inputs/{node}"
    if path not in hf:
        return None
    arr = np.asarray(hf[path], float)
    if arr.ndim != 2 or arr.shape[0] <= ch:
        return None
    return arr[ch, i0:i1 + 1]


def read_scalars(hf, i0, i1, nt):
    """Equilibrium scalar labels ``S`` of shape ``(nt, N_SCALARS)`` over the window.

    Absent/short channels become an all-NaN column (folded into validity later).
    """
    S = np.full((nt, N_SCALARS), np.nan, np.float32)
    for k, (_name, node, ch, scale) in enumerate(SCALAR_NODES):
        col = _read_scalar(hf, node, ch, i0, i1)
        if col is not None and col.shape == (nt,):
            S[:, k] = (np.asarray(col, float) * scale).astype(np.float32)
    return S


def periodic_interp(query, xp, fp):
    """Linear interp of a 2*pi-periodic function sampled at sorted ``xp`` in [0, 2pi).

    ``xp`` ascending; one sample is wrapped across the 0/2pi seam so queries in the
    last interval (xp[-1] .. xp[0]+2pi) are covered.
    """
    xp = np.asarray(xp, float)
    fp = np.asarray(fp, float)
    xe = np.concatenate([xp[-1:] - 2 * np.pi, xp, xp[:1] + 2 * np.pi])
    fe = np.concatenate([fp[-1:], fp, fp[:1]])
    return np.interp(query, xe, fe)


def radii_on_grid(R, Z, origin, theta_grid):
    """Radius profile of one LCFS slice on a fixed angle grid.

    ``R``, ``Z`` are the native vertices (m). Returns ``r`` at each ``theta_grid``
    angle, where ``theta`` is measured from ``origin`` (theta=0 outboard, CCW).

    **Precondition:** the vertices must be angularly monotonic about ``origin``, i.e.
    the boundary must be star-shaped about it. This is *not* automatic -- it fails for
    roughly a third of raw GMagH5 slices -- and when it fails ``argsort`` silently
    reorders the vertices and the result is a boundary that never existed. Criteria
    S3/S4 in :mod:`src.data.filter` are what establish it; call this only on slices
    that pass them, with ``origin`` the same center those criteria were judged about.
    """
    R0, Z0 = origin
    th = np.arctan2(Z - Z0, R - R0) % (2 * np.pi)
    r = np.hypot(R - R0, Z - Z0)
    o = np.argsort(th)
    return periodic_interp(theta_grid, th[o], r[o])


PE_BASE = 5.0          # replaces the transformer's 10000: shots are seconds, not tokens
PE_PAIRS = 5           # sin/cos pairs -> 2 * PE_PAIRS feature columns


def time_encoding(t, pairs=PE_PAIRS, base=PE_BASE):
    """Sinusoidal positional encoding of ignitron **time in seconds**.

    ``PE[:, 2i] = sin(t / base**(2i/d))``, ``PE[:, 2i+1] = cos(...)`` for
    ``i = 0..pairs-1``. Returns ``(nt, 2 * pairs)``.

    ``d = 2 * pairs`` is the width of the PE block itself, playing the role of the
    transformer's ``d_model`` -- the convention where the encoding fills the whole
    embedding. It is a choice, not a given: the PE here is 10 columns appended to a
    22-column feature vector, so there is no embedding of width 10 to point at. It is
    also the choice that spreads the frequencies. With ``d = 2 * pairs`` the exponent
    reaches 0.8 and the periods span 6.3 -> 22.8 s; using the full 32-column feature
    width instead would cap the exponent at 0.25 and squeeze all five periods into
    6.3 -> 9.4 s, i.e. five near-duplicates.

    ``base=5`` rather than the usual 10000 because the positions here are physical
    seconds over a ~10-70 s discharge, not token indices over thousands of steps.
    Both sin and cos are emitted so the phase is unambiguous -- sin alone cannot
    distinguish a rising from a falling instant.
    """
    t = np.asarray(t, float).reshape(-1)
    d = 2 * pairs
    i = np.arange(pairs)
    denom = base ** (2 * i / d)                      # (pairs,)
    ang = t[:, None] / denom[None, :]                # (nt, pairs)
    out = np.empty((t.size, d), np.float32)
    out[:, 0::2] = np.sin(ang)
    out[:, 1::2] = np.cos(ang)
    return out


def time_encoding_names(pairs=PE_PAIRS, base=PE_BASE):
    """Column names for :func:`time_encoding`, in the same order."""
    names = []
    for k in range(pairs):
        period = 2 * np.pi * base ** (2 * k / (2 * pairs))
        names += [f"pe_sin_{k}_T{period:.1f}s", f"pe_cos_{k}_T{period:.1f}s"]
    return names


def read_center(hf, nt):
    """``(2, nt)`` plasma geometric center ``(Rgeom, Zgeom)`` in **metres**.

    ``inputs/GMAG_GEOM`` is ``(16, nt)`` with channels ``[0, 1]`` = (Rgeom, Zgeom) in
    millimetres. Absent or misshaped -> all-NaN, which fails criterion S3 for every
    slice rather than silently substituting a default center.
    """
    center = np.full((2, nt), np.nan)
    if "inputs/GMAG_GEOM" in hf:
        gm = hf["inputs/GMAG_GEOM"]
        if (getattr(gm, "shape", None) is not None and gm.ndim == 2
                and gm.shape[0] >= 2 and gm.shape[1] == nt):
            center = np.asarray(gm, float)[:2] * 1e-3
    return center


def discharge_window(bnd):
    """First/last column index where the LCFS is physically plausible.

    ``bnd`` is ``(64, nt)`` interleaved ``[R0,Z0,...,R31,Z31]``. A slice counts when
    finite, non-zero, R in (1.8, 3.3), R-span > 0.3 m and |Z| < 1.2 m. Returns the
    inclusive ``(i0, i1)`` span, or ``None``.

    Superseded for dataset building by the criteria in :mod:`src.data.filter` (this
    bounding box accepts boundaries up to 1 cm outside the real vessel); retained
    because ``src/data/sweep_inputs.py`` still windows with it.
    """
    nt = bnd.shape[1]
    g = bnd.reshape(N_POINTS, 2, nt)              # [point, (R,Z), time]
    R = g[:, 0, :]
    Z = g[:, 1, :]
    finite = np.isfinite(g).all(axis=(0, 1))
    nonzero = np.any(g != 0, axis=(0, 1))
    with np.errstate(invalid="ignore"):
        ok = (finite & nonzero
              & (R.min(0) > 1.8) & (R.max(0) < 3.3)
              & ((R.max(0) - R.min(0)) > 0.3) & (np.abs(Z).max(0) < 1.2))
    idx = np.where(ok)[0]
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])


def keep_span(keep):
    """Inclusive ``(i0, i1)`` span covering every kept slice, or ``None``.

    Trimming to this span keeps the stored arrays small and preserves the historical
    behaviour that ``time`` starts at the equilibrium start rather than at t=0.
    Slices *inside* the span that fail the filters stay in the arrays but are marked
    invalid, so the time base stays uniform for sequence models.
    """
    idx = np.where(np.asarray(keep, bool))[0]
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])


def build_node_map(config_path):
    """Resolve base.yml's ``data.input_list`` to ``{input_name: [(ds_path, node)]}``.

    Convention (mirrors ``src.data.input_stats``): ``*_real`` -> ``dcs/<node>/actual``,
    ``*_ref`` -> ``dcs/<node>/ref``. The stem is a key under ``nodes:`` whose list
    gives the member scope signals. These are the actuator/diagnostic inputs
    (PF/CS coils, heating power+phase, Ip reference, line density, measured Ip).
    """
    cfg = yaml.safe_load(pathlib.Path(config_path).read_text())
    node_groups, input_names = cfg["nodes"], cfg["data"]["input_list"]
    mapping = {}
    for name in input_names:
        if name.endswith("_real"):
            group_key, sub = name[:-5], "actual"
        elif name.endswith("_ref"):
            group_key, sub = name[:-4], "ref"
        else:
            raise ValueError(f"cannot infer node group for '{name}'")
        if group_key not in node_groups:
            raise KeyError(f"node group '{group_key}' not in {config_path}")
        mapping[name] = [(f"dcs/{n}/{sub}", n) for n in node_groups[group_key]]
    return mapping


def input_layout(node_map):
    """Flatten node_map to the per-channel ``X`` layout.

    Returns ``(channels, layout, F)``: ``channels`` is the ordered list of
    ``(input_name, ds_path, node)`` (one per feature column) and ``layout`` is a
    list of ``{input_name, channels, cols, nodes, unit}`` for ``meta.json``.
    """
    channels, layout, start = [], [], 0
    for input_name, paths in node_map.items():
        for ds_path, node in paths:
            channels.append((input_name, ds_path, node))
        layout.append({"input_name": input_name, "channels": len(paths),
                       "cols": [start, start + len(paths)],
                       "nodes": [n for _, n in paths], "unit": ""})
        start += len(paths)
    return channels, layout, start


def _read_channel(hf, ds_path, i0, i1):
    """Read one 1-D scope channel ``[i0:i1+1]`` from ``ds_path`` (``dcs/<scope>/<sub>``),
    falling back to ``inputs/<node>`` for MDS+ signals (e.g. SMAG_IP). ``None`` if absent.
    """
    path = ds_path if ds_path in hf else f"inputs/{ds_path.split('/')[1]}"
    if path not in hf:
        return None
    arr = np.asarray(hf[path], float).reshape(-1)   # scope traces are 1-D (nt,)
    return arr[i0:i1 + 1]


def reduce_stats(per_shot, eps):
    """Fold per-shot streaming stats into mean/std for X and Y.

    Each entry of ``per_shot`` carries ``X_sum`` / ``X_sumsq`` / ``X_cnt`` (length
    F; summed over that shot's valid rows, finite-only per column since X may hold
    NaN), ``Y_sum`` / ``Y_sumsq`` (length n_angles) and scalar ``n_valid``. Returns
    ``{X_mean, X_std, Y_mean, Y_std, n_valid}``; std is the population std, floored
    at ``eps``. X columns with no finite values collapse to mean 0 / std eps.
    """
    x_sum = np.sum([np.asarray(d["X_sum"], float) for d in per_shot], axis=0)
    x_sumsq = np.sum([np.asarray(d["X_sumsq"], float) for d in per_shot], axis=0)
    x_cnt = np.sum([np.asarray(d["X_cnt"], float) for d in per_shot], axis=0)
    y_sum = np.sum([np.asarray(d["Y_sum"], float) for d in per_shot], axis=0)
    y_sumsq = np.sum([np.asarray(d["Y_sumsq"], float) for d in per_shot], axis=0)
    n = float(sum(int(d["n_valid"]) for d in per_shot))
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.maximum(x_cnt, 1.0)
        x_mean = np.where(x_cnt > 0, x_sum / denom, 0.0)
        x_std = np.sqrt(np.maximum(np.where(x_cnt > 0, x_sumsq / denom, 0.0)
                                   - x_mean ** 2, 0.0))
    y_mean = y_sum / n
    y_std = np.sqrt(np.maximum(y_sumsq / n - y_mean ** 2, 0.0))
    out = {"X_mean": x_mean, "X_std": np.maximum(x_std, eps),
           "Y_mean": y_mean, "Y_std": np.maximum(y_std, eps), "n_valid": n}
    if per_shot and "S_sum" in per_shot[0]:
        s_sum = np.sum([np.asarray(d["S_sum"], float) for d in per_shot], axis=0)
        s_sumsq = np.sum([np.asarray(d["S_sumsq"], float) for d in per_shot], axis=0)
        n_s = float(sum(int(d["n_scalar_valid"]) for d in per_shot))
        s_mean = s_sum / max(n_s, 1.0)
        s_std = np.sqrt(np.maximum(s_sumsq / max(n_s, 1.0) - s_mean ** 2, 0.0))
        out["S_mean"] = s_mean
        out["S_std"] = np.maximum(s_std, eps)
        out["n_scalar_valid"] = n_s
    return out


def theta_grid(n_angles=N_ANGLES):
    """Fixed uniform angle grid in [0, 2*pi); theta=0 outboard (+R), CCW."""
    return np.arange(n_angles) / n_angles * 2 * np.pi


def _build_one(args):
    """Worker: build one ProjDB/Npz/<shot>.npz. Returns (shot, ok, payload|err)."""
    shot, merged_dir, npz_dir, channels, theta = args
    src = pathlib.Path(merged_dir) / f"{shot}.h5"
    out = pathlib.Path(npz_dir) / f"{shot}.npz"
    try:
        with h5py.File(src, "r") as hf:
            bnd = np.asarray(hf["targets/GMAG_BND"][:], float)      # (64, N)
            t_all = np.asarray(hf["time"], float).reshape(-1)
            center_all = read_center(hf, bnd.shape[1])

            # --- per-slice quality filters (src/data/filter.py) --------------------
            # Evaluated over the whole record, then trimmed to the span of survivors.
            q = SliceQuality(bnd, center_all, t_all)
            win = keep_span(q.keep)
            if win is None:
                return shot, False, "no slice passes the quality filters"
            i0, i1 = win
            keep = q.keep[i0:i1 + 1]
            fail = q.fail_code()[i0:i1 + 1]
            only = q.only_code()[i0:i1 + 1]
            flags = q.flags()[i0:i1 + 1]
            center = center_all[:, i0:i1 + 1]
            time = t_all[i0:i1 + 1]
            nt = time.size

            # V2 (uniform lattice) ships per-slice interpolation provenance. Absent
            # from the NpzOrigin/NpzGeom merges, so this stays optional.
            gap = None
            if "src_gap_ms" in hf:
                g_all = np.asarray(hf["src_gap_ms"], float).reshape(-1)
                if g_all.size == bnd.shape[1]:
                    gap = g_all[i0:i1 + 1].astype(np.float32)
            # Captured here while the file is open; the return payload below runs
            # after the ``with`` closes, so hf.attrs cannot be read there.
            grid = (None if "grid_source" not in hf.attrs else {
                "source": str(hf.attrs["grid_source"]),
                "hz": float(hf.attrs.get("grid_hz", float("nan"))),
                "dt_ms": 1e3 * float(hf.attrs.get("grid_dt", float("nan"))),
                "clip_gap_ms": float(hf.attrs.get("clip_gap_ms", float("nan"))),
                "k0": int(hf.attrs.get("grid_k0", 0)),
                "clip_dropped_n": int(hf.attrs.get("clip_dropped_n", 0)),
                "clip_dropped_s": float(hf.attrs.get("clip_dropped_s", 0.0)),
            })

            # X: one column per input channel (1-D scope trace on the shared grid),
            # then the time positional encoding appended on the right.
            # Absent / short channels are kept as NaN (faithful); nothing is imputed.
            n_ch = len(channels)
            F = n_ch + 2 * PE_PAIRS
            X = np.full((nt, F), np.nan, np.float32)
            for k, (_inp, ds_path, _node) in enumerate(channels):
                ch = _read_channel(hf, ds_path, i0, i1)
                if ch is not None and ch.shape == (nt,):
                    X[:, k] = ch.astype(np.float32)
            X[:, n_ch:] = time_encoding(time)

            g = bnd[:, i0:i1 + 1].reshape(N_POINTS, 2, nt)          # [pt,(R,Z),t]
            R = g[:, 0, :]
            Z = g[:, 1, :]
            # Y = r(theta) about each slice's OWN center (Rgeom, Zgeom). The filters
            # judge enclosure and star-shapedness about that same point, so the
            # projection's precondition holds exactly where ``keep`` is true -- and
            # only there, hence the NaN elsewhere rather than a fabricated profile.
            Y = np.full((nt, theta.size), np.nan, np.float32)
            for t in np.where(keep)[0]:
                Y[t] = radii_on_grid(R[:, t], Z[:, t],
                                     (center[0, t], center[1, t]), theta)
            bnd_RZ = np.transpose(g, (2, 0, 1)).astype(np.float32)  # (nt,32,2)

            # A slice is usable when it passes every filter. Faithful NaN in
            # individual X channels is preserved for downstream to handle -- it is
            # not a reason to drop a slice here.
            valid = keep & np.isfinite(Y).all(axis=1)
            if not valid.any():
                return shot, False, "zero valid slices"

            S = read_scalars(hf, i0, i1, nt)                 # (nt, N_SCALARS)
            s_valid = valid & np.isfinite(S).all(axis=1)

            # streaming stats over valid rows (float64 for stable accumulation).
            # X may carry NaN (absent channels) -> accumulate per column over its
            # finite values only, tracked via X_cnt; Y/S are finite on valid rows.
            Xv = X[valid].astype(np.float64)
            Yv = Y[valid].astype(np.float64)
            Sv = S[s_valid].astype(np.float64)
            fin_X = np.isfinite(Xv)

            extra = {} if gap is None else {"src_gap_ms": gap}
            np.savez(out, X=X.astype(np.float32), Y=Y, S=S, bnd_RZ=bnd_RZ,
                     time=time.astype(np.float32), valid=valid,
                     center=center.T.astype(np.float32),   # (nt, 2) polar origin per slice
                     fail=fail, only=only, flags=flags, **extra)
        return shot, True, {
            "shot": int(shot), "n_slices": int(nt),
            "t_start": float(time[0]), "t_end": float(time[-1]),
            "reject": {c.code: int(np.sum(q.first_failure() == c.code))
                       for c in FILTER_REGISTRY},
            "X_sum": np.where(fin_X, Xv, 0.0).sum(0),
            "X_sumsq": np.where(fin_X, Xv * Xv, 0.0).sum(0),
            "X_cnt": fin_X.sum(axis=0),
            "Y_sum": Yv.sum(0), "Y_sumsq": (Yv * Yv).sum(0),
            "n_valid": int(valid.sum()),
            "n_fabricated": (0 if gap is None
                             else int((valid & (gap > FAB_GAP_MS)).sum())),
            "grid": grid,
            "S_sum": Sv.sum(0), "S_sumsq": (Sv * Sv).sum(0),
            "n_scalar_valid": int(s_valid.sum()),
        }
    except Exception as exc:  # noqa: BLE001
        if out.exists():
            out.unlink()
        return shot, False, str(exc)


def run(merged_dir=None, npz_dir=None, config_path=None,
        n_angles=N_ANGLES, workers=1):
    """Build ProjDB/Npz/<shot>.npz for every Merged shot + write meta.json.

    Inputs come from ``configs/base.yml`` (``data.input_list`` -> actuator/
    diagnostic scope traces). Target is always ``targets/GMAG_BND``.
    """
    cfg = get_proj_config()
    merged_dir = pathlib.Path(merged_dir) if merged_dir else cfg.mergedh5_dir
    npz_dir = pathlib.Path(npz_dir) if npz_dir else cfg.mergednpz_dir
    config_path = (pathlib.Path(config_path) if config_path else cfg.base_config_f)
    target_node = "GMAG_BND"

    node_map = build_node_map(config_path)
    channels, layout, n_ch = input_layout(node_map)
    # the time positional encoding occupies the columns after the scope channels
    pe_names = time_encoding_names()
    layout = layout + [{"input_name": "time_pe", "channels": len(pe_names),
                        "cols": [n_ch, n_ch + len(pe_names)],
                        "nodes": pe_names, "unit": ""}]
    F = n_ch + len(pe_names)
    theta = theta_grid(n_angles)
    print(f"inputs ({F} channels = {n_ch} scope + {len(pe_names)} time-PE): "
          + ", ".join(f"{k}={len(v)}" for k, v in node_map.items()))

    if npz_dir.exists():
        shutil.rmtree(npz_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)

    shots = sorted(int(p.stem) for p in merged_dir.glob("*.h5"))
    items = [(s, merged_dir, npz_dir, channels, theta)
             for s in shots]
    results = pmap(_build_one, items, workers, "build_npz")

    ok = [payload for _, success, payload in results if success]
    fail = [(s, payload) for s, success, payload in results if not success]

    # split the per-shot summary (into meta["shots"]) from the streaming stats
    summary_keys = ("shot", "n_slices", "t_start", "t_end")
    shots_meta = sorted([{k: d[k] for k in summary_keys} for d in ok],
                        key=lambda d: d["shot"])
    # fold the per-shot sums into dataset-wide mean/std (eps floors std)
    norm = (reduce_stats(ok, cfg.eps) if ok
            else {"X_mean": [], "X_std": [], "Y_mean": [], "Y_std": [],
                  "n_valid": 0})

    reject = {c.code: sum(d["reject"][c.code] for d in ok) for c in FILTER_REGISTRY}
    meta = {
        # Y is r(theta) about each slice's own (Rgeom, Zgeom). There is no fixed
        # origin in this pipeline; a reader keying on a constant would be wrong.
        "origin": "per_slice_gmag_geom",
        "theta_deg": list(np.degrees(theta)),
        "target": {"node": target_node, "repr": "polar_r_theta",
                   "n_angles": n_angles, "unit": "m",
                   "center": "per-slice, stored in the 'center' array"},
        "filters": {
            "module": "src.data.filter",
            "criteria": [{"code": c.code, "title": c.title, "detail": c.detail,
                          "requires": list(c.requires)} for c in FILTER_REGISTRY],
            "thresholds": dict(FILTER_DEFAULTS),
            "rejected_by_first_failure": reject,
            "note": ("'fail'/'only' arrays are 1-based criterion indices (0 = kept); "
                     "'flags' bit i set = criterion i passes, order-free"),
        },
        "inputs": layout,
        "n_features": F,
        "time_pe": {"pairs": PE_PAIRS, "base": PE_BASE, "pos": "ignitron time [s]",
                    "formula": "PE[:,2i]=sin(t/base**(2i/(2*pairs))), 2i+1=cos(...)",
                    "columns": pe_names},
        "arrays": {"X": ["nt", F], "Y": ["nt", n_angles], "S": ["nt", N_SCALARS],
                   "bnd_RZ": ["nt", 32, 2], "time": ["nt"], "valid": ["nt"],
                   "center": ["nt", 2], "fail": ["nt"], "only": ["nt"],
                   "flags": ["nt"]},
        "norm": {"X_mean": np.asarray(norm["X_mean"]).tolist(),
                 "X_std": np.asarray(norm["X_std"]).tolist(),
                 "Y_mean": np.asarray(norm["Y_mean"]).tolist(),
                 "Y_std": np.asarray(norm["Y_std"]).tolist(),
                 "n_valid": int(norm["n_valid"])},
        "shots": shots_meta,
        "n_shots": len(ok),
        "total_slices": int(sum(d["n_slices"] for d in ok)),
    }
    grids = [d["grid"] for d in ok if d.get("grid")]
    if grids:
        meta["grid"] = {
            "source": sorted({g["source"] for g in grids}),
            "hz": sorted({g["hz"] for g in grids}),
            "dt_ms": sorted({round(g["dt_ms"], 6) for g in grids}),
            "clip_gap_ms": sorted({g["clip_gap_ms"] for g in grids}),
            "clip_dropped_n_total": int(sum(g["clip_dropped_n"] for g in grids)),
            "clip_dropped_s_total": float(sum(g["clip_dropped_s"] for g in grids)),
            "k0_per_shot": {str(d["shot"]): d["grid"]["k0"]
                            for d in ok if d.get("grid")},
            # spec C5: the measured cost of filtering interpolated geometry
            "fabricated_valid_slices": int(sum(d.get("n_fabricated", 0) for d in ok)),
            "fabricated_gap_threshold_ms": FAB_GAP_MS,
        }
    if any("src_gap_ms" in np.load(npz_dir / f"{d['shot']}.npz").files for d in ok[:1]):
        meta["arrays"]["src_gap_ms"] = ["nt"]
    if "S_mean" in norm:
        meta["norm"]["S_mean"] = np.asarray(norm["S_mean"]).tolist()
        meta["norm"]["S_std"] = np.asarray(norm["S_std"]).tolist()
        meta["norm"]["n_scalar_valid"] = int(norm["n_scalar_valid"])
        meta["scalars"] = [{"name": n, "node": node, "channel": ch, "scale": sc}
                           for (n, node, ch, sc) in SCALAR_NODES]
    with open(npz_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\ndone: {len(ok)} npz, {len(fail)} failed")
    for s, err in fail[:20]:
        print(f"  {s}: {err}")
    if len(fail) > 20:
        print(f"  ... and {len(fail) - 20} more")
    return len(ok), len(fail)
