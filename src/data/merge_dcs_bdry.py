# -*- coding: utf-8 -*-
"""Build the resampled, time-aligned MergedH5 dataset for selected shots.

Inputs (both on the **ignitron** time base, t=0 at the ignitron — no ref-onset
shift needed, so the old ``ip_align`` step is gone):

* DCS scopes from ``DCSHeating/DCS_archive_<shot>.mat`` (raw DCS archive; the 30
  ``*_scope`` nodes, all sharing ``Ip_scope.time``, columns 0=ref / 3=actual).
* GMAG boundary + inputs from ``GMagH5/<shot>.h5`` (``targets/GMAG_BND`` etc.).

For each shot passing the quality masks (nonzero bnd, fs > 400 Hz, DCS ok) AND
total Ip flat-top > ``cfg.flat_top_min_s`` (from ``flat_top.csv``), the shared
grid is the DCS ignitron time clipped to the ``GMAG_BND`` time span. DCS scopes
are kept native (sliced to the window); every GMAG input/target is resampled
onto that grid. Output ``MergedH5/<shot>.h5`` has ONE shared ``time`` axis
(no per-signal ``_time`` datasets) — the same layout the old ``Merged/`` used, so
``build_npz`` / ``input_stats`` / ``sweep_inputs`` consume it unchanged.

The old output dir is removed and regenerated.

Importable as a package module; driven by the root runner::

    python scripts/run_data_pre.py --workers 8
"""
import pathlib
import shutil

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from ..proj_config import get_proj_config
from ..utils import pmap

REF_COL = 0   # DCS scope column: reference (commanded setpoint)
ACT_COL = 3   # DCS scope column: actual measured value


def resample_to_grid(grid, ts, values):
    """Linearly interpolate ``values`` (sampled at ``ts``) onto ``grid``.

    ``values`` is 1-D ``(N,)`` or multi-channel ``(C, N)`` with time on the last
    axis. Grid points outside the source span are set to NaN (no extrapolation).

    ``ts`` is sorted and de-duplicated here rather than assumed ascending: 27 of the
    759 selected shots carry a GMAG time axis with a single non-increasing sample
    (sub-millisecond jitter), and ``np.interp`` requires a monotonic ``xp`` -- given a
    non-monotonic one it returns silently wrong values instead of failing. That
    corrupted the boundary for 5 shots before this guard existed.
    """
    grid = np.asarray(grid, float)
    ts = np.asarray(ts, float)
    values = np.asarray(values, float)
    if np.any(np.diff(ts) <= 0):
        order = np.argsort(ts, kind="stable")
        ts = ts[order]
        values = values[..., order]
        uniq = np.r_[True, np.diff(ts) > 0]
        ts = ts[uniq]
        values = values[..., uniq]
    outside = (grid < ts[0]) | (grid > ts[-1])
    if values.ndim == 1:
        out = np.interp(grid, ts, values)
        out[outside] = np.nan
        return out
    out = np.empty((values.shape[0], grid.size), float)
    for c in range(values.shape[0]):
        row = np.interp(grid, ts, values[c])
        row[outside] = np.nan
        out[c] = row
    return out


def uniform_grid(bnd_t, dcs_t, hz=500.0, clip_gap_ms=16.0, t_min=0.0):
    """Uniform lattice ``t_k = k / hz`` over the usable window, plus an audit dict.

    V2's whole point is that this axis is *generated*, not taken from data: the
    native ``GMAG_BND_time`` is non-uniform (2.048 ms is only the modal step, with
    dropouts to ~33 ms and 0 of 1346 shots uniformly sampled), which a step-indexed
    GRU silently misreads as equal spacing.

    The window is the **longest contiguous run** of ``bnd_t`` samples whose
    consecutive spacing is <= ``clip_gap_ms``, intersected with the DCS span and with
    ``t >= t_min``. Longest-run rather than first-gap, and 16 ms rather than ~2x the
    modal step, both matter: shot 57295 carries an isolated 7.179 ms interval at
    t=0.089 s inside its fast window, and a 4.1 ms first-gap rule truncates it to 45
    grid points. 16 ms sits between the 2.048 ms cadence and the 32.768 ms idle
    tier, so only the idle tier and genuine long dropouts terminate the grid.

    Measured on the 759 selected shots this clip removes nothing (0 of 7 302 943
    samples at t >= 0): ``t >= 0`` and the DCS span already bound the grid inside the
    fast-acquisition window. It is kept as a guard for a future campaign or a
    different ``hz``, and ``clip_dropped_*`` turn "it never fires" into a
    per-build measurement rather than an assumption.

    Returns ``(grid, info)``. ``grid`` is empty when no usable window survives --
    the caller drops that shot.
    """
    dt = 1.0 / float(hz)
    info = {"grid_hz": float(hz), "grid_dt": dt, "grid_k0": 0, "grid_n": 0,
            "clip_gap_ms": float(clip_gap_ms), "clip_t_start": float("nan"),
            "clip_t_end": float("nan"), "clip_dropped_n": 0, "clip_dropped_s": 0.0}
    # np.unique sorts and de-duplicates: 27 of the 759 shots carry a non-increasing
    # GMAG timestamp (sub-ms jitter), which would make np.diff-based gaps meaningless.
    ts = np.unique(np.asarray(bnd_t, float).reshape(-1))
    dcs_t = np.asarray(dcs_t, float).reshape(-1)
    if ts.size < 2 or dcs_t.size < 2:
        return np.empty(0), info

    brk = np.where(np.diff(ts) > clip_gap_ms / 1000.0)[0]
    starts = np.r_[0, brk + 1]
    stops = np.r_[brk, ts.size - 1]
    lo = np.maximum(ts[starts], max(float(t_min), float(dcs_t.min())))
    hi = np.minimum(ts[stops], float(dcs_t.max()))
    dur = np.where(hi > lo, hi - lo, -1.0)
    if not np.any(dur > 0):
        return np.empty(0), info
    j = int(np.argmax(dur))
    t_lo, t_hi = float(lo[j]), float(hi[j])

    # eps guards the lattice arithmetic: t_lo/dt for a t_lo already on the lattice
    # can land a hair above the integer and push ceil() one step too far.
    k0 = int(np.ceil(t_lo / dt - 1e-9))
    k1 = int(np.floor(t_hi / dt + 1e-9))
    if k1 < k0:
        return np.empty(0), info
    grid = np.arange(k0, k1 + 1, dtype=float) * dt

    in_win = (ts >= t_lo) & (ts <= t_hi)
    pos = ts >= t_min
    pos_ptp = (ts[pos].max() - ts[pos].min()) if pos.any() else 0.0
    info.update(grid_k0=k0, grid_n=int(grid.size),
                clip_t_start=t_lo, clip_t_end=t_hi,
                clip_dropped_n=int(pos.sum() - (in_win & pos).sum()),
                clip_dropped_s=float(max(0.0, pos_ptp - (t_hi - t_lo))))
    return grid, info


def load_selected_shots(status_csv, flat_top_csv, min_s):
    """Shots passing all criteria (nonzero bnd, fs > 400 Hz, DCS ok, flat-top long enough)."""
    ss = pd.read_csv(status_csv)
    status_mask = (ss["bnd_category"].str.contains("nonzero", case=False, na=False)
                   & (ss["bnd_fs_hz"] > 400)
                   & ss["dcs_category"].str.contains("ok"))
    status_shots = set(ss.loc[status_mask, "shot"].astype(int))

    ft = pd.read_csv(flat_top_csv)
    ft_ok = ft[(ft["flag"] == "ok") & (ft["total_flat_top_s"] > min_s)]
    ft_shots = set(ft_ok["shot"].astype(int))

    return sorted(status_shots & ft_shots)


def read_dcs_mat(mat_path):
    """Return ``(dcs_time, {scope: (ref_1d, actual_1d)})`` from a DCS archive .mat.

    All ``*_scope`` nodes share ``Ip_scope.time`` (the PCS grid); only nodes
    whose values matrix matches that length are kept, so index-slicing is safe.
    """
    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    def signals(obj):
        try:
            return np.asarray(obj.signals.values, dtype=float)
        except Exception:
            try:
                return np.asarray(obj[0, 0]["signals"]["values"][0, 0], dtype=float)
            except Exception:
                return None

    def time_of(obj):
        try:
            return np.asarray(obj.time, dtype=float).reshape(-1)
        except Exception:
            return np.asarray(obj[0, 0]["time"], dtype=float).reshape(-1)

    ip = data["Ip_scope"]
    dcs_time = time_of(ip)

    scopes = {}
    for name, obj in data.items():
        if name.startswith("__") or not name.endswith("_scope"):
            continue
        y = signals(obj)
        if y is None or y.ndim < 2 or y.shape[1] <= ACT_COL:
            continue
        if y.shape[0] != dcs_time.size:        # guard: must share the Ip_scope grid
            continue
        scopes[name] = (y[:, REF_COL], y[:, ACT_COL])
    return dcs_time, scopes


def _merge_one(args):
    """Worker: build one resampled MergedH5/<shot>.h5. Returns (shot, ok, err)."""
    shot, bnd_dir, dcs_dir, merged_dir, time_base = args
    bnd_path = bnd_dir / f"{shot}.h5"
    dcs_path = dcs_dir / f"DCS_archive_{shot}.mat"
    out_path = merged_dir / f"{shot}.h5"

    if not bnd_path.exists():
        return shot, False, f"boundary h5 missing: {bnd_path}"
    if not dcs_path.exists():
        return shot, False, f"DCS mat missing: {dcs_path}"

    try:
        dcs_time, scopes = read_dcs_mat(dcs_path)

        with h5py.File(bnd_path, "r") as hf_bnd:
            gb_t = np.asarray(hf_bnd["targets/GMAG_BND_time"][:], float).reshape(-1)
            if time_base == "gmag":
                # The LCFS is what we predict, so it must never be interpolated: the
                # grid IS the GMAG_BND reconstruction time, clipped to the DCS span so
                # the scope interpolation never extrapolates. Every surviving slice is
                # then a real reconstruction, which is also what makes the per-slice
                # quality filters meaningful.
                #
                # This axis is NOT uniform: 2.048 ms (488 Hz) is only the modal step,
                # and every shot contains longer gaps (up to ~33 ms) where the
                # reconstruction dropped out. A few shots additionally carry a single
                # non-increasing sample (sub-ms jitter). Time must be strictly
                # increasing -- np.interp's xp requires it, and np.diff(t) feeds
                # cumulative features -- so keep a strictly increasing subsequence
                # rather than a contiguous slice.
                inwin = np.where((gb_t >= dcs_time.min()) & (gb_t <= dcs_time.max()))[0]
                if inwin.size < 2:
                    return shot, False, "empty DCS/GMAG_BND overlap"
                span = np.arange(int(inwin[0]), int(inwin[-1]) + 1)
                t_span = gb_t[span]
                strict = np.r_[True, np.diff(t_span) > 0]
                sel = span[strict]
                if sel.size < 2:
                    return shot, False, "GMAG_BND time axis not increasing"
                grid = gb_t[sel]
            elif time_base == "dcs":
                lo, hi = float(gb_t.min()), float(gb_t.max())
                win = np.where((dcs_time >= lo) & (dcs_time <= hi))[0]
                if win.size < 2:
                    return shot, False, "empty DCS/GMAG_BND overlap"
                grid = dcs_time[int(win[0]):int(win[-1]) + 1]
            else:
                return shot, False, f"unknown time_base {time_base!r}"

            with h5py.File(out_path, "w") as hf_out:
                for ak, av in hf_bnd.attrs.items():
                    hf_out.attrs[ak] = av
                hf_out.attrs["time_base"] = "ignitron"
                hf_out.attrs["grid_source"] = time_base
                hf_out.attrs["grid_n"] = grid.size
                hf_out.create_dataset("time", data=grid)

                # DCS scopes: resampled onto the grid (identity when time_base='dcs',
                # since the grid is then a contiguous slice of the scope's own axis)
                dcs_grp = hf_out.create_group("dcs")
                for scope, (ref, act) in scopes.items():
                    g = dcs_grp.create_group(scope)
                    two = resample_to_grid(grid, dcs_time,
                                           np.stack([ref, act], axis=0))
                    g.create_dataset("ref", data=two[0])
                    g.create_dataset("actual", data=two[1])

                # GMAG inputs/targets: resampled onto the grid
                for grp_name in ("inputs", "targets"):
                    out_grp = hf_out.create_group(grp_name)
                    src = hf_bnd[grp_name]
                    for name in src:
                        if name.endswith("_time"):
                            continue
                        tname = f"{name}_time"
                        if tname not in src:
                            continue
                        ts = np.asarray(src[tname][:], float).reshape(-1)
                        vals = np.asarray(src[name][:], float)
                        if ts.size < 2 or vals.shape[-1] != ts.size:
                            continue
                        out_grp.create_dataset(
                            name, data=resample_to_grid(grid, ts, vals))
        return shot, True, None
    except Exception as exc:  # noqa: BLE001
        if out_path.exists():
            out_path.unlink()
        return shot, False, str(exc)


def run(bnd_dir=None, dcs_dir=None, merged_dir=None, status_csv=None,
        flat_top_csv=None, workers=1, time_base="gmag"):
    """Build MergedH5 for every selected shot.

    ``time_base='gmag'`` (default) makes the shared grid the native 488 Hz
    ``GMAG_BND_time``, so the LCFS target is never interpolated and every slice is a
    real reconstruction. ``'dcs'`` reproduces the historical ~1 kHz DCS grid, which
    interpolated the boundary -- kept only to regenerate the old baseline.
    """
    cfg = get_proj_config()
    bnd_dir = pathlib.Path(bnd_dir) if bnd_dir else cfg.gmagh5_dir
    dcs_dir = pathlib.Path(dcs_dir) if dcs_dir else cfg.dcsheating_dir
    merged_dir = pathlib.Path(merged_dir) if merged_dir else cfg.mergedh5_dir
    status_csv = pathlib.Path(status_csv) if status_csv else cfg.shot_status_csv
    flat_top_csv = (pathlib.Path(flat_top_csv) if flat_top_csv
                    else cfg.stats_dir / "flat_top.csv")

    sel = load_selected_shots(status_csv, flat_top_csv, cfg.flat_top_min_s)
    print(f"selected shots: {len(sel)}")
    print(f"  DCS .mat  : {dcs_dir}")
    print(f"  boundary  : {bnd_dir}")
    print(f"  output    : {merged_dir}")
    print(f"  time base : {time_base} "
          f"({'GMAG_BND native, target not interpolated' if time_base == 'gmag' else 'DCS grid, boundary interpolated'})")

    # regenerate the dataset (new format incompatible with the old Merged files)
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    items = [(s, bnd_dir, dcs_dir, merged_dir, time_base) for s in sel]
    results = pmap(_merge_one, items, workers, "merging")

    ok = sum(1 for _, success, _ in results if success)
    fail = len(results) - ok
    errors = [(s, err) for s, success, err in results if not success]

    print(f"\ndone: {ok} merged, {fail} failed")
    if errors:
        print("errors:")
        for s, err in errors[:20]:
            print(f"  {s}: {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    return ok, fail
