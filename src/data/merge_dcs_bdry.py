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
    axis. Grid points outside ``[ts[0], ts[-1]]`` are set to NaN (no
    extrapolation). ``ts`` must be ascending.
    """
    grid = np.asarray(grid, float)
    ts = np.asarray(ts, float)
    values = np.asarray(values, float)
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
    shot, bnd_dir, dcs_dir, merged_dir = args
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
            lo, hi = float(gb_t.min()), float(gb_t.max())
            win = np.where((dcs_time >= lo) & (dcs_time <= hi))[0]
            if win.size < 2:
                return shot, False, "empty DCS/GMAG_BND overlap"
            w0, w1 = int(win[0]), int(win[-1])
            grid = dcs_time[w0:w1 + 1]            # ignitron time, clipped to GMAG_BND span

            with h5py.File(out_path, "w") as hf_out:
                for ak, av in hf_bnd.attrs.items():
                    hf_out.attrs[ak] = av
                hf_out.attrs["time_base"] = "ignitron"
                hf_out.attrs["grid_n"] = grid.size
                hf_out.create_dataset("time", data=grid)

                # DCS scopes: native, sliced to the window (shared Ip_scope grid)
                dcs_grp = hf_out.create_group("dcs")
                for scope, (ref, act) in scopes.items():
                    g = dcs_grp.create_group(scope)
                    g.create_dataset("ref", data=ref[w0:w1 + 1])
                    g.create_dataset("actual", data=act[w0:w1 + 1])

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
        flat_top_csv=None, workers=1):
    cfg = get_proj_config()
    bnd_dir = pathlib.Path(bnd_dir) if bnd_dir else cfg.gmagh5_dir
    dcs_dir = pathlib.Path(dcs_dir) if dcs_dir else cfg.dcsheating_dir
    merged_dir = pathlib.Path(merged_dir) if merged_dir else cfg.mergedh5_dir
    status_csv = pathlib.Path(status_csv) if status_csv else cfg.shot_status_csv
    flat_top_csv = (pathlib.Path(flat_top_csv) if flat_top_csv
                    else cfg.stats_dir / "flat_top.csv")

    sel = load_selected_shots(status_csv, flat_top_csv, cfg.flat_top_min_s)
    print(f"selected shots: {len(sel)}")
    print(f"  DCS .mat : {dcs_dir}")
    print(f"  boundary : {bnd_dir}")
    print(f"  output   : {merged_dir}")

    # regenerate the dataset (new format incompatible with the old Merged files)
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    items = [(s, bnd_dir, dcs_dir, merged_dir) for s in sel]
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
