# -*- coding: utf-8 -*-
"""Build the resampled, time-aligned Merged dataset for selected shots.

For each shot passing the quality masks (nonzero bnd, fs > 400 Hz, DCS ok) AND
total Ip flat-top > ``cfg.flat_top_min_s`` (from flat_top.csv), build a single
calibrated time grid ``time = dcs_time - ref_start_time`` (t=0 at the Ip-ref
onset), clip it to the GMAG_BND time span, keep DCS scopes native on that grid,
and resample every DataOrg input/target onto it. Output ``Merged/<shot>.h5`` has
ONE shared ``time`` axis (no per-signal ``_time`` datasets).

The old Merged dir is removed and regenerated (the format is incompatible).

Importable as a package module; driven by the root runner::

    python scripts/run_data_pre.py --workers 8
"""
import pathlib
import shutil

import h5py
import numpy as np
import pandas as pd

from ..proj_config import get_proj_config
from ..utils import pmap


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
    """Return ``(sorted_shots, {shot: ref_start_time})`` passing all criteria."""
    ss = pd.read_csv(status_csv)
    status_mask = (ss["bnd_category"].str.contains("nonzero", case=False, na=False)
                   & (ss["bnd_fs_hz"] > 400)
                   & ss["dcs_category"].str.contains("ok"))
    status_shots = set(ss.loc[status_mask, "shot"].astype(int))

    ft = pd.read_csv(flat_top_csv)
    ft_ok = ft[(ft["flag"] == "ok") & (ft["total_flat_top_s"] > min_s)]
    ft_shots = set(ft_ok["shot"].astype(int))
    ref_start = dict(zip(ft_ok["shot"].astype(int),
                         ft_ok["ref_start_time"].astype(float)))

    sel = sorted(status_shots & ft_shots)
    return sel, ref_start


def _merge_one(args):
    """Worker: build one resampled Merged/<shot>.h5. Returns (shot, ok, err)."""
    shot, bnd_dir, dcs_dir, merged_dir, ref_start_time = args
    bnd_path = bnd_dir / f"{shot}.h5"
    dcs_path = dcs_dir / f"{shot}.h5"
    out_path = merged_dir / f"{shot}.h5"

    if not bnd_path.exists():
        return shot, False, f"boundary h5 missing: {bnd_path}"
    if not dcs_path.exists():
        return shot, False, f"DCS h5 missing: {dcs_path}"

    try:
        with h5py.File(dcs_path, "r") as hf_dcs, h5py.File(bnd_path, "r") as hf_bnd:
            dcs_time = np.asarray(hf_dcs["time"][:], float).reshape(-1)
            grid_full = dcs_time - ref_start_time

            gb_t = np.asarray(hf_bnd["targets/GMAG_BND_time"][:], float).reshape(-1)
            lo, hi = float(gb_t.min()), float(gb_t.max())
            win = np.where((grid_full >= lo) & (grid_full <= hi))[0]
            if win.size < 2:
                return shot, False, "empty DCS/GMAG_BND overlap"
            w0, w1 = int(win[0]), int(win[-1])
            grid = grid_full[w0:w1 + 1]

            with h5py.File(out_path, "w") as hf_out:
                for ak, av in hf_bnd.attrs.items():
                    hf_out.attrs[ak] = av
                hf_out.attrs["ref_start_time"] = ref_start_time
                hf_out.attrs["grid_n"] = grid.size
                hf_out.create_dataset("time", data=grid)

                # DCS scopes: native, sliced to the window
                dcs_grp = hf_out.create_group("dcs")
                for scope in hf_dcs:
                    obj = hf_dcs[scope]
                    if not isinstance(obj, h5py.Group):
                        continue
                    g = dcs_grp.create_group(scope)
                    for col in obj:                       # ref / actual
                        arr = np.asarray(obj[col][:], float).reshape(-1)
                        g.create_dataset(col, data=arr[w0:w1 + 1])

                # DataOrg inputs/targets: resampled onto the grid
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
    bnd_dir = pathlib.Path(bnd_dir) if bnd_dir else cfg.data_org_dir
    dcs_dir = pathlib.Path(dcs_dir) if dcs_dir else cfg.dcs_h5_dir
    merged_dir = pathlib.Path(merged_dir) if merged_dir else cfg.merged_dir
    status_csv = pathlib.Path(status_csv) if status_csv else cfg.shot_status_csv
    flat_top_csv = (pathlib.Path(flat_top_csv) if flat_top_csv
                    else cfg.stats_dir / "flat_top.csv")

    sel, ref_start = load_selected_shots(status_csv, flat_top_csv, cfg.flat_top_min_s)
    print(f"selected shots: {len(sel)}")

    # regenerate the dataset (new format incompatible with the old Merged files)
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    items = [(s, bnd_dir, dcs_dir, merged_dir, ref_start[s]) for s in sel]
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
