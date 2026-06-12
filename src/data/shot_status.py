# -*- coding: utf-8 -*-
"""Per-shot availability/health status for boundary h5 AND DCS h5, plus the
per-signal sampling-frequency scan over the boundary h5.

Two tables are written per run:

1. ``shot_status.csv`` -- one row per shot present in either source:
    shot            shot number (key)
    bnd_category    boundary GMAG_BND: nonzero | allzero | absent | corrupt | missing
    bnd_fs_hz       boundary GMAG_BND sampling rate (blank if absent/corrupt)
    dcs_category    DCS record:        ok | truncated | missing
    dcs_n_samples   length of the DCS /time axis (0 if missing)
    dcs_fs_hz       DCS sampling rate from /time  (blank if missing/truncated)

2. ``shot_freq_stats.csv`` -- the per-signal frequency scan (merged in from the
   former shot_stats.py): for every signal with a ``<name>_time`` companion, the
   rate is computed via ``private_modules.utils.com_tools.calc_sample_frequency``.
   Each shot row reports status / n_signals / n_downsampled / downsampled plus
   one column per signal; the nominal rate per signal is the median over all
   shots, so the downsample flag is data-driven.

The boundary h5 is opened only once per shot, so the GMAG_BND verdict and the
per-signal frequencies come from a single read pass.

Sources (resolved from configs/base.yml via proj_config):
    boundary h5 : <DATABASE_dir>/DataOrg/<shot>.h5
    DCS h5      : <DATABASE_dir>/DCSH5/<shot>.h5
    status out  : <proj_db_dir>/Stats/shot_status.csv
    freq out    : <proj_db_dir>/Stats/shot_freq_stats.csv

Consolidates the former bnd_status_stats.py, dcs_freq_stats.py and shot_stats.py.

Note: proj_config + calc_sample_frequency pull in pandas (via private_modules),
which needs the conda env's libstdc++ on the linker path. Run under an activated
`torch` env.

Importable as a package module (relative imports); drive it via the root runner:
    python scripts/run_data_pre.py            # drives this + other prep steps
"""
import collections
import csv
import os
import pathlib

import h5py
import numpy as np

from private_modules.utils.com_tools import calc_sample_frequency

from ..proj_config import get_proj_config
from ..utils import pmap
from .paths import data_paths

# Single source of truth: src/proj_config.py (ProjConfig fields/ClassVars).
_cfg = get_proj_config()
BND = _cfg.gmag_bnd_key
TRUNC_MIN = _cfg.dcs_trunc_min          # DCS records shorter than this = "truncated"
DOWNSAMPLE_TOL = _cfg.downsample_tol    # < this * nominal => "downsampled"


def default_paths():
    """Derive source/output dirs from configs/base.yml via proj_config."""
    d = data_paths()
    return {
        **d,
        "status_csv": d["stats_dir"] / "shot_status.csv",
        "freq_csv": d["stats_dir"] / "shot_freq_stats.csv",
    }


def bnd_scan(path, method="median"):
    """Open one boundary h5 once; return ``(shot, bnd_category, status, freqs)``.

    ``bnd_category`` is the GMAG_BND verdict (nonzero/allzero/absent/corrupt) and
    ``status`` is the per-signal frequency-scan verdict (ok/corrupt:<msg>); the
    two are independent, so a bad GMAG_BND read does not poison the freq scan or
    vice versa. ``freqs`` maps ``"{group}/{signal}"`` to its rate in Hz.
    """
    shot = pathlib.Path(path).stem
    freqs = {}
    status = "ok"
    try:
        with h5py.File(path, "r") as hf:
            try:
                if BND not in hf:
                    bcat = "absent"
                else:
                    data = hf[BND][:]
                    bcat = "nonzero" if np.any(data != 0) else "allzero"
            except Exception:  # noqa: BLE001 - GMAG_BND read failed but keep scanning
                bcat = "corrupt"
            try:
                for group_name in hf:
                    group = hf[group_name]
                    if not isinstance(group, h5py.Group):
                        continue
                    for name in group:
                        if name.endswith("_time"):
                            continue
                        time_name = f"{name}_time"
                        if time_name not in group:
                            continue
                        times = group[time_name][:]
                        if times.size < 2:
                            continue
                        freqs[f"{group_name}/{name}"] = calc_sample_frequency(
                            times, method=method)
            except Exception as exc:  # noqa: BLE001 - record and keep the GMAG_BND verdict
                status = f"corrupt: {exc}"
    except Exception as exc:  # noqa: BLE001 - whole file unreadable
        return shot, "corrupt", f"corrupt: {exc}", {}
    return shot, bcat, status, freqs


def dcs_status(path):
    """Return ``(shot, dcs_category, n_samples, fs_hz)`` for a DCS h5 file."""
    shot = pathlib.Path(path).stem
    try:
        with h5py.File(path, "r") as hf:
            if "time" not in hf:
                return shot, "truncated", 0, np.nan
            t = np.asarray(hf["time"][:], dtype=float).reshape(-1)
    except Exception:  # noqa: BLE001
        return shot, "corrupt", 0, np.nan
    n = t.size
    if n < 2:
        return shot, "truncated", n, np.nan
    dt = np.diff(t)
    dt = dt[dt > 0]
    fs = float(1.0 / np.median(dt)) if dt.size else np.nan
    return shot, ("ok" if n >= TRUNC_MIN else "truncated"), n, fs


def collect(bnd_dir, dcs_dir, workers, method="median"):
    """Scan both sources; return ``(status_rows, freq_records)``.

    ``status_rows``  : list of ``(shot, bnd, dcs_cat, n_samples, fs)``
    ``freq_records`` : ``{shot: (status, {signal: fs_hz})}`` for every shot that
                       has a boundary h5 present
    """
    bnd = pmap(bnd_scan, sorted(pathlib.Path(bnd_dir).glob("*.h5")),
               workers, "boundary")
    bnd_cat = {s: b for s, b, _, _ in bnd}
    freq_records = {s: (st, fr) for s, _, st, fr in bnd}

    dcs = {s: (c, n, f) for s, c, n, f in pmap(
        dcs_status, sorted(pathlib.Path(dcs_dir).glob("*.h5")), workers, "dcs")}

    shots = sorted(set(bnd_cat) | set(dcs),
                   key=lambda s: (int(s) if s.isdigit() else 1 << 62, s))
    status_rows = []
    for s in shots:
        dcat, n, f = dcs.get(s, ("missing", 0, np.nan))
        # GMAG_BND rate is already in the freq scan; reuse it (no extra read).
        bfs = freq_records.get(s, ("", {}))[1].get(BND, np.nan)
        status_rows.append((s, bnd_cat.get(s, "missing"), bfs, dcat, n, f))
    return status_rows, freq_records


def write_status_csv(rows, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["shot", "bnd_category", "bnd_fs_hz", "dcs_category",
                    "dcs_n_samples", "dcs_fs_hz"])
        for s, bcat, bfs, dcat, n, f in rows:
            w.writerow([s, bcat,
                        "" if np.isnan(bfs) else f"{bfs:.3f}",
                        dcat, n, "" if np.isnan(f) else f"{f:.3f}"])
    print(f"  wrote {path}  ({len(rows)} shots)")


def write_freq_csv(records, path):
    """Write the per-signal frequency table; return the sorted ``signals`` list
    and the per-signal ``nominal`` rate (median over shots)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signals = sorted({s for _, fr in records.values() for s in fr})
    nominal = {}
    for sig in signals:
        vals = [fr[sig] for _, fr in records.values() if sig in fr]
        nominal[sig] = float(np.median(vals)) if vals else float("nan")

    fieldnames = (["shot", "status", "n_signals", "n_downsampled", "downsampled"]
                  + signals)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for shot in sorted(records, key=lambda s: (len(s), s)):
            status, freqs = records[shot]
            n_down = sum(1 for sig, fs in freqs.items()
                         if nominal[sig] > 0
                         and fs < DOWNSAMPLE_TOL * nominal[sig])
            row = {"shot": shot, "status": status, "n_signals": len(freqs),
                   "n_downsampled": n_down, "downsampled": n_down > 0}
            for sig in signals:
                row[sig] = f"{freqs[sig]:.3f}" if sig in freqs else ""
            w.writerow(row)
    print(f"  wrote {path}  ({len(records)} shots, {len(signals)} signals)")
    return signals, nominal


def print_summary(status_rows, freq_records, nominal):
    bnd = collections.Counter(r[1] for r in status_rows)
    dcs = collections.Counter(r[3] for r in status_rows)
    print("\nboundary GMAG_BND: " + ", ".join(f"{k}={v}" for k, v in sorted(bnd.items())))
    bfs = [r[2] for r in status_rows if np.isfinite(r[2])]
    if bfs:
        rates = collections.Counter(int(round(x)) for x in bfs)
        print("GMAG_BND fs (Hz): " + ", ".join(f"{r}Hz={c}" for r, c in sorted(rates.items())))
    print("DCS record      : " + ", ".join(f"{k}={v}" for k, v in sorted(dcs.items())))
    fs = [r[5] for r in status_rows if np.isfinite(r[5])]
    if fs:
        rates = collections.Counter(int(round(x)) for x in fs)
        print("DCS fs (Hz)     : " + ", ".join(f"{r}Hz={c}" for r, c in sorted(rates.items())))

    n_ok = sum(1 for st, _ in freq_records.values() if st == "ok")
    n_corrupt = len(freq_records) - n_ok
    n_with = sum(1 for _, fr in freq_records.values()
                 if any(nominal.get(s, 0) > 0
                        and fs < DOWNSAMPLE_TOL * nominal[s]
                        for s, fs in fr.items()))
    print(f"\nboundary freq scan: ok={n_ok}, corrupt={n_corrupt}, with downsample={n_with}")
    print("nominal rate per signal (median over shots):")
    for sig in nominal:
        present = sum(1 for _, fr in freq_records.values() if sig in fr)
        print(f"  {sig:<26} {nominal[sig]:>10.3f} Hz   present in {present} shots")


def run(bnd_dir=None, dcs_dir=None, status_csv=None, freq_csv=None,
        workers=None, method="median"):
    """Build both status tables; defaults come from proj_config when omitted.

    Programmatic entry point (no argparse) so callers like ``run_data_pre`` can
    drive it directly.
    """
    d = default_paths()
    status_rows, freq_records = collect(
        bnd_dir or d["bnd_dir"], dcs_dir or d["dcs_dir"],
        workers or min(16, os.cpu_count() or 1), method=method)
    write_status_csv(status_rows, status_csv or d["status_csv"])
    _, nominal = write_freq_csv(freq_records, freq_csv or d["freq_csv"])
    print_summary(status_rows, freq_records, nominal)
    return status_rows, freq_records
