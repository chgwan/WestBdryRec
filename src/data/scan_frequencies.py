# -*- coding: utf-8 -*-
"""Sampling-frequency survey for boundary signals and the DCS ``.mat`` time axis.

Consolidates three frequency concerns so the root runner (``run_data_pre``) has a
single entry point:

1. Per-signal aggregate summary. ``shot_status`` already scans every boundary h5
   per-signal and writes ``shot_freq_stats.csv`` (rows = shots, cols = signals,
   values = fs_hz). This module DERIVES the per-signal view from that CSV -- no
   re-scan -- producing a per-signal table (n / min / median / max / n_distinct)
   and writing it to ``signal_freq_summary.csv``.

2. Per-shot boundary h5 integrity check. For one representative shot, compute
   each signal's fs straight from its ``<name>_time`` vector via
   ``calc_sample_frequency`` and compare it to the ``median_fs_hz`` attribute
   stored on the signal -- a quick "computed matches stored" sanity check
   (absorbed from the former ``calc_frequencies.py``).

3. DCS ``.mat`` time-axis reader. Each ``DCS_archive_<shot>.mat`` stores one
   shared time grid under every ``*_scope`` struct; ``mat_time`` unwraps it to the
   1-D axis (fs ~ 1 kHz), printed for a representative shot.

scipy's ``loadmat`` turns every ``*_scope`` key into a (1,1) object array around
a struct with fields ``('time', 'signals')``; the time axis is reached by
unwrapping that wrapper:

    data = loadmat(path)
    data["Ip_scope"][0, 0]          -> the struct record (numpy.void)
    data["Ip_scope"][0, 0]["time"]  -> the time vector, flattened to 1-D

Importable as a package module; driven by the root runner::

    python scripts/run_data_pre.py
"""
import csv
import pathlib

import h5py
import numpy as np
from private_modules.utils.com_tools import calc_sample_frequency
from scipy.io import loadmat

from ..proj_config import get_proj_config

_cfg = get_proj_config()

# shot_freq_stats.csv columns that are per-shot metadata, not signals.
META_COLS = {"shot", "status", "n_signals", "n_downsampled", "downsampled"}

# Representative shots for the one-file source-level checks.
DEFAULT_DCS_MAT = "DCS_archive_57269.mat"
DEFAULT_BOUNDARY_SHOT = "57993"


def mat_time(mat_path, scope="Ip_scope"):
    """Return the 1-D time axis (seconds) stored under ``scope`` in a DCS .mat.

    All scopes share one time grid, so any ``*_scope`` key works; ``Ip_scope`` is
    used by convention.
    """
    data = loadmat(str(mat_path))
    record = data[scope][0, 0]            # unwrap the (1,1) object array
    return np.asarray(record["time"]).reshape(-1)


def iter_signals(group):
    """Yield ``(name, signal_dataset, time_dataset)`` for signals with a time vec."""
    for name in group:
        if name.endswith("_time"):
            continue
        time_name = f"{name}_time"
        if time_name in group:
            yield name, group[name], group[time_name]


def shot_signal_freqs(h5_path, method="median"):
    """Compute per-signal fs for one boundary h5.

    Returns ``{(group_name, name): (calc_fs, stored_fs)}`` where ``calc_fs`` is
    from ``calc_sample_frequency`` on the time vector and ``stored_fs`` is the
    signal's ``median_fs_hz`` attribute (``None`` if absent).

    ``method="median"`` matches ``shot_status`` and the stored attr; the function
    default ("mean") is derailed by the few large dt gaps in these time vectors,
    so it is not used here.
    """
    out = {}
    with h5py.File(h5_path, "r") as hf:
        for group_name in hf:
            group = hf[group_name]
            if not isinstance(group, h5py.Group):
                continue
            for name, signal, time_ds in iter_signals(group):
                times = time_ds[:]
                if times.size < 2:
                    continue
                out[(group_name, name)] = (
                    calc_sample_frequency(times, method=method),
                    signal.attrs.get("median_fs_hz", None))
    return out


def load_signal_freqs(csv_path):
    """Read ``shot_freq_stats.csv`` -> ``{signal: np.ndarray of fs_hz}``.

    Blank cells mean the signal was absent in that shot, so they are skipped.
    """
    with pathlib.Path(csv_path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        signals = [c for c in reader.fieldnames if c not in META_COLS]
        data = {sig: [] for sig in signals}
        for row in reader:
            for sig in signals:
                val = row[sig]
                if val:  # blank => signal absent in that shot
                    data[sig].append(float(val))
    return {sig: np.array(v, dtype=float) for sig, v in data.items()}


def aggregate(records):
    """Return per-signal rows ``(signal, n, min, median, max, n_distinct)``."""
    rows = []
    for sig in sorted(records):
        vals = records[sig]
        n = vals.size
        if n == 0:
            rows.append((sig, 0, np.nan, np.nan, np.nan, 0))
            continue
        rows.append((sig, n, float(vals.min()), float(np.median(vals)),
                     float(vals.max()),
                     int(np.unique(np.round(vals, 3)).size)))
    return rows


def write_summary(rows, path):
    """Write the per-signal aggregate table to ``path``."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["signal", "n_files", "min_hz", "median_hz", "max_hz",
                    "n_distinct"])
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} signals)")


def print_summary(rows, freq_csv):
    header = (f"{'signal':<26} {'n':>5} {'min':>12} {'median':>12} "
              f"{'max':>12} {'#distinct':>10}")
    print(f"\nper-signal sampling frequency (from {pathlib.Path(freq_csv).name}):")
    print(header)
    print("-" * len(header))
    for sig, n, vmin, vmed, vmax, ndist in rows:
        if n == 0:
            print(f"{sig:<26} {n:>5} {'-':>12} {'-':>12} {'-':>12} {ndist:>10}")
            continue
        print(f"{sig:<26} {n:>5} {vmin:>12.3f} {vmed:>12.3f} "
              f"{vmax:>12.3f} {ndist:>10}")


def print_dcs_mat_fs(mat_path, scope="Ip_scope"):
    """Print the DCS ``.mat`` time-axis length + fs for one representative shot."""
    mat_path = pathlib.Path(mat_path)
    if not mat_path.exists():
        print(f"  (skip DCS .mat check: {mat_path} not found)")
        return
    t = mat_time(mat_path, scope)
    dt = np.median(np.diff(t))
    print(f"  DCS .mat {mat_path.name}: n={t.size}  duration={t[-1] - t[0]:.4g} s  "
          f"fs={1.0 / dt:.3f} Hz  (scope={scope})")


def print_shot_freqs(h5_path, method="median"):
    """Print a calc-vs-stored fs integrity check for one boundary h5.

    ``calc_fs`` comes from ``calc_sample_frequency`` on each signal's time vector;
    ``stored`` is the ``median_fs_hz`` attribute. Only mismatches (>1%) are
    listed; otherwise a one-line summary is printed.
    """
    h5_path = pathlib.Path(h5_path)
    if not h5_path.exists():
        print(f"  (skip boundary h5 check: {h5_path} not found)")
        return
    freqs = shot_signal_freqs(h5_path, method=method)
    mismatches = []
    n_with_attr = 0
    for (group_name, name), (calc_fs, stored) in sorted(freqs.items()):
        if stored is None or stored <= 0:
            continue
        n_with_attr += 1
        if abs(calc_fs - stored) / stored > 0.01:
            mismatches.append((group_name, name, calc_fs, stored))
    if mismatches:
        print(f"  boundary h5 {h5_path.name}: {len(mismatches)}/{n_with_attr} "
              f"signals (with median_fs_hz) disagree:")
        for group_name, name, calc_fs, stored in mismatches:
            print(f"    [{group_name}] {name:<18} calc={calc_fs:.3f}  "
                  f"stored={stored:.3f}")
    else:
        print(f"  boundary h5 {h5_path.name}: {n_with_attr}/{len(freqs)} signals "
              f"have median_fs_hz, all match calc_fs")


def run(freq_csv=None, summary_csv=None, mat_path=None, scope="Ip_scope",
        shot_h5=None, method="median"):
    """Build the per-signal frequency summary and print both source-level checks.

    No directory scan: per-shot frequencies already live in ``shot_freq_stats.csv``
    (written by ``shot_status``), so the aggregate is derived from it; the
    boundary h5 and DCS .mat checks each open just one representative file.
    """
    freq_csv = pathlib.Path(freq_csv) if freq_csv else _cfg.shot_freq_stats_csv
    summary_csv = (pathlib.Path(summary_csv) if summary_csv
                   else _cfg.signal_freq_summary_csv)
    mat_path = (pathlib.Path(mat_path) if mat_path
                else _cfg.dcs_org_dir / DEFAULT_DCS_MAT)
    shot_h5 = (pathlib.Path(shot_h5) if shot_h5
               else _cfg.data_org_dir / f"{DEFAULT_BOUNDARY_SHOT}.h5")

    records = load_signal_freqs(freq_csv)
    rows = aggregate(records)
    write_summary(rows, summary_csv)
    print_summary(rows, freq_csv)

    print("\nDCS .mat time axis:")
    print_dcs_mat_fs(mat_path, scope)

    print("\nboundary h5 per-signal fs (calc vs stored):")
    print_shot_freqs(shot_h5, method=method)
    return rows
