# -*- coding: utf-8 -*-
"""Readers for WEST DCS archive ``.mat`` files and per-node statistics.

Each ``DCS_archive_<shot>.mat`` stores ~129 top-level keys; the per-signal
scopes used for boundary reconstruction are MATLAB structs with two fields:

    ``time``    1-D time axis, shape (n_samples,)
    ``signals`` (1,1) struct -> ``values``  float matrix, shape (n_samples, n_channels)

Across WEST DCS every scope's ``signals.values`` carries 4 columns:
``0`` = reference (the commanded setpoint), ``1``/``2`` = derived, and
``3`` = the value actually measured in the experiment. Statistics are computed
for BOTH the reference (col 0) and the actual (col 3) so they can be told
apart; each output row carries a ``column`` field (``ref`` / ``actual``).

Builds a shot-by-shot, per-node ``{length, mean, variance}`` table over every
DCS file, keyed by shot number. Importable as a package module; the table is
produced via ``run()`` and driven from outside (e.g. the root data-prep runner).
"""
import csv
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml
from scipy.io import loadmat
from tqdm import tqdm

from ..proj_config import get_proj_config

_cfg = get_proj_config()

DEFAULT_DCS_DIR = _cfg.dcs_org_dir
DEFAULT_CFG = _cfg.base_config_f
DEFAULT_OUTDIR = _cfg.stats_dir
# (label, index) pairs for the columns of each scope's values matrix.
# Column 0 = reference (commanded setpoint); column 3 = actual measured value.
COLUMNS = _cfg.dcs_columns


def get_ref_interval(mat_file):
    data_dict = read_mat_file(mat_file, ['Ip_scope_0'])
    Ip_ref_data = data_dict['Ip_scope_0']
    eps = 1e-5
    set_value = 1.0
    ids = Ip_ref_data > (set_value + eps)
    start_idx = np.min(np.arange(len(Ip_ref_data))[ids])
    start_idx = start_idx - 1
    end_idx = len(Ip_ref_data)
    return start_idx, end_idx

def read_mat_file(mat_file: os.PathLike, nodes):
    data = loadmat(mat_file)
    node_stat_dict = dict({})
    # assume same time axis. 
    Ip_struct = data['Ip_scope'][0,0]
    time_field = Ip_struct['time'].squeeze()
    node_stat_dict['time'] = time_field
    for node in nodes:
        scope_name = node[:-2]
        slice = int(node[-1])
        scope_struct = data[scope_name][0,0]
        # time_field = scope_struct['time'].squeeze()
        # Extract signals and values
        signals_field = scope_struct['signals']
        signal_values = signals_field['values'][0,0] 
        # num_signals = signal_values.shape[1] 
        signal_data = signal_values[:, slice]
        node_stat_dict[node] = signal_data
    return node_stat_dict

def scope_values(struct):
    """Return the (n_samples, n_channels) float matrix held by a scope struct."""
    return struct["signals"]["values"][0, 0]


def read_scope_signal(data, node, column):
    """Return the 1-D signal for scope ``node`` at matrix ``column``.

    Raises ``KeyError`` if the scope is absent and ``ValueError`` if its values
    matrix is empty or has no such column.
    """
    if node not in data:
        raise KeyError(node)
    vals = np.asarray(scope_values(data[node][0, 0]))
    if vals.ndim == 0 or vals.size == 0:
        raise ValueError(f"{node}: empty values matrix")
    if vals.ndim < 2 or vals.shape[1] <= column:
        raise ValueError(f"{node}: values {vals.shape} has no column {column}")
    return vals[:, column]


def node_stats(signal):
    """Return ``(length, n_nan, mean, variance)`` for a 1-D signal, NaN-aware."""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    finite = np.isfinite(signal)
    length = signal.size
    n_nan = int((~finite).sum())
    if not finite.any():
        return length, n_nan, np.nan, np.nan
    return length, n_nan, float(signal[finite].mean()), float(signal[finite].var())


def read_shot(path, nodes, columns=COLUMNS):
    """Return ``{(node, label): signal}`` for every present scope/column.

    A scope (or a specific column of it) that is missing or too narrow is
    simply omitted, so callers can treat absence explicitly.
    """
    data = loadmat(str(path))
    out = {}
    for node in nodes:
        for label, col in columns:
            try:
                out[(node, label)] = read_scope_signal(data, node, col)
            except (KeyError, ValueError):
                continue
    return out


def parse_shot(path):
    """Extract the shot number from a ``DCS_archive_<shot>.mat`` filename."""
    return pathlib.Path(path).stem.replace("DCS_archive_", "")


def _process_file(path, nodes, columns):
    """Worker: load one file -> ``(shot, {(node,label): stats}, None)`` or error."""
    shot = parse_shot(path)
    try:
        signals = read_shot(path, nodes, columns)
    except Exception as exc:  # noqa: BLE001  (corrupt/truncated .mat)
        return shot, None, str(exc)
    return shot, {key: node_stats(sig) for key, sig in signals.items()}, None


def collect_stats(directory, nodes, workers, columns=COLUMNS):
    """Compute per-node, per-column stats for every ``DCS_archive_*.mat`` file.

    Returns ``(rows, errors)`` where each row is
    ``(shot, node, column, length, n_nan, mean, variance)`` and ``errors`` lists
    files that could not be read. ``shot`` is the main key: rows are sorted by
    shot number, then node order, then column order.
    """
    files = sorted(pathlib.Path(directory).glob("DCS_archive_*.mat"))
    if not files:
        raise SystemExit(f"No DCS_archive_*.mat files found in {directory}")

    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process_file, str(f), nodes, columns) for f in files]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="DCS stats"):
            shot, stats, err = fut.result()
            if err is not None:
                errors.append((shot, err))
                continue
            for node in nodes:
                for label, _ in columns:
                    length, n_nan, mean, var = stats.get(
                        (node, label), (0, 0, np.nan, np.nan))
                    rows.append((shot, node, label, length, n_nan, mean, var))

    node_order = {n: i for i, n in enumerate(nodes)}
    col_order = {lab: i for i, (lab, _) in enumerate(columns)}
    big = 1 << 62
    rows.sort(key=lambda r: (int(r[0]) if r[0].isdigit() else big,
                             node_order.get(r[1], len(nodes)),
                             col_order.get(r[2], len(columns))))
    return rows, errors


def write_csv(rows, path):
    """Write the per-node, per-column stats table to ``path`` (tidy/long)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shot", "node", "column", "length", "n_nan", "mean", "variance"])
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def load_nodes(cfg_path=DEFAULT_CFG, key="scan_nodes"):
    """Read the node list under ``nodes.<key>`` from a YAML config."""
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    return list(cfg["nodes"][key])


def run(directory=DEFAULT_DCS_DIR, cfg=DEFAULT_CFG, node_key="scan_nodes",
        workers=None, outdir=DEFAULT_OUTDIR, csv_path=None):
    """Build the per-node, per-column DCS stats table; defaults from proj_config.

    Programmatic entry point (no argparse) so external callers can drive it
    directly. Writes ``<outdir>/dcs_node_stats.csv`` unless ``csv_path`` is given
    and returns ``(rows, errors)``.
    """
    workers = workers or min(16, os.cpu_count() or 1)

    nodes = load_nodes(cfg, node_key)
    print(f"nodes ({len(nodes)}): {', '.join(nodes[:6])}, ...")
    print(f"columns: {', '.join(f'{lab}=col{idx}' for lab, idx in COLUMNS)}")

    rows, errors = collect_stats(directory, nodes, workers)
    shots = {r[0] for r in rows} | {e[0] for e in errors}
    print(f"\nprocessed {len(shots)} shots -> {len(rows)} node-column-rows")
    if errors:
        print(f"  {len(errors)} file(s) failed:")
        for shot, err in errors[:10]:
            print(f"    {shot}: {err}")

    csv_path = csv_path or (pathlib.Path(outdir) / "dcs_node_stats.csv")
    write_csv(rows, csv_path)
    return rows, errors
