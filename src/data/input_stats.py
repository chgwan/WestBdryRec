# -*- coding: utf-8 -*-
"""Per-shot, per-channel mean, variance and standard deviation of the inputs.

For every merged h5 shot, reads the ``dcs/<node>/actual`` (or ``ref``) datasets
for the nodes listed under ``data.input_list`` (resolved via the ``nodes``
groups in ``configs/base.yml``) and writes ONE tidy/long table:

    input_stats.csv — one row per (shot, input_name, channel):
        shot, input_name, channel, n_samples, mean, variance, std

Nodes absent under ``dcs/`` fall back to the boundary ``inputs/<node>``
dataset (e.g. SMAG_IP, which is an MDS+ signal, not a DCS scope).

Stats are shot-by-shot: each row is that shot's own mean/std over its finite
samples, NOT pooled across shots. The former across-shot rollups
(``input_stats_ch.csv``) are gone — derive any aggregate downstream if needed.

Entry point::

    python -m src.data.input_stats
"""
import csv
import pathlib

import h5py
import numpy as np
import yaml

from ..proj_config import get_proj_config
from ..utils import pmap


def build_node_map(cfg_path):
    """Return ``{input_name: [(ds_path, node_name), ...]}`` from base.yml.

    Convention:
      - ``*_real`` → ``actual`` sub-dataset
      - ``*_ref``  → ``ref`` sub-dataset
    The stem is looked up as a key under ``nodes:``.
    """
    cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
    node_groups = cfg["nodes"]
    input_names = cfg["data"]["input_list"]

    mapping = {}
    for name in input_names:
        if name.endswith("_real"):
            group_key, sub = name.replace("_real", ""), "actual"
        elif name.endswith("_ref"):
            group_key, sub = name.replace("_ref", ""), "ref"
        else:
            raise ValueError(f"cannot infer node group for '{name}'")
        if group_key not in node_groups:
            raise KeyError(f"node group '{group_key}' not in base.yml")
        nodes = node_groups[group_key]
        mapping[name] = [(f"dcs/{n}/{sub}", n) for n in nodes]
    return mapping


def _scan_shot(args):
    """Worker: return ``(shot, {ds_path: (sum, sum_sq, n)})`` for one merged h5.

    ``n`` is the count of finite samples; missing/empty channels are omitted so
    the caller only sees channels that are actually present in that shot.
    """
    shot, mdir, all_ds_paths = args
    path = mdir / f"{shot}.h5"
    if not path.exists():
        return shot, None
    try:
        result = {}
        with h5py.File(path, "r") as hf:
            for ds_path in all_ds_paths:
                real_path = ds_path
                if real_path not in hf:
                    # MDS+ signals (e.g. SMAG_IP) live under inputs/, not dcs/
                    real_path = "inputs/" + ds_path.split("/")[1]
                    if real_path not in hf:
                        continue
                data = np.asarray(hf[real_path], dtype=float)
                finite = data[np.isfinite(data)]
                if finite.size == 0:
                    continue
                result[ds_path] = (float(finite.sum()),
                                   float((finite ** 2).sum()),
                                   finite.size)
        return shot, result
    except Exception:  # noqa: BLE001
        return shot, None


def run(workers=1):
    cfg = get_proj_config()
    node_map = build_node_map(cfg.base_config_f)

    print("input -> node mapping:")
    for name, paths in node_map.items():
        print(f"  {name}: {[n for _, n in paths]}")

    shots = sorted(int(p.stem) for p in cfg.merged_dir.glob("*.h5"))
    print(f"\nscanning {len(shots)} merged shots ...")

    # every unique ds_path across all inputs (read once per shot)
    all_ds_paths = sorted({p for paths in node_map.values() for p, _ in paths})
    items = [(s, cfg.merged_dir, all_ds_paths) for s in shots]
    results = pmap(_scan_shot, items, workers, "input_stats")

    # ── per-shot, per-channel rows ───────────────────────────────────────
    # (shot, input_order, ch_order, input_name, node, n_samples, mean, var, std)
    input_order = {n: i for i, n in enumerate(node_map)}
    rows = []
    for shot, data in results:
        if data is None:
            continue
        for name, paths in node_map.items():
            for ch_i, (ds_path, node_name) in enumerate(paths):
                if ds_path not in data:
                    continue
                s, sq, n = data[ds_path]
                if n == 0:
                    m, var, sd = np.nan, np.nan, np.nan
                else:
                    m = s / n
                    var = max(sq / n - m ** 2, 0.0)
                    sd = np.sqrt(var)
                rows.append((shot, input_order[name], ch_i, name, node_name,
                             n, m, var, sd))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    # ── write tidy CSV (one row per shot × channel) ──────────────────────
    cfg.stats_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.stats_dir / "input_stats.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shot", "input_name", "channel", "n_samples",
                    "mean", "variance", "std"])
        for shot, _, _, name, node, n, m, v, s in rows:
            w.writerow([shot, name, node, n,
                        "" if np.isnan(m) else f"{m:.6g}",
                        "" if np.isnan(v) else f"{v:.6g}",
                        "" if np.isnan(s) else f"{s:.6g}"])
    n_chan = sum(len(p) for p in node_map.values())
    print(f"\nwrote {out_path}  ({len(rows)} rows = {len(shots)} shots "
          f"× up to {n_chan} channels)")

    # ── preview ──────────────────────────────────────────────────────────
    fmt = lambda v: "" if np.isnan(v) else f"{v:.6g}"
    print(f"\n{'shot':<8}{'input_name':<20}{'channel':<16}"
          f"{'n_samples':>12}{'mean':>14}{'variance':>14}{'std':>14}")
    print("-" * 98)
    for shot, _, _, name, node, n, m, v, s in rows[:8]:
        print(f"{shot:<8}{name:<20}{node:<16}{n:>12}"
              f"{fmt(m):>14}{fmt(v):>14}{fmt(s):>14}")
    if len(rows) > 8:
        print(f"... ({len(rows) - 8} more rows)")

    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(workers=args.workers)


if __name__ == "__main__":
    main()
