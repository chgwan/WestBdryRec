# -*- coding: utf-8 -*-
"""Find the DCS<->MDS time-alignment delta from the Ip signals.

The DCS scopes and the MDS boundary signals use different time origins. DCS
actual Ip (``dcs/Ip_scope/actual``) and the MDS SMAG_IP (``inputs/SMAG_IP``) are
the SAME plasma current, so the integer sample shift that best matches them is
the residual delay between the two clocks after a coarse Ip-ref-onset alignment.

For each merged shot:
  1. coarse-align: set t=0 at the DCS Ip-ref onset (ref first exceeds 1 A);
  2. for each candidate ``delta`` (sample offset applied to that onset index),
     interpolate SMAG_IP onto the shifted DCS time grid and compute the MSE
     over the overlapping window;
  3. the best delta is the argmin-MSE shift.

Writes ``ip_align_delta.csv`` (one row per shot) and prints the aggregate best
delta over the plasma shots.

Importable as a package module; driven from ``scripts/run_find_delta.py``.
"""
import collections
import csv
import os
import pathlib

import h5py
import numpy as np

from ..proj_config import get_proj_config
from ..utils import pmap

REF_THRESH = 1.0        # Ip ref (A) must exceed this to mark the onset
REF_EPS = 1e-5

CSV_FIELDS = ["shot", "ref_start_index", "best_delta", "best_delta_ms",
              "best_mse", "mse_at_0", "ip_peak_ka", "n_overlap", "flag"]


def ref_start_index(ip_ref):
    """Index where the Ip ref first exceeds REF_THRESH, minus one (or None)."""
    ids = np.where(ip_ref > REF_THRESH + REF_EPS)[0]
    if ids.size == 0:
        return None
    return max(int(ids[0]) - 1, 0)


def _mse(t_shifted, ip_act, smag_t, smag_ip):
    """MSE of actual Ip vs SMAG_IP over their overlap, or None if they miss."""
    lo, hi = max(t_shifted[0], smag_t[0]), min(t_shifted[-1], smag_t[-1])
    if hi <= lo:
        return None
    m = (t_shifted >= lo) & (t_shifted <= hi)
    if int(m.sum()) < 2:
        return None
    smag_on_dcs = np.interp(t_shifted[m], smag_t, smag_ip)
    return float(np.mean((ip_act[m] - smag_on_dcs) ** 2)), int(m.sum())


def _load(shot, merged_dir):
    """Return ``(t_dcs, ip_ref_A, ip_act_kA, smag_t, smag_kA)`` or None."""
    path = pathlib.Path(merged_dir) / f"{shot}.h5"
    if not path.exists():
        return None
    try:
        with h5py.File(path, "r") as hf:
            t = np.asarray(hf["dcs/time"][:], float).reshape(-1)
            ip_ref = np.asarray(hf["dcs/Ip_scope/ref"][:], float).reshape(-1)
            ip_act = np.asarray(hf["dcs/Ip_scope/actual"][:], float).reshape(-1)
            smag = np.asarray(hf["inputs/SMAG_IP"][:], float).reshape(-1)
            smag_t = np.asarray(hf["inputs/SMAG_IP_time"][:], float).reshape(-1)
    except Exception:  # noqa: BLE001 - missing scope / unreadable file
        return None
    if t.size < 2 or smag_t.size < 2:
        return None
    order = np.argsort(smag_t)               # np.interp needs ascending x
    return t, ip_ref, ip_act / 1e3, smag_t[order], smag[order]


def best_delta_one(args):
    """Worker: scan deltas for one shot; return a result dict.

    ``flag`` is ``ok`` (plasma shot, usable), ``low_ip`` (peak below the plasma
    threshold, excluded from the aggregate), or an error tag.
    """
    shot, merged_dir, deltas, min_ip_ka = args
    loaded = _load(shot, merged_dir)
    if loaded is None:
        return {"shot": shot, "flag": "unreadable"}
    t, ip_ref, ip_act, smag_t, smag = loaded

    ip_peak = float(np.nanmax(np.abs(ip_act))) if ip_act.size else 0.0
    dt_ms = float(np.median(np.diff(t))) * 1e3
    si = ref_start_index(ip_ref)
    if si is None:
        return {"shot": shot, "ip_peak_ka": ip_peak, "flag": "no_ref"}

    n = t.size
    best = None  # (mse, delta, n_overlap)
    for d in deltas:
        j = si + d
        if j < 0 or j >= n:
            continue
        res = _mse(t - t[j], ip_act, smag_t, smag)
        if res is None:
            continue
        mse, n_ov = res
        if d == 0:
            mse_at_0 = mse
        if best is None or mse < best[0]:
            best = (mse, d, n_ov)
    if best is None:
        return {"shot": shot, "ref_start_index": si, "ip_peak_ka": ip_peak,
                "flag": "no_overlap"}

    best_mse, best_d, n_ov = best
    return {
        "shot": shot,
        "ref_start_index": si,
        "best_delta": best_d,
        "best_delta_ms": best_d * dt_ms,
        "best_mse": best_mse,
        "mse_at_0": locals().get("mse_at_0", float("nan")),
        "ip_peak_ka": ip_peak,
        "n_overlap": n_ov,
        "flag": "ok" if ip_peak >= min_ip_ka else "low_ip",
    }


def _fmt(v, spec):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return format(v, spec)


def write_csv(rows, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["shot"]):
            w.writerow({
                "shot": r["shot"],
                "ref_start_index": r.get("ref_start_index", ""),
                "best_delta": r.get("best_delta", ""),
                "best_delta_ms": _fmt(r.get("best_delta_ms"), ".4g"),
                "best_mse": _fmt(r.get("best_mse"), ".6g"),
                "mse_at_0": _fmt(r.get("mse_at_0"), ".6g"),
                "ip_peak_ka": _fmt(r.get("ip_peak_ka"), ".5g"),
                "n_overlap": r.get("n_overlap", ""),
                "flag": r.get("flag", ""),
            })
    print(f"  wrote {path}  ({len(rows)} shots)")


def print_summary(rows, max_delta):
    flags = collections.Counter(r.get("flag") for r in rows)
    print("\nflags: " + ", ".join(f"{k}={v}" for k, v in sorted(flags.items())))

    ok = [r for r in rows if r.get("flag") == "ok"]
    if not ok:
        print("no usable plasma shots")
        return

    deltas = np.array([r["best_delta"] for r in ok])
    dt_ms = np.median([r["best_delta_ms"] / r["best_delta"]
                       for r in ok if r["best_delta"] != 0] or [1.0])
    mode_delta = collections.Counter(deltas.tolist()).most_common(1)[0][0]
    at_bound = int(np.sum(np.abs(deltas) == max_delta))

    print(f"\nbest delta over {len(ok)} plasma shots (sample units, dt~{dt_ms:.2f} ms):")
    print(f"  median = {int(np.median(deltas)):+d}  "
          f"({np.median(deltas) * dt_ms:+.2f} ms)")
    print(f"  mean   = {deltas.mean():+.2f}  ({deltas.mean() * dt_ms:+.2f} ms)")
    print(f"  mode   = {int(mode_delta):+d}  ({mode_delta * dt_ms:+.2f} ms)")
    print(f"  range  = [{int(deltas.min()):+d}, {int(deltas.max()):+d}]")
    if at_bound:
        print(f"  WARNING: {at_bound} shots hit the scan edge "
              f"(|delta|={max_delta}); widen --max-delta")

    tot0 = sum(r["mse_at_0"] for r in ok if not np.isnan(r["mse_at_0"]))
    totb = sum(r["best_mse"] for r in ok)
    if tot0 > 0:
        print(f"  sum MSE: delta=0 -> {tot0:.4g}, best -> {totb:.4g} "
              f"({100 * (tot0 - totb) / tot0:.1f}% lower)")


def run(merged_dir=None, max_delta=150, min_ip_ka=50.0, workers=None,
        csv_path=None):
    """Scan every merged shot for its best DCS<->MDS Ip alignment delta.

    Programmatic entry point (no argparse). Writes ``ip_align_delta.csv`` and
    returns the list of per-shot result dicts.
    """
    cfg = get_proj_config()
    merged_dir = pathlib.Path(merged_dir) if merged_dir else cfg.merged_dir
    workers = workers or min(16, os.cpu_count() or 1)
    deltas = list(range(-max_delta, max_delta + 1))

    shots = sorted(int(p.stem) for p in merged_dir.glob("*.h5"))
    print(f"scanning {len(shots)} merged shots, "
          f"delta in [{-max_delta}, {max_delta}] ...")

    items = [(s, merged_dir, deltas, min_ip_ka) for s in shots]
    rows = pmap(best_delta_one, items, workers, "ip_align")

    csv_path = csv_path or (cfg.stats_dir / "ip_align_delta.csv")
    write_csv(rows, csv_path)
    print_summary(rows, max_delta)
    return rows
