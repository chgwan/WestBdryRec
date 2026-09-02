# -*- coding: utf-8 -*-
"""Per-shot Ip flat-top detector (ported from EASTAnalyzer, adapted for WEST).

DCS actual Ip (``Ip_scope_3``) and ref Ip (``Ip_scope_0``) share one time axis.
The ref baseline (~1 A) dominates a full DCS record, so detection is restricted
to the plasma window ``[ref-onset, ref-offset]`` and the ref slope is normalized
by the *plateau* value (median over that window), not the whole-record median.

Pipeline: ref-flat segments -> refine each edge with the actual-Ip q20 threshold
(EAST ``get_flat_top_times``) -> total flat-top duration, then split the rest of
the window into ramp-up / ramp-down gaps by the sign of the average ref slope.
Writes ``flat_top.csv`` (per-shot summary) and ``flat_top_segments.csv``
(per-phase rows).

Importable as a package module; driven from ``scripts/run_data_pre.py``.
"""
import collections
import csv
import os
import pathlib

import numpy as np

from ..proj_config import get_proj_config
from ..utils import pmap
from . import mat_io

REF_THRESH = 1.0        # Ip ref (A) must exceed this to count as "on"
REF_EPS = 1e-5
FLAT_THRESHOLD = 1e-3   # |normalized ref slope| < this is "flat"
MIN_SEGMENT_S = 0.5     # minimum flat-segment duration to keep
Q_PCT = 20              # actual-Ip percentile used to refine flat-top edges

CSV_FIELDS = ["shot", "ref_start_index", "ref_start_time", "n_segments",
              "flat_top_start", "flat_top_end", "total_flat_top_s",
              "pass_3s", "ip_peak_ka", "flag"]

SEGMENT_CSV_FIELDS = ["shot", "i_segment", "start", "end", "flag"]


def ref_start_index(ip_ref):
    """First index where ref exceeds REF_THRESH, minus one (or None)."""
    ids = np.where(np.asarray(ip_ref, float) > REF_THRESH + REF_EPS)[0]
    if ids.size == 0:
        return None
    return max(int(ids[0]) - 1, 0)


def plasma_window(ip_ref):
    """Return ``(i0, i1)`` of the ref-active window, or None if ref never on."""
    ip_ref = np.asarray(ip_ref, float)
    ids = np.where(ip_ref > REF_THRESH + REF_EPS)[0]
    if ids.size == 0:
        return None
    i0 = max(int(ids[0]) - 1, 0)
    i1 = min(int(ids[-1]) + 1, ip_ref.size - 1)
    return i0, i1


def find_flat_segments(t, ip_ref, i0, i1,
                       flat_threshold=FLAT_THRESHOLD, min_segment_s=MIN_SEGMENT_S):
    """Return absolute-index ``(start, end)`` pairs of flat ref runs in [i0, i1]."""
    t = np.asarray(t, float)
    ip_ref = np.asarray(ip_ref, float)
    tw = t[i0:i1 + 1]
    rw = ip_ref[i0:i1 + 1]
    if tw.size < 3:
        return []
    plateau = float(np.median(rw))
    if plateau == 0:
        return []
    nd = np.diff(rw) / np.diff(tw) / plateau          # aligned to rw[:-1]
    idx = np.where(np.abs(nd) < flat_threshold)[0]    # local indices into window
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) != 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    segments = []
    for s, e in zip(starts, ends):
        a = int(idx[s])
        b = int(idx[e]) + 1          # right sample of the last flat diff-interval
        if tw[b] - tw[a] > min_segment_s:
            segments.append((i0 + a, i0 + b))
    return segments


def refine_flat_top(t, ip_act, segments, q_pct=Q_PCT, eps=1e-7):
    """Per segment: q-percentile of actual Ip; first/last time actual >= q."""
    t = np.asarray(t, float)
    ip_act = np.asarray(ip_act, float)
    out = []
    for a, b in segments:
        seg = ip_act[a:b + 1]
        segt = t[a:b + 1]
        if seg.size == 0:
            continue
        q = np.percentile(seg, q_pct)
        hit = np.where(seg >= q - eps)[0]
        if hit.size == 0:
            continue
        out.append((float(segt[hit[0]]), float(segt[hit[-1]])))
    return out


def phase_segments(t, ref, i0, i1, tops):
    """Chronological ramp-up / flat-top / ramp-down split of the plasma window.

    ``tops`` are the refined flat-top ``(start_time, end_time)`` spans (see
    :func:`refine_flat_top`). The rest of ``[t[i0], t[i1]]`` is gaps; each gap
    is ``ramp-up`` when the ref rises across it (``ref[end] - ref[start] > 0``),
    ``ramp-down`` otherwise. Zero-length gaps are skipped; the emitted segments
    tile the window with shared boundaries.

    Returns ``[{"i_segment", "start", "end", "flag"}, ...]`` in time order.
    """
    t = np.asarray(t, float)
    ref = np.asarray(ref, float)

    def gap_flag(start, end):
        a = int(np.searchsorted(t, start))               # first sample >= start
        b = int(np.searchsorted(t, end, "right")) - 1    # last sample <= end
        rise = ref[b] - ref[a] if b > a else 0.0
        return "ramp-up" if rise > 0 else "ramp-down"

    spans = []
    cursor = float(t[i0])
    for top_start, top_end in tops:
        if top_start > cursor:
            spans.append((cursor, float(top_start), gap_flag(cursor, top_start)))
        spans.append((float(top_start), float(top_end), "flat-top"))
        cursor = float(top_end)
    if float(t[i1]) > cursor:
        spans.append((cursor, float(t[i1]), gap_flag(cursor, float(t[i1]))))
    return [{"i_segment": k, "start": s, "end": e, "flag": f}
            for k, (s, e, f) in enumerate(spans)]


def load_ip(shot, dcs_org_dir):
    """Return ``(time, ip_ref, ip_act)`` (native A) for one shot, or None."""
    mat_file = pathlib.Path(dcs_org_dir) / f"DCS_archive_{shot}.mat"
    if not mat_file.exists():
        return None
    try:
        d = mat_io.read_mat_file(str(mat_file), ["Ip_scope_0", "Ip_scope_3"])
    except Exception:  # noqa: BLE001 - corrupt/unreadable .mat
        return None
    return (np.asarray(d["time"], float).reshape(-1),
            np.asarray(d["Ip_scope_0"], float).reshape(-1),
            np.asarray(d["Ip_scope_3"], float).reshape(-1))


def phase_stats(shot, dcs_org_dir):
    """Detect the phase split for one shot; return a result dict (see CSV_FIELDS).

    ``flag == "ok"`` rows additionally carry ``segments``: the
    :func:`phase_segments` list for ``flat_top_segments.csv``."""
    loaded = load_ip(shot, dcs_org_dir)
    if loaded is None:
        return {"shot": shot, "flag": "unreadable"}
    t, ref, act = loaded
    ip_peak = float(np.nanmax(np.abs(act))) / 1e3 if act.size else 0.0

    pw = plasma_window(ref)
    if pw is None:
        return {"shot": shot, "ip_peak_ka": ip_peak, "flag": "no_ref"}
    i0, i1 = pw
    segments = find_flat_segments(t, ref, i0, i1)
    base = {"shot": shot, "ref_start_index": i0, "ref_start_time": float(t[i0]),
            "ip_peak_ka": ip_peak}
    if not segments:
        return {**base, "n_segments": 0, "total_flat_top_s": 0.0, "flag": "no_flat"}
    tops = refine_flat_top(t, act, segments)
    if not tops:
        return {**base, "n_segments": 0, "total_flat_top_s": 0.0, "flag": "no_flat"}
    total = float(sum(e - s for s, e in tops))
    return {**base, "n_segments": len(tops),
            "flat_top_start": tops[0][0], "flat_top_end": tops[-1][1],
            "total_flat_top_s": total, "flag": "ok",
            "segments": phase_segments(t, ref, i0, i1, tops)}


def _stats_worker(args):
    shot, dcs_org_dir = args
    return phase_stats(shot, dcs_org_dir)


def _fmt(v, spec):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return format(v, spec)


def write_csv(rows, path, min_s):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["shot"]):
            total = r.get("total_flat_top_s")
            w.writerow({
                "shot": r["shot"],
                "ref_start_index": r.get("ref_start_index", ""),
                "ref_start_time": _fmt(r.get("ref_start_time"), ".6f"),
                "n_segments": r.get("n_segments", ""),
                "flat_top_start": _fmt(r.get("flat_top_start"), ".6f"),
                "flat_top_end": _fmt(r.get("flat_top_end"), ".6f"),
                "total_flat_top_s": _fmt(total, ".4f"),
                "pass_3s": "" if total is None else (total > min_s),
                "ip_peak_ka": _fmt(r.get("ip_peak_ka"), ".5g"),
                "flag": r.get("flag", ""),
            })
    print(f"  wrote {path}  ({len(rows)} shots)")


def write_segments_csv(rows, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEGMENT_CSV_FIELDS)
        w.writeheader()
        n = 0
        for r in sorted(rows, key=lambda r: r["shot"]):
            for seg in r.get("segments") or []:
                w.writerow({"shot": r["shot"], "i_segment": seg["i_segment"],
                            "start": _fmt(seg["start"], ".6f"),
                            "end": _fmt(seg["end"], ".6f"),
                            "flag": seg["flag"]})
                n += 1
    print(f"  wrote {path}  ({n} phase segments)")


def print_summary(rows, min_s):
    flags = collections.Counter(r.get("flag") for r in rows)
    print("\nflags: " + ", ".join(f"{k}={v}" for k, v in sorted(flags.items())))
    totals = [r["total_flat_top_s"] for r in rows if r.get("flag") == "ok"]
    n_pass = sum(1 for x in totals if x > min_s)
    if totals:
        arr = np.array(totals)
        print(f"flat-top (ok shots): n={len(totals)} "
              f"median={np.median(arr):.2f}s max={arr.max():.2f}s")
    print(f"pass (> {min_s} s): {n_pass}")


def run(dcs_org_dir=None, workers=None, csv_path=None, min_s=None):
    """Scan every DCS .mat for its phase split; write both CSVs.

    Programmatic entry point (no argparse). Returns the per-shot result dicts.
    """
    cfg = get_proj_config()
    dcs_org_dir = pathlib.Path(dcs_org_dir) if dcs_org_dir else cfg.dcs_org_dir
    workers = workers or min(16, os.cpu_count() or 1)
    min_s = cfg.flat_top_min_s if min_s is None else min_s

    shots = sorted(int(mat_io.parse_shot(p))
                   for p in dcs_org_dir.glob("DCS_archive_*.mat"))
    print(f"scanning {len(shots)} DCS .mat files for Ip flat-top ...")
    rows = pmap(_stats_worker, [(s, dcs_org_dir) for s in shots], workers, "flat_top")

    csv_path = csv_path or (cfg.stats_dir / "flat_top.csv")
    write_csv(rows, csv_path, min_s)
    write_segments_csv(rows, cfg.stats_dir / "flat_top_segments.csv")
    print_summary(rows, min_s)
    return rows
