# -*- coding: utf-8 -*-
"""Flat-top window from the IMAS equilibrium plasma current (replaces the DCS
Ip-scope detector in src/data/flat_top.py). Single signal: no ref/act split."""
import numpy as np

FLAT_THRESHOLD = 1e-3    # |normalized slope| below this is "flat"
MIN_SEGMENT_S = 0.5      # discard flat runs shorter than this
Q_PCT = 20               # percentile used to refine each segment's edges


def find_flat_segments(t, ip, flat_threshold=FLAT_THRESHOLD,
                       min_segment_s=MIN_SEGMENT_S):
    """Absolute-index (start, end) pairs of flat runs of ``ip`` (plateau-normalized
    slope under ``flat_threshold``)."""
    t = np.asarray(t, float)
    ip = np.asarray(ip, float)
    if t.size < 3:
        return []
    plateau = float(np.nanmedian(ip))
    if plateau == 0 or not np.isfinite(plateau):
        return []
    dt = np.diff(t)
    nd = np.diff(ip) / dt / plateau            # normalized slope, len = t.size-1
    idx = np.where(np.abs(nd) < flat_threshold)[0]
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) != 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    segs = []
    for s, e in zip(starts, ends):
        a, b = int(idx[s]), int(idx[e]) + 1
        if t[b] - t[a] > min_segment_s:
            segs.append((a, b))
    return segs


def flat_top_window(t, ip, min_total_s=3.0, q_pct=Q_PCT):
    """Inclusive ``(i0, i1)`` index span of the sustained flat-top, or ``None``.

    Flat segments refined by the q-percentile of ``ip`` within each; total refined
    duration must exceed ``min_total_s``."""
    t = np.asarray(t, float)
    ip = np.asarray(ip, float)
    segs = find_flat_segments(t, ip)
    if not segs:
        return None
    refined = []
    for a, b in segs:
        seg = ip[a:b + 1]
        if seg.size == 0:
            continue
        q = np.percentile(seg, q_pct)
        hit = np.where(seg >= q - 1e-7)[0]
        if hit.size:
            refined.append((a + int(hit[0]), a + int(hit[-1])))
    if not refined:
        return None
    total = float(sum(t[b] - t[a] for a, b in refined))
    if total < min_total_s:
        return None
    return int(refined[0][0]), int(refined[-1][1])