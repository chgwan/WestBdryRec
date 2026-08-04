# -*- coding: utf-8 -*-
"""Plot the LCFS target for a single (shot, time) slice — DCS MergedNpz.

Two panels:
  left  : (R,Z) boundary from bnd_RZ, with the reconstruction origin marked.
  right : r(theta) = Y (the model target), with near-flat edges highlighted in
          red (the 折线 / flat-shelf symptom) and the minimum-radius vertex
          ringed in green (the elongation / origin-near-edge symptom).

Prints the diagnostics used by the LCFS-quality filter for that slice:
  longest flat run /32, r_min, r_max, r_max/r_min, origin-inside?

Usage:
  python exploration/plot_lcfs_slice.py 58303 67.443
  python exploration/plot_lcfs_slice.py 58035 0.050 --tol 0.005 --rmin-floor 0.15
  python exploration/plot_lcfs_slice.py 57620 39.1 --out /tmp/check.png
"""
import argparse
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path  # noqa: E402

from src.proj_config import get_proj_config  # noqa: E402

CFG = get_proj_config()
INK, BLUE, RED, GREEN = "#0b0b0b", "#256abf", "#c0392b", "#27ae60"
FLAT_RUN_THRESH = 8  # flat-run reject threshold shown in the printed flag line


def longest_cyclic_run(flat):
    """Longest cyclic run of True in a 1-D bool array, capped at its length."""
    f2 = np.r_[flat, flat]
    cnt = 0
    best = 0
    for v in f2:
        cnt = cnt + 1 if v else 0
        best = max(best, cnt)
    return min(best, flat.size)


def plot_lcfs_slice(shot, t, tol=0.005, rmin_floor=0.15, out=None):
    p = CFG.mergednpz_dir / f"{shot}.npz"
    if not p.exists():
        raise SystemExit(f"no NPZ for shot {shot}: {p}")
    d = np.load(p)
    time = d["time"].astype(float)
    bnd = d["bnd_RZ"].astype(float)
    Y = d["Y"].astype(float)
    meta = json.loads((CFG.mergednpz_dir / "meta.json").read_text())
    origin = tuple(float(x) for x in meta["origin"])

    i = int(np.argmin(np.abs(time - t)))
    R = bnd[i, :, 0]
    Z = bnd[i, :, 1]
    r = Y[i].astype(float)

    if not np.isfinite(r).all():
        print(f"WARNING: shot {shot} idx={i} t={time[i]:.4f}s has non-finite r(theta)")

    flat = np.abs(np.diff(np.r_[r, r[0]])) < tol
    lr = longest_cyclic_run(flat)
    rmin, rmax = float(np.nanmin(r)), float(np.nanmax(r))
    inside = bool(Path(np.c_[R, Z]).contains_point(origin)) if np.isfinite(R).all() else False
    jmin = int(np.nanargmin(r))

    print(f"shot {shot}  idx={i}  t={time[i]:.4f}s  (requested {t})")
    print(f"  flat_run={lr}/32   r_min={rmin:.4f}  r_max={rmax:.4f}  "
          f"r_max/r_min={rmax / max(rmin, 1e-9):.3f}")
    print(f"  origin {origin} inside? {inside}")
    print(f"  flag?  flat_run>={FLAT_RUN_THRESH}: {'YES' if lr >= FLAT_RUN_THRESH else 'no'}   "
          f"r_min<{rmin_floor}: {'YES' if rmin < rmin_floor else 'no'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    # left: (R,Z) boundary + origin
    ax1.plot(np.r_[R, R[0]], np.r_[Z, Z[0]], "-", color=INK, lw=2)
    ax1.scatter(R, Z, s=16, color=BLUE, zorder=3)
    ax1.scatter(*origin, marker="*", s=240, color=RED, zorder=4, label=f"origin {origin}")
    ax1.set_aspect("equal", "box")
    ax1.grid(True, ls=":", alpha=.5)
    ax1.set_xlabel("R (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_title(f"{shot}  t={time[i]:.3f}s  (R,Z)", fontsize=11)
    ax1.legend(fontsize=8, frameon=False)

    # right: r(theta), flat edges red, r_min vertex ringed
    yy = np.r_[r, r[0]]
    for j in range(32):
        ax2.plot([j, j + 1], [yy[j], yy[j + 1]],
                 color=(RED if flat[j] else BLUE), lw=2.2)
    ax2.scatter(np.arange(32), r, s=16, color=INK, zorder=3)
    ax2.scatter([jmin], [rmin], s=110, facecolors="none", edgecolors=GREEN,
                linewidths=2, zorder=4, label=f"r_min={rmin:.3f}")
    if rmin_floor is not None:
        ax2.axhline(rmin_floor, ls="--", lw=1, color=GREEN, alpha=.6,
                    label=f"floor {rmin_floor}")
    ax2.set_xticks(np.arange(0, 33, 4))
    ax2.grid(True, ls=":", alpha=.5)
    ax2.set_xlabel(r"$\theta$ sample index (0..31, cyclic)")
    ax2.set_ylabel(r"$r(\theta)$ [m]")
    ax2.set_title(f"r(theta)  flat run={lr}/32", fontsize=11)
    ax2.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    out = pathlib.Path(out) if out else (CFG.lcfs_fig_dir / f"{shot}_t{t}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  saved {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("shot", type=int, help="shot number (e.g. 58303)")
    ap.add_argument("time", type=float, help="requested time in seconds")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="flat-edge |dr| tolerance [m] (default 0.005)")
    ap.add_argument("--rmin-floor", type=float, default=0.15,
                    help="r_min elongation floor [m] shown + used in flag line (default 0.15)")
    ap.add_argument("--out", type=str, default=None,
                    help="output png (default figs/LCFS/<shot>_t<time>.png)")
    args = ap.parse_args()
    plot_lcfs_slice(args.shot, args.time, tol=args.tol,
                    rmin_floor=args.rmin_floor, out=args.out)


if __name__ == "__main__":
    main()
