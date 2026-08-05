# -*- coding: utf-8 -*-
"""Plot an LCFS slice directly from GMagH5/<shot>.h5.

Reads, all from the native GMag H5 on its shared ignitron ``*_time`` grid at the
sample nearest the requested time:
  * the boundary  -> targets/GMAG_BND  (32 ordered (R,Z) vertices)
  * plasma-current gravity center -> inputs/GMAG_BARY   (R, Z) [m]
  * geometric center -> inputs/GMAG_GEOM[0:2] = (Rgeom, Zgeom) [mm -> /1000 -> m]

Left panel marks: reconstruction origin (2.5, 0), GMAG_BARY, (Rgeom, Zgeom).
Right panel: r(theta) about the fixed origin (the polar target), near-flat edges
in red (折线 / flat-shelf) and the r_min vertex ringed (elongation). Sentinel
centers (no equilibrium, e.g. breakdown) are detected and omitted with a note.

Usage:
  python scripts/plot_lcfs_slice.py 58303 67.443
  python scripts/plot_lcfs_slice.py 57620 39.1 --tol 0.005 --rmin-floor 0.15
  python scripts/plot_lcfs_slice.py 58035 0.050 --out /tmp/check.png
"""
import argparse
import json
import pathlib

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.proj_config import get_proj_config  # noqa: E402
from src.data.build_npz import radii_on_grid  # noqa: E402  (same r(theta) as the model target)

CFG = get_proj_config()
INK, BLUE, RED, GREEN, PURPLE = "#0b0b0b", "#256abf", "#c0392b", "#27ae60", "#7d3c98"
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


def _valid_center(R, Z):
    """A real WEST equilibrium center sits near (2.5, 0); reject sentinels."""
    return np.isfinite(R) and np.isfinite(Z) and (2.0 < R < 3.0) and (abs(Z) < 0.6)


def plot_lcfs_slice(shot, t, tol=0.005, rmin_floor=0.15, out=None):
    p = CFG.gmagh5_dir / f"{shot}.h5"
    if not p.exists():
        raise SystemExit(f"no GMagH5 for shot {shot}: {p}")
    meta = json.loads((CFG.mergednpz_dir / "meta.json").read_text())
    origin = tuple(float(x) for x in meta["origin"])
    theta = np.deg2rad(np.asarray(meta["theta_deg"], float))

    with h5py.File(p, "r") as hf:
        bnd_t = np.asarray(hf["targets/GMAG_BND_time"], float).reshape(-1)
        j = int(np.argmin(np.abs(bnd_t - t)))
        bnd = np.asarray(hf["targets/GMAG_BND"], float)[:, j].reshape(32, 2)
        R, Z = bnd[:, 0], bnd[:, 1]

        def center(node, tkey, scale=1.0):
            """(R, Z) of a GMAG center signal at the nearest sample, or None."""
            if f"inputs/{node}" not in hf:
                return None
            vals = np.asarray(hf[f"inputs/{node}"], float)
            if vals.ndim != 2 or vals.shape[0] < 2:
                return None
            if f"inputs/{tkey}" in hf:
                tt = np.asarray(hf[f"inputs/{tkey}"], float).reshape(-1)
                jj = int(np.argmin(np.abs(tt - t)))
            else:                       # shared grid fallback
                jj = j
            return float(vals[0, jj]) * scale, float(vals[1, jj]) * scale

        bary_RZ = center("GMAG_BARY", "GMAG_BARY_time", scale=1.0)      # already [m]
        geom_RZ = center("GMAG_GEOM", "GMAG_GEOM_time", scale=1e-3)     # [mm] -> [m]

    r = radii_on_grid(R, Z, origin, theta)
    if not np.isfinite(r).all():
        print(f"WARNING: shot {shot} t={bnd_t[j]:.4f}s has non-finite r(theta)")
    flat = np.abs(np.diff(np.r_[r, r[0]])) < tol
    lr = longest_cyclic_run(flat)
    rmin, rmax = float(np.nanmin(r)), float(np.nanmax(r))
    jmin = int(np.nanargmin(r))

    bary_ok = bary_RZ is not None and _valid_center(*bary_RZ)
    geom_ok = geom_RZ is not None and _valid_center(*geom_RZ)

    print(f"shot {shot}  GMagH5 idx={j}  t={bnd_t[j]:.4f}s  (requested {t})")
    print(f"  flat_run={lr}/32   r_min={rmin:.4f}  r_max={rmax:.4f}  "
          f"r_max/r_min={rmax / max(rmin, 1e-9):.3f}")
    print(f"  origin            = {tuple(round(v, 3) for v in origin)}")
    brys = "(sentinel / no equilibrium)" if bary_RZ and not bary_ok else ""
    gmys = "(sentinel / no equilibrium)" if geom_RZ and not geom_ok else ""
    print(f"  GMAG_BARY (R,Z)   = {None if not bary_RZ else tuple(round(v, 3) for v in bary_RZ)}  {brys}")
    print(f"  Rgeom, Zgeom      = {None if not geom_RZ else tuple(round(v, 3) for v in geom_RZ)}  {gmys}")
    print(f"  flag?  flat_run>={FLAT_RUN_THRESH}: {'YES' if lr >= FLAT_RUN_THRESH else 'no'}   "
          f"r_min<{rmin_floor}: {'YES' if rmin < rmin_floor else 'no'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5))
    # left: (R,Z) boundary + origin + centers
    ax1.plot(np.r_[R, R[0]], np.r_[Z, Z[0]], "-", color=INK, lw=2, label="LCFS (GMAG_BND)")
    ax1.scatter(R, Z, s=14, color=BLUE, zorder=3)
    ax1.scatter(*origin, marker="*", s=260, color=RED, zorder=5,
                edgecolors=INK, linewidths=.4,
                label=f"origin {tuple(round(v, 2) for v in origin)}")
    if bary_ok:
        ax1.scatter(*bary_RZ, marker="s", s=95, color=GREEN, zorder=5,
                    edgecolors=INK, linewidths=.4,
                    label=f"BARY grav. {tuple(round(v, 2) for v in bary_RZ)}")
    if geom_ok:
        ax1.scatter(*geom_RZ, marker="^", s=115, color=PURPLE, zorder=5,
                    edgecolors=INK, linewidths=.4,
                    label=f"plasma geom center (Rgeom,Zgeom) "
                         f"{tuple(round(v, 2) for v in geom_RZ)}")
    elif geom_RZ is not None:  # signal present but sentinel (no equilibrium)
        ax1.text(0.02, 0.02, "Rgeom,Zgeom: no equilibrium (sentinel)",
                 transform=ax1.transAxes, ha="left", va="bottom",
                 fontsize=7.5, color=PURPLE)
    ax1.set_aspect("equal", "box")
    ax1.grid(True, ls=":", alpha=.5)
    ax1.set_xlabel("R (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_title(f"{shot}  t={bnd_t[j]:.3f}s  (R,Z)", fontsize=11)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper left")

    # right: r(theta), flat edges red, r_min vertex ringed
    yy = np.r_[r, r[0]]
    for k in range(32):
        ax2.plot([k, k + 1], [yy[k], yy[k + 1]],
                 color=(RED if flat[k] else BLUE), lw=2.2)
    ax2.scatter(np.arange(32), r, s=14, color=INK, zorder=3)
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
    ax2.legend(fontsize=7.5, frameon=False)

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
    ap.add_argument("time", type=float, help="requested time in seconds (ignitron)")
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
