import argparse
from pathlib import Path

import numpy as np
import h5py
from scipy.io import loadmat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


REF_COL = 0
ACT_COL = 3

# --- chart chrome + palette (dataviz skill, light surface, CVD-safe) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SMAG_COLOR = "#e34948"   # GMAG SMAG_IP (distinct from DCS Ip black + power/BND blues)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#184f95", "#0d366b"]

# --- Ip flat-top detection (ported from src/data/flat_top.py, inlined to stay standalone) ---
REF_THRESH = 1.0       # Ip ref (A) must exceed this to count as "on"
FLAT_THRESHOLD = 1e-3  # |normalized ref slope| < this is "flat"
MIN_SEGMENT_S = 0.5    # minimum flat-segment duration to keep
Q_PCT = 20             # actual-Ip percentile used to refine flat-top edges


def get_scope(data, name):
    if name not in data:
        return None, None

    scope = data[name]

    try:
        t = np.asarray(scope.time, dtype=float).reshape(-1)
        y = np.asarray(scope.signals.values, dtype=float)
    except Exception:
        try:
            scope = scope[0, 0]
            t = np.asarray(scope["time"], dtype=float).reshape(-1)
            y = np.asarray(scope["signals"]["values"][0, 0], dtype=float)
        except Exception:
            return None, None

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return t, y


def sum_scope_column(data, names, column):
    time_base = None
    total = None
    used = []

    for name in names:
        t, y = get_scope(data, name)

        if t is None or y is None:
            continue

        if y.shape[1] <= column:
            continue

        values = np.abs(y[:, column])

        if time_base is None:
            time_base = t
            total = values.copy()
        else:
            total = total + np.interp(time_base, t, values, left=np.nan, right=np.nan)

        used.append(name)

    return time_base, total, used


def remove_tiny_values(values, threshold=1.0):
    if values is None:
        return None

    values = np.asarray(values, dtype=float)
    values[np.abs(values) < threshold] = 0.0
    return values


def peak_mw(values):
    if values is None or len(values) == 0:
        return 0.0

    if not np.isfinite(values).any():
        return 0.0

    return float(np.nanmax(np.abs(values)) / 1e6)


def detect_flat_top(t, ip_ref, ip_act):
    """Return ``(ref_start_time, flat_top_start, flat_top_end)`` or ``None``.

    Mirrors src/data/flat_top.py: ref-active window -> flat ref segments ->
    edges refined by the actual-Ip q-percentile.
    """
    t = np.asarray(t, float)
    ip_ref = np.asarray(ip_ref, float)
    ip_act = np.asarray(ip_act, float)

    ids = np.where(ip_ref > REF_THRESH + 1e-5)[0]
    if ids.size == 0:
        return None
    i0 = max(int(ids[0]) - 1, 0)
    i1 = min(int(ids[-1]) + 1, ip_ref.size - 1)

    tw = t[i0:i1 + 1]
    rw = ip_ref[i0:i1 + 1]
    if tw.size < 3:
        return None
    plateau = float(np.median(rw))
    if plateau == 0:
        return None
    nd = np.diff(rw) / np.diff(tw) / plateau
    idx = np.where(np.abs(nd) < FLAT_THRESHOLD)[0]
    if idx.size == 0:
        return None
    breaks = np.where(np.diff(idx) != 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    segments = []
    for s, e in zip(starts, ends):
        a = int(idx[s])
        b = int(idx[e]) + 1
        if tw[b] - tw[a] > MIN_SEGMENT_S:
            segments.append((i0 + a, i0 + b))
    if not segments:
        return None

    tops = []
    for a, b in segments:
        seg = ip_act[a:b + 1]
        segt = t[a:b + 1]
        if seg.size == 0:
            continue
        q = np.percentile(seg, Q_PCT)
        hit = np.where(seg >= q - 1e-7)[0]
        if hit.size:
            tops.append((float(segt[hit[0]]), float(segt[hit[-1]])))
    if not tops:
        return None

    return float(t[i0]), tops[0][0], tops[-1][1]


def default_instants(ref_start, ft0, ft1):
    """[(label, time_s), ...]: middle of ramp-up, then q30/q60/q90 of flat-top."""
    return [("ramp-mid", 0.5 * (ref_start + ft0)),
            ("FT q30", ft0 + 0.30 * (ft1 - ft0)),
            ("FT q60", ft0 + 0.60 * (ft1 - ft0)),
            ("FT q90", ft0 + 0.90 * (ft1 - ft0))]


def slice_colors(n):
    if n <= 1:
        return ["#256abf"]
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP, N=max(n, 2))
    return [cmap(i / (n - 1)) for i in range(n)]


def load_h5_signals(h5_path):
    """Load GMAG_BND and SMAG_IP from one boundary h5 (opened once).

    Returns ``(bnd, smag)``:
      ``bnd``  = ``(t, R, Z)`` shapes ``(N,), (32, N), (32, N)`` or ``None``.
      ``smag`` = ``(t, vals_kA)`` shapes ``(N,), (N,)`` or ``None``.

    GMAG_BND is stored (64, N) = 32 points x 2 coords (R, Z), interleaved
    channel-major -> reshape (32, 2, N): R = b[:, 0, :], Z = b[:, 1, :].
    SMAG_IP is scalar plasma current, already in kA; its time base matches
    GMAG_BND_time (same ignitron base as the DCS Ip_scope).
    """
    try:
        with h5py.File(h5_path, "r") as f:
            bnd = None
            if "targets/GMAG_BND" in f:
                v = np.asarray(f["targets/GMAG_BND"][:], float)
                t = np.asarray(f["targets/GMAG_BND_time"][:], float).reshape(-1)
                n = v.shape[-1]
                b = v.reshape(32, 2, n)
                bnd = (t, b[:, 0, :], b[:, 1, :])
            smag = None
            if "inputs/SMAG_IP" in f:
                sv = np.asarray(f["inputs/SMAG_IP"][:], float).reshape(-1)
                st = np.asarray(f["inputs/SMAG_IP_time"][:], float).reshape(-1)
                smag = (st, sv)
            return bnd, smag
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--dcs-dir", default="ProjDB/datasets/DCSHeating")
    parser.add_argument("--h5-dir", default="ProjDB/datasets/GMagH5")
    parser.add_argument("--out-dir", default="figs/dcs_mat")
    parser.add_argument("--bnd-instants", type=float, nargs="*", default=None,
                        help="explicit BND slice times [s]; default = ramp-mid + FT q30/q60/q90")
    parser.add_argument("--t-start", type=float, default=-1.0,
                        help="heating time-axis lower bound [s] (default -1)")
    parser.add_argument("--t-end", type=float, default=None,
                        help="heating time-axis upper bound [s] (default = end of Ip record)")
    args = parser.parse_args()

    dcs_dir = Path(args.dcs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_file = dcs_dir / f"DCS_archive_{args.shot}.mat"
    if not mat_file.is_file():
        raise FileNotFoundError(f"File not found: {mat_file}")

    data = loadmat(mat_file, squeeze_me=True, struct_as_record=False)

    t_ip, ip_values = get_scope(data, "Ip_scope")
    if t_ip is None or ip_values is None:
        raise RuntimeError("Ip_scope is missing or unreadable.")

    if ip_values.shape[1] > ACT_COL:
        ip_ref = ip_values[:, REF_COL]
        ip_act = ip_values[:, ACT_COL]
    else:
        ip_ref = ip_values[:, 0]
        ip_act = ip_values[:, 0]
    ip_ka = ip_act / 1e3

    t_lh_ref, lh_ref, used_lh_ref = sum_scope_column(data, ["PowLH1_scope", "PowLH2_scope"], REF_COL)
    t_lh_act, lh_act, used_lh_act = sum_scope_column(data, ["PowLH1_scope", "PowLH2_scope"], ACT_COL)
    t_ic_ref, ic_ref, used_ic_ref = sum_scope_column(data, ["PowIC1_scope", "PowIC2_scope", "PowIC3_scope"], REF_COL)
    t_ic_act, ic_act, used_ic_act = sum_scope_column(data, ["PowIC1_scope", "PowIC2_scope", "PowIC3_scope"], ACT_COL)

    lh_ref = remove_tiny_values(lh_ref)
    lh_act = remove_tiny_values(lh_act)
    ic_ref = remove_tiny_values(ic_ref)
    ic_act = remove_tiny_values(ic_act)

    # --- choose BND slice instants ---
    bnd, smag = load_h5_signals(Path(args.h5_dir) / f"{args.shot}.h5")
    instants = []
    if args.bnd_instants is not None:
        instants = [(f"slice {i + 1}", float(x)) for i, x in enumerate(args.bnd_instants)]
    else:
        ft = detect_flat_top(t_ip, ip_ref, ip_act)
        if ft is not None:
            instants = default_instants(*ft)
        elif bnd is not None:
            # fallback: evenly spaced over the valid (finite, in-vessel) window
            t_b, R, Z = bnd
            m = (np.isfinite(R).all(axis=0) & (R.min(axis=0) > 1.8)
                 & (R.max(axis=0) < 3.2) & (np.abs(Z).max(axis=0) < 1.0))
            idx = np.where(m)[0]
            if idx.size >= 4:
                pick = np.linspace(0, idx.size - 1, 4).astype(int)
                instants = [(f"q{int(round(p / (idx.size - 1) * 100))}",
                             float(t_b[idx[p]])) for p in pick]

    print(f"Shot: {args.shot}")
    print(f"DCS .mat: {mat_file}")
    print(f"LH ref scopes: {used_lh_ref}   actual scopes: {used_lh_act}")
    print(f"IC ref scopes: {used_ic_ref}   actual scopes: {used_ic_act}")
    print(f"LH actual peak: {peak_mw(lh_act):.3f} MW   IC actual peak: {peak_mw(ic_act):.3f} MW")
    if smag is not None:
        print(f"SMAG_IP peak: {float(np.nanmax(np.abs(smag[1]))):.1f} kA "
              f"(GMAG; DCS Ip peak = {float(np.nanmax(np.abs(ip_ka))):.1f} kA)")
    if instants:
        print(f"BND slices ({len(instants)}): " + ", ".join(f"{lab}={t:.2f}s" for lab, t in instants))
    else:
        print("BND slices: none (no flat-top detected and no H5Target boundary available)")

    # --- figure: top row = LH / IC heating, bottom row = BND overlay (equal aspect) ---
    has_bnd = bnd is not None and len(instants) > 0
    fig = plt.figure(figsize=(13, 9 if has_bnd else 7), constrained_layout=True)

    if has_bnd:
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
        ax_lh = fig.add_subplot(gs[0, 0])
        ax_ic = fig.add_subplot(gs[0, 1], sharex=ax_lh)
        ax_bnd = fig.add_subplot(gs[1, :])
    else:
        ax_lh = fig.add_subplot(2, 1, 1)
        ax_ic = fig.add_subplot(2, 1, 2, sharex=ax_lh)
        ax_bnd = None

    colors = slice_colors(len(instants)) if has_bnd else []

    def plot_heating(ax, t_p, p_ref, p_act, ref_label, act_label, ylabel, title):
        ax.plot(t_ip, ip_ka, color="#0b0b0b", label="Ip actual [kA]", linewidth=1.2)
        if smag is not None:
            ax.plot(smag[0], smag[1], "--", color=SMAG_COLOR,
                    label="SMAG_IP [kA]", linewidth=1.1)
        ax.set_ylabel("Ip [kA]", color=MUTED)
        ax.grid(True, color=GRID, lw=0.8, alpha=0.7)
        for s in ax.spines.values():
            s.set_color(AXIS)
        ax.tick_params(colors=MUTED)

        ax_p = ax.twinx()
        if p_ref is not None:
            ax_p.plot(t_p, p_ref / 1e6, "--", color=MUTED, label=ref_label, linewidth=1.1)
        if p_act is not None:
            ax_p.plot(t_p, p_act / 1e6, "-", color="#2a78d6", label=act_label, linewidth=1.3)
        ax_p.set_ylabel(ylabel, color=MUTED)
        for s in ax_p.spines.values():
            s.set_color(AXIS)
        ax_p.tick_params(colors=MUTED)

        # mark the BND slice instants (same colors as the boundary curves below)
        for c, (_, tt) in zip(colors, instants):
            ax.axvline(tt, color=c, lw=1.0, alpha=0.8, zorder=0)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_p.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=False)
        ax.set_title(title, color=INK, fontsize=10)

    plot_heating(ax_lh, t_lh_act, lh_ref, lh_act,
                 "LH ref [MW]", "LH actual [MW]", "LH power [MW]", f"Shot {args.shot} - LH heating")
    plot_heating(ax_ic, t_ic_act, ic_ref, ic_act,
                 "IC ref [MW]", "IC actual [MW]", "IC power [MW]", f"Shot {args.shot} - IC heating")
    ax_ic.set_xlabel("Time after ignitron [s]", color=MUTED)

    # time window: from --t-start (default -1 s) to the end of the Ip record
    t_hi = float(args.t_end) if args.t_end is not None else float(np.nanmax(t_ip))
    ax_lh.set_xlim(args.t_start, t_hi)   # ax_ic shares this x-axis

    if has_bnd:
        t_b, R, Z = bnd
        # collect the plotted (closed) curves first, skipping non-finite slices
        curves = []
        for lab, tt in instants:
            ti = int(np.argmin(np.abs(t_b - tt)))
            rr = np.append(R[:, ti], R[0, ti])
            zz = np.append(Z[:, ti], Z[0, ti])
            if np.isfinite(rr).all():
                curves.append((lab, float(t_b[ti]), rr, zz))
        # equal scale AND equal range: both axes span the same length, centered on data
        if curves:
            Ra = np.concatenate([c[2] for c in curves])
            Za = np.concatenate([c[3] for c in curves])
            span = max(Ra.max() - Ra.min(), Za.max() - Za.min())
            span = span if span > 0 else 1.0   # guard degenerate (all-zero) frame
            half = span / 2 * 1.1              # 10% margin
            rc = (Ra.max() + Ra.min()) / 2
            zc = (Za.max() + Za.min()) / 2
            ax_bnd.set_xlim(rc - half, rc + half)
            ax_bnd.set_ylim(zc - half, zc + half)
        for c_col, (lab, tb_ti, rr, zz) in zip(colors, curves):
            ax_bnd.plot(rr, zz, "-o", color=c_col, ms=4, lw=1.5,
                        label=f"{lab}  (t={tb_ti:.2f} s)")
        ax_bnd.set_aspect("equal", adjustable="box")
        ax_bnd.set_xlabel("R [m]", color=MUTED)
        ax_bnd.set_ylabel("Z [m]", color=MUTED)
        ax_bnd.grid(True, ls=":", color=GRID, alpha=0.5)
        for s in ax_bnd.spines.values():
            s.set_color(AXIS)
        ax_bnd.tick_params(colors=MUTED)
        ax_bnd.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=MUTED)
        ax_bnd.set_title("LCFS (GMAG_BND) at selected instants", color=INK, fontsize=10)

    fig.suptitle(
        f"Shot {args.shot} | LH actual peak = {peak_mw(lh_act):.3f} MW | "
        f"IC actual peak = {peak_mw(ic_act):.3f} MW",
        fontsize=11, color=INK,
    )
    fig.patch.set_facecolor(SURFACE)

    output_file = out_dir / f"DCS_heating_bnd_{args.shot}.png"
    fig.savefig(output_file, dpi=150, facecolor=SURFACE)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
