"""Plot the WEST tokamak in-vessel configuration.

Geometry provenance
-------------------
Polygons come from ToFu (https://github.com/ToFuProject/tofu), the CEA/IRFM
open-source tomography library, file set ``tofu/geom/inputs/TFG_*_ExpWEST_*.txt``.
ToFu derives them from ``WEST_Geometry_Rev_Nov2016_V2.xlsx`` (WEST CAD revision
Nov 2016). Config composition (which components make up WEST-V1..V4) is taken
from ``tofu/geom/_def_config.py``.

File format (tofu ``Struct.from_txt``):
    row 0        : (npts, noccur)   noccur = 0 -> axisymmetric
    rows 1..npts : poloidal polygon, columns (R, Z) in metres
    rows npts+1: : (pos, extent) per toroidal occurrence, radians

Encoding rule
-------------
A poloidal cross-section cannot show toroidal coverage, so a filled polygon
would imply a component wraps the whole torus. Coverage is therefore computed
from the (pos, extent) records and drives the draw style:
    coverage >= COVERAGE_FILL  -> filled  (quasi-continuous toroidal ring)
    coverage <  COVERAGE_FILL  -> outline (toroidally localised)

Outputs (../figs/west_invessel/, PNG at 200 dpi + vector SVG):
    west_invessel_v4.*         poloidal cross-section + divertor zoom + top view
    west_config_evolution.*    WEST-V1 -> V4 in-vessel configurations
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# glyphs as paths: SVG stays label-exact on machines without the same fonts
plt.rcParams["svg.fonttype"] = "path"
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

HERE = Path(__file__).resolve().parent
GEOM = HERE / "west_geom"                       # downloaded ToFu inputs
OUT = HERE.parent / "figs" / "west_invessel"    # rendered figures

COVERAGE_FILL = 0.60          # toroidal coverage above which a shape is filled

# ── palette (validated categorical slots, light surface) ───────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948",
)
SHIELD = "#a9a79e"      # recessive structure — thermal shields
CASING = "#c9c7be"      # recessive structure — divertor casing / supports


# ── loading ────────────────────────────────────────────────────────────────
def load(cls, name):
    """Return (poly (n,2) in (R, Z) metres, occurrences (m,2) rad or None)."""
    arr = np.loadtxt(GEOM / f"TFG_{cls}_ExpWEST_{name}.txt", comments="#")
    npts, noccur = int(arr[0, 0]), int(arr[0, 1])
    poly = arr[1:npts + 1, :]
    occ = arr[npts + 1:npts + 1 + noccur, :] if noccur > 0 else None
    return poly, occ


def close(poly):
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def coverage(occ):
    """Fraction of the torus covered. None (axisymmetric) -> 1.0."""
    if occ is None:
        return 1.0
    return float(np.sum(occ[:, 1]) / (2.0 * np.pi))


# ── component table: (group, colour, cls, name) ────────────────────────────
COMPONENTS_V4 = [
    ("Lower divertor — ITER-like W monoblocks", BLUE, "PFC", "DivLowITERV3"),
    ("Upper divertor — W-coated graphite", VIOLET, "PFC", "DivUpV3"),
    ("Baffle", ORANGE, "PFC", "BaffleV2"),
    ("Inner / outer bumpers", AQUA, "PFC", "BumperInnerV3"),
    ("Inner / outer bumpers", AQUA, "PFC", "BumperOuterV3"),
    ("ICRH antennas (×3)", MAGENTA, "PFC", "IC1V1"),
    ("ICRH antennas (×3)", MAGENTA, "PFC", "IC2V1"),
    ("ICRH antennas (×3)", MAGENTA, "PFC", "IC3V1"),
    ("LHCD launchers (×2)", YELLOW, "PFC", "LH1V1"),
    ("LHCD launchers (×2)", YELLOW, "PFC", "LH2V1"),
    ("Ripple / VDE protections", RED, "PFC", "RippleV1"),
    ("Ripple / VDE protections", RED, "PFC", "VDEV0"),
    ("Thermal shields", SHIELD, "PFC", "ThermalShieldHFSV0"),
    ("Thermal shields", SHIELD, "PFC", "ThermalShieldLFSSlimV0"),
    ("Thermal shields", SHIELD, "PFC", "ThermalShieldLFSWideV0"),
    ("Thermal shields", SHIELD, "PFC", "ThermalShieldLFSLowV0"),
    ("Thermal shields", SHIELD, "PFC", "ThermalShieldLFSUpV0"),
    ("Divertor casing / supports", CASING, "PFC", "CasingLDivV0"),
    ("Divertor casing / supports", CASING, "PFC", "CasingUDivV0"),
    ("Divertor casing / supports", CASING, "PFC", "CasingCoverLDivV0"),
    ("Divertor casing / supports", CASING, "PFC", "CasingCoverUDivV0"),
    ("Divertor casing / supports", CASING, "PFC", "CasingPFUPlateLDivV0"),
    ("Divertor casing / supports", CASING, "PFC", "CasingPFUPlateUDivV0"),
    ("Divertor casing / supports", CASING, "PFC", "CasingPJLDivV0"),
    ("Divertor casing / supports", CASING, "PFC", "CasingPJUDivV0"),
]

DIV_COILS = ["DivLow1V0", "DivLow2V0", "DivUp1V0", "DivUp2V0"]


def draw_vessel(ax, lw=1.2):
    """Vacuum vessel wall: the shell between the outer and inner contours."""
    outer, _ = load("Ves", "OuterV0")
    inner, _ = load("Ves", "InnerV0")
    ax.add_patch(MplPolygon(outer, closed=True, facecolor="#dedcd4",
                            edgecolor="none", zorder=1))
    ax.add_patch(MplPolygon(inner, closed=True, facecolor=SURFACE,
                            edgecolor="none", zorder=2))
    for p in (outer, inner):
        p = close(p)
        ax.plot(p[:, 0], p[:, 1], color=INK, lw=lw, zorder=3,
                solid_joinstyle="round")


def draw_components(ax, components, lw=1.0, zorder=5, ring=True):
    """Draw each component, style chosen by its toroidal coverage."""
    for _, color, cls, name in components:
        poly, occ = load(cls, name)
        if coverage(occ) >= COVERAGE_FILL:
            ax.add_patch(MplPolygon(poly, closed=True, facecolor=color,
                                    alpha=0.8, edgecolor=color, lw=lw,
                                    zorder=zorder))
            if ring:   # surface ring keeps adjacent fills separable
                p = close(poly)
                ax.plot(p[:, 0], p[:, 1], color=SURFACE, lw=lw + 1.4,
                        zorder=zorder - 0.5, solid_joinstyle="round")
        else:
            p = close(poly)
            ax.plot(p[:, 0], p[:, 1], color=color, lw=lw + 0.2, zorder=zorder,
                    solid_joinstyle="round")


def style_axes(ax, xlabel="R  [m]", ylabel="Z  [m]"):
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color=GRID, lw=0.5, zorder=0)


# ══════════════════════════════════════════════════════════════════════════
# Figure 1
# ══════════════════════════════════════════════════════════════════════════
# rows of the toroidal strip panel: (label, colour, [components])
TOROIDAL_ROWS = [
    ("Lower divertor PFUs", BLUE, [("PFC", "DivLowITERV3")]),
    ("Upper divertor PFUs", VIOLET, [("PFC", "DivUpV3")]),
    ("Baffle sectors", ORANGE, [("PFC", "BaffleV2")]),
    ("Divertor casing", CASING, [("PFC", "CasingLDivV0")]),
    ("Divertor casing (PJ)", CASING, [("PFC", "CasingPJLDivV0")]),
    ("LFS shield — wide / slim", SHIELD,
     [("PFC", "ThermalShieldLFSWideV0"), ("PFC", "ThermalShieldLFSSlimV0")]),
    ("LFS shield — upper / lower", SHIELD,
     [("PFC", "ThermalShieldLFSUpV0"), ("PFC", "ThermalShieldLFSLowV0")]),
    ("HFS shield (axisymmetric)", SHIELD, [("PFC", "ThermalShieldHFSV0")]),
    ("Inner bumper", AQUA, [("PFC", "BumperInnerV3")]),
    ("Outer bumper", AQUA, [("PFC", "BumperOuterV3")]),
    ("VDE protections", RED, [("PFC", "VDEV0")]),
    ("Ripple protections", RED, [("PFC", "RippleV1")]),
    ("ICRH antennas", MAGENTA,
     [("PFC", "IC1V1"), ("PFC", "IC2V1"), ("PFC", "IC3V1")]),
    ("LHCD launchers", YELLOW, [("PFC", "LH1V1"), ("PFC", "LH2V1")]),
]


def _bars(ax, y, occ, color, h=0.34):
    """Draw one row of toroidal occurrences, wrapping across +/-180 deg."""
    for pos, ext in np.degrees(occ):
        a, b = pos - ext / 2.0, pos + ext / 2.0
        for lo, hi in (((a, b),) if (a >= -180 and b <= 180)
                       else ((max(a, -180), min(b, 180)),
                             (a + 360, 180) if a < -180 else (-180, b - 360))):
            if hi > lo:
                ax.add_patch(MplPolygon(
                    [[lo, y - h], [hi, y - h], [hi, y + h], [lo, y + h]],
                    closed=True, facecolor=color, edgecolor="none", zorder=4))


def figure_invessel():
    fig = plt.figure(figsize=(14.6, 9.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.06, 1.0],
                          height_ratios=[1.0, 1.02],
                          left=0.055, right=0.985, top=0.875, bottom=0.135,
                          wspace=0.20, hspace=0.34)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    # ---- (a) poloidal cross-section -------------------------------------
    draw_vessel(axA)
    draw_components(axA, COMPONENTS_V4)

    for cn in DIV_COILS:                    # inside the divertor casing
        poly, _ = load("CoilPF", cn)
        axA.add_patch(MplPolygon(poly, closed=True, facecolor="#f4f3f0",
                                 edgecolor=INK2, lw=0.7, zorder=6))

    sep = close(load("PlasmaDomain", "Sep")[0])
    axA.plot(sep[:, 0], sep[:, 1], color=INK, lw=1.4, ls=(0, (5, 3)), zorder=8)
    axA.plot([2.5], [0.0], marker="+", ms=9, mew=1.4, color=INK2, zorder=8)

    style_axes(axA)
    axA.set_xlim(1.30, 4.16)
    axA.set_ylim(-1.12, 1.12)
    axA.set_xticks([1.5, 2.0, 2.5, 3.0, 3.5])
    axA.set_title("(a)  Poloidal cross-section", loc="left", color=INK,
                  fontsize=11, fontweight="bold", pad=8)

    ann = dict(fontsize=8.4, color=INK, zorder=10,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7,
                               shrinkA=1, shrinkB=2))
    axA.annotate("vacuum vessel\n(two shells)", xy=(3.12, 0.745),
                 xytext=(3.06, 1.075), ha="left", va="top", **ann)
    axA.annotate("inner bumper", xy=(1.84, 0.34), xytext=(1.31, 0.62),
                 ha="left", **ann)
    axA.annotate("HFS thermal\nshield", xy=(1.80, -0.24), xytext=(1.31, -0.50),
                 ha="left", va="top", **ann)
    axA.annotate("upper divertor", xy=(2.20, 0.72), xytext=(2.15, 0.99),
                 ha="left", **ann)
    axA.annotate("ripple / VDE", xy=(2.66, 0.84), xytext=(2.52, 1.075),
                 ha="left", va="top", **ann)
    axA.annotate("lower divertor\n(ITER-like)", xy=(2.14, -0.68),
                 xytext=(1.60, -0.86), ha="left", va="top", **ann)
    axA.annotate("baffle", xy=(2.74, -0.760), xytext=(2.98, -0.95),
                 ha="left", **ann)
    axA.annotate("divertor casing\n& PF coils", xy=(2.06, -0.86),
                 xytext=(1.62, -1.02), ha="left", va="top", **ann)
    # low-field side is crowded — park these labels in the right margin
    axA.annotate("ICRH antenna", xy=(3.38, 0.42), xytext=(3.80, 0.66),
                 ha="left", **ann)
    axA.annotate("LHCD launcher", xy=(3.24, 0.26), xytext=(3.80, 0.44),
                 ha="left", **ann)
    axA.annotate("outer bumper", xy=(3.50, 0.04), xytext=(3.80, 0.22),
                 ha="left", **ann)
    axA.annotate("LFS thermal\nshield", xy=(3.30, -0.42), xytext=(3.80, -0.16),
                 ha="left", va="top", **ann)
    axA.text(2.55, -0.02, "magnetic axis", fontsize=7.8, color=INK2,
             ha="left", va="top", zorder=10)
    axA.text(2.30, 0.42, "separatrix\n(example\nequilibrium)", fontsize=7.8,
             color=INK2, ha="center", va="center", zorder=10)

    # zoom marker for panel (b)
    zx0, zx1, zy0, zy1 = 1.83, 2.86, -1.02, -0.50
    axA.add_patch(MplPolygon([[zx0, zy0], [zx1, zy0], [zx1, zy1], [zx0, zy1]],
                             closed=True, facecolor="none", edgecolor=INK2,
                             lw=0.8, ls=(0, (3, 2)), zorder=9))
    axA.text(zx1 + 0.03, zy0 - 0.01, "(b)", fontsize=8.4, color=INK2,
             ha="left", va="top", zorder=10, fontweight="bold")

    # ---- (b) lower divertor zoom ---------------------------------------
    draw_vessel(axB, lw=1.0)
    draw_components(axB, COMPONENTS_V4, lw=1.1)
    for cn, lbl in (("DivLow1V0", "PF1"), ("DivLow2V0", "PF2")):
        poly, _ = load("CoilPF", cn)
        axB.add_patch(MplPolygon(poly, closed=True, facecolor="#f4f3f0",
                                 edgecolor=INK2, lw=0.8, zorder=6))
        axB.text(poly[:, 0].mean(), poly[:, 1].mean(), lbl, fontsize=7.2,
                 color=INK2, ha="center", va="center", zorder=7)

    style_axes(axB)
    axB.set_xlim(zx0, zx1)
    axB.set_ylim(zy0, zy1)
    axB.set_title("(b)  Lower divertor — 456 actively-cooled PFUs", loc="left",
                  color=INK, fontsize=11, fontweight="bold", pad=8)
    axB.annotate("W monoblock\ntarget", xy=(2.12, -0.660),
                 xytext=(2.20, -0.560), ha="left", va="top", **ann)
    axB.annotate("baffle", xy=(2.56, -0.730), xytext=(2.62, -0.620),
                 ha="left", **ann)
    axB.annotate("casing / supports", xy=(2.02, -0.900),
                 xytext=(2.12, -0.980), ha="left", **ann)
    axB.annotate("vessel", xy=(2.44, -0.952), xytext=(2.50, -0.900),
                 ha="left", **ann)

    # ---- (c) toroidal placement strip ----------------------------------
    axC.axvspan(-180, 180, color="#f6f5f2", zorder=0)
    labels = []
    for i, (label, color, comps) in enumerate(TOROIDAL_ROWS):
        y = len(TOROIDAL_ROWS) - 1 - i
        n, cov, axisym = 0, 0.0, False
        for cls, name in comps:
            _, occ = load(cls, name)
            if occ is None:                 # axisymmetric: full-width band
                axC.add_patch(MplPolygon(
                    [[-180, y - 0.34], [180, y - 0.34],
                     [180, y + 0.34], [-180, y + 0.34]],
                    closed=True, facecolor=color, edgecolor="none", zorder=4))
                axisym, cov = True, 1.0
                continue
            _bars(axC, y, occ, color)
            n += occ.shape[0]
            cov += coverage(occ)
        axC.text(196, y, "—" if axisym else f"{n:>3d}", fontsize=8,
                 color=INK2, ha="right", va="center", family="monospace")
        axC.text(258, y, f"{min(cov, 1.0) * 100:>3.0f}%", fontsize=8,
                 color=INK2, ha="right", va="center", family="monospace")
        labels.append(label)

    axC.set_yticks(range(len(TOROIDAL_ROWS)))
    axC.set_yticklabels(labels[::-1], fontsize=8.4, color=INK2)
    axC.set_xlim(-180, 268)
    axC.set_ylim(-0.8, len(TOROIDAL_ROWS) - 0.2)
    axC.set_xticks([-180, -90, 0, 90, 180])
    axC.set_xticklabels(["−180°", "−90°", "0°", "90°", "180°"])
    axC.set_xlabel("toroidal angle  φ", color=INK2, fontsize=9)
    axC.set_facecolor(SURFACE)
    axC.tick_params(colors=MUTED, labelsize=8, length=3, width=0.7)
    axC.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        axC.spines[s].set_visible(False)
    axC.spines["bottom"].set_color(GRID)
    for xg in (-90, 0, 90):
        axC.axvline(xg, color=GRID, lw=0.5, zorder=1)
    axC.text(196, len(TOROIDAL_ROWS) - 0.55, "n", fontsize=8, color=MUTED,
             ha="right", va="center", style="italic")
    axC.text(258, len(TOROIDAL_ROWS) - 0.55, "cover", fontsize=8, color=MUTED,
             ha="right", va="center", style="italic")
    axC.set_title("(c)  Toroidal placement of each component", loc="left",
                  color=INK, fontsize=11, fontweight="bold", pad=8)

    # ---- legend ---------------------------------------------------------
    seen, handles = set(), []
    for label, color, cls, name in COMPONENTS_V4:
        if label in seen:
            continue
        seen.add(label)
        filled = any(coverage(load(c, n)[1]) >= COVERAGE_FILL
                     for lb, _, c, n in COMPONENTS_V4 if lb == label)
        handles.append(
            MplPolygon([[0, 0]], facecolor=color, alpha=0.8, edgecolor=color,
                       label=label) if filled
            else Line2D([0], [0], color=color, lw=1.7, label=label))
    handles += [
        Line2D([0], [0], color=INK, lw=1.4, label="Vacuum vessel"),
        Line2D([0], [0], color=INK, lw=1.4, ls=(0, (5, 3)),
               label="Separatrix (example equilibrium)"),
        MplPolygon([[0, 0]], facecolor="#f4f3f0", edgecolor=INK2,
                   label="Divertor PF coils"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.108), ncol=4, frameon=False,
               fontsize=8.6, labelcolor=INK2, handlelength=1.6,
               columnspacing=1.8, handletextpad=0.6).set_zorder(20)

    fig.suptitle("WEST tokamak — in-vessel configuration (WEST-V4)", x=0.055,
                 ha="left", y=0.972, color=INK, fontsize=15.5,
                 fontweight="bold")
    fig.text(0.055, 0.930,
             "Full-tungsten, actively-cooled plasma-facing components.   "
             "$R_0 \\approx 2.5$ m,  $a \\approx 0.5$ m,  circular two-shell "
             "vacuum vessel.   Geometry: ToFu WEST-V4 (CAD rev. Nov 2016).",
             ha="left", color=INK2, fontsize=9.8)
    fig.text(0.055, 0.902,
             "Filled = component wraps the torus quasi-continuously "
             "(coverage ≥ 60%);  outline = toroidally localised — see panel "
             "(c). Outlined shapes are therefore present only at some φ.",
             ha="left", color=MUTED, fontsize=9.0)

    out = OUT / "west_invessel_v4.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Figure 2 — configuration evolution V1 -> V4
# ══════════════════════════════════════════════════════════════════════════
# NB: V1..V4 are *description* versions of one CAD revision, not machine
# states over time. Verified: across V0->V3 every component keeps an identical
# (R, Z) bounding box, finer versions retain the coarser points, and the last
# version's polygon is identical to the previous one with only toroidal
# occurrences added. ToFu added V1-V3 in a single commit (2019-04-04) and the
# V4 extras on 2021-02-15.
CONFIGS = [
    ("WEST-V1", "coarse outlines, no segmentation", "2019-04-04", "StandardV2", [
        ("PFC", "BaffleV0", ORANGE), ("PFC", "DivUpV1", VIOLET),
        ("PFC", "DivLowITERV1", BLUE)]),
    ("WEST-V2", "+ bumpers, ICRH antennas", "2019-04-04", "StandardV2", [
        ("PFC", "BaffleV1", ORANGE), ("PFC", "DivUpV2", VIOLET),
        ("PFC", "DivLowITERV2", BLUE), ("PFC", "BumperInnerV1", AQUA),
        ("PFC", "BumperOuterV1", AQUA), ("PFC", "IC1V1", MAGENTA),
        ("PFC", "IC2V1", MAGENTA), ("PFC", "IC3V1", MAGENTA)]),
    ("WEST-V3", "+ LHCD launchers, ripple / VDE guards", "2019-04-04",
     "StandardV2", [
        ("PFC", "BaffleV2", ORANGE), ("PFC", "DivUpV3", VIOLET),
        ("PFC", "DivLowITERV3", BLUE), ("PFC", "BumperInnerV3", AQUA),
        ("PFC", "BumperOuterV3", AQUA), ("PFC", "IC1V1", MAGENTA),
        ("PFC", "IC2V1", MAGENTA), ("PFC", "IC3V1", MAGENTA),
        ("PFC", "LH1V1", YELLOW), ("PFC", "LH2V1", YELLOW),
        ("PFC", "RippleV1", RED), ("PFC", "VDEV0", RED)]),
    ("WEST-V4", "+ thermal shields, casing, true vessel", "2021-02-15", None,
     [(c, n, col) for _, col, c, n in COMPONENTS_V4]),
]

# WEST campaign record — independent of the geometry files above.
# (start, end, label, note, colour); end=None for the running phase.
MACHINE_PHASES = [
    (2017.0, 2021.0, "PHASE 1 · C1–C5",
     "a few ITER-grade PFUs", "#b9d4f3"),
    (2022.0, 2026.35, "PHASE 2 · C6–C11 →",
     "lower divertor fully ITER-grade", BLUE),
]
MACHINE_MARKS = [(2016.95, "first plasma\nDec 2016")]
RECORD_MARKS = [(2024.92, "824 s"), (2025.20, "1337 s")]
# when each description version actually entered ToFu
TOFU_MARKS = [
    (2016.87, "D", "CAD rev. Nov 2016\nsource of all four versions"),
    (2019.26, "o", "V1–V3 added\n2019-04-04"),
    (2021.12, "o", "V4 added\n2021-02-15"),
]


def draw_timeline(ax):
    """Two decoupled lanes: the machine's campaigns vs ToFu's version dates."""
    ybar, ytofu = 1.95, 0.55

    for x0, x1, label, note, color in MACHINE_PHASES:
        ax.add_patch(MplPolygon(
            [[x0, ybar - .15], [x1, ybar - .15], [x1, ybar + .15],
             [x0, ybar + .15]], closed=True, facecolor=color,
            edgecolor="none", zorder=3))
        dark = color == BLUE
        ax.text((x0 + x1) / 2, ybar, label, fontsize=7.4,
                color=SURFACE if dark else INK2, ha="center", va="center",
                fontweight="bold", zorder=4)
        ax.text((x0 + x1) / 2, ybar - .42, note, fontsize=7.2, color=MUTED,
                ha="center", va="center", zorder=4)

    for x, label in MACHINE_MARKS:
        ax.plot([x], [ybar], marker="o", ms=5, color=INK2, zorder=5)
        ax.text(x, ybar + .40, label, fontsize=7.2, color=INK2, ha="center",
                va="bottom", linespacing=1.4, zorder=5)
    for x, label in RECORD_MARKS:
        ax.plot([x, x], [ybar + .15, ybar + .34], color=INK2, lw=.8, zorder=5)
    ax.text(sum(x for x, _ in RECORD_MARKS) / len(RECORD_MARKS), ybar + .40,
            "record pulses  " + " → ".join(t for _, t in RECORD_MARKS),
            fontsize=7.2, color=INK2, ha="center", va="bottom", zorder=5)

    for x, mk, label in TOFU_MARKS:
        ax.plot([x], [ytofu], marker=mk, ms=6 if mk == "D" else 5.5,
                color=INK, zorder=5)
        ax.text(x, ytofu - .34, label, fontsize=7.2, color=INK2, ha="center",
                va="top", linespacing=1.4, zorder=5)
    # everything derives from the one CAD revision
    ax.plot([TOFU_MARKS[0][0]] * 2, [ytofu, ybar + .55], color=GRID, lw=.9,
            ls=(0, (3, 2)), zorder=1)

    for y, txt in ((ybar, "MACHINE"), (ytofu, "DESCRIPTION")):
        ax.text(2015.62, y, txt, fontsize=6.8, color=MUTED, ha="left",
                va="center", family="monospace",
                fontweight="bold", zorder=5)

    ax.set_xlim(2015.55, 2026.6)
    ax.set_ylim(-0.62, 3.05)
    ax.set_xticks([2016, 2018, 2020, 2022, 2024, 2026])
    ax.set_xticklabels([str(t) for t in (2016, 2018, 2020, 2022, 2024, 2026)])
    ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    for t in (2018, 2020, 2022, 2024, 2026):
        ax.axvline(t, color=GRID, lw=0.5, zorder=0)
    ax.set_title("Timing — the two are independent: no WEST-V number "
                 "corresponds to a campaign", loc="left", color=INK,
                 fontsize=9.6, fontweight="bold", pad=6)


def figure_evolution():
    fig = plt.figure(figsize=(15.0, 7.05), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.34],
                          left=0.045, right=0.99, top=0.815, bottom=0.075,
                          wspace=0.20, hspace=0.30)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    draw_timeline(fig.add_subplot(gs[1, :]))

    for ax, (name, note, added, ves, comps) in zip(axes, CONFIGS):
        if ves is None:
            draw_vessel(ax, lw=1.0)
        else:
            p = close(load("Ves", ves)[0])
            ax.plot(p[:, 0], p[:, 1], color=INK, lw=1.2, zorder=3)
        draw_components(ax, [(None, col, c, n) for c, n, col in comps], lw=0.9)

        # resolution readout: polygon points and toroidal occurrences
        npts = nocc = 0
        for c, n, _ in comps:
            poly, occ = load(c, n)
            npts += len(poly)
            nocc += 0 if occ is None else occ.shape[0]

        style_axes(ax, ylabel="Z  [m]" if ax is axes[0] else "")
        if ax is not axes[0]:
            ax.set_yticklabels([])
        ax.set_xlim(1.32, 3.66)
        ax.set_ylim(-1.10, 1.10)
        ax.set_title(f"{name}\n{note}", loc="left", color=INK, fontsize=10.5,
                     fontweight="bold", pad=8)
        # the vessel interior is empty here, but only ~21 chars wide between
        # the inner bumper and the LFS components — keep these lines short
        ax.text(0.46, 0.50,
                f"{len(comps)} components\n{npts} polygon points\n"
                f"{nocc} toroidal occ.\nadded {added}",
                transform=ax.transAxes, ha="center", va="center", fontsize=7.6,
                color=MUTED, linespacing=1.6)

    fig.suptitle("WEST in-vessel description — the ToFu reference geometries, "
                 "V1 → V4", x=0.045, ha="left", y=0.955, color=INK,
                 fontsize=14.5, fontweight="bold")
    fig.text(0.045, 0.888,
             "Not a machine timeline: all four describe the same Nov 2016 CAD "
             "revision at increasing completeness — outlines are refined and "
             "toroidal segmentation added, never repositioned.",
             ha="left", color=INK2, fontsize=9.4)
    fig.text(0.045, 0.850,
             "Every shape keeps an identical $(R, Z)$ bounding box across "
             "versions. V1–V3 use one D-shaped wall envelope; only V4 has the "
             "true circular vessel. Fill vs outline as in the main figure.",
             ha="left", color=MUTED, fontsize=9.0)

    out = OUT / "west_config_evolution.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (figure_invessel(), figure_evolution()):
        print(f"wrote {p}  (+ {p.with_suffix('.svg').name})")
