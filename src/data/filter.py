# -*- coding: utf-8 -*-
"""Per-slice LCFS quality filters for ``targets/GMAG_BND``.

A registry of independent criteria plus a ``SliceQuality`` class that evaluates them
over one shot, vectorized in time. Full evidence, thresholds and rejected
alternatives: ``docs/lcfs_filters.md`` and
``docs/superpowers/specs/2026-08-05-lcfs-slice-shape-gate-design.md``.

  S0 in the discharge   ignitron time >= 0 -- nothing exists before the shot fires
  S1 finite & sane      all (R,Z) finite, not all-zero, no zero-length edge
  S2 inside the vessel  every vertex within ``wall_tol_cm`` of the WEST vessel
                        contour, and R-span > ``r_span_min``
  S3 center enclosed    winding number of the vertex loop about (Rgeom, Zgeom) = +-1
  S4 simply connected   theta about (Rgeom, Zgeom) strictly monotonic
  S5 convexity          ``convex_min`` <= |A_poly| / |A_hull| <= 1 + ``convex_eps``

## Adding a criterion

Write one decorated function; nothing else needs editing. It receives the metric
context and returns a bool array of length ``nt``:

    @criterion("s6", "up-down symmetry", "reject strongly asymmetric boundaries",
               requires=("s0", "s1"))
    def _s6(c):
        return np.abs(c.Z.max(axis=0) + c.Z.min(axis=0)) < c.thresholds["z_skew_max"]

``requires`` names the criteria that must pass for this one's metrics to be
meaningful -- metric-bearing criteria are evaluated only on that mask, and are False
elsewhere (a slice with a non-finite vertex has no convexity to compare). Order in
the file is evaluation order, which is what ``first_failure`` attributes to. Add any
new threshold to ``DEFAULTS`` so it stays overridable per call.

## Two subtleties that cost real debugging -- do not "simplify" them away

* **S5 is two-sided.** A simple polygon always has |A_poly| <= |A_hull|; a ratio
  above 1 means the boundary self-crosses and the shoelace area double-counts the
  overlap. A one-sided test admits those (measured: ratios to 1.26 on double-wrap
  spirals, 3511 slices campaign-wide).
* **Only integral shape measures work.** Vertex-count convexity, notch depth, turn
  angle and local sagitta were all tested and rejected -- each fires on the normal
  X-point cusp, because a 32-point discretization of a high-curvature cusp is
  genuinely kinked at the centimetre scale. See ``docs/lcfs_filters.md`` §5.

I/O-free apart from ``load_wall`` reading the vessel contour once per process.
"""
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial import ConvexHull

from ..proj_config import get_proj_config

N_POINTS = 32          # native LCFS vertices per slice
# No fixed origin lives here: every criterion is judged about the slice's own
# (Rgeom, Zgeom). A module constant would only tempt a caller to project about it.

# --- tunable thresholds (rationale in docs/lcfs_filters.md) ---
DEFAULTS = {
    "convex_min": 0.995,    # S5 lower bound: below this a visible step/notch is present
    "convex_eps": 1e-6,     # S5 upper bound is 1 + this; above it the boundary crosses
    "wall_file": "TFG_Ves_ExpWEST_StandardV2.txt",   # S2 contour (the D-shaped envelope,
                                                     # strictly inside true Ves_InnerV0)
    "wall_tol_cm": 0.5,     # S2: limiter plasmas touch the bumper; reconstruction errs
    "r_span_min": 0.3,      # S2: below this extent it is not a plasma
    "min_edge_cm": 1e-4,    # S1: shorter than this is a duplicated vertex
    "winding_tol": 1e-6,    # S3: |winding| must equal 1 to within this
}

# module-level aliases, for callers that just want the numbers
CONVEX_MIN = DEFAULTS["convex_min"]
CONVEX_EPS = DEFAULTS["convex_eps"]
WALL_FILE = DEFAULTS["wall_file"]
WALL_TOL_CM = DEFAULTS["wall_tol_cm"]
R_SPAN_MIN = DEFAULTS["r_span_min"]

_WALL_CACHE = {}


# ------------------------------------------------------------------- geometry ---
def load_wall(name=WALL_FILE):
    """``(V, 2)`` R-Z vertices of a ToFu-format WEST contour, cached per process.

    File layout: ``# key = value`` comments, a ``count 0`` row, then ``count`` rows
    of ``R Z`` in metres. ``TFG_Ves_ExpWEST_StandardV2`` is 59 points spanning
    R 1.788-3.298 m, Z **-0.798 to +0.869** m -- the Z asymmetry is real, and is why
    a symmetric bounding box is not a substitute.
    """
    if name not in _WALL_CACHE:
        path = get_proj_config().proj_db_dir / "datasets" / "west_geom" / name
        rows = [ln.split() for ln in path.read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
        nums = np.array([[float(a) for a in r] for r in rows], float)
        n = int(round(nums[0, 0]))
        pts = nums[1:n + 1]
        if pts.shape[0] != n:
            raise ValueError(f"{name}: header says {n} points, found {pts.shape[0]}")
        _WALL_CACHE[name] = pts
    return _WALL_CACHE[name]


def as_grid(bnd):
    """Normalise a boundary array to ``(32, 2, nt)``.

    Accepts the raw ``(64, nt)`` interleaved ``[R0,Z0,...,R31,Z31]`` layout or an
    already-reshaped ``(32, 2, nt)``.
    """
    a = np.asarray(bnd, float)
    if a.ndim == 2 and a.shape[0] == 2 * N_POINTS:
        return a.reshape(N_POINTS, 2, -1)
    if a.ndim == 3 and a.shape[:2] == (N_POINTS, 2):
        return a
    raise ValueError(f"expected (64, nt) or (32, 2, nt) boundary, got {a.shape}")


def hull_metrics(R, Z):
    """``(convexity, depth_cm)`` for one closed 32-gon.

    ``convexity`` is |shoelace area| / |convex-hull area|; ``depth_cm`` is how far
    the most re-entrant vertex sits inside the hull. NaN if the hull is degenerate.
    """
    P = np.c_[R, Z]
    try:
        h = ConvexHull(P)
    except Exception:                                    # noqa: BLE001
        return np.nan, np.nan
    area = abs(0.5 * np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z))
    eq = h.equations                                     # n.x + b <= 0 inside
    depth = -np.max(P @ eq[:, :2].T + eq[:, 2], axis=1)   # > 0 => inside the hull
    return area / h.volume, float(np.max(depth) * 100)


def theta_walk(R, Z, cR, cZ):
    """``(winding, monotonic)`` of theta about a per-slice center.

    ``R``/``Z`` are ``(32, nt)``, ``cR``/``cZ`` are ``(nt,)``. ``winding`` is the
    signed number of turns the vertex loop makes about the center (0 => outside,
    +-1 => enclosed once, +-2 => the vertex order is scrambled into a double wrap);
    ``monotonic`` is whether theta advances strictly one way, i.e. star-shaped.
    """
    th = np.arctan2(Z - cZ, R - cR)
    d = (np.diff(np.r_[th, th[:1]], axis=0) + np.pi) % (2 * np.pi) - np.pi
    return (np.sum(d, axis=0) / (2 * np.pi),
            np.all(d > 0, axis=0) | np.all(d < 0, axis=0))


def center_margin_cm(R, Z, cR, cZ):
    """``(nt,)`` distance from the center to the nearest boundary edge [cm].

    A small value means the geometric center sits almost on the boundary.
    """
    bR, bZ = np.roll(R, -1, 0), np.roll(Z, -1, 0)
    eR, eZ = bR - R, bZ - Z
    t = np.clip(((cR - R) * eR + (cZ - Z) * eZ)
                / np.maximum(eR ** 2 + eZ ** 2, 1e-18), 0.0, 1.0)
    return np.min(np.hypot(cR - (R + t * eR), cZ - (Z + t * eZ)), axis=0) * 100


def wall_outside_cm(R, Z, wall):
    """``(nt,)`` largest outward excursion past ``wall`` [cm]; 0 where contained.

    Inside/outside is the even-odd crossing rule; the excursion is the
    point-to-segment distance to the nearest wall edge. Accumulated edge-by-edge on
    purpose: the fully broadcast ``(V, 32, nt)`` form costs ~250 MB per intermediate
    on a long shot, which does not survive one worker per core.
    """
    wR, wZ = wall[:, 0], wall[:, 1]
    crossings = np.zeros(R.shape, np.int32)
    dmin = np.full(R.shape, np.inf)
    for i in range(wall.shape[0]):
        aR, aZ = wR[i], wZ[i]
        j = (i + 1) % wall.shape[0]
        bR, bZ = wR[j], wZ[j]
        if bZ != aZ:                                     # horizontal edges cast no ray
            straddle = (aZ > Z) != (bZ > Z)
            xcross = (bR - aR) * (Z - aZ) / (bZ - aZ) + aR
            crossings += (straddle & (R < xcross))
        eR, eZ = bR - aR, bZ - aZ
        L2 = max(eR * eR + eZ * eZ, 1e-18)
        tt = np.clip(((R - aR) * eR + (Z - aZ) * eZ) / L2, 0.0, 1.0)
        np.minimum(dmin, np.hypot(R - (aR + tt * eR), Z - (aZ + tt * eZ)), out=dmin)
    return np.max(np.where((crossings % 2) == 1, 0.0, dmin), axis=0) * 100


# ------------------------------------------------------------------- registry ---
@dataclass(frozen=True)
class Criterion:
    """One filter: its code, what it means, what it needs, and how to test it."""
    code: str                       # "s0" ... ; also the bit position and fail code
    title: str                      # short human label for reports
    detail: str                     # one-line rationale
    fn: Callable                    # (MetricContext) -> bool ndarray (nt,)
    requires: tuple = ()            # codes that must pass for this to be evaluable

    @property
    def index(self):
        return int(self.code[1:])


REGISTRY = []


def criterion(code, title, detail, requires=()):
    """Register a criterion. Declaration order is evaluation order."""
    def deco(fn):
        REGISTRY.append(Criterion(code, title, detail, fn, tuple(requires)))
        return fn
    return deco


def codes():
    """Registered criterion codes, in evaluation order."""
    return tuple(c.code for c in REGISTRY)


def exclusive_candidates():
    """Codes eligible for the 'fails only this one' view.

    A criterion that other criteria depend on cannot have a meaningful exclusive
    set: when it fails, its dependents are unevaluable rather than passing.
    """
    depended_on = {r for c in REGISTRY for r in c.requires}
    return tuple(c.code for c in REGISTRY if c.code not in depended_on)


# ------------------------------------------------------------------- context ---
class MetricContext:
    """Shared per-slice geometry and metrics, computed once and cached.

    Metric-bearing criteria are evaluated only where their ``requires`` all pass, so
    each metric is cached per gate mask -- two criteria with different ``requires``
    get correctly-scoped metrics rather than silently sharing one.
    """

    def __init__(self, bnd, geom, t=None, **thresholds):
        bad = set(thresholds) - set(DEFAULTS)
        if bad:
            raise ValueError(f"unknown threshold(s): {sorted(bad)}")
        self.thresholds = {**DEFAULTS, **thresholds}
        g = as_grid(bnd)
        self.g = g
        self.R, self.Z = g[:, 0, :], g[:, 1, :]
        self.nt = self.R.shape[1]
        geom = np.asarray(geom, float)
        if geom.shape != (2, self.nt):
            raise ValueError(f"geom must be (2, {self.nt}), got {geom.shape}")
        self.geom = geom
        self.cR, self.cZ = geom[0], geom[1]
        self.t = None if t is None else np.asarray(t, float)
        self.gate = np.ones(self.nt, bool)      # set by the evaluator per criterion
        self._cache = {}

        # cheap, always-defined descriptors. All-NaN slices are normal here (the
        # merge writes NaN outside a signal's own span) and reduce to NaN, which
        # fails S1/S2 -- so the numpy warning is expected noise, not a signal.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            warnings.filterwarnings("ignore", message="All-NaN axis encountered")
            e_len = np.hypot(np.roll(self.R, -1, 0) - self.R,
                             np.roll(self.Z, -1, 0) - self.Z)
            self.min_edge_cm = np.nanmin(e_len, axis=0) * 100
            self.r_lo = np.nanmin(self.R, axis=0)
            self.r_hi = np.nanmax(self.R, axis=0)
            self.absz_max = np.nanmax(np.abs(self.Z), axis=0)
        self.r_span = self.r_hi - self.r_lo
        self.n_nonfinite = np.sum(~np.isfinite(g), axis=(0, 1))
        self.allzero = ~np.any(g != 0, axis=(0, 1))
        # theta metrics need no gate: arctan2 is defined wherever the inputs are
        self.winding, self.monotonic = theta_walk(self.R, self.Z, self.cR, self.cZ)
        self.margin_cm = center_margin_cm(self.R, self.Z, self.cR, self.cZ)

    def _gated(self, name, compute):
        """Cache ``compute`` per (metric, gate), NaN outside the gate."""
        key = (name, self.gate.tobytes())
        if key not in self._cache:
            out = np.full(self.nt, np.nan)
            idx = np.where(self.gate)[0]
            if idx.size:
                out[idx] = compute(idx)
            self._cache[key] = out
        return self._cache[key]

    @property
    def outside_cm(self):
        """Largest excursion past the vessel wall [cm]; needs finite vertices."""
        wall = load_wall(self.thresholds["wall_file"])
        return self._gated(
            "outside_cm",
            lambda idx: wall_outside_cm(self.R[:, idx], self.Z[:, idx], wall))

    @property
    def convexity(self):
        return self._hull()[0]

    @property
    def depth_cm(self):
        return self._hull()[1]

    def _hull(self):
        key = ("hull", self.gate.tobytes())
        if key not in self._cache:
            cx = np.full(self.nt, np.nan)
            dp = np.full(self.nt, np.nan)
            for j in np.where(self.gate)[0]:
                cx[j], dp[j] = hull_metrics(self.R[:, j], self.Z[:, j])
            self._cache[key] = (cx, dp)
        return self._cache[key]


# ------------------------------------------------------------------ criteria ---
# Declaration order is evaluation order, and is what first_failure attributes to.

@criterion("s0", "in the discharge",
           "ignitron time >= 0; before the shot fires there is no plasma")
def _s0(c):
    if c.t is None:
        return np.ones(c.nt, bool)
    return c.t >= 0.0


@criterion("s1", "finite & non-degenerate",
           "all (R,Z) finite, not all-zero, no duplicated vertex")
def _s1(c):
    with np.errstate(invalid="ignore"):
        return ((c.n_nonfinite == 0) & ~c.allzero
                & (c.min_edge_cm > c.thresholds["min_edge_cm"]))


@criterion("s2", "inside the vessel",
           "every vertex within tolerance of the WEST vessel contour, and big enough",
           requires=("s0", "s1"))
def _s2(c):
    with np.errstate(invalid="ignore"):
        return ((c.outside_cm <= c.thresholds["wall_tol_cm"])
                & (c.r_span > c.thresholds["r_span_min"]))


@criterion("s3", "center enclosed",
           "winding of the vertex loop about (Rgeom, Zgeom) is +-1; "
           "a sentinel or absent center lands outside and fails here")
def _s3(c):
    with np.errstate(invalid="ignore"):
        return (np.isfinite(c.geom).all(axis=0)
                & (np.abs(np.abs(c.winding) - 1.0) < c.thresholds["winding_tol"]))


@criterion("s4", "simply connected",
           "theta about the center advances strictly one way, so r(theta) is "
           "single-valued and the curve cannot self-intersect")
def _s4(c):
    return c.monotonic


@criterion("s5", "convexity",
           "area/hull ratio within [convex_min, 1]; below rejects notches, "
           "above rejects self-crossing boundaries",
           requires=("s0", "s1"))
def _s5(c):
    with np.errstate(invalid="ignore"):
        return ((c.convexity >= c.thresholds["convex_min"])
                & (c.convexity <= 1.0 + c.thresholds["convex_eps"]))


CODES = codes()
EXCLUSIVE_KEYS = exclusive_candidates()


# ------------------------------------------------------------- SliceQuality ---
class SliceQuality:
    """Evaluate every registered criterion over one shot, vectorized in time.

    ``bnd`` is ``(64, nt)`` or ``(32, 2, nt)`` in metres; ``geom`` is ``(2, nt)``
    (Rgeom, Zgeom) in **metres** (GMagH5 stores mm -- scale by 1e-3 first); ``t`` is
    the ignitron time axis, or None to make S0 a no-op.

    Behaves like a mapping so ``q["s2"]`` / ``q["convexity"]`` work, which keeps the
    dict-based call sites unchanged.
    """

    METRICS = ("convexity", "depth_cm", "winding", "margin_cm", "outside_cm",
               "r_span", "r_lo", "r_hi", "absz_max", "min_edge_cm",
               "n_nonfinite", "allzero")

    def __init__(self, bnd, geom, t=None, **thresholds):
        self.ctx = MetricContext(bnd, geom, t, **thresholds)
        self.nt = self.ctx.nt
        self.results = {}
        for c in REGISTRY:
            gate = np.ones(self.nt, bool)
            for req in c.requires:
                gate &= self.results[req]
            self.ctx.gate = gate
            out = np.asarray(c.fn(self.ctx), bool)
            if out.shape != (self.nt,):
                raise ValueError(f"criterion {c.code} returned {out.shape}, "
                                 f"expected {(self.nt,)}")
            # a criterion is only asserted where its prerequisites hold
            self.results[c.code] = out & gate if c.requires else out
        # Leave the context gated to the prerequisite mask so that reading a metric
        # afterwards reuses the cache the criteria were evaluated against: same NaN
        # pattern, and no second pass of 7.5 M needless convex hulls.
        self.ctx.gate = self.prereq_mask()

    # --- mapping-ish access, so existing q["..."] call sites keep working ---
    def __getitem__(self, key):
        if key in self.results:
            return self.results[key]
        if key == "keep":
            return self.keep
        if key in self.METRICS:
            return getattr(self.ctx, key)
        raise KeyError(key)

    def __contains__(self, key):
        return key in self.results or key == "keep" or key in self.METRICS

    def keys(self):
        return (*self.results, "keep", *self.METRICS)

    def to_dict(self):
        """Plain dict of every criterion, ``keep`` and every metric."""
        return {k: self[k] for k in self.keys()}

    # --- derived views ---
    def prereq_mask(self):
        """Mask where every criterion that others depend on passes (the metric gate)."""
        out = np.ones(self.nt, bool)
        for r in {r for c in REGISTRY for r in c.requires}:
            out &= self.results[r]
        return out

    @property
    def keep(self):
        out = np.ones(self.nt, bool)
        for v in self.results.values():
            out &= v
        return out

    def first_failure(self):
        """``(nt,)`` object array naming the first criterion each slice fails.

        Order-dependent by construction; use it for reporting, not for judging how
        much a criterion contributes (see ``exclusive``).
        """
        lab = np.array([""] * self.nt, dtype=object)
        for c in REGISTRY:
            lab[(lab == "") & ~self.results[c.code]] = c.code
        return lab

    def fail_code(self):
        """``(nt,)`` int8: 0 kept, else 1-based index of the first failing criterion."""
        lab = self.first_failure()
        code = np.zeros(self.nt, np.int8)
        for i, c in enumerate(REGISTRY, start=1):
            code[lab == c.code] = i
        return code

    def exclusive(self):
        """Per-criterion mask of slices that fail **only** that criterion.

        The marginal-necessity view: a criterion with an empty exclusive set rejects
        nothing the others would not have caught. Only criteria nothing depends on
        are eligible -- when a prerequisite fails, its dependents are unevaluable
        rather than passing, so "only S1" would be vacuous.

        Measured campaign-wide (2026-08-05): S2 151, S3 0, S4 0, S5 6148. S3 and S4
        cover *each other*, so dropping either is free but dropping both is not.
        """
        prereqs = {r for c in REGISTRY for r in c.requires}
        base = np.ones(self.nt, bool)
        for r in prereqs:
            base &= self.results[r]
        out = {}
        for k in EXCLUSIVE_KEYS:
            m = base & ~self.results[k]
            for other in EXCLUSIVE_KEYS:
                if other != k:
                    m = m & self.results[other]
            out[k] = m
        # A prerequisite has no peers to compare against -- when it fails, everything
        # downstream is unevaluable. Report it as "everything before it passed".
        for c in REGISTRY:
            if c.code not in prereqs:
                continue
            m = ~self.results[c.code]
            for earlier in REGISTRY:
                if earlier.code == c.code:
                    break
                m = m & self.results[earlier.code]
            out[c.code] = m
        return out

    def only_code(self):
        """``(nt,)`` int8: criterion that rejects a slice alone (0 = not unique)."""
        excl = self.exclusive()
        code = np.zeros(self.nt, np.int8)
        for i, c in enumerate(REGISTRY, start=1):
            if c.code in excl:
                code[excl[c.code]] = i
        return code

    def flags(self):
        """``(nt,)`` uint8 bitmask, bit ``i`` set = criterion ``i`` **passes**.

        Order-free, so arbitrary set algebra survives a round-trip to disk: e.g.
        ``(flags >> 3) & 1 == 0`` selects every slice failing S3 regardless of what
        else it fails.
        """
        out = np.zeros(self.nt, np.uint8)
        for c in REGISTRY:
            out |= (self.results[c.code].astype(np.uint8) << c.index)
        return out


# --------------------------------------------------- function-style shortcuts ---
def slice_quality(bnd, geom, t=None, **thresholds):
    """``SliceQuality(...).to_dict()`` -- the dict-returning form."""
    return SliceQuality(bnd, geom, t, **thresholds).to_dict()


class _DictView(SliceQuality):
    """Adapter so the module-level helpers accept a plain dict too.

    Subclasses ``SliceQuality`` to inherit every derived view; the views only read
    ``results`` and ``nt``, so no metric context is needed.
    """

    def __init__(self, d):                               # noqa: D107 (see class doc)
        self.results = {c.code: np.asarray(d[c.code], bool) for c in REGISTRY}
        self.nt = next(iter(self.results.values())).size
        self.ctx = None


def _as_obj(q):
    """Accept either a ``SliceQuality`` or the dict it produces."""
    return q if isinstance(q, SliceQuality) else _DictView(q)


def first_failure(q):
    """First failing criterion per slice ('' = kept). Accepts a dict or an object."""
    return SliceQuality.first_failure(_as_obj(q))


def fail_code(q):
    return SliceQuality.fail_code(_as_obj(q))


def exclusive_masks(q):
    return SliceQuality.exclusive(_as_obj(q))


def only_code(q):
    return SliceQuality.only_code(_as_obj(q))


def quality_flags(q):
    return SliceQuality.flags(_as_obj(q))
