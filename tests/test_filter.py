# -*- coding: utf-8 -*-
"""Unit tests for the per-slice LCFS filters (synthetic geometry + real references).

The synthetic cases pin each criterion to one intended failure mode. The
retired-criterion guard is deliberate: a normal diverted plasma carries ~2 concave
vertices ~1 mm deep at the X-point cusp, and every count- or depth-based convexity
test that was tried rejected those. If someone reintroduces one, that test fails.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data.filter import (  # noqa: E402
    CODES, CONVEX_MIN, REGISTRY, SliceQuality, as_grid, codes, criterion,
    exclusive_candidates, exclusive_masks, fail_code, first_failure, hull_metrics,
    load_wall, only_code, quality_flags, slice_quality, theta_walk,
)

N = 32


def _circle(r=0.5, cR=2.4, cZ=0.0, n=N):
    """A clean convex n-gon, counter-clockwise, centred at (cR, cZ)."""
    th = np.arange(n) / n * 2 * np.pi
    return cR + r * np.cos(th), cZ + r * np.sin(th)


def _slice(R, Z, center=None, t=1.0):
    """Pack one slice into the ``(32, 2, 1)`` / ``(2, 1)`` / ``(1,)`` call shape."""
    g = np.stack([np.asarray(R, float), np.asarray(Z, float)], axis=1)[:, :, None]
    if center is None:
        center = (float(np.mean(R)), float(np.mean(Z)))
    geom = np.array([[center[0]], [center[1]]], float)
    return g, geom, np.array([t], float)


def _q(R, Z, center=None, t=1.0):
    return slice_quality(*_slice(R, Z, center=center, t=t))


# ------------------------------------------------------------------ the happy path ---
def test_clean_circle_passes_everything():
    q = _q(*_circle())
    for k in CODES:
        assert bool(q[k][0]), f"{k} should pass for a clean circle"
    assert bool(q["keep"][0])
    assert q["convexity"][0] == pytest.approx(1.0, abs=1e-6)
    assert q["winding"][0] == pytest.approx(1.0, abs=1e-9)
    assert q["outside_cm"][0] == 0.0
    assert first_failure(q)[0] == ""
    assert fail_code(q)[0] == 0


def test_flags_roundtrip_and_keep_agrees():
    q = _q(*_circle())
    flags = quality_flags(q)
    assert flags[0] == 0b111111                          # all six criteria pass
    for bit, k in enumerate(CODES):                      # bit i mirrors criterion i
        assert bool((flags[0] >> bit) & 1) == bool(q[k][0])


# --------------------------------------------------------------- one failure each ---
def test_s0_rejects_pre_ignitron():
    q = _q(*_circle(), t=-1.0)
    assert not q["s0"][0] and not q["keep"][0]
    assert first_failure(q)[0] == "s0"
    # S1/S3/S4 need no prerequisite, so clean geometry still passes them
    for k in ("s1", "s3", "s4"):
        assert bool(q[k][0]), f"{k} should still pass"
    # S2 and S5 declare requires=("s0","s1"), so with S0 failing their metrics are
    # never computed and they report False rather than a vacuous pass
    for k in ("s2", "s5"):
        assert not bool(q[k][0])
        assert np.isnan(q["outside_cm"][0]) or np.isnan(q["convexity"][0])


def test_s1_rejects_all_zero_and_nonfinite():
    q = slice_quality(np.zeros((N, 2, 1)), np.zeros((2, 1)), np.array([1.0]))
    assert not q["s1"][0] and bool(q["allzero"][0])
    assert first_failure(q)[0] == "s1"

    R, Z = _circle()
    R = R.copy()
    R[3] = np.nan
    q = _q(R, Z)
    assert not q["s1"][0] and q["n_nonfinite"][0] == 1


def test_s1_rejects_duplicated_vertex():
    R, Z = _circle()
    R, Z = R.copy(), Z.copy()
    R[5], Z[5] = R[4], Z[4]                              # zero-length edge
    q = _q(R, Z)
    assert not q["s1"][0]


def test_s2_rejects_outside_the_vessel():
    # a plasma shifted onto the outboard wall: the vessel ends at R = 3.298
    q = _q(*_circle(r=0.5, cR=3.1))
    assert not q["s2"][0]
    assert q["outside_cm"][0] > 10.0
    assert first_failure(q)[0] == "s2"


def test_s2_rejects_too_small_but_contained():
    q = _q(*_circle(r=0.1))                              # R-span 0.2 m < 0.3 m
    assert not q["s2"][0]
    assert q["outside_cm"][0] == 0.0                     # fully inside the wall
    assert q["r_span"][0] == pytest.approx(0.2, abs=1e-6)


def test_s2_tolerance_admits_a_grazing_vertex():
    """Limiter plasmas touch the bumper; 2 mm past the wall must still pass."""
    wall = load_wall()
    r_in = wall[:, 0].min()                              # inboard-most wall point
    R, Z = _circle(r=0.5, cR=r_in + 0.5 - 0.002)         # pokes ~2 mm inboard
    q = _q(R, Z)
    assert 0.0 < q["outside_cm"][0] <= 0.5
    assert bool(q["s2"][0])


def test_s3_rejects_center_outside_the_boundary():
    R, Z = _circle(r=0.3, cR=2.4)
    q = _q(R, Z, center=(2.0, 0.0))                      # center left of the plasma
    assert not q["s3"][0]
    assert q["winding"][0] == pytest.approx(0.0, abs=1e-9)


def test_s3_rejects_sentinel_center():
    q = _q(*_circle(), center=(0.0, 0.0))                # GMAG_GEOM sentinel
    assert not q["s3"][0]


def test_s4_rejects_a_wedge_reaching_past_the_center():
    """One vertex dragged across the center makes theta backtrack."""
    R, Z = _circle(r=0.5, cR=2.4)
    R, Z = R.copy(), Z.copy()
    R[8], Z[8] = 2.4 - 0.4, 0.0                          # spike through the middle
    q = _q(R, Z, center=(2.4, 0.0))
    assert not q["s4"][0]


def test_s5_rejects_a_notch():
    R, Z = _circle(r=0.5, cR=2.4)
    R, Z = R.copy(), Z.copy()
    R[10] = 2.4 + 0.5 * np.cos(10 / N * 2 * np.pi) * 0.55   # pull one vertex inward
    Z[10] = 0.5 * np.sin(10 / N * 2 * np.pi) * 0.55
    q = _q(R, Z, center=(2.4, 0.0))
    assert q["convexity"][0] < CONVEX_MIN
    assert not q["s5"][0]


def test_s5_is_two_sided_and_rejects_self_crossing():
    """A ratio above 1 is impossible for a simple polygon -- it means self-crossing.

    A one-sided ``convexity >= 0.995`` waved these through; campaign-wide that was
    3511 slices, including the only two that S3 and S4 uniquely caught.
    """
    th = np.arange(N) / N * 2 * np.pi * 2                 # wind twice -> spiral
    R = 2.4 + (0.2 + 0.01 * np.arange(N)) * np.cos(th)
    Z = (0.2 + 0.01 * np.arange(N)) * np.sin(th)
    q = _q(R, Z, center=(2.4, 0.0))
    assert q["convexity"][0] > 1.0
    assert not q["s5"][0], "two-sided S5 must reject a self-crossing boundary"


# -------------------------------------------------------- retired-criterion guard ---
def test_normal_xpoint_cusp_is_kept():
    """~2 concave vertices ~1 mm deep is what a real diverted plasma looks like.

    Guards against reintroducing a count-based convexity test (`n_concave == 0`
    keeps only 13.6% of good slices) or a notch-depth cap (legitimate cusps reach
    7 cm).
    """
    R, Z = _circle(r=0.5, cR=2.4)
    R, Z = R.copy(), Z.copy()
    for k in (23, 24):                                   # dent two vertices by 1 mm
        rr = 0.5 - 0.001
        ang = k / N * 2 * np.pi
        R[k], Z[k] = 2.4 + rr * np.cos(ang), rr * np.sin(ang)
    q = _q(R, Z, center=(2.4, 0.0))
    assert q["convexity"][0] > CONVEX_MIN
    assert bool(q["keep"][0]), "a normal X-point cusp must not be filtered out"


# ----------------------------------------------------------------- derived views ---
def test_exclusive_masks_isolate_a_single_criterion():
    q = _q(*_circle(r=0.1))                              # fails S2 (too small) only
    excl = exclusive_masks(q)
    assert bool(excl["s2"][0])
    for k in ("s3", "s4", "s5"):
        assert not bool(excl[k][0])
    assert only_code(q)[0] == CODES.index("s2") + 1


def test_exclusive_is_empty_when_two_criteria_fire():
    R, Z = _circle(r=0.3, cR=2.4)
    q = _q(R, Z, center=(2.0, 0.0))                      # S3 and S4 both fail
    assert not q["s3"][0] and not q["s4"][0]
    excl = exclusive_masks(q)
    assert not bool(excl["s3"][0]) and not bool(excl["s4"][0])
    assert only_code(q)[0] == 0


# ----------------------------------------------------------------------- helpers ---
def test_as_grid_accepts_both_layouts():
    R, Z = _circle()
    interleaved = np.empty((2 * N, 1))
    interleaved[0::2, 0] = R
    interleaved[1::2, 0] = Z
    g = as_grid(interleaved)
    assert g.shape == (N, 2, 1)
    assert np.allclose(g[:, 0, 0], R) and np.allclose(g[:, 1, 0], Z)
    assert as_grid(g).shape == (N, 2, 1)
    with pytest.raises(ValueError):
        as_grid(np.zeros((17, 3)))


def test_hull_metrics_on_a_square():
    R = np.array([0.0, 1.0, 1.0, 0.0])
    Z = np.array([0.0, 0.0, 1.0, 1.0])
    cx, depth = hull_metrics(R, Z)
    assert cx == pytest.approx(1.0, abs=1e-12)
    assert depth == pytest.approx(0.0, abs=1e-9)


def test_theta_walk_direction_and_winding():
    R, Z = _circle()
    w, mono = theta_walk(R[:, None], Z[:, None],
                         np.array([np.mean(R)]), np.array([np.mean(Z)]))
    assert w[0] == pytest.approx(1.0, abs=1e-9) and bool(mono[0])
    w, mono = theta_walk(R[::-1, None], Z[::-1, None],          # reversed = clockwise
                         np.array([np.mean(R)]), np.array([np.mean(Z)]))
    assert w[0] == pytest.approx(-1.0, abs=1e-9) and bool(mono[0])


def test_vessel_contour_shape_and_extent():
    w = load_wall()
    assert w.shape == (59, 2)
    assert w[:, 0].min() == pytest.approx(1.788, abs=1e-3)
    assert w[:, 0].max() == pytest.approx(3.298, abs=1e-3)
    # the vessel is asymmetric in Z -- a symmetric |Z| box is not a substitute
    assert w[:, 1].min() == pytest.approx(-0.798, abs=1e-3)
    assert w[:, 1].max() == pytest.approx(0.869, abs=1e-3)


def test_geom_shape_is_validated():
    g, _, t = _slice(*_circle())
    with pytest.raises(ValueError):
        slice_quality(g, np.zeros((2, 5)), t)


# -------------------------------------------------------- class / registry API ---
def test_class_and_dict_forms_agree():
    g, geom, t = _slice(*_circle())
    q_obj = SliceQuality(g, geom, t)
    q_dict = slice_quality(g, geom, t)
    for k in q_dict:
        a, b = np.asarray(q_obj[k]), np.asarray(q_dict[k])
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), k
    assert set(q_dict) == set(q_obj.keys())


def test_thresholds_are_overridable_per_call():
    R, Z = _circle(r=0.1)                                 # R-span 0.2 m
    g, geom, t = _slice(R, Z)
    assert not SliceQuality(g, geom, t)["s2"][0]          # default r_span_min = 0.3
    assert SliceQuality(g, geom, t, r_span_min=0.1)["s2"][0]


def test_unknown_threshold_is_rejected():
    g, geom, t = _slice(*_circle())
    with pytest.raises(ValueError, match="unknown threshold"):
        SliceQuality(g, geom, t, convexity_min=0.9)       # typo for convex_min


def test_registry_is_ordered_and_self_consistent():
    assert codes() == ("s0", "s1", "s2", "s3", "s4", "s5")
    assert [c.index for c in REGISTRY] == [0, 1, 2, 3, 4, 5]
    # S0/S1 are prerequisites, so they are not exclusive-eligible
    assert exclusive_candidates() == ("s2", "s3", "s4", "s5")
    for c in REGISTRY:                                    # requires must precede
        seen = []
        for other in REGISTRY:
            if other.code == c.code:
                break
            seen.append(other.code)
        assert set(c.requires) <= set(seen), f"{c.code} requires a later criterion"
    for c in REGISTRY:
        assert c.title and c.detail, f"{c.code} needs a title and rationale"


def test_a_new_criterion_is_picked_up_everywhere():
    """Adding a criterion must need one decorated function and nothing else."""
    n_before = len(REGISTRY)
    try:
        @criterion("s9", "test-only", "always fails, to prove registration works")
        def _s9(c):
            return np.zeros(c.nt, bool)

        g, geom, t = _slice(*_circle())
        q = SliceQuality(g, geom, t)
        assert "s9" in q.results and not q.keep[0]
        assert q.first_failure()[0] == "s9"               # last in order
        assert (q.flags()[0] >> 9) & 1 == 0               # its bit is clear
        assert codes()[-1] == "s9"
    finally:
        del REGISTRY[n_before:]                           # keep module state clean
