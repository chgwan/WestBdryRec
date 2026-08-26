# -*- coding: utf-8 -*-
"""Native closed-contour distance and geometry metrics (R-Z, metres).

Compares two closed polylines directly in absolute R-Z space, with no shared
angle parameterisation: every metric is built from vertex-to-polyline
distances (``point_to_closed_polyline``) and from shoelace polygon geometry
of each contour on its own.

Distance metrics (per time slice, ``d_pt`` = each predicted vertex's distance
to the truth polyline, ``d_tp`` = each truth vertex's distance to the
predicted polyline, ``combined`` = both pooled):

- ``mean_symmetric_mm`` = 0.5 * (mean(d_pt) + mean(d_tp)) * 1000
- ``p95_mm``            = 95th percentile of ``combined`` * 1000
- ``chamfer_rms_mm``    = sqrt(0.5 * (mean(d_pt^2) + mean(d_tp^2))) * 1000
- ``hausdorff_mm``      = max(``combined``) * 1000

Geometry metrics (shoelace signed area ``A`` and area-weighted centroid
``C``; both are winding-independent because the sign of ``A`` cancels):

- ``area_abs_m2``   = |enclosed(pred) - enclosed(true)|, enclosed = |A|
- ``centroid_mm``   = ||C_pred - C_true|| * 1000
- ``elongation_abs`` = |kappa_pred - kappa_true| with the extrema form
  kappa = (Z_max - Z_min) / (R_max - R_min)
- ``triangularity_upper_abs`` = |delta_u(pred) - delta_u(true)| with
  delta_u = (R_c - R[Z_max]) / a
- ``triangularity_lower_abs`` = |delta_l(pred) - delta_l(true)| with
  delta_l = (R_c - R[Z_min]) / a

where ``R_c`` is the shoelace-centroid major radius, ``R[Z_max]`` /
``R[Z_min]`` are the major radii of the maximum-/minimum-Z vertices and
``a = (R_max - R_min) / 2`` is the half-width used as the minor radius.
The sign convention makes delta positive when the Z-extreme vertex is
displaced toward the inboard (low-field, smaller-R) side, so standard
D-shaped plasmas come out with positive triangularity (the geometric limit
of the Miller parameterisation R(theta) = R_0 + a*cos(theta + delta*sin
theta), whose theta = +/-pi/2 points sit at R_0 -/+ a*sin(delta)).

Degenerate contours (zero area or zero R-width, guarded only by a 1e-12
width floor mirroring the 1e-24 segment guard) propagate NaN/Inf rather
than raising.
"""
import numpy as np

_WIDTH_EPS = 1e-12


def point_to_closed_polyline(points, contour):
    """Euclidean distance from each point to a closed polyline (N, 2).

    The polyline is closed through its last->first edge: segments are
    ``(a_i, a_{i+1})`` with ``b = roll(a, -1)``, so ``b[-1] == a[0]``.
    Returns (M,) distances in the input units (metres for R-Z contours).
    """
    p = np.asarray(points, float)
    a = np.asarray(contour, float)
    b = np.roll(a, -1, axis=0)
    ab = b - a
    den = np.maximum((ab * ab).sum(1), 1e-24)
    ap = p[:, None, :] - a[None, :, :]
    u = np.clip(
        (ap * ab[None, :, :]).sum(2) / den[None, :],
        0.0, 1.0)
    nearest = a[None, :, :] + u[:, :, None] * ab[None, :, :]
    return np.sqrt(
        ((p[:, None, :] - nearest) ** 2).sum(2)
    ).min(1)


def _shoelace_area_centroid(contour):
    """(signed_area, centroid (R, Z)) of a closed polyline via the shoelace
    formula. The polyline is closed through its last->first edge and must
    not repeat the first vertex at the end."""
    a = np.asarray(contour, float)
    b = np.roll(a, -1, axis=0)
    cross = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
    area = 0.5 * cross.sum()
    centroid = np.array([
        ((a[:, 0] + b[:, 0]) * cross).sum() / (6.0 * area),
        ((a[:, 1] + b[:, 1]) * cross).sum() / (6.0 * area),
    ])
    return area, centroid


def _contour_geometry(contour):
    """(enclosed_area, centroid, kappa, delta_upper, delta_lower) of one
    closed contour; see the module docstring for the formulas."""
    a = np.asarray(contour, float)
    area, centroid = _shoelace_area_centroid(a)
    width = np.maximum(a[:, 0].max() - a[:, 0].min(), _WIDTH_EPS)
    kappa = (a[:, 1].max() - a[:, 1].min()) / width
    half_width = 0.5 * width
    delta_u = (centroid[0] - a[np.argmax(a[:, 1]), 0]) / half_width
    delta_l = (centroid[0] - a[np.argmin(a[:, 1]), 0]) / half_width
    return abs(area), centroid, kappa, delta_u, delta_l


def native_contour_metrics(pred, true):
    """Bidirectional contour-distance and geometry-error metrics per slice.

    ``pred`` / ``true`` are (B, N, 2) closed R-Z polylines in metres (a
    single (N, 2) contour is treated as B = 1); the two contours of a slice
    may differ in vertex count and winding. Returns a dict of (B,) float
    arrays keyed exactly ``mean_symmetric_mm, p95_mm, chamfer_rms_mm,
    hausdorff_mm, area_abs_m2, centroid_mm, elongation_abs,
    triangularity_upper_abs, triangularity_lower_abs`` (see the module
    docstring for each definition).
    """
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    if pred.ndim == 2:
        pred = pred[None]
    if true.ndim == 2:
        true = true[None]
    if len(pred) != len(true):
        raise ValueError(
            f"pred/true batch mismatch: {len(pred)} vs {len(true)}")
    keys = ("mean_symmetric_mm", "p95_mm", "chamfer_rms_mm", "hausdorff_mm",
            "area_abs_m2", "centroid_mm", "elongation_abs",
            "triangularity_upper_abs", "triangularity_lower_abs")
    out = {k: np.zeros(len(pred)) for k in keys}
    for i, (pc, tc) in enumerate(zip(pred, true)):
        d_pt = point_to_closed_polyline(pc, tc)
        d_tp = point_to_closed_polyline(tc, pc)
        combined = np.concatenate([d_pt, d_tp])
        out["mean_symmetric_mm"][i] = 500.0 * (d_pt.mean() + d_tp.mean())
        out["p95_mm"][i] = 1000.0 * np.percentile(combined, 95)
        out["chamfer_rms_mm"][i] = 1000.0 * np.sqrt(
            0.5 * ((d_pt ** 2).mean() + (d_tp ** 2).mean()))
        out["hausdorff_mm"][i] = 1000.0 * combined.max()
        (area_p, c_p, kap_p, du_p, dl_p) = _contour_geometry(pc)
        (area_t, c_t, kap_t, du_t, dl_t) = _contour_geometry(tc)
        out["area_abs_m2"][i] = abs(area_p - area_t)
        out["centroid_mm"][i] = 1000.0 * np.linalg.norm(c_p - c_t)
        out["elongation_abs"][i] = abs(kap_p - kap_t)
        out["triangularity_upper_abs"][i] = abs(du_p - du_t)
        out["triangularity_lower_abs"][i] = abs(dl_p - dl_t)
    return out
