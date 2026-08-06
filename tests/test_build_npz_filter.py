# -*- coding: utf-8 -*-
"""Tests for the quality-filter wiring in build_npz (synthetic MergedH5 fixture).

Covers the contract the filters impose on the built NPZ: which slices are valid,
that ``Y`` is projected about each slice's own stored center, and that no invalid
slice carries a fabricated target.
"""
import json
import pathlib
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data.build_npz import (  # noqa: E402
    N_ANGLES, _build_one, keep_span, read_center, run, theta_grid,
)
from src.data.filter import REGISTRY  # noqa: E402

N = 32


def _ring(r, cR, cZ, n=N):
    th = np.arange(n) / n * 2 * np.pi
    return cR + r * np.cos(th), cZ + r * np.sin(th)


def _write_merged(path, nt=12, bad_at=(), center=(2.4, 0.05), t0=0.0, dt=0.01):
    """A minimal MergedH5 with one clean boundary per sample, plus injected defects.

    ``bad_at`` maps index -> defect name, so a test can place exactly one failure
    mode at a known slice.
    """
    bnd = np.zeros((64, nt))
    geom = np.zeros((16, nt))
    for j in range(nt):
        R, Z = _ring(0.45, *center)
        defect = dict(bad_at).get(j)
        if defect == "zero":
            R, Z = np.zeros(N), np.zeros(N)
        elif defect == "outside":
            R, Z = _ring(0.45, 3.15, 0.05)          # pushed past the outboard wall
        elif defect == "notch":
            R, Z = R.copy(), Z.copy()
            R[7] = center[0] + 0.45 * np.cos(7 / N * 2 * np.pi) * 0.5
            Z[7] = center[1] + 0.45 * np.sin(7 / N * 2 * np.pi) * 0.5
        bnd[0::2, j] = R
        bnd[1::2, j] = Z
        if defect == "sentinel":
            geom[0, j], geom[1, j] = 0.0, 0.0      # GMAG_GEOM sentinel
        else:
            geom[0, j], geom[1, j] = center[0] * 1e3, center[1] * 1e3   # mm
    with h5py.File(path, "w") as hf:
        hf["time"] = t0 + dt * np.arange(nt)
        hf.create_dataset("targets/GMAG_BND", data=bnd)
        hf.create_dataset("inputs/GMAG_GEOM", data=geom)
    return bnd, geom


def _build(tmp_path, **kw):
    merged = tmp_path / "merged"
    merged.mkdir(exist_ok=True)
    out = tmp_path / "npz"
    out.mkdir(exist_ok=True)
    _write_merged(merged / "999.h5", **kw)
    shot, ok, payload = _build_one(
        (999, merged, out, [], theta_grid(N_ANGLES)))
    return ok, payload, (np.load(out / "999.npz") if ok else None)


# --------------------------------------------------------------------- helpers ---
def test_keep_span_trims_to_survivors():
    keep = np.array([False, False, True, False, True, False])
    assert keep_span(keep) == (2, 4)
    assert keep_span(np.zeros(5, bool)) is None


def test_read_center_converts_mm_and_handles_absence(tmp_path):
    p = tmp_path / "a.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("inputs/GMAG_GEOM", data=np.full((16, 3), 2400.0))
    with h5py.File(p, "r") as hf:
        c = read_center(hf, 3)
    assert c.shape == (2, 3) and np.allclose(c, 2.4)      # mm -> m

    with h5py.File(p, "w") as hf:
        hf.create_dataset("other", data=[1])
    with h5py.File(p, "r") as hf:
        c = read_center(hf, 3)
    assert c.shape == (2, 3) and np.isnan(c).all()        # absent -> fails S3, no default


# ----------------------------------------------------------------- the contract ---
def test_clean_shot_is_fully_valid(tmp_path):
    ok, payload, d = _build(tmp_path)
    assert ok, payload
    assert d["valid"].all()
    assert d["Y"].shape == (12, N_ANGLES)
    assert d["center"].shape == (12, 2)
    assert np.allclose(d["center"][:, 0], 2.4) and np.allclose(d["center"][:, 1], 0.05)
    assert (d["fail"] == 0).all() and (d["only"] == 0).all()
    # a ring about its own center has constant r(theta)
    assert np.allclose(d["Y"], 0.45, atol=1e-6)


def test_invalid_slices_are_nan_not_fabricated(tmp_path):
    ok, _payload, d = _build(tmp_path, bad_at={5: "notch"})
    assert ok
    assert not d["valid"][5] and d["valid"].sum() == 11
    assert np.isnan(d["Y"][5]).all(), "a rejected slice must not carry a target"
    assert np.isfinite(d["Y"][d["valid"]]).all()
    s5 = next(i for i, c in enumerate(REGISTRY, start=1) if c.code == "s5")
    assert d["fail"][5] == s5 and d["only"][5] == s5


def test_each_defect_is_attributed_to_its_criterion(tmp_path):
    idx = {3: "zero", 5: "outside", 7: "sentinel", 9: "notch"}
    ok, _payload, d = _build(tmp_path, bad_at=idx)
    assert ok
    code = {c.code: i for i, c in enumerate(REGISTRY, start=1)}
    assert d["fail"][3] == code["s1"]                    # all-zero vertices
    assert d["fail"][5] == code["s2"]                    # outside the vessel
    assert d["fail"][7] == code["s3"]                    # sentinel center
    assert d["fail"][9] == code["s5"]                    # notched
    assert d["valid"].sum() == 12 - len(idx)


def test_span_is_trimmed_to_kept_slices(tmp_path):
    # first two and last two slices unusable -> span shrinks to the middle
    ok, payload, d = _build(tmp_path, bad_at={0: "zero", 1: "zero",
                                             10: "zero", 11: "zero"})
    assert ok
    assert payload["n_slices"] == 8
    assert d["valid"].all(), "everything inside the trimmed span survives here"


def test_negative_time_is_excluded(tmp_path):
    ok, payload, d = _build(tmp_path, nt=12, t0=-0.05, dt=0.01)
    assert ok
    assert (d["time"] >= 0).all(), "S0 must keep the span clear of pre-ignitron slices"
    assert payload["n_slices"] == 7                      # t = 0.00 .. 0.06


def test_flags_agree_with_valid(tmp_path):
    ok, _payload, d = _build(tmp_path, bad_at={4: "notch"})
    assert ok
    all_pass = (1 << len(REGISTRY)) - 1
    assert ((d["flags"] == all_pass) == d["valid"]).all()


def test_shot_with_no_usable_slice_is_rejected(tmp_path):
    ok, payload, _ = _build(tmp_path, bad_at={j: "zero" for j in range(12)})
    assert not ok and "quality filters" in payload


# ----------------------------------------------------------------------- meta ---
def test_meta_records_filters_and_per_slice_origin(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    _write_merged(merged / "999.h5")
    npz = tmp_path / "npz"
    cfg = tmp_path / "base.yml"
    cfg.write_text("nodes:\n  dummy: []\ndata:\n  input_list: []\n")
    n_ok, n_fail = run(merged_dir=merged, npz_dir=npz, config_path=cfg, workers=1)
    assert (n_ok, n_fail) == (1, 0)
    meta = json.loads((npz / "meta.json").read_text())
    assert meta["origin"] == "per_slice_gmag_geom"
    assert "origin_legacy_fixed" not in meta, "the fixed origin is gone for good"
    assert meta["target"]["center"].startswith("per-slice")
    codes = [c["code"] for c in meta["filters"]["criteria"]]
    assert codes == [c.code for c in REGISTRY]
    assert "convex_min" in meta["filters"]["thresholds"]
    assert set(meta["filters"]["rejected_by_first_failure"]) == set(codes)
    for key in ("center", "fail", "only", "flags"):
        assert key in meta["arrays"]
