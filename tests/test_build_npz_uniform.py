# -*- coding: utf-8 -*-
"""The V2 (uniform 500 Hz) build contract.

build_npz is shared with NpzOrigin/NpzGeom, so everything V2 adds must be
conditional on src_gap_ms being present. tests/test_build_npz_filter.py -- whose
fixture has no src_gap_ms -- is the regression guard for that; do not edit it.
"""
import json
import pathlib
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data.build_npz import (  # noqa: E402
    FAB_GAP_MS, N_ANGLES, PE_PAIRS, _build_one, theta_grid, time_encoding,
)

N = 32
DT = 0.002


def _write_uniform_merged(path, nt=600, gap_at=None):
    """A MergedH5Uni500 fixture: clean ring per slice on an exact 2 ms lattice.

    ``gap_at`` marks one slice as dropout-interpolated by widening its src_gap_ms.
    """
    t = np.arange(nt) * DT
    bnd = np.zeros((64, nt))
    geom = np.zeros((16, nt))
    th = np.arange(N) / N * 2 * np.pi
    for j in range(nt):
        bnd[0::2, j] = 2.4 + 0.45 * np.cos(th)
        bnd[1::2, j] = 0.05 + 0.45 * np.sin(th)
        geom[0, j], geom[1, j] = 2400.0, 50.0
    gap = np.full(nt, 2.048)
    if gap_at is not None:
        gap[gap_at] = 33.0
    with h5py.File(path, "w") as hf:
        hf.attrs["time_base"] = "ignitron"
        hf.attrs["grid_source"] = "uniform500"
        hf.attrs["grid_hz"] = 500.0
        hf.attrs["grid_dt"] = DT
        hf.attrs["grid_k0"] = 0
        hf.attrs["clip_gap_ms"] = 16.0
        hf.attrs["clip_dropped_n"] = 0
        hf.attrs["clip_dropped_s"] = 0.0
        hf["time"] = t
        hf["src_gap_ms"] = gap
        hf.create_dataset("targets/GMAG_BND", data=bnd)
        hf.create_dataset("inputs/GMAG_GEOM", data=geom)
    return t, gap


def _build(tmp_path, **kw):
    merged = tmp_path / "merged"; merged.mkdir(exist_ok=True)
    out = tmp_path / "npz"; out.mkdir(exist_ok=True)
    _write_uniform_merged(merged / "999.h5", **kw)
    shot, ok, payload = _build_one((999, merged, out, [], theta_grid(N_ANGLES)))
    return ok, payload, (np.load(out / "999.npz") if ok else None)


def test_src_gap_ms_reaches_the_npz(tmp_path):
    ok, payload, d = _build(tmp_path, gap_at=300)
    assert ok, payload
    assert "src_gap_ms" in d.files
    assert d["src_gap_ms"].shape == d["time"].shape
    assert d["src_gap_ms"].dtype == np.float32
    assert np.isclose(d["src_gap_ms"][300], 33.0)


def test_fabrication_census_counts_valid_wide_gap_slices(tmp_path):
    ok, payload, d = _build(tmp_path, gap_at=300)
    assert ok
    assert payload["n_fabricated"] == 1, "one valid slice sits across a 33 ms gap"
    assert FAB_GAP_MS == pytest.approx(3.072)


def test_time_is_the_exact_lattice_in_the_npz(tmp_path):
    ok, _, d = _build(tmp_path)
    t = d["time"].astype(float)
    dt = np.diff(t)
    assert np.all(np.abs(dt - DT) < 1e-5), "float32 rounding only"


def test_pe_columns_come_from_the_uniform_time(tmp_path):
    """X's last 2*PE_PAIRS columns must be time_encoding of the lattice."""
    merged = tmp_path / "m"; merged.mkdir()
    out = tmp_path / "n"; out.mkdir()
    _write_uniform_merged(merged / "999.h5")
    shot, ok, payload = _build_one((999, merged, out, [], theta_grid(N_ANGLES)))
    assert ok, payload
    d = np.load(out / "999.npz")
    assert d["X"].shape[1] == 2 * PE_PAIRS, "no scope channels in this fixture"
    assert np.allclose(d["X"], time_encoding(d["time"].astype(float)), atol=1e-6)


def test_schema_matches_npzgeom_plus_one_array(tmp_path):
    ok, _, d = _build(tmp_path)
    for k in ("X", "Y", "S", "bnd_RZ", "time", "valid", "center",
              "fail", "only", "flags", "src_gap_ms"):
        assert k in d.files, f"missing {k}"
    assert d["valid"].all() and np.isfinite(d["Y"]).all()


def test_grid_meta_block_is_written(tmp_path):
    from src.data.build_npz import run
    merged = tmp_path / "merged"; merged.mkdir()
    out = tmp_path / "npz"
    _write_uniform_merged(merged / "999.h5")
    n_ok, n_fail = run(merged_dir=merged, npz_dir=out, workers=1)
    assert (n_ok, n_fail) == (1, 0)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["grid"]["source"] == ["uniform500"]
    assert meta["grid"]["hz"] == [500.0]
    assert meta["grid"]["fabricated_gap_threshold_ms"] == pytest.approx(3.072)
    assert meta["arrays"]["src_gap_ms"] == ["nt"]


GEOM = pathlib.Path("ProjDB/datasets/NpzGeom")
UNI = pathlib.Path("ProjDB/datasets/NpzUni500")


@pytest.mark.skipif(not (GEOM.exists() and UNI.exists()),
                    reason="needs both built datasets")
def test_uniform_agrees_with_native_at_coincident_times():
    """Catches phase and off-by-one errors: where a V2 grid point lands on a native
    reconstruction timestamp, its Y must match V1's to within resampling noise."""
    shot = sorted(int(p.stem) for p in UNI.glob("*.npz"))[0]
    a = np.load(GEOM / f"{shot}.npz")
    b = np.load(UNI / f"{shot}.npz")
    ta, tb = a["time"].astype(float), b["time"].astype(float)
    j = np.searchsorted(ta, tb)
    j = np.clip(j, 0, ta.size - 1)
    close = np.abs(ta[j] - tb) < 1e-4                    # coincident to 0.1 ms
    both = close & b["valid"] & a["valid"][j]
    assert both.sum() > 100, f"only {both.sum()} coincident valid slices"
    err = np.abs(b["Y"][both] - a["Y"][j[both]])
    assert np.median(err) < 5e-3, f"median |dY| = {np.median(err) * 1e3:.2f} mm"


def _run():
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as d:
                fn(pathlib.Path(d)) if fn.__code__.co_argcount else fn()
            print(f"  {name} OK")
    print("OK")


if __name__ == "__main__":
    _run()
