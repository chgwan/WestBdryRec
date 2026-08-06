# -*- coding: utf-8 -*-
"""Unit tests for the V2 uniform-500 Hz grid helpers (synthetic arrays).

The grid is generated arithmetically rather than taken from data, so these are
pure-function tests in the style of test_merge_resample.py -- no HDF5 fixtures.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data.merge_dcs_bdry import native_gap_on_grid, uniform_grid  # noqa: E402

DT = 0.002
MODAL = 0.002048          # the native GMAG_BND cadence


def _native(t0, t1, step=MODAL):
    """A native reconstruction axis at the modal cadence."""
    return np.arange(t0, t1, step)


def test_grid_is_the_exact_lattice():
    bnd = _native(-30.0, 20.0)
    dcs = np.arange(-29.5, 12.6, 0.001)
    grid, info = uniform_grid(bnd, dcs)
    k = np.arange(info["grid_k0"], info["grid_k0"] + grid.size)
    assert np.array_equal(grid, k * DT), "grid must be exactly k*dt"
    assert grid.size == info["grid_n"]
    assert np.all(np.diff(grid) > 0)
    spread = np.diff(grid).max() - np.diff(grid).min()
    assert spread <= 1e-12, f"float64 lattice spread {spread} exceeds one ULP"
    assert info["grid_hz"] == 500.0 and info["grid_dt"] == DT


def test_grid_starts_at_ignitron_and_stays_inside_both_spans():
    bnd = _native(-30.0, 20.0)
    dcs = np.arange(-29.5, 12.6, 0.001)
    grid, info = uniform_grid(bnd, dcs)
    assert info["grid_k0"] == 0 and grid[0] == 0.0, "t<0 is S0's job, grid starts at 0"
    assert grid[-1] <= dcs.max() + 1e-12
    assert grid[-1] <= bnd.max() + 1e-12


def test_grid_respects_a_late_starting_source():
    """ceil() must not place a grid point before a late source's first sample."""
    bnd = _native(1.0, 5.0)
    dcs = np.arange(1.0, 5.0, 0.001)
    grid, info = uniform_grid(bnd, dcs)
    assert grid[0] >= 1.0 and info["grid_k0"] == 500     # 1.0 s / 2 ms


def test_clip_terminates_at_a_long_dropout():
    """A 33 ms idle gap ends the window; the audit records what was dropped."""
    bnd = np.r_[_native(0.0, 5.0), np.arange(5.033, 8.0, 0.032768)]
    dcs = np.arange(-29.5, 8.0, 0.001)
    grid, info = uniform_grid(bnd, dcs)
    assert grid[-1] <= 5.0, "grid must not enter the 30 Hz region"
    assert info["clip_t_end"] <= 5.0
    assert info["clip_dropped_n"] > 50, "the slow-tier samples are recorded as dropped"
    assert info["clip_dropped_s"] > 2.0


def test_isolated_hiccup_does_not_clip():
    """Regression: a 4.1 ms first-gap rule cut shot 57295 to 45 grid points.

    57295 carries one isolated 7.179 ms interval at t=0.089 s, well inside its
    fast-acquisition window. A 16 ms threshold plus longest-run must ride over it.
    """
    bnd = np.r_[_native(0.0, 0.089), np.array([0.0962]), _native(0.0982, 18.0)]
    dcs = np.arange(-29.5, 18.0, 0.001)
    grid, info = uniform_grid(bnd, dcs)
    assert grid.size > 8000, f"hiccup truncated the shot to {grid.size} points"
    assert info["clip_t_end"] > 17.0


def test_longest_run_wins_not_first_run():
    """Split by one long gap: the longer side is kept, not the earlier one."""
    bnd = np.r_[_native(0.0, 1.0), _native(4.0, 18.0)]
    dcs = np.arange(-29.5, 18.0, 0.001)
    grid, info = uniform_grid(bnd, dcs)
    assert info["clip_t_start"] >= 4.0 and info["clip_t_end"] > 17.0


def test_empty_grid_when_nothing_usable():
    grid, info = uniform_grid(_native(-30.0, -20.0), np.arange(-30.0, -20.0, 0.001))
    assert grid.size == 0, "an all-negative record yields no grid; caller drops the shot"
    grid, _ = uniform_grid(np.array([1.0]), np.arange(0.0, 2.0, 0.001))
    assert grid.size == 0


def test_gap_is_the_bracketing_native_spacing():
    """Half-open convention: a grid point in [ts[j], ts[j+1]) reports that interval."""
    ts = np.array([0.0, 0.002, 0.004, 0.040])       # a 36 ms dropout after 0.004
    grid = np.array([0.001, 0.003, 0.010, 0.030])
    gap = native_gap_on_grid(grid, ts)
    assert np.allclose(gap, [2.0, 2.0, 36.0, 36.0])


def test_gap_reads_the_modal_cadence_inside_a_normal_run():
    bnd = _native(0.0, 4.0)
    grid = np.arange(0, 1500) * DT
    gap = native_gap_on_grid(grid, bnd)
    assert np.allclose(gap, MODAL * 1e3), "normal cadence must read ~2.048 ms"


def test_gap_handles_a_degenerate_axis():
    gap = native_gap_on_grid(np.array([0.0, 0.002]), np.array([1.0]))
    assert gap.shape == (2,) and np.isnan(gap).all()


def test_gap_is_finite_at_and_beyond_the_native_edges():
    ts = np.array([0.0, 0.002, 0.004])
    gap = native_gap_on_grid(np.array([-0.5, 0.0, 0.004, 9.0]), ts)
    assert np.all(np.isfinite(gap)), "edge points clamp to the nearest interval"


# ---------------------------------------------------------------------------
# Integration: time_base='uniform' wiring (Task 3). These monkeypatch
# ``read_dcs_mat`` rather than building a synthetic ``.mat``, so the uniform
# branch is exercised end-to-end without scipy fixtures.
# ---------------------------------------------------------------------------
import h5py
import pytest

from src.data import merge_dcs_bdry as M  # noqa: E402


def _write_gmag(path, t0=-30.0, t1=18.0, step=MODAL):
    """A minimal GMagH5: a linearly growing ring so interpolation is predictable."""
    t = np.arange(t0, t1, step)
    n = t.size
    bnd = np.zeros((64, n))
    geom = np.zeros((16, n))
    th = np.arange(32) / 32 * 2 * np.pi
    for j in range(n):
        r = 0.40 + 0.01 * t[j]              # radius grows 1 cm per second
        bnd[0::2, j] = 2.4 + r * np.cos(th)
        bnd[1::2, j] = 0.05 + r * np.sin(th)
        geom[0, j], geom[1, j] = 2400.0, 50.0        # mm
    with h5py.File(path, "w") as hf:
        hf.attrs["time_reference"] = "ignitron"
        hf.create_dataset("targets/GMAG_BND", data=bnd)
        hf.create_dataset("targets/GMAG_BND_time", data=t)
        hf.create_dataset("inputs/GMAG_GEOM", data=geom)
        hf.create_dataset("inputs/GMAG_GEOM_time", data=t)
    return t


def _merge(tmp_path, monkeypatch, dcs_t1=12.6):
    bnd_dir = tmp_path / "gmag"; bnd_dir.mkdir()
    dcs_dir = tmp_path / "dcs"; dcs_dir.mkdir()
    out_dir = tmp_path / "merged"; out_dir.mkdir()
    _write_gmag(bnd_dir / "999.h5")
    (dcs_dir / "DCS_archive_999.mat").write_text("stub")   # existence check only

    dcs_t = np.arange(-29.5, dcs_t1, 0.001)
    ramp = np.linspace(0.0, 100.0, dcs_t.size)
    monkeypatch.setattr(M, "read_dcs_mat",
                        lambda p: (dcs_t, {"Ip_scope": (ramp, 2 * ramp)}))
    shot, ok, err = M._merge_one((999, bnd_dir, dcs_dir, out_dir,
                                  "uniform", 500.0, 16.0))
    assert ok, err
    return out_dir / "999.h5", dcs_t, ramp


def test_merged_uniform_attrs_and_lattice(tmp_path, monkeypatch):
    path, _, _ = _merge(tmp_path, monkeypatch)
    with h5py.File(path) as hf:
        t = hf["time"][:]
        assert hf.attrs["grid_source"] == "uniform500"
        assert hf.attrs["time_base"] == "ignitron"
        assert hf.attrs["grid_hz"] == 500.0
        k0 = int(hf.attrs["grid_k0"])
        assert np.array_equal(t, (np.arange(k0, k0 + t.size)) * DT)
        assert "src_gap_ms" in hf and hf["src_gap_ms"].shape == t.shape


def test_target_is_interpolated_to_the_exact_blend(tmp_path, monkeypatch):
    """V2's defining behaviour. V1's suite pins the target bit-identical to the raw
    reconstruction; V2 must pin the opposite -- a real linear blend."""
    path, _, _ = _merge(tmp_path, monkeypatch)
    with h5py.File(path) as hf:
        t = hf["time"][:]
        bnd = hf["targets/GMAG_BND"][:]
    i = t.size // 2
    # the fixture's outboard vertex is R = 2.4 + (0.40 + 0.01 t), exactly linear in t,
    # so a correct linear interpolation reproduces it to float precision at any t.
    assert np.isclose(bnd[0, i], 2.4 + 0.40 + 0.01 * t[i], atol=1e-9)
    assert not np.isclose(np.ptp(np.diff(t)), MODAL), "the axis must not be the native one"


def test_dcs_scope_is_interpolated_once_from_its_own_axis(tmp_path, monkeypatch):
    """No 1 kHz -> 488 Hz -> 500 Hz double hop: merge reads the original sources."""
    path, dcs_t, ramp = _merge(tmp_path, monkeypatch)
    with h5py.File(path) as hf:
        t = hf["time"][:]
        got = hf["dcs/Ip_scope/ref"][:]
    assert np.allclose(got, np.interp(t, dcs_t, ramp), atol=1e-9, equal_nan=True)


def test_uniform_shot_dropped_when_no_window(tmp_path, monkeypatch):
    bnd_dir = tmp_path / "g2"; bnd_dir.mkdir()
    dcs_dir = tmp_path / "d2"; dcs_dir.mkdir()
    out_dir = tmp_path / "m2"; out_dir.mkdir()
    _write_gmag(bnd_dir / "998.h5", t0=-30.0, t1=-20.0)     # entirely pre-ignitron
    (dcs_dir / "DCS_archive_998.mat").write_text("stub")
    dcs_t = np.arange(-29.5, -20.0, 0.001)
    monkeypatch.setattr(M, "read_dcs_mat",
                        lambda p: (dcs_t, {"Ip_scope": (dcs_t * 0, dcs_t * 0)}))
    shot, ok, err = M._merge_one((998, bnd_dir, dcs_dir, out_dir,
                                  "uniform", 500.0, 16.0))
    assert not ok and "uniform window" in err
    assert not (out_dir / "998.h5").exists(), "a failed merge leaves no partial file"


def test_existing_time_bases_still_work(tmp_path, monkeypatch):
    """The 7-tuple arity change must not break 'gmag' or 'dcs'."""
    for base in ("gmag", "dcs"):
        bnd_dir = tmp_path / f"g_{base}"; bnd_dir.mkdir()
        dcs_dir = tmp_path / f"d_{base}"; dcs_dir.mkdir()
        out_dir = tmp_path / f"m_{base}"; out_dir.mkdir()
        _write_gmag(bnd_dir / "997.h5")
        (dcs_dir / "DCS_archive_997.mat").write_text("stub")
        dcs_t = np.arange(-29.5, 12.6, 0.001)
        monkeypatch.setattr(M, "read_dcs_mat",
                            lambda p: (dcs_t, {"Ip_scope": (dcs_t * 0, dcs_t * 0)}))
        shot, ok, err = M._merge_one((997, bnd_dir, dcs_dir, out_dir, base, 500.0, 16.0))
        assert ok, f"{base}: {err}"
        with h5py.File(out_dir / "997.h5") as hf:
            assert hf.attrs["grid_source"] == base
            assert "src_gap_ms" not in hf, "only the uniform base writes provenance"


def test_proj_config_exposes_the_v2_dirs():
    from src.proj_config import get_proj_config
    cfg = get_proj_config()
    assert cfg.mergedh5_uni500_dir.name == "MergedH5Uni500"
    assert cfg.npzuni500_dir.name == "NpzUni500"


def _run():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} OK")
    print("OK")


if __name__ == "__main__":
    _run()
