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


def _run():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} OK")
    print("OK")


if __name__ == "__main__":
    _run()
