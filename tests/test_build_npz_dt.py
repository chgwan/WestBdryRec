# -*- coding: utf-8 -*-
"""The optional Δt (sample-spacing) input column for the dt ablation."""
import pathlib
import sys

import h5py
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data.build_npz import N_ANGLES, _build_one, theta_grid, time_gap_column  # noqa: E402

N = 32


def _ring(r, cR, cZ, n=N):
    th = np.arange(n) / n * 2 * np.pi
    return cR + r * np.cos(th), cZ + r * np.sin(th)


def _write_merged(path, nt=12, dt=0.01, t0=0.0):
    bnd = np.zeros((64, nt))
    geom = np.zeros((16, nt))
    for j in range(nt):
        R, Z = _ring(0.45, 2.4, 0.05)
        bnd[0::2, j] = R
        bnd[1::2, j] = Z
        geom[0, j], geom[1, j] = 2400.0, 50.0
    t = t0 + dt * np.arange(nt)         # uniform, but Δt must still be recorded
    with h5py.File(path, "w") as hf:
        hf["time"] = t
        hf.create_dataset("targets/GMAG_BND", data=bnd)
        hf.create_dataset("inputs/GMAG_GEOM", data=geom)
    return t


def test_time_gap_column_matches_diff():
    t = np.array([0.0, 0.002, 0.005, 0.005, 0.038])   # gaps 2,3,0,33 ms; a repeat + a dropout
    g = time_gap_column(t)
    assert g.shape == (5, 1)
    expected = np.array([0.002, 0.002, 0.003, 0.000, 0.033])
    assert np.allclose(g.ravel(), expected)


def test_dt_column_is_optional_and_last(tmp_path):
    """add_dt=False leaves X as before; add_dt=True appends dt_gap_s as the last column."""
    from src.data.build_npz import run
    merged = tmp_path / "m"; merged.mkdir()
    out_off = tmp_path / "off"; out_dt = tmp_path / "dt"
    _write_merged(merged / "999.h5", nt=12)
    n_ok, n_fail = run(merged_dir=merged, npz_dir=out_off, workers=1, add_dt=False)
    assert (n_ok, n_fail) == (1, 0)
    n_ok, n_fail = run(merged_dir=merged, npz_dir=out_dt, workers=1, add_dt=True)
    assert (n_ok, n_fail) == (1, 0)
    off = np.load(out_off / "999.npz"); dt = np.load(out_dt / "999.npz")
    # actuators identical, Y identical; dt build has one extra X column
    assert off["X"].shape[1] + 1 == dt["X"].shape[1]
    assert np.array_equal(off["Y"], dt["Y"])
    # the dt column == time_gap_column(time)
    t = dt["time"].astype(float)
    assert np.allclose(dt["X"][:, -1], time_gap_column(t).ravel())
    # meta records the dt_gap_s node so node_col_map can select it
    import json
    meta = json.loads((out_dt / "meta.json").read_text())
    nodes = [n for g in meta["inputs"] for n in g["nodes"]]
    assert "dt_gap_s" in nodes
    off_meta = json.loads((out_off / "meta.json").read_text())
    off_nodes = [n for g in off_meta["inputs"] for n in g["nodes"]]
    assert "dt_gap_s" not in off_nodes


def _run():
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            args = fn.__code__.co_argcount
            with tempfile.TemporaryDirectory() as d:
                (fn(pathlib.Path(d)) if args else fn())
            print(f"  {name} OK")
    print("OK")


if __name__ == "__main__":
    _run()
