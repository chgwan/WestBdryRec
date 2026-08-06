# -*- coding: utf-8 -*-
"""Checks that must hold on the real dataset before six trainings are worth starting."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.axis_frame import reconstruct_absolute  # noqa: E402
from src.ml.dcs_features import load_dcs_config, load_meta, node_col_map  # noqa: E402
from src.ml.target import N_OUT, load_target, target_mean_std  # noqa: E402
from src.proj_config import get_proj_config  # noqa: E402

CFG = get_proj_config()
pytestmark = pytest.mark.skipif(not CFG.npzgeom_dir.exists(), reason="NpzGeom not built")


def _some_shots(n=6):
    return sorted(int(p.stem) for p in CFG.npzgeom_dir.glob("*.npz"))[:n]


def test_target_is_34_wide_and_centre_is_absolute():
    for s in _some_shots(3):
        T, finite = load_target(CFG.npzgeom_dir / f"{s}.npz")
        assert T.shape[1] == N_OUT
        assert finite.any()
        assert 2.0 < np.nanmedian(T[finite, 32]) < 3.0, "Rgeom must be absolute metres"


def test_truth_reconstruction_is_well_formed():
    """Truth boundary rebuilt from (centre, rho) on the uniform theta grid.

    There is no longer a representation floor to probe: the absolute metric
    compares pred-vs-truth reconstructions that are BOTH built on the uniform
    grid, i.e. like-for-like. The former round-trip-against-bnd_RZ check
    compared a uniform-grid reconstruction against native-angle raw vertices
    and was the bug itself, so it is retired. This test just guards against
    garbled truth columns (NaNs, wrong units) by checking the reconstruction
    is finite and inside the WEST vessel envelope.
    """
    theta = np.deg2rad(np.asarray(load_meta(CFG.npzgeom_dir)["theta_deg"], float))
    for s in _some_shots(4):
        d = np.load(CFG.npzgeom_dir / f"{s}.npz")
        T, finite = load_target(CFG.npzgeom_dir / f"{s}.npz")
        v = finite & d["valid"].astype(bool)
        R, Z = reconstruct_absolute(T[v, 32], T[v, 33], T[v, :32], theta)
        assert np.isfinite(R).all() and np.isfinite(Z).all(), f"shot {s}: non-finite truth recon"
        assert 1.8 < R.min() and R.max() < 3.3, f"shot {s}: R outside vessel envelope"
        assert np.abs(Z).max() < 1.2, f"shot {s}: |Z| outside vessel envelope"


def test_pe_channels_resolve_for_the_headline_config():
    meta = load_meta(CFG.npzgeom_dir)
    ncm = node_col_map(meta)
    mc = load_dcs_config("configs/dcs_model_geom.yml")
    missing = [c["node"] for c in mc["channels"] if c["node"] not in ncm]
    assert not missing, f"config lists channels absent from the dataset: {missing}"
    assert sum(1 for c in mc["channels"] if c["kind"] == "pe") == 10


def test_target_stats_come_from_train_shots_only():
    from src.ml import bench
    train, _val, test = bench.load_filtered_split(CFG.npzgeom_dir)
    mean, std = target_mean_std(CFG.npzgeom_dir, train[:20])
    assert mean.shape == (N_OUT,) and (std > 0).all()
    assert set(train).isdisjoint(test), "split must not overlap"
