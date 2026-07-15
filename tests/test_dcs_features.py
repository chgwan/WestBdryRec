# -*- coding: utf-8 -*-
"""Tests for DCS feature config + meta mapping."""
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.ml.dcs_features import load_dcs_config, strict_channels, node_col_map  # noqa: E402


def test_config_has_18_strict_channels():
    cfg = load_dcs_config()
    ch = strict_channels(cfg)
    assert len(ch) == 18
    kinds = {kd for _, kd, _ in ch}
    assert {"lh", "lh_phase", "ic", "ip", "pf"} <= kinds


def test_node_col_map_matches_layout():
    meta = {"inputs": [
        {"input_name": "LHW_real", "cols": [0, 4], "nodes": ["PowLH1_scope", "PowLH2_scope", "PhaLH1_scope", "PhaLH2_scope"]},
        {"input_name": "PF_real", "cols": [12, 22], "nodes": ["IBb_scope", "IXh_scope"]},
    ]}
    ncm = node_col_map(meta)
    assert ncm["PowLH1_scope"] == 0 and ncm["PhaLH2_scope"] == 3 and ncm["IBb_scope"] == 12


def _write_npz(path, X, valid, time):
    np.savez(path, X=X.astype(np.float32),
             Y=np.zeros((X.shape[0], 32), np.float32),
             valid=valid.astype(bool), time=time.astype(np.float32),
             bnd_RZ=np.zeros((X.shape[0], 32, 2), np.float32), S=np.zeros((X.shape[0], 2), np.float32))


def test_read_snapshot_nan_policy_and_shape(tmp_path):
    from src.ml.dcs_features import read_snapshot, load_dcs_config
    cfg = load_dcs_config()
    ncm = {"PowLH1_scope": 0, "PowLH2_scope": 1, "PhaLH1_scope": 2, "PhaLH2_scope": 3,
           "PowIC1_scope": 7, "PowIC2_scope": 8, "PowIC3_scope": 9, "Ip_scope": 10,
           "IBb_scope": 12, "IDb_scope": 13, "IEb_scope": 14, "IFb_scope": 15,
           "IFh_scope": 16, "IEh_scope": 17, "IDh_scope": 18, "IBh_scope": 19,
           "IXb_scope": 20, "IXh_scope": 21}
    X = np.ones((6, 22))
    X[:, 0] = np.nan          # heating (LH power) NaN -> 0
    X[3, 12] = np.nan         # essential (PF) NaN on slice 3 -> drop slice
    _write_npz(tmp_path / "1.npz", X, np.ones(6, bool), np.arange(6, dtype=float))
    feats, mask = read_snapshot(tmp_path / "1.npz", cfg, ncm)
    n_chan = 18
    assert feats.shape == (6, n_chan + 4)          # +4 derived
    assert feats.shape[1] == 22
    assert np.isfinite(feats).all()
    assert mask.sum() == 5 and not mask[3]          # essential-NaN slice dropped
    assert abs(feats[0, 0]) < 1e-9                  # LH NaN -> 0


def test_read_series_shape_and_mask(tmp_path):
    from src.ml.dcs_features import read_series, load_dcs_config
    cfg = load_dcs_config()
    ncm = {"PowLH1_scope": 0, "PowLH2_scope": 1, "PhaLH1_scope": 2, "PhaLH2_scope": 3,
           "PowIC1_scope": 7, "PowIC2_scope": 8, "PowIC3_scope": 9, "Ip_scope": 10,
           "IBb_scope": 12, "IDb_scope": 13, "IEb_scope": 14, "IFb_scope": 15,
           "IFh_scope": 16, "IEh_scope": 17, "IDh_scope": 18, "IBh_scope": 19,
           "IXb_scope": 20, "IXh_scope": 21}
    X = np.ones((4, 22)); X[1, 12] = np.nan
    _write_npz(tmp_path / "2.npz", X, np.ones(4, bool), np.arange(4, dtype=float))
    A, mask = read_series(tmp_path / "2.npz", cfg, ncm)
    assert A.shape == (4, 18) and mask.sum() == 3 and not mask[1]


def _run():
    import inspect, tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as td:
                    fn(pathlib.Path(td))
            else:
                fn()
            print(f"  {name} OK")
    print("OK")


if __name__ == "__main__":
    _run()
