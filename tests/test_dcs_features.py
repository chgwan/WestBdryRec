# -*- coding: utf-8 -*-
"""Tests for DCS feature config + meta mapping."""
import pathlib, sys
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


def _run():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} OK")
    print("OK")


if __name__ == "__main__":
    _run()
