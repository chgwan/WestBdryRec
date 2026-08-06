# -*- coding: utf-8 -*-
"""Saved predictions must be 34 wide and in metres, not standardized units."""
import importlib.util
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.target import N_OUT, destandardize  # noqa: E402


def _load_train_dcs():
    spec = importlib.util.spec_from_file_location(
        "train_dcs", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "train_dcs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_destandardize_is_the_inverse_used_by_predict():
    rng = np.random.default_rng(0)
    mean = np.r_[np.full(32, 0.5), 2.44, -0.02]
    std = np.r_[np.full(32, 0.05), 0.06, 0.03]
    metres = rng.normal(mean, std, size=(7, N_OUT))
    z = (metres - mean) / std
    assert np.allclose(destandardize(z, mean, std), metres)


def test_predict_helpers_reference_target_stats():
    """Every predict path must de-standardize; a missing call is a silent unit error."""
    mod = _load_train_dcs()
    src = pathlib.Path(mod.__file__).read_text()
    for fn in ("_pred_m0", "_pred_m1", "_pred_m2"):
        body = src.split(f"def {fn}(")[1].split("\ndef ")[0]
        assert "destandardize" in body, f"{fn} does not de-standardize its output"
        assert "tgt_mean" in body and "tgt_std" in body, f"{fn} ignores the target stats"
