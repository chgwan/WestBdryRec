# -*- coding: utf-8 -*-
"""34-column scoring: rho metrics stay baseline-comparable; centre reported separately."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.predictions import save_predictions  # noqa: E402
from src.ml.target import N_OUT, N_RHO  # noqa: E402


def _fixture(tmp_path, n_shots=3, nt=20, err_m=0.0):
    """Write NpzGeom-shaped shots plus predictions offset by ``err_m`` metres."""
    theta = np.arange(N_RHO) / N_RHO * 2 * np.pi
    rng = np.random.default_rng(0)
    shots, preds = [], {}
    for k in range(n_shots):
        s = 100 + k
        rho = 0.5 + 0.02 * np.sin(theta)[None, :] + rng.normal(0, 1e-3, (nt, N_RHO))
        C = np.column_stack([np.full(nt, 2.44), np.full(nt, -0.02)])
        R = C[:, :1] + rho * np.cos(theta)[None, :]
        Z = C[:, 1:2] + rho * np.sin(theta)[None, :]
        bnd = np.stack([R, Z], axis=-1).astype(np.float32)
        np.savez(tmp_path / f"{s}.npz", Y=rho.astype(np.float32),
                 center=C.astype(np.float32), bnd_RZ=bnd,
                 valid=np.ones(nt, bool), time=np.arange(nt, dtype=np.float32))
        preds[s] = np.concatenate([rho + err_m, C + err_m], axis=1).astype(np.float32)
        shots.append(s)
    (tmp_path / "meta.json").write_text(
        '{"shots": [' + ",".join(f'{{"shot": {s}}}' for s in shots) + '],'
        ' "theta_deg": ' + str(list(np.degrees(theta))) + '}')
    save_predictions(tmp_path / "p.npz", preds)
    return shots, theta


def test_perfect_prediction_scores_perfectly(tmp_path):
    from src.ml.score34 import score_dcs34
    shots, theta = _fixture(tmp_path, err_m=0.0)
    m = score_dcs34(tmp_path / "p.npz", tmp_path, shots, shots, theta)
    assert m["n_shots"] == len(shots)
    assert m["ccc"] > 0.999 and m["r2"] > 0.999
    assert m["rgeom_mae_mm"] < 1e-6 and m["zgeom_mae_mm"] < 1e-6
    # abs_bnd rebuilds (R,Z) from float32-stored centre+rho vs a float32-stored
    # bnd_RZ; the reconstruction carries ~1 ulp of float32 noise (~1e-4 mm), so
    # the floor is the storage precision, not zero.
    assert m["abs_bnd_rmse_mm"] < 0.01


def test_known_offset_lands_in_the_right_units(tmp_path):
    from src.ml.score34 import score_dcs34
    shots, theta = _fixture(tmp_path, err_m=0.001)      # 1 mm on every column
    m = score_dcs34(tmp_path / "p.npz", tmp_path, shots, shots, theta)
    # Tolerances are float32 storage noise (~2.4e-4 mm at 2.44 m), not logic
    # slack: a metres/mm unit bug would land 1000x off.
    assert m["rgeom_mae_mm"] == pytest.approx(1.0, abs=0.01)
    assert m["zgeom_mae_mm"] == pytest.approx(1.0, abs=0.01)
    assert m["centre_rmse_mm"] == pytest.approx(np.sqrt(2.0), abs=0.01)
    assert 0.5 < m["abs_bnd_rmse_mm"] < 5.0


def test_wrong_width_raises_instead_of_scoring_zero_shots(tmp_path):
    from src.ml.score34 import score_dcs34
    shots, theta = _fixture(tmp_path)
    bad = {s: np.zeros((20, N_RHO), np.float32) for s in shots}   # 32, not 34
    save_predictions(tmp_path / "bad.npz", bad)
    with pytest.raises(ValueError, match=str(N_OUT)):
        score_dcs34(tmp_path / "bad.npz", tmp_path, shots, shots, theta)
