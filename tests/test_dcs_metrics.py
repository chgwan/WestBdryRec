# -*- coding: utf-8 -*-
"""Unit tests for the pooled CCC metric."""
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.ml.metrics import ccc, boundary_metrics  # noqa: E402


def test_ccc_perfect_is_one():
    y = np.random.default_rng(0).normal(size=100)
    assert abs(ccc(y, y) - 1.0) < 1e-9


def test_ccc_anticorrelated_is_negative():
    y = np.arange(50, dtype=float)
    assert ccc(-y, y) < 0.0


def test_ccc_bias_lowers_value():
    y = np.arange(100, dtype=float)
    assert ccc(y + 10.0, y) < ccc(y, y)


def test_ccc_formula_handcheck():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.1, 1.9, 3.2, 3.8])
    mx, my = x.mean(), y.mean()
    sxy = ((x - mx) * (y - my)).mean()
    ref = 2 * sxy / (x.var() + y.var() + (mx - my) ** 2)
    assert abs(ccc(x, y) - ref) < 1e-12


def test_boundary_metrics_includes_ccc():
    yt = np.random.default_rng(1).normal(size=(5, 32))
    m = boundary_metrics(yt, yt)
    assert "ccc" in m and abs(m["ccc"] - 1.0) < 1e-9


def test_ccc_row_mask_perfect():
    y = np.random.default_rng(4).normal(size=(6, 32))
    rowmask = np.array([True, True, False, True, True, True])
    assert abs(ccc(y, y, mask=rowmask) - 1.0) < 1e-9


def test_score_predictions_reports_ccc(tmp_path=None):
    import json
    import tempfile
    from src.ml.bench import score_predictions
    from src.ml.predictions import save_predictions
    if tmp_path is None:
        tmp_path = pathlib.Path(tempfile.mkdtemp())
    npz = tmp_path / "data"; npz.mkdir()
    yt = np.random.default_rng(2).normal(size=(6, 32))
    ytr = np.random.default_rng(3).normal(size=(10, 32))
    for i, arr in enumerate([yt, ytr], start=1):
        np.savez(npz / f"{i}.npz", Y=arr.astype(np.float32), valid=np.ones(len(arr), dtype=bool))
    json.dump({"shots": [{"shot": 1}, {"shot": 2}]}, (npz / "meta.json").open("w"))
    pred_path = tmp_path / "pred.npz"
    save_predictions(pred_path, {1: yt.copy()})  # shot 1 = test
    m = score_predictions(pred_path, str(npz), train_shots=[2], test_shots=[1])
    assert "ccc" in m and "per_shot_ccc" in m and len(m["per_shot_ccc"]) == 1


def _run():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} OK")
    print("OK")


if __name__ == "__main__":
    _run()
