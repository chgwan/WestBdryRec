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


def _run():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} OK")
    print("OK")


if __name__ == "__main__":
    _run()
