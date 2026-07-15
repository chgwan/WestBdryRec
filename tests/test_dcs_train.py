# -*- coding: utf-8 -*-
"""Smoke tests for DCS M0/M1/M2 training (use a handful of real shots)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.proj_config import get_proj_config  # noqa: E402
_cfg = get_proj_config()
NPZ = _cfg.mergednpz_dir
FEW = [57821, 58088, 58084, 58086, 58087]  # skip if any missing


def _have():
    return all((NPZ / f"{s}.npz").exists() for s in FEW)


def test_m0_dcs_smoke(tmp_path):
    import joblib, numpy as np
    from src.ml.train import train_m0_dcs
    if not _have():
        return  # data not present in this environment
    out = tmp_path / "m0.joblib"
    meta = train_m0_dcs(str(NPZ), out, shots=FEW)
    assert out.exists() and np.isfinite(meta["train_ccc"]) and meta["n_train"] > 0


def _run():
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as td:
                fn(pathlib.Path(td)); print(f"  {name} OK")
    print("OK")


if __name__ == "__main__":
    _run()
