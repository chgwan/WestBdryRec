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


def test_m1_dcs_smoke(tmp_path):
    import numpy as np
    from src.ml.train import train_m1_dcs
    if not _have():
        return
    out = tmp_path / "m1.pt"
    cfg_override = None
    import src.ml.dcs_features as df
    cfg_override = df.load_dcs_config()
    cfg_override["hp"]["m1"]["epochs"] = 2
    meta = train_m1_dcs(str(NPZ), out, cfg=cfg_override, shots=FEW)
    assert out.exists() and np.isfinite(meta["best_val_mse"])


def test_m2_dcs_smoke(tmp_path):
    import numpy as np
    from src.ml.train import train_m2_dcs
    if not _have():
        return
    out = tmp_path / "m2.pt"
    import src.ml.dcs_features as df
    cfg_override = df.load_dcs_config()
    cfg_override["hp"]["m2"]["epochs"] = 2
    meta = train_m2_dcs(str(NPZ), out, cfg=cfg_override, shots=FEW)
    assert out.exists() and np.isfinite(meta["best_val_mse"])


def test_train_dcs_cli_m0(tmp_path):
    import subprocess, sys, csv
    if not _have():
        return  # data not present in this environment
    env_out = tmp_path / "out.csv"
    rc = subprocess.run([sys.executable, "scripts/train_dcs.py", "m0",
                         "--shots", "57821", "58088", "58084", "58086", "58087",
                         "--bench-out", str(env_out)], check=False).returncode
    assert rc == 0
    rows = list(csv.DictReader(env_out.open()))
    assert rows and rows[0]["model"] == "m0" and float(rows[0]["ccc"]) == float(rows[0]["ccc"])


def _run():
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as td:
                fn(pathlib.Path(td)); print(f"  {name} OK")
    print("OK")


if __name__ == "__main__":
    _run()
