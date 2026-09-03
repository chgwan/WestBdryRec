# -*- coding: utf-8 -*-
"""Smoke test for the exploratory dense-angle (n_rho=64) driver.

The driver is NOT the frozen matrix runner: it must reuse the frozen
training components (build_context_loaders / run_epochs / wrap_ddp) while
skipping validate_context_config's pinned 7-context x 5-seed publication
matrix, and write a plain exploratory artifact per (context, seed).
"""
import os

# Login-node courtesy: cap CPU threading before torch is first imported
# (torch reads OMP/MKL thread counts at import; the smoke run is tiny).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import importlib.util  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402


def _pfctx_helpers():
    """The fixture writer from tests/test_pf_context_data.py, loaded by file
    path: the torch env's site-packages ships a regular top-level ``tests``
    package that shadows the repo's namespace package, so
    ``from tests.test_pf_context_data import ...`` cannot resolve here."""
    path = (pathlib.Path(__file__).resolve().parent /
            "test_pf_context_data.py")
    spec = importlib.util.spec_from_file_location("test_pf_context_data",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_config(tmp_path):
    with open("configs/dcs_pf_context_a64.yml") as f:
        cfg = yaml.safe_load(f)
    cfg["hp"]["epochs"] = 1
    cfg["hp"]["microbatch_per_rank"] = 2
    p = tmp_path / "cfg.yml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _tiny_split(tmp_path, shot):
    """A one-shot manifest mirroring pfobs_random_pilot.json's schema
    (top-level train/validation/test role lists) with the fixture shot in
    every role."""
    manifest = {
        "name": "a64_exploratory_smoke",
        "version": 1,
        "time_axis": "native_gmag_bnd",
        "claim_scope": "pilot_only",
        "train": [shot],
        "validation": [shot],
        "test": [shot],
        "shot_metadata": None,
        "slice_strata_dir": None,
    }
    p = tmp_path / "split.json"
    p.write_text(json.dumps(manifest, indent=2))
    return p


def test_driver_smoke_world1_cpu(tmp_path, monkeypatch):
    helpers = _pfctx_helpers()
    PFCTX_SHOT, write_pfctx_fixture = helpers.PFCTX_SHOT, \
        helpers.write_pfctx_fixture
    import scripts.run_a64_exploratory as drv

    target_dir, sidecar_dir = write_pfctx_fixture(tmp_path, nt=700)
    # The frozen build_context_loaders hashes <dir>/meta.json for
    # provenance, so the fixture directories must carry the file.
    (target_dir / "meta.json").write_text('{"fixture": "target"}\n')
    (sidecar_dir / "meta.json").write_text('{"fixture": "sidecar"}\n')
    cfg = _tiny_config(tmp_path)
    split = _tiny_split(tmp_path, PFCTX_SHOT)
    monkeypatch.setenv("WORLD_SIZE", "1")          # CPU world-1 dist env
    out = tmp_path / "trains"
    rc = drv.main([
        "--config", str(cfg), "--split", str(split),
        "--target-dir", str(target_dir), "--sidecar-dir", str(sidecar_dir),
        "--out-root", str(out), "--contexts", "h0128", "--seeds", "0",
        "--epochs", "1", "--max-train-shots", "1", "--max-validation-shots",
        "1", "--n-rho", "64"])
    assert rc == 0
    art = torch.load(out / "a64_h0128_s0" / "m3.pt", weights_only=False)
    assert art["exploratory"] is True and art["n_out"] == 66
    assert art["study"] == "a64_exploratory"
    assert art["n_rho"] == 64 and art["context_label"] == "h0128"
    assert art["seed"] == 0 and art["n_act"] == 21
    # later tasks rebuild model and attention masks from the artifact alone
    assert art["depth"] == 6
    assert art["hp"]["epochs"] == 1
    assert art["hp"]["microbatch_per_rank"] == 2
    from src.ml.models import ActSeqAttn
    model = ActSeqAttn(n_act=21, n_out=66, d=256, heads=8, depth=6,
                       ffn=1024, dropout=0.1, pe="rope_time")
    model.load_state_dict(art["state"])            # strict reload


def test_driver_refuses_a_config_without_exploratory(tmp_path):
    import scripts.run_a64_exploratory as drv

    plain = yaml.safe_load(open("configs/dcs_pf_context_a64.yml"))
    del plain["exploratory"]
    cfg = tmp_path / "plain.yml"
    cfg.write_text(yaml.safe_dump(plain))
    with pytest.raises(SystemExit, match="exploratory"):
        drv.main(["--config", str(cfg), "--split", "unused.json",
                  "--out-root", str(tmp_path / "trains")])
