# -*- coding: utf-8 -*-
"""Tests for the a64 (dense-angle) exploratory evaluation module.

Covers the three plan-mandated surfaces of ``src.ml.a64_eval``: the
representation floor (a perfectly representable contour floors at ~0), the
paired bootstrap convention (median + percentile interval, reproducible), and
a small end-to-end smoke of the prediction path over a synthetic fixture
(plumbing only -- a randomly initialized model asserts shapes, row gather
and metric wiring, never accuracy).
"""
import os

# Login-node courtesy: cap CPU threading before torch is first imported.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import importlib.util  # noqa: E402
import pathlib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.ml.a64_eval import (  # noqa: E402
    paired_bootstrap, predict_shot, read_baseline_csv, reconstruct_and_floor,
    write_per_shot_csv,
)
from src.ml.target import uniform_theta  # noqa: E402


def _pfctx_helpers():
    """The fixture writer from tests/test_pf_context_data.py, loaded by file
    path: the torch env's site-packages ships a regular top-level ``tests``
    package that shadows the repo's namespace package (Task 4 hit this)."""
    path = (pathlib.Path(__file__).resolve().parent /
            "test_pf_context_data.py")
    spec = importlib.util.spec_from_file_location("test_pf_context_data",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_floor_of_circle_representation_is_zero():
    # The brief's 32-vertex circle fails its own threshold (rays between
    # vertices hit chords 2.2e-3 m low; Task 2 documented this), so the truth
    # polyline sits ON the reconstruction grid: the theta-grid reconstruction
    # then reproduces it vertex-for-vertex and the floor is exactly 0.
    ang = uniform_theta(64)
    c = np.array([2.4, 0.05])
    bnd = np.stack([c[0] + 0.45 * np.cos(ang), c[1] + 0.45 * np.sin(ang)], 1)
    radii = np.full((1, 64), 0.45)
    mm = reconstruct_and_floor(bnd[None], radii, c[None, :], uniform_theta(64))
    assert mm.shape == (1,) and mm[0] < 1e-6


def test_floor_of_circle_representation_is_zero_at_32_angles():
    ang = uniform_theta(32)
    c = np.array([2.4, 0.05])
    bnd = np.stack([c[0] + 0.45 * np.cos(ang), c[1] + 0.45 * np.sin(ang)], 1)
    radii = np.full((1, 32), 0.45)
    mm = reconstruct_and_floor(bnd[None], radii, c[None, :], uniform_theta(32))
    assert mm.shape == (1,) and mm[0] < 1e-6


def test_reconstruct_and_floor_matches_native_metric_construction():
    """The per-slice construction must be exactly ``pfobs_infer.score_shot``'s
    (500*(d_pt.mean()+d_tp.mean()), verified bit-exact upstream): compare
    against ``native_contour_metrics``' mean_symmetric_mm on a non-trivial
    contour pair (ellipse reconstruction vs a dense circle polyline)."""
    from src.ml.axis_frame import reconstruct_absolute
    from src.ml.native_metrics import native_contour_metrics

    theta = uniform_theta(32)
    c = np.array([2.4, 0.05])
    radii = 0.45 + 0.08 * np.cos(2 * theta)[None, :]      # ellipse-ish
    ang = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    bnd = np.stack([c[0] + 0.45 * np.cos(ang), c[1] + 0.45 * np.sin(ang)], 1)
    mm = reconstruct_and_floor(bnd[None], radii, c[None, :], theta)
    R, Z = reconstruct_absolute(c[None, 0], c[None, 1], radii, theta)
    contour = np.stack([R, Z], axis=-1)
    reference = native_contour_metrics(contour, bnd[None])["mean_symmetric_mm"]
    assert mm.shape == reference.shape == (1,)
    assert np.allclose(mm, reference, rtol=0.0, atol=1e-9)


def test_paired_bootstrap_matches_convention():
    rng = np.random.default_rng(0)
    diffs = rng.normal(-1.0, 0.5, 76)
    med, lo, hi = paired_bootstrap(diffs)
    assert np.isclose(med, np.median(diffs))
    assert lo < med < hi


def test_paired_bootstrap_reproducible():
    d = np.arange(76, dtype=float)
    assert paired_bootstrap(d) == paired_bootstrap(d)


def test_paired_bootstrap_interval_convention():
    """The interval is the 2.5/97.5 percentile bootstrap of the median with
    the frozen PCG64 seed -- the established selection/final-test
    convention."""
    rng = np.random.default_rng(0)
    diffs = rng.normal(-1.0, 0.5, 76)
    _med, lo, hi = paired_bootstrap(diffs)
    ref = np.random.default_rng(20_260_820)
    idx = ref.integers(0, len(diffs), (10_000, len(diffs)))
    draws = np.median(np.asarray(diffs)[idx], axis=1)
    assert np.isclose(lo, np.quantile(draws, 0.025))
    assert np.isclose(hi, np.quantile(draws, 0.975))


def test_predict_shot_end_to_end_smoke(tmp_path):
    """Plumbing smoke on the synthetic Task-2 fixture: a tiny artifact with a
    randomly initialized ActSeqAttn state, one shot, three scored windows.
    Asserts the gather contract (exactly the score-valid rows, strictly
    increasing), shapes, and the per-slice metric wiring -- not accuracy."""
    helpers = _pfctx_helpers()
    from src.ml.a64_eval import build_model, load_artifact
    from src.ml.axis_frame import reconstruct_absolute
    from src.ml.models import ActSeqAttn
    from src.ml.native_metrics import native_contour_metrics
    from src.ml.pf_context import context_level
    from src.ml.pfctx_data import PFContextDataset, load_context_series

    target_dir, sidecar_dir = helpers.write_pfctx_fixture(tmp_path, nt=300)
    shot = helpers.PFCTX_SHOT
    torch.manual_seed(0)
    model = ActSeqAttn(n_act=21, n_out=66, d=32, heads=4, depth=2,
                       ffn=64, dropout=0.1, pe="rope_time")
    artifact = {
        "study": "a64_exploratory", "exploratory": True,
        "n_act": 21, "n_out": 66, "n_rho": 64, "depth": 2,
        "hp": {"d_model": 32, "heads": 4, "ffn": 64, "dropout": 0.1,
               "epochs": 1},
        "context_label": "h0032", "seed": 0,
        "state": model.state_dict(),
        "feature_mean": np.zeros(21, np.float32),
        "feature_std": np.ones(21, np.float32),
        "target_mean": np.zeros(66, np.float32),
        "target_std": np.ones(66, np.float32),
        "best_val_mse": float("nan"),
    }
    artifact_path = tmp_path / "m3.pt"
    torch.save(artifact, artifact_path)

    art = load_artifact(artifact_path)
    rebuilt = build_model(art)
    dataset = PFContextDataset(
        target_dir, sidecar_dir, [shot], context_level("h0032"),
        artifact["feature_mean"], artifact["feature_std"],
        artifact["target_mean"], artifact["target_std"],
        score_block=128, n_rho=64)
    assert len(dataset) == 3                    # three scored windows
    predictions = predict_shot(art, dataset, rebuilt, uniform_theta(64),
                               target_dir)
    assert sorted(predictions) == [shot]
    record = predictions[shot]

    series = load_context_series(
        pathlib.Path(target_dir) / f"{shot}.npz", sidecar_dir, n_rho=64)
    expected_rows = np.flatnonzero(series.score_valid)
    assert np.array_equal(record.row_index, expected_rows)
    assert np.all(np.diff(record.row_index) > 0)
    assert record.prediction.shape == (len(expected_rows), 66)
    assert record.timestamp.shape == (len(expected_rows),)
    assert np.array_equal(record.timestamp, series.time[expected_rows])
    assert record.per_slice_mm.shape == (len(expected_rows),)
    assert np.isfinite(record.prediction).all()
    assert np.isfinite(record.per_slice_mm).all()

    # the per-slice values ARE the native mean-symmetric metric of the
    # destandardized prediction contours against the truth polyline
    with np.load(pathlib.Path(target_dir) / f"{shot}.npz") as data:
        bnd = data["bnd_RZ"].astype(np.float64)[expected_rows]
    radii = record.prediction[:, :64].astype(np.float64)
    center = record.prediction[:, 64:].astype(np.float64)
    R, Z = reconstruct_absolute(center[:, 0], center[:, 1], radii,
                                uniform_theta(64))
    contour = np.stack([R, Z], axis=-1)
    reference = native_contour_metrics(contour, bnd)["mean_symmetric_mm"]
    assert np.allclose(record.per_slice_mm, reference, rtol=0.0, atol=1e-9)


def test_load_artifact_rejects_a_non_exploratory_artifact(tmp_path):
    from src.ml.a64_eval import load_artifact
    from src.ml.models import ActSeqAttn

    torch.manual_seed(0)
    model = ActSeqAttn(n_act=21, n_out=66, d=32, heads=4, depth=2,
                       ffn=64, dropout=0.1, pe="rope_time")
    artifact = {
        "study": "a64_exploratory", "exploratory": False,
        "n_act": 21, "n_out": 66, "n_rho": 64, "depth": 2,
        "hp": {"d_model": 32, "heads": 4, "ffn": 64, "dropout": 0.1},
        "context_label": "h0032", "seed": 0,
        "state": model.state_dict(),
        "feature_mean": np.zeros(21, np.float32),
        "feature_std": np.ones(21, np.float32),
        "target_mean": np.zeros(66, np.float32),
        "target_std": np.ones(66, np.float32),
    }
    path = tmp_path / "plain.pt"
    torch.save(artifact, path)
    try:
        load_artifact(path)
    except ValueError as exc:
        assert "exploratory" in str(exc)
    else:
        raise AssertionError("a non-exploratory artifact must be refused")


def test_write_per_shot_csv_round_trips_through_read_baseline(tmp_path):
    art = {"context_label": "h0128", "seed": 0}
    per_shot = {58293: (120, 5.5), 58295: (300, 6.25)}
    path = write_per_shot_csv(tmp_path / "a360_h0128_s0_per_shot.csv",
                              art=art, per_shot=per_shot)
    lines = path.read_text().splitlines()
    assert lines[0] == "context,seed,shot,n_slices,mean_symmetric_mm"
    assert lines[1] == "h0128,0,58293,120,5.500000"
    assert read_baseline_csv(path) == {58293: 5.5, 58295: 6.25}


def test_main_routes_baseline_and_default_out_csv(tmp_path):
    """main() end to end on the smoke fixture: a pre-written exploratory
    baseline (the run's own per-shot value plus a constant offset, so the
    expected paired median, run minus baseline, is analytically the negated
    offset) must reach the md's paired section and provenance line, and the
    omitted ``--out-csv`` must default to ``<out-md stem>_per_shot.csv``."""
    import json
    import re

    helpers = _pfctx_helpers()
    from src.ml.a64_eval import (
        build_model, floor_per_shot, load_artifact, main, read_baseline_csv)
    from src.ml.models import ActSeqAttn
    from src.ml.pf_context import context_level
    from src.ml.pfctx_data import PFContextDataset

    target_dir, sidecar_dir = helpers.write_pfctx_fixture(tmp_path, nt=300)
    shot = helpers.PFCTX_SHOT
    torch.manual_seed(0)
    model = ActSeqAttn(n_act=21, n_out=66, d=32, heads=4, depth=2,
                       ffn=64, dropout=0.1, pe="rope_time")
    artifact = {
        "study": "a64_exploratory", "exploratory": True,
        "n_act": 21, "n_out": 66, "n_rho": 64, "depth": 2,
        "hp": {"d_model": 32, "heads": 4, "ffn": 64, "dropout": 0.1,
               "epochs": 1},
        "context_label": "h0032", "seed": 0,
        "state": model.state_dict(),
        "feature_mean": np.zeros(21, np.float32),
        "feature_std": np.ones(21, np.float32),
        "target_mean": np.zeros(66, np.float32),
        "target_std": np.ones(66, np.float32),
        "best_val_mse": float("nan"),
    }
    artifact_path = tmp_path / "m3.pt"
    torch.save(artifact, artifact_path)

    # The run's own per-shot value from exactly the pipeline main() drives
    # (same eval-mode weights, default score_block): main() recomputes it
    # identically, so the paired median below is analytically -offset.
    art = load_artifact(artifact_path)
    dataset = PFContextDataset(
        target_dir, sidecar_dir, [shot], context_level("h0032"),
        artifact["feature_mean"], artifact["feature_std"],
        artifact["target_mean"], artifact["target_std"], n_rho=64)
    record = predict_shot(art, dataset, build_model(art), uniform_theta(64),
                          target_dir)[shot]
    own_mm = float(record.per_slice_mm.mean())

    offset = 1.25
    baseline_path = write_per_shot_csv(
        tmp_path / "baseline_per_shot.csv",
        art={"context_label": "h0032", "seed": 0},
        per_shot={shot: (len(record.row_index), own_mm + offset)})

    # the fixture's own 32-theta floor, so main()'s strict self-check
    # against --floor-csv agrees by construction
    floor_value, floor_rows = floor_per_shot(
        pathlib.Path(target_dir) / f"{shot}.npz", sidecar_dir, 32)
    floor_csv = tmp_path / "floor.csv"
    floor_csv.write_text("shot,mean_symmetric_mm,n_slices\n"
                         f"{shot},{floor_value:.12f},{floor_rows}\n")

    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": [], "validation": [],
                                      "test": [shot]}))
    out_dir = tmp_path / "out"
    rc = main(["--artifact", str(artifact_path),
               "--out-md", str(out_dir / "results.md"),
               "--out-png", str(out_dir / "fig.png"),
               "--split", str(split_path),
               "--target-dir", str(target_dir),
               "--sidecar-dir", str(sidecar_dir),
               "--baseline", str(baseline_path),
               "--floor-csv", str(floor_csv)])
    assert rc == 0

    # --out-csv omitted: the CSV must default to <out-md stem>_per_shot.csv
    derived = (out_dir / "results.md").with_name("results_per_shot.csv")
    assert derived.exists()
    lines = derived.read_text().splitlines()
    assert lines[0] == "context,seed,shot,n_slices,mean_symmetric_mm"
    assert abs(read_baseline_csv(derived)[shot] - own_mm) < 1e-6
    assert (out_dir / "fig.png").exists()

    md = (out_dir / "results.md").read_text()
    assert "minus baseline, per shot" in md      # the paired section header
    assert f"`{baseline_path}`" in md            # the provenance line
    median = float(re.search(r"median difference \*\*([+-]?\d+\.\d+) mm",
                             md).group(1))
    # baseline = own + offset, and the paired diff is run - baseline
    assert abs(median + offset) < 5e-5           # .4f rendering tolerance
