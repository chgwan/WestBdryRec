import numpy as np
import pytest
import yaml

from src.ml.pf_context import (
    ATTENTION_DEPTH, CADENCE_REF_SECONDS, CONTEXT_LEVELS, SCORE_BLOCK,
    ContextLevel, context_level, scored_windows,
)


def test_context_grid_is_frozen_in_seconds():
    got = [(x.label, x.nominal_samples, x.seconds) for x in CONTEXT_LEVELS]
    assert got == [
        ("h0001", 1, 0.0),
        ("h0032", 32, 0.063488),
        ("h0128", 128, 0.260096),
        ("h0256", 256, 0.522240),
        ("h0512", 512, 1.046528),
        ("h1024", 1024, 2.095104),
        ("h2048", 2048, 4.192256),
    ]


def test_context_lookup_rejects_unknown_label():
    assert context_level("h0512").nominal_samples == 512
    with pytest.raises(ValueError, match="context must be one of"):
        context_level("h0064")


@pytest.mark.parametrize("nt", [1, 7, 511, 512, 513, 1024, 1703, 5000])
def test_scored_blocks_cover_each_row_once(nt):
    t = 0.053 + np.arange(nt, dtype=np.float64) * 0.002048
    for level in CONTEXT_LEVELS:
        blocks = scored_windows(t, level)
        count = np.zeros(nt, dtype=np.int64)
        for w in blocks:
            assert 0 <= w.window_start <= w.block_start < w.block_end <= nt
            assert w.window_end == w.block_end
            count[w.block_start:w.block_end] += 1
        assert np.array_equal(count, np.ones(nt, dtype=np.int64))


def test_native_gap_changes_prefix_rows_not_physical_horizon():
    t = np.array([0.000, 0.002, 0.004, 0.100, 0.102, 0.104], np.float64)
    level = ContextLevel("probe", 32, 0.063488)
    block = scored_windows(t, level, score_block=4)[1]
    assert block.block_start == 4
    assert block.window_start == 3
    assert t[block.block_start] - t[block.window_start] <= level.seconds


def test_config_yaml_matches_frozen_constants():
    """Configuration drift must fail before training."""
    with open("configs/dcs_pf_context_sweep.yml") as f:
        cfg = yaml.safe_load(f)

    # Assert contexts match Python constants
    yaml_contexts = [(c["label"], c["nominal_samples"], c["seconds"])
                     for c in cfg["contexts"]]
    py_contexts = [(x.label, x.nominal_samples, x.seconds)
                   for x in CONTEXT_LEVELS]
    assert yaml_contexts == py_contexts

    # Assert critical constants match
    assert cfg["hp"]["depth"] == ATTENTION_DEPTH
    assert cfg["score_block"] == SCORE_BLOCK
    assert cfg["cadence_reference_seconds"] == CADENCE_REF_SECONDS

    # Assert remaining frozen fields match exact literal values
    assert cfg["study"] == "pf_context"
    assert cfg["arm"] == "B"
    assert cfg["model"] == "ActSeqAttn"
    assert cfg["pe"] == "rope_time"
    assert cfg["time_axis"] == "native_gmag_bnd"
    assert cfg["input_width"] == 21
    assert cfg["seeds"] == [0, 1, 2, 3, 4]
    assert cfg["hp"]["d_model"] == 256
    assert cfg["hp"]["heads"] == 8
    assert cfg["hp"]["ffn"] == 1024
    assert cfg["hp"]["dropout"] == 0.1
    assert cfg["hp"]["effective_global_batch"] == 16
    assert cfg["hp"]["microbatch_per_rank"] == 4
    assert cfg["hp"]["production_gradient_accumulation"] == 1
    assert cfg["hp"]["lr"] == 0.0003
    assert cfg["hp"]["warmup"] == 5
    assert cfg["hp"]["epochs"] == 80
    assert cfg["hp"]["patience"] == 12
    assert cfg["hp"]["gradient_clip"] == 1.0
    assert cfg["selection"]["anchor"] == "h2048"
    assert cfg["selection"]["practical_margin_mm"] == 1.0
    assert cfg["selection"]["validation_family_alpha"] == 0.05
    assert cfg["selection"]["bootstrap_resamples"] == 10000
    assert cfg["selection"]["required_seed_agreement"] == 4


# ── Task 2: the Arm B native-time joined series and fixed scored blocks ──
import pathlib  # noqa: E402

import torch  # noqa: E402

from src.ml.pf_context import context_level  # noqa: E402
from src.ml.pfctx_data import (  # noqa: E402
    PFContextDataset, load_context_series, pad_context_collate,
)

PFCTX_SHOT = 7
PFCTX_CADENCE = 0.002048          # the synthetic native grid's step, seconds


def fixture_ref(nt):
    """Deterministic (nt, 10) float32 PF reference: column k is (k+1) + row."""
    row = np.arange(nt, dtype=np.float32)
    return np.stack([np.float32(k + 1) + row for k in range(10)], axis=1)


def fixture_actual(nt):
    """Deterministic (nt, 10) float32 PF actual: col k is 100*(k+1) + row."""
    row = np.arange(nt, dtype=np.float32)
    return np.stack([np.float32(100 * (k + 1)) + row for k in range(10)],
                    axis=1)


def fixture_ip_ref(nt):
    """Deterministic (nt,) float32 Ip reference: 500 + row."""
    return np.float32(500.0) + np.arange(nt, dtype=np.float32)


def write_pfctx_fixture(tmp_path, nt=520, target_bad=(), actual_bad=(),
                        ref_bad=()):
    """One deterministic NpzGeom target + NpzGeomPFObs sidecar pair on a
    single native float32 axis.

    The target holds exactly what ``load_target`` consumes (``Y`` (nt, 32),
    ``center`` (nt, 2)) plus an all-True ``valid`` and the axis; the sidecar
    holds exactly what ``build_pf_observability.build_one`` writes (``pf_ref``
    / ``pf_actual`` (nt, 10) float32, ``ip_ref`` (nt, 1) float32,
    ``common_valid`` (nt,) bool and a byte-identical ``time``). ``target_bad``
    rows carry a NaN target (target not finite, Arm B history untouched);
    ``actual_bad`` rows carry a NaN PF actual (Arm B history invalid);
    ``ref_bad`` rows carry a NaN PF reference only -- common-invalid while the
    Arm B history stays valid, so a reference dropout can never gate Arm B
    history. Returns ``(target_dir, sidecar_dir)``.
    """
    target_dir = pathlib.Path(tmp_path) / "NpzGeom"
    sidecar_dir = tmp_path / "NpzGeomPFObs"
    target_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    t = (PFCTX_CADENCE * np.arange(nt, dtype=np.float64)).astype(np.float32)
    ref = fixture_ref(nt).copy()
    actual = fixture_actual(nt).copy()
    ip = fixture_ip_ref(nt).copy()
    Y = np.full((nt, 32), 0.45, np.float32)
    center = np.tile(np.array([2.4, 0.05], np.float32), (nt, 1))
    for r in target_bad:
        Y[r] = np.nan
    for r in actual_bad:
        actual[r] = np.nan
    for r in ref_bad:
        ref[r] = np.nan

    # Dense circle polyline consistent with Y=0.45 about the centre: 4096
    # vertices on the r=0.45 circle, so ray-cast derivation at any uniform
    # angle grid returns 0.45 to within the sagitta 0.45*(1-cos(pi/4096)) ~
    # 1.3e-7 m, far inside the tests' atol=1e-6.  (The plan's original
    # 32-vertex circle fails its own test: rays between vertices hit chords
    # at 0.45*cos(pi/32), which is 2.2e-3 low.)
    ang = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    # (nt, 1) + (4096,) broadcasts to (nt, 4096); the plan's
    # ``center[:, 0] + ...`` does not broadcast at all.
    bnd = np.stack([center[:, 0:1] + 0.45 * np.cos(ang),
                    center[:, 1:2] + 0.45 * np.sin(ang)], axis=2)  # (nt, 4096, 2)

    target_valid = np.ones(nt, bool)
    target_finite = np.isfinite(Y).all(1) & np.isfinite(center).all(1)
    common = (
        target_valid
        & (t >= 0)
        & np.isfinite(ref).all(1)
        & np.isfinite(actual).all(1)
        & np.isfinite(ip)
    )

    np.savez(target_dir / f"{PFCTX_SHOT}.npz", time=t, valid=target_valid,
             Y=Y, center=center, bnd_RZ=bnd)
    np.savez(sidecar_dir / f"{PFCTX_SHOT}.npz", time=t, pf_ref=ref,
             pf_actual=actual, ip_ref=ip.reshape(-1, 1), common_valid=common)
    return target_dir, sidecar_dir


def make_context_dataset(target_dir, sidecar_dir, label):
    """A one-shot PFContextDataset over the fixture, neutral normalization."""
    return PFContextDataset(
        target_dir, sidecar_dir, [PFCTX_SHOT], context_level(label),
        feature_mean=np.zeros(21), feature_std=np.ones(21),
        target_mean=np.zeros(34), target_std=np.ones(34))


def test_load_context_series_uses_fixed_arm_b(tmp_path):
    target_dir, sidecar_dir = write_pfctx_fixture(tmp_path, nt=520)
    series = load_context_series(target_dir / "7.npz", sidecar_dir)
    assert series.features.shape == (520, 21)
    # numpy 2 array_equal demands identical shapes, so the zero-padded
    # reference block is compared against a same-shaped zeros array
    assert np.array_equal(series.features[:, :10],
                          np.zeros_like(series.features[:, :10]))
    assert np.array_equal(series.features[:, 10:20], fixture_actual(520))
    assert np.array_equal(series.features[:, 20], fixture_ip_ref(520))


def test_history_and_score_validity_are_distinct(tmp_path):
    target_dir, sidecar_dir = write_pfctx_fixture(
        tmp_path, nt=8, target_bad=(2,), actual_bad=(4,), ref_bad=(6,))
    s = load_context_series(target_dir / "7.npz", sidecar_dir)
    assert not s.score_valid[2]
    assert s.history_valid[2]
    assert not s.history_valid[4]
    assert not s.score_valid[4]
    assert s.history_valid[6]
    assert not s.score_valid[6]
    assert np.isfinite(s.features).all()


def test_dataset_scores_each_common_valid_row_once(tmp_path):
    target_dir, sidecar_dir = write_pfctx_fixture(tmp_path, nt=1300)
    ds = PFContextDataset(
        target_dir, sidecar_dir, [7], context_level("h0512"),
        feature_mean=np.zeros(21), feature_std=np.ones(21),
        target_mean=np.zeros(34), target_std=np.ones(34))
    count = np.zeros(1300, dtype=np.int64)
    prefix_lengths = []
    for item, (_series_index, index) in zip(
            (ds[i] for i in range(len(ds))), ds.index):
        local_start = index.block_start - index.window_start
        n_score = index.block_end - index.block_start
        # history rows are attended but never scored
        assert not item.loss_mask[:local_start].any()
        prefix_lengths.append(local_start)
        count[index.block_start:index.block_end] += (
            item.loss_mask[local_start:local_start + n_score].numpy())
    assert max(prefix_lengths) > 0, "h0512 windows must have history prefixes"
    with np.load(target_dir / "7.npz") as d:
        expected = d["valid"].astype(np.int64)
    assert np.array_equal(count, expected)


def test_score_invalid_rows_inside_a_scored_block_stay_unscored(tmp_path):
    """A score-ineligible row inside a scored block must come out False in
    the dataset's loss mask -- through the dataset path, not only at series
    level -- whatever made it ineligible: a NaN target (row 600) or a NaN
    measured signal (row 700)."""
    target_dir, sidecar_dir = write_pfctx_fixture(
        tmp_path, nt=1300, target_bad=(600,), actual_bad=(700,))
    ds = make_context_dataset(target_dir, sidecar_dir, "h0512")
    scored = np.zeros(1300, dtype=bool)
    for item, (_si, w) in zip((ds[i] for i in range(len(ds))), ds.index):
        local = w.block_start - w.window_start
        assert not item.loss_mask[:local].any()
        mask = np.zeros(1300, dtype=bool)
        mask[w.block_start:w.block_end] = item.loss_mask[local:].numpy()
        scored |= mask
    assert not scored[600]
    assert not scored[700]
    assert scored.sum() == 1298


def test_collate_preserves_float64_native_time_and_masks_padding(tmp_path):
    target_dir, sidecar_dir = write_pfctx_fixture(tmp_path, nt=700)
    ds = make_context_dataset(target_dir, sidecar_dir, "h0032")
    batch = pad_context_collate([ds[0], ds[1]])
    assert batch.time.dtype == torch.float64
    assert batch.real.dtype == torch.bool
    assert batch.history_valid.dtype == torch.bool
    assert not batch.loss_mask[~batch.real].any()


def test_context_level_leaves_blocks_and_loss_masks_invariant(tmp_path):
    """The sweep may vary ONLY the history window: the scored blocks and
    their loss masks are properties of the native axis and the score block,
    so all seven context levels must tile the shot identically."""
    assert len(CONTEXT_LEVELS) == 7
    target_dir, sidecar_dir = write_pfctx_fixture(tmp_path, nt=1300)
    block_lists, mask_blobs, window_extents = [], [], set()
    for level in CONTEXT_LEVELS:
        ds = make_context_dataset(target_dir, sidecar_dir, level.label)
        shot = ds.series[ds.index[0][0]][0]
        block_lists.append(
            [(shot, w.block_start, w.block_end) for _si, w in ds.index])
        mask_blobs.append(b"".join(
            item.loss_mask[w.block_start - w.window_start:].numpy().tobytes()
            for item, (_si, w) in zip(
                (ds[i] for i in range(len(ds))), ds.index)))
        window_extents.update(
            (w.window_start, w.window_end) for _si, w in ds.index)
    assert all(b == block_lists[0] for b in block_lists)
    assert all(m == mask_blobs[0] for m in mask_blobs)
    assert len(window_extents) > 1, "levels must actually vary the window"


def test_load_context_series_rejects_a_one_bit_time_mismatch(tmp_path):
    """The join may not silently repair even a single-ULP axis drift: a
    sidecar row pairs only with the target row at the exact same time."""
    target_dir, sidecar_dir = write_pfctx_fixture(tmp_path, nt=64)
    path = sidecar_dir / "7.npz"
    with np.load(path) as d:
        arrays = {k: d[k] for k in d.files}
    t = arrays["time"]
    arrays["time"] = t.copy()
    arrays["time"][17] = np.nextafter(t[17], np.float32(np.inf))
    assert arrays["time"][17].view(np.uint32) != t[17].view(np.uint32)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="native time mismatch"):
        load_context_series(target_dir / "7.npz", sidecar_dir)


def test_pfctx_data_never_calls_an_interpolation_function():
    """Native-time pairing is exact-row, never repaired: no call in the
    module may interpolate, resample or regrid the time axis (checked at the
    AST call level, so a docstring word can never mask a real call)."""
    import ast
    import inspect

    import src.ml.pfctx_data as pfctx_module

    tree = ast.parse(inspect.getsource(pfctx_module))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
        if isinstance(node, ast.Import):
            called.update(a.name for a in node.names)
        if isinstance(node, ast.ImportFrom):
            called.add(node.module or "")
    forbidden = sorted(name for name in called if any(
        s in name.lower() for s in
        ("interp", "resample", "spline", "pchip", "griddata", "scipy")))
    assert not forbidden, (
        f"pfctx_data must stay native-time: found {forbidden}")


# ── Task 2: n_rho-parameterized targets with load-time radii derivation ──
def test_n_rho_none_returns_stored_y_bit_identical(tmp_path):
    target_dir, sidecar_dir = write_pfctx_fixture(tmp_path, nt=520)
    a = load_context_series(target_dir / f"{PFCTX_SHOT}.npz", sidecar_dir)
    b = load_context_series(target_dir / f"{PFCTX_SHOT}.npz", sidecar_dir,
                            n_rho=32)
    assert a.target.shape == b.target.shape == (520, 34)
    assert np.array_equal(a.target, b.target)
    assert np.array_equal(a.score_valid, b.score_valid)


def test_n_rho_64_derives_circle_radii(tmp_path):
    target_dir, sidecar_dir = write_pfctx_fixture(
        tmp_path, nt=520, target_bad=(10,))
    s = load_context_series(target_dir / f"{PFCTX_SHOT}.npz", sidecar_dir,
                            n_rho=64)
    assert s.target.shape == (520, 66)
    ok = s.score_valid
    assert ok[10] is np.False_ or not ok[10]          # bad row stays unscored
    assert np.allclose(s.target[ok][:, :64], 0.45, atol=1e-6)
    # ... and the passthrough columns really are the fixture centre
    assert np.allclose(s.target[ok][:, 64], np.float32(2.4))
    assert np.allclose(s.target[ok][:, 65], np.float32(0.05))


def test_derivation_requires_bnd_rz(tmp_path):
    target_dir, sidecar_dir = write_pfctx_fixture(tmp_path, nt=8)
    path = target_dir / f"{PFCTX_SHOT}.npz"
    with np.load(path) as d:
        arrays = {k: d[k] for k in d.files if k != "bnd_RZ"}
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="bnd_RZ"):
        load_context_series(path, sidecar_dir, n_rho=64)


def test_dataset_and_loaders_pass_n_rho_through(tmp_path):
    target_dir, sidecar_dir = write_pfctx_fixture(tmp_path, nt=600)
    ds = make_context_dataset(target_dir, sidecar_dir, "h0128")
    item = ds[0]
    assert item.target.shape[1] == 34
    ds64 = PFContextDataset(
        target_dir, sidecar_dir, [PFCTX_SHOT], context_level("h0128"),
        feature_mean=np.zeros(21), feature_std=np.ones(21),
        target_mean=np.zeros(66), target_std=np.ones(66), n_rho=64)
    assert ds64[0].target.shape[1] == 66


# ── Task 3: the exploratory 64-angle config's drift guard ──
def test_a64_config_pins_exploratory_contract():
    with open("configs/dcs_pf_context_a64.yml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["exploratory"] is True
    assert cfg["n_rho"] == 64
    assert cfg["input_width"] == 21
    assert cfg["score_block"] == 512
    assert [c["label"] for c in cfg["contexts"]] == ["h0512"]
    assert cfg["seeds"] == [0]
    assert (cfg["study"], cfg["arm"], cfg["model"], cfg["pe"],
            cfg["time_axis"]) == ("pf_context", "B", "ActSeqAttn",
                                  "rope_time", "native_gmag_bnd")
    assert cfg["hp"] == {
        "d_model": 256, "heads": 8, "depth": 6, "ffn": 1024,
        "dropout": 0.1, "effective_global_batch": 16,
        "microbatch_per_rank": 4, "production_gradient_accumulation": 1,
        "lr": 0.0003, "warmup": 5, "epochs": 80, "patience": 12,
        "gradient_clip": 1.0,
    }
    assert cfg["selection"]["anchor"] == "h2048"   # unchanged bookkeeping


def test_a128_config_pins_exploratory_contract():
    with open("configs/dcs_pf_context_a128.yml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["exploratory"] is True
    assert cfg["n_rho"] == 128
    assert [c["label"] for c in cfg["contexts"]] == ["h0512"]
    assert cfg["seeds"] == [0]
    assert (cfg["study"], cfg["arm"], cfg["model"], cfg["pe"],
            cfg["time_axis"]) == ("pf_context", "B", "ActSeqAttn",
                                  "rope_time", "native_gmag_bnd")
    assert cfg["input_width"] == 21 and cfg["score_block"] == 512
    assert cfg["hp"]["d_model"] == 256 and cfg["hp"]["epochs"] == 80


def test_a256_config_pins_exploratory_contract():
    with open("configs/dcs_pf_context_a256.yml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["exploratory"] is True
    assert cfg["n_rho"] == 256
    assert [c["label"] for c in cfg["contexts"]] == ["h0512"]
    assert cfg["seeds"] == [0]
    assert (cfg["study"], cfg["arm"], cfg["model"], cfg["pe"],
            cfg["time_axis"]) == ("pf_context", "B", "ActSeqAttn",
                                  "rope_time", "native_gmag_bnd")
    assert cfg["input_width"] == 21 and cfg["score_block"] == 512
    assert cfg["hp"]["d_model"] == 256 and cfg["hp"]["epochs"] == 80


def test_a360_config_pins_exploratory_contract():
    with open("configs/dcs_pf_context_a360.yml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["exploratory"] is True
    assert cfg["n_rho"] == 360
    assert [c["label"] for c in cfg["contexts"]] == ["h0512"]
    assert cfg["seeds"] == [0]
    assert (cfg["study"], cfg["arm"], cfg["model"], cfg["pe"],
            cfg["time_axis"]) == ("pf_context", "B", "ActSeqAttn",
                                  "rope_time", "native_gmag_bnd")
    assert cfg["input_width"] == 21 and cfg["score_block"] == 512
    assert cfg["hp"]["d_model"] == 256 and cfg["hp"]["epochs"] == 80
