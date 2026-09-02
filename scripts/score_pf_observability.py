# -*- coding: utf-8 -*-
"""The one-shot final test evaluation for the PF-observability matrix.

Loads NO test target until every cell of the requested matrix (production:
all 20 = 4 arms x 5 seeds) validates: artifact + validation JSON present AND
the stored ``run_fingerprint`` equal to the freshly computed one. A missing
run, any hash mismatch, a smoke ``--epochs`` artifact (the override is not
fingerprinted), a matrix mixing world_size/global_batch, or -- without
``--allow-subset`` -- a subset matrix aborts before a single test shot is
opened. Only artifacts carrying the fixed base contract (the ``ActSeqAttn``
windowed model, ``pe = rope_time``, the ``native_gmag_bnd`` time axis behind
the frozen split, the 21-column arm input) are ever scored.

The evaluation is a transaction under the stats root
(default ``ProjDB/Stats/pf_observability``): everything is written into
``final_test.staging`` (a crashed run is disposable and resumable -- cells
whose two output files already exist are not rescored, while the model-free
representation floor is always rewritten from the frozen split), then
verified complete, then the staging directory is atomically renamed to
``final_test`` and only THEN is the ``FINAL_TEST_EVALUATED.json`` marker
written, last. An existing marker (or an existing ``final_test`` directory)
refuses the run: the final scoring happens exactly once.

Per run the scorer writes ``final_test/<run_name>/m3_pred.npz`` (per shot,
the destandardized 34-column predictions on exactly the common-valid rows)
and ``final_test/<run_name>/per_shot_metrics.csv``; once per evaluation it
writes ``final_test/representation_floor_per_shot.csv`` (model-free).

Usage (from the repo root, after the matrix runner completed):
  python scripts/score_pf_observability.py
  python scripts/score_pf_observability.py --verify-only
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import stat
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.dcs_features import load_dcs_config, load_meta  # noqa: E402
from src.ml.pf_observability import (  # noqa: E402
    ARMS, run_fingerprint, sha256_file,
)
from src.ml.pfobs_provenance import (  # noqa: E402
    PER_SHOT_METRICS_HEADER,
    REPRESENTATION_FLOOR_HEADER,
    ValidationFreezeLock,
    build_work2_final_marker,
    canonical_work2_final_marker_bytes,
    validate_representation_floor,
    validate_source_audit_identity,
    validate_work2_final_marker,
    validate_work2_validation_decision,
    write_exclusive_destination,
)
from src.ml.publication_split import (  # noqa: E402
    PUBLICATION_MANIFEST_PATH,
    PUBLICATION_OUT_ROOT,
    PUBLICATION_SEEDS,
    PUBLICATION_SIDECAR_DIR,
    PUBLICATION_TARGET_DIR,
    PUBLICATION_WORK2_ARMS,
    PUBLICATION_WORK2_AUDIT_IDENTITY,
    PUBLICATION_WORK2_CONFIG,
    PUBLICATION_WORK2_MARKER,
    PUBLICATION_WORK2_PREFIX,
    PUBLICATION_WORK2_STATS_ROOT,
    canonical_shot_list_sha256,
    load_split_for_mode,
    publication_disclosure,
    require_publication_paths,
)
from src.ml.pfobs_infer import (  # noqa: E402
    build_model, floor_shot, load_artifact, predict_shot, scored_row_indices,
    score_shot,
)
from src.ml.pfobs_train import _available_shots, validate_base_contract  # noqa: E402
from src.ml.predictions import save_predictions  # noqa: E402
from src.ml.target import load_target  # noqa: E402
from src.proj_config import get_proj_config  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NPZ_DIR = PUBLICATION_TARGET_DIR
SIDECAR_DIR = PUBLICATION_SIDECAR_DIR
CONFIG = PUBLICATION_WORK2_CONFIG
SPLIT = PUBLICATION_MANIFEST_PATH
OUT_ROOT = PUBLICATION_OUT_ROOT
STATS_ROOT = PUBLICATION_WORK2_STATS_ROOT
AUDIT_IDENTITY = PUBLICATION_WORK2_AUDIT_IDENTITY

FINAL_DIR_NAME = "final_test"
STAGING_NAME = FINAL_DIR_NAME + ".staging"
MARKER_NAME = "FINAL_TEST_EVALUATED.json"
# The one-shot transaction scores exactly this matrix; anything smaller needs
# an explicit --allow-subset (diagnostics only, never the committed result).
PRODUCTION_ARMS = tuple(ARMS)
PRODUCTION_SEEDS = (0, 1, 2, 3, 4)


def _require_publication_invocation(args, *, operation):
    """Reject every publication namespace/matrix downgrade before data access."""
    mode = getattr(args, "manifest_mode", "publication")
    if mode != "publication":
        return
    require_publication_paths(
        mode,
        {
            "npz_dir": args.npz_dir,
            "sidecar_dir": args.sidecar_dir,
            "config": args.config,
            "split": args.split,
            "out_root": args.out_root,
            "stats_root": args.stats_root,
            "audit_identity": args.audit_identity,
        },
        {
            "npz_dir": NPZ_DIR,
            "sidecar_dir": SIDECAR_DIR,
            "config": CONFIG,
            "split": SPLIT,
            "out_root": OUT_ROOT,
            "stats_root": STATS_ROOT,
            "audit_identity": AUDIT_IDENTITY,
        },
    )
    if args.allow_subset:
        raise ValueError(
            "publication mode rejects --allow-subset; use explicit generic mode")
    if tuple(args.arms) != tuple(PUBLICATION_WORK2_ARMS):
        raise ValueError("publication --arms must be the canonical A B C D matrix")
    if tuple(int(seed) for seed in args.seeds) != tuple(PUBLICATION_SEEDS):
        raise ValueError(
            "publication --seeds must be the canonical 0 1 2 3 4 matrix")
    if str(args.run_prefix) != PUBLICATION_WORK2_PREFIX:
        raise ValueError(
            "publication --run-prefix must be the frozen 'pfobs' prefix")
    canonical_marker = pathlib.Path(STATS_ROOT) / MARKER_NAME
    if operation == "score" and os.path.lexists(canonical_marker):
        raise RuntimeError(
            f"{canonical_marker} exists: the final test was already evaluated -- "
            "the canonical publication transaction is globally final")


def _load_manifest(path, manifest_mode, available_shots=None):
    """Load the requested split contract once at scorer preflight."""
    return load_split_for_mode(
        path,
        manifest_mode=manifest_mode,
        available_shots=available_shots,
        project_root=REPO_ROOT,
    )


PER_SHOT_HEADER = list(PER_SHOT_METRICS_HEADER)
FLOOR_HEADER = list(REPRESENTATION_FLOOR_HEADER)


def _load_runner():
    """scripts/ is not a package; load the validation runner by path so the
    matrix naming and hash-gate logic has exactly one implementation."""
    spec = importlib.util.spec_from_file_location(
        "run_pf_observability",
        pathlib.Path(__file__).resolve().parent / "run_pf_observability.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUNNER = _load_runner()


def _load_analysis():
    """Load the immutable validation-decision derivation by maintained path."""
    spec = importlib.util.spec_from_file_location(
        "pf_observability_analysis_for_scorer",
        REPO_ROOT / "exploration" / "pf_observability_analysis.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PF-observability validation analysis")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ANALYSIS = _load_analysis()


def _atomic_json_dump(obj, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=header, extrasaction="raise", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _load_top_level_split(args):
    """Validate the selected manifest before any scorer state shortcut."""
    npz_dir = pathlib.Path(args.npz_dir)
    sidecar_dir = pathlib.Path(args.sidecar_dir)
    return _load_manifest(
        args.split,
        getattr(args, "manifest_mode", "publication"),
        available_shots=_available_shots(npz_dir, sidecar_dir),
    )


def _preflight(args, split=None):
    """Symmetric config/split/dataset preflight -- no test target is opened."""
    npz_dir = pathlib.Path(args.npz_dir)
    sidecar_dir = pathlib.Path(args.sidecar_dir)
    if split is None:
        split = _load_top_level_split(args)
    validate_base_contract(load_dcs_config(args.config))
    meta = load_meta(npz_dir)
    if "theta_deg" not in meta:
        raise ValueError(f"{npz_dir}/meta.json: no 'theta_deg' -- the uniform "
                         "angle grid is required to rebuild contours")
    theta = np.deg2rad(np.asarray(meta["theta_deg"], float))
    return npz_dir, sidecar_dir, split, theta


def validate_matrix(args):
    """Every cell stored and hash-valid, and its artifact contract-checked.

    Runs over ALL cells of the requested matrix (production: all 20) before
    the caller touches any test target. A missing run raises; a fingerprint
    mismatch raises through the runner's ``resume_decision`` (never overwrite
    a stored run). Beyond the hash gate every artifact must AGREE with the
    shared config and with its peers: ``hp.epochs`` equal to the config's
    ``hp.m3.epochs`` (``--epochs`` smoke runs are deliberately NOT
    fingerprinted, so the recorded hp is the only witness) and ONE shared
    ``world_size``/``global_batch`` across the whole matrix. Returns the
    cells with their loaded artifacts.
    """
    config_epochs = int(load_dcs_config(args.config)["hp"]["m3"]["epochs"])
    runs = RUNNER.matrix_runs(arms=tuple(args.arms), seeds=tuple(args.seeds),
                              prefix=args.run_prefix)
    cells = []
    world_sizes, global_batches = set(), set()
    for run in runs:
        run_dir = pathlib.Path(args.out_root) / run["run_name"]
        if not RUNNER.is_validation_done(run_dir):
            raise RuntimeError(
                f"final matrix incomplete: {run['run_name']} has no stored "
                "artifact + validation JSON -- train the matrix before "
                "scoring the final test")
        stored = RUNNER._stored_fingerprint(run_dir)
        fingerprint = run_fingerprint(
            run["arm"], run["seed"], args.config, args.split,
            args.sidecar_dir, args.npz_dir,
            normalization_sha256=stored["normalization_sha256"],
            source_audit_identity_sha256=stored[
                "source_audit_identity_sha256"],
            project_root=REPO_ROOT,
        )
        if RUNNER.resume_decision(run_dir, fingerprint) != RUNNER.DECISION_SKIP:
            raise RuntimeError(          # defensive: resume_decision raises
                f"{run_dir}: stored run_fingerprint is not a full match")
        art = load_artifact(run_dir / "m3.pt")
        if int(art["hp"]["epochs"]) != config_epochs:
            raise RuntimeError(
                f"{run['run_name']}: artifact hp.epochs "
                f"{int(art['hp']['epochs'])} != config hp.m3.epochs "
                f"{config_epochs} -- a smoke --epochs run under this prefix "
                "is not the production matrix (the override is recorded in "
                "the artifact but not fingerprinted)")
        world_sizes.add(int(art["world_size"]))
        global_batches.add(int(art["global_batch"]))
        if len(world_sizes) > 1 or len(global_batches) > 1:
            raise RuntimeError(
                f"the scored matrix mixes world_size {sorted(world_sizes)} "
                f"and/or global_batch {sorted(global_batches)} -- every "
                "production cell must share one distributed contract")
        cells.append({**run, "dir": run_dir, "artifact": art})
    return cells


def _state_path_exists(path):
    """Recognize regular, symlink, broken-symlink, and directory state."""
    return os.path.lexists(path)


def _require_uncommitted_final_state(stats_root):
    final_dir = pathlib.Path(stats_root) / FINAL_DIR_NAME
    marker = pathlib.Path(stats_root) / MARKER_NAME
    if _state_path_exists(marker):
        raise RuntimeError(
            f"{marker} exists: the final test was already evaluated -- a "
            "one-shot transaction never reruns")
    if _state_path_exists(final_dir):
        raise RuntimeError(
            f"{final_dir} exists but its marker is missing -- resolve the "
            "half-finished transaction manually; never overwrite it")


def _publication_final_readiness(args, split=None):
    """Validate all publication provenance before the first test-target open."""
    stats_root = pathlib.Path(args.stats_root)
    if split is None:
        split = _load_top_level_split(args)
    _require_uncommitted_final_state(stats_root)
    expected_audit = pathlib.Path(os.path.abspath(
        stats_root / "source_audit_identity.json"))
    selected_audit = pathlib.Path(os.path.abspath(args.audit_identity))
    if selected_audit != expected_audit:
        raise ValueError(
            "--audit-identity must be source_audit_identity.json directly "
            "under --stats-root")
    if selected_audit.is_symlink() or not selected_audit.is_file():
        raise ValueError(
            f"{selected_audit}: source audit identity must be a regular "
            "no-follow file")
    audit = validate_source_audit_identity(
        selected_audit,
        split_path=args.split,
        split=split,
        npz_dir=args.npz_dir,
        sidecar_dir=args.sidecar_dir,
        project_root=REPO_ROOT,
    )
    audit_sha256 = sha256_file(selected_audit)

    derived = ANALYSIS.derive_validation_decision(args)
    decision_path = stats_root / "validation_decision.json"
    if decision_path.is_symlink() or not decision_path.is_file():
        raise RuntimeError(
            f"{decision_path}: validation decision must be a regular no-follow file")
    decision = validate_work2_validation_decision(
        decision_path,
        expected_payload=derived["payload"],
        validation_matrix_path=derived["table_path"],
    )
    decision_sha256 = sha256_file(decision_path)
    cells = validate_matrix(args)
    expected_fingerprints = {
        (record["arm"], int(record["seed"])): record["run_fingerprint"]
        for record in decision["runs"]
    }
    if len(expected_fingerprints) != 20:
        raise RuntimeError("validation decision does not contain exact 20 cells")
    for cell in cells:
        key = (cell["arm"], int(cell["seed"]))
        if cell["artifact"]["run_fingerprint"] != expected_fingerprints.get(key):
            raise RuntimeError(
                f"{cell['run_name']}: artifact fingerprint differs from the "
                "immutable validation decision")
    common_fp = cells[0]["artifact"]["run_fingerprint"]
    disclosure = publication_disclosure(split)
    disclosure.update({
        "manifest_sha256": sha256_file(args.split),
        "test_shots_sha256": canonical_shot_list_sha256(split.test),
        "excluded_shots_sha256": canonical_shot_list_sha256(split.excluded),
    })
    return {
        "split": split,
        "cells": cells,
        "split_disclosure": disclosure,
        "dataset": {
            "target_meta_sha256": common_fp["target_meta_sha256"],
            "sidecar_meta_sha256": common_fp["sidecar_meta_sha256"],
            "shot_metadata_sha256": common_fp["shot_metadata_sha256"],
            "slice_strata_sha256": common_fp["slice_strata_sha256"],
        },
        "source_audit": {
            **audit,
            "source_audit_identity_sha256": audit_sha256,
        },
        "validation_decision": {
            **decision,
            "validation_decision_sha256": decision_sha256,
        },
    }


def _truth_rows(npz_dir, sidecar_dir, shot, art=None):
    """(T, bnd, idx) -- truth block, native polyline and native row indices
    of the scored rows for one shot. With ``art`` the indices come from the
    artifact's own (finite-wrapped) arm reader; without it the arm-free
    ``common_valid & finite`` intersection (the representation floor)."""
    p = pathlib.Path(npz_dir) / f"{int(shot)}.npz"
    T_all, finite = load_target(p)
    if art is None:
        with np.load(pathlib.Path(sidecar_dir) / f"{int(shot)}.npz") as s:
            common = s["common_valid"].astype(bool)
        idx = np.flatnonzero(common & finite)
    else:
        idx = scored_row_indices(art, sidecar_dir, p)
    with np.load(p) as d:
        bnd = d["bnd_RZ"].astype(float)[idx]
    return T_all[idx], bnd, idx


RUN_MEMBER_NAMES = {"m3_pred.npz", "per_shot_metrics.csv"}
FLOOR_MEMBER_NAME = "representation_floor_per_shot.csv"


def _require_nofollow_type(path, *, directory):
    path = pathlib.Path(path)
    if not os.path.lexists(path):
        raise RuntimeError(f"{path}: required final member is absent")
    mode = path.lstat().st_mode
    valid = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    label = "directory" if directory else "regular file"
    if not valid:
        raise RuntimeError(f"{path}: final member must be a no-follow {label}")


def _validate_run_tree(run_dir, *, require_complete):
    run_dir = pathlib.Path(run_dir)
    _require_nofollow_type(run_dir, directory=True)
    names = {entry.name for entry in run_dir.iterdir()}
    extras = names - RUN_MEMBER_NAMES
    if extras:
        raise RuntimeError(
            f"{run_dir}: unexpected run members {sorted(extras)}")
    if require_complete and names != RUN_MEMBER_NAMES:
        raise RuntimeError(
            f"{run_dir}: run tree membership {sorted(names)} is incomplete")
    for name in names:
        _require_nofollow_type(run_dir / name, directory=False)


def _validate_final_tree(root, *, expected_run_names, require_complete):
    """Validate exact no-follow staging/final tree membership and node types."""
    root = pathlib.Path(root)
    _require_nofollow_type(root, directory=True)
    expected_runs = set(expected_run_names)
    allowed = expected_runs | {FLOOR_MEMBER_NAME}
    names = {entry.name for entry in root.iterdir()}
    extras = names - allowed
    if extras:
        raise RuntimeError(f"{root}: unexpected final-tree members {sorted(extras)}")
    if require_complete and names != allowed:
        raise RuntimeError(
            f"{root}: final-tree membership is not exact "
            f"(missing={sorted(allowed - names)})")
    for run_name in names & expected_runs:
        _validate_run_tree(
            root / run_name, require_complete=require_complete)
    if FLOOR_MEMBER_NAME in names:
        _require_nofollow_type(root / FLOOR_MEMBER_NAME, directory=False)


def _run_complete(run_dir):
    """A staged cell counts as done only when both exact regular members exist."""
    run_dir = pathlib.Path(run_dir)
    if not os.path.lexists(run_dir):
        return False
    _validate_run_tree(run_dir, require_complete=False)
    names = {entry.name for entry in run_dir.iterdir()}
    if names != RUN_MEMBER_NAMES:
        return False
    _validate_run_tree(run_dir, require_complete=True)
    return True


def _canonical_csv_int(value, *, path, field, minimum=0):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{path}: {field} is not an integer") from exc
    if parsed < minimum or str(parsed) != str(value).strip():
        raise RuntimeError(
            f"{path}: {field} must be a canonical integer >= {minimum}")
    return parsed


def _finite_csv_float(value, *, path, field):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{path}: {field} is not numeric") from exc
    if not np.isfinite(parsed):
        raise RuntimeError(f"{path}: non-finite {field}")
    return parsed


def _validate_staged_cell(cell, split, staging, expected_counts):
    """Validate one reusable/committable staged metrics and prediction pair."""
    run_dir = pathlib.Path(staging) / cell["run_name"]
    metrics_path = run_dir / "per_shot_metrics.csv"
    predictions_path = run_dir / "m3_pred.npz"
    if not _run_complete(run_dir):
        raise RuntimeError(f"{run_dir}: incomplete staged cell")
    try:
        metrics_bytes = metrics_path.read_bytes()
        metrics_text = metrics_bytes.decode("utf-8")
        reader = csv.DictReader(io.StringIO(metrics_text, newline=""))
        if tuple(reader.fieldnames or ()) != tuple(PER_SHOT_HEADER):
            raise RuntimeError(
                f"{metrics_path}: metrics header is not the frozen contract")
        rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RuntimeError(f"cannot read staged metrics at {metrics_path}: {exc}") \
            from exc
    expected_shots = tuple(int(shot) for shot in split.test)
    if len(rows) != len(expected_shots):
        raise RuntimeError(
            f"{metrics_path}: {len(rows)} rows for {len(expected_shots)} test shots")
    parsed_shots = []
    n_pred_rows = 0
    for row_index, row in enumerate(rows, start=2):
        if None in row or set(row) != set(PER_SHOT_HEADER):
            raise RuntimeError(
                f"{metrics_path}: row {row_index} has the wrong schema")
        shot = _canonical_csv_int(
            row["shot"], path=metrics_path, field=f"row {row_index} shot")
        seed = _canonical_csv_int(
            row["seed"], path=metrics_path, field=f"row {row_index} seed")
        n_slices = _canonical_csv_int(
            row["n_slices"], path=metrics_path,
            field=f"row {row_index} n_slices", minimum=1)
        cold_start_n = _canonical_csv_int(
            row["cold_start_n"], path=metrics_path,
            field=f"row {row_index} cold_start_n")
        if (row["run"] != cell["run_name"]
                or row["arm"] != cell["arm"]
                or seed != int(cell["seed"])):
            raise RuntimeError(
                f"{metrics_path}: row {row_index} claims another run/cell")
        contract = expected_counts.get(shot)
        if not isinstance(contract, dict):
            raise RuntimeError(
                f"{metrics_path}: shot {shot} has no expected row contract")
        expected_n_slices = int(contract["n_slices"])
        expected_cold_start_n = int(contract["cold_start_n"])
        if n_slices != expected_n_slices:
            raise RuntimeError(
                f"{metrics_path}: shot {shot} has {n_slices} rows, expected "
                f"{expected_n_slices}")
        if cold_start_n != expected_cold_start_n:
            raise RuntimeError(
                f"{metrics_path}: shot {shot} cold_start_n {cold_start_n} != "
                f"expected {expected_cold_start_n}")
        for field in PER_SHOT_HEADER[5:]:
            if field in {"cold_start_n", "cold_start_mean_symmetric_mm"}:
                continue
            _finite_csv_float(
                row[field], path=metrics_path,
                field=f"row {row_index} {field}")
        cold_mean_raw = row["cold_start_mean_symmetric_mm"]
        if expected_cold_start_n == 0:
            if str(cold_mean_raw).strip() != "nan":
                raise RuntimeError(
                    f"{metrics_path}: shot {shot} zero cold_start_n requires "
                    "cold_start_mean_symmetric_mm nan sentinel")
        else:
            _finite_csv_float(
                cold_mean_raw, path=metrics_path,
                field=f"row {row_index} cold_start_mean_symmetric_mm")
        parsed_shots.append(shot)
        n_pred_rows += n_slices
    if len(set(parsed_shots)) != len(parsed_shots):
        raise RuntimeError(f"{metrics_path}: duplicate per-shot metrics rows")
    if tuple(parsed_shots) != expected_shots:
        raise RuntimeError(
            f"{metrics_path}: metrics shot membership/order differs from the "
            "frozen test split")

    try:
        predictions_bytes = predictions_path.read_bytes()
        with np.load(io.BytesIO(predictions_bytes), allow_pickle=False) as predictions:
            keys = tuple(predictions.files)
            expected_keys = tuple(str(shot) for shot in expected_shots)
            if keys != expected_keys:
                raise RuntimeError(
                    f"{predictions_path}: prediction key set/order differs from "
                    "the frozen test split")
            for shot in expected_shots:
                values = predictions[str(shot)]
                expected_shape = (int(expected_counts[shot]["n_slices"]), 34)
                if values.shape != expected_shape:
                    raise RuntimeError(
                        f"{predictions_path}: shot {shot} shape {values.shape} "
                        f"!= {expected_shape}")
                if values.dtype != np.float32:
                    raise RuntimeError(
                        f"{predictions_path}: shot {shot} dtype must be float32")
                if not np.isfinite(values).all():
                    raise RuntimeError(
                        f"{predictions_path}: shot {shot} has non-finite values")
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot read staged predictions at {predictions_path}: {exc}") from exc
    return {
        "run_name": cell["run_name"],
        "arm": cell["arm"],
        "seed": int(cell["seed"]),
        "n_shots": len(rows),
        "n_pred_rows": n_pred_rows,
        "best_val_mse": float(cell["artifact"]["best_val_mse"]),
        "run_fingerprint": cell["artifact"]["run_fingerprint"],
        "per_shot_metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "m3_pred_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
    }


def _expected_prediction_counts(cell, split, npz_dir, sidecar_dir):
    """Exact scored and cold-prefix row counts from producer mask semantics."""
    artifact = cell["artifact"]
    ctx = int(artifact["hp"]["ctx"])
    contracts = {}
    for shot in split.test:
        path = pathlib.Path(npz_dir) / f"{int(shot)}.npz"
        indices = scored_row_indices(artifact, sidecar_dir, path)
        with np.load(path, allow_pickle=False) as target:
            nt = int(target["time"].size)
        contracts[int(shot)] = {
            "n_slices": int(len(indices)),
            "cold_start_n": int(np.count_nonzero(
                indices < min(ctx, nt))),
        }
    return contracts


def score_run(cell, split, npz_dir, sidecar_dir, theta, staging):
    """Predict + score one validated cell into the staging directory."""
    art = cell["artifact"]
    model = build_model(art)
    ctx = int(art["hp"]["ctx"])
    run_dir = staging / cell["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    preds, rows = {}, []
    for shot in split.test:
        P, n_scored = predict_shot(art, model, npz_dir, sidecar_dir, shot)
        T, bnd, idx = _truth_rows(npz_dir, sidecar_dir, shot, art=art)
        if P.shape[0] != len(idx) or n_scored != len(idx):
            raise RuntimeError(
                f"{cell['run_name']} shot {shot}: {P.shape[0]} predicted "
                f"rows vs {len(idx)} common-valid rows -- the gather and the "
                "predict mask disagree")
        with np.load(pathlib.Path(npz_dir) / f"{int(shot)}.npz") as d:
            nt = d["time"].size
        row, mean_sym = score_shot(P, T, bnd, theta, art["tgt_mean"],
                                   art["tgt_std"])
        cold = np.flatnonzero(idx < min(ctx, nt))   # first scored block's head
        row.update(run=cell["run_name"], arm=art["arm"], seed=int(art["seed"]),
                   shot=int(shot), cold_start_n=int(cold.size),
                   cold_start_mean_symmetric_mm=(
                       float(mean_sym[cold].mean()) if cold.size
                       else float("nan")))
        rows.append(row)
        preds[int(shot)] = P
    save_predictions(run_dir / "m3_pred.npz", preds)
    _write_csv(run_dir / "per_shot_metrics.csv", PER_SHOT_HEADER, rows)


def write_floor(split, npz_dir, sidecar_dir, theta, staging):
    """Write the model-free floor and return its exact common-valid counts."""
    rows = []
    counts = {}
    for shot in split.test:
        T, bnd, indices = _truth_rows(npz_dir, sidecar_dir, shot)
        counts[int(shot)] = int(len(indices))
        rows.append({"shot": int(shot), **floor_shot(T, bnd, theta)})
    _write_csv(staging / "representation_floor_per_shot.csv", FLOOR_HEADER,
               rows)
    return counts


def _verify_staging(cells, split, staging, expected_counts):
    """Validate every staged cell before final-directory publication."""
    return [
        _validate_staged_cell(
            cell, split, staging, expected_counts[cell["run_name"]])
        for cell in cells
    ]


def _require_production_matrix(args):
    """The one-shot transaction scores the FULL 4x5 production matrix.

    A completed subset would write the marker and consume the transaction,
    so anything but all four arms over all five production seeds is refused
    unless ``--allow-subset`` is passed explicitly.
    """
    if args.allow_subset:
        if getattr(args, "manifest_mode", "publication") == "publication":
            raise ValueError(
                "publication mode rejects --allow-subset; use explicit generic mode")
        return
    if (sorted(args.arms) != sorted(PRODUCTION_ARMS)
            or sorted(args.seeds) != sorted(PRODUCTION_SEEDS)):
        raise RuntimeError(
            f"refusing a subset matrix (arms {args.arms}, seeds "
            f"{args.seeds}): the one-shot final test scores the full "
            f"{len(PRODUCTION_ARMS)}x{len(PRODUCTION_SEEDS)} production "
            "matrix -- pass --allow-subset explicitly for a diagnostic run")


def _prepare_staging(staging):
    if _state_path_exists(staging):
        if not pathlib.Path(staging).is_dir() or os.path.islink(staging):
            raise RuntimeError(
                f"{staging}: staging must be a real directory, not redirected state")
        return
    pathlib.Path(staging).mkdir(parents=True)


def _ordered_publication_summaries(summaries, run_prefix):
    by_name = {summary["run_name"]: summary for summary in summaries}
    expected = [
        f"{run_prefix}_{arm.lower()}_s{seed}"
        for arm in PRODUCTION_ARMS for seed in PRODUCTION_SEEDS
    ]
    if set(by_name) != set(expected):
        raise RuntimeError("staged summary set is not the exact 20-cell matrix")
    return [by_name[name] for name in expected]


def _publication_marker_payload(args, readiness, floor, summaries):
    summaries = _ordered_publication_summaries(summaries, args.run_prefix)
    return build_work2_final_marker(
        split=readiness["split_disclosure"],
        dataset=readiness["dataset"],
        source_audit=readiness["source_audit"],
        validation_decision=readiness["validation_decision"],
        representation_floor={
            "path": f"{FINAL_DIR_NAME}/representation_floor_per_shot.csv",
            **floor,
        },
        matrix={
            "config_sha256": summaries[0]["run_fingerprint"]["config_sha256"],
            "arms": list(PRODUCTION_ARMS),
            "seeds": list(PRODUCTION_SEEDS),
            "run_prefix": args.run_prefix,
            "n_runs": len(summaries),
        },
        runs=summaries,
    )


def _execute_final_transaction(args, split, cells, theta, readiness=None,
                               transaction=None):
    if (getattr(args, "manifest_mode", "publication") == "publication"
            and len(cells) != len(PRODUCTION_ARMS) * len(PRODUCTION_SEEDS)):
        raise RuntimeError(
            "publication readiness requires the exact 20-cell matrix before "
            "any test-row count or target access")
    stats_root = pathlib.Path(args.stats_root)
    final_dir = stats_root / FINAL_DIR_NAME
    staging = stats_root / STAGING_NAME
    marker = stats_root / MARKER_NAME
    npz_dir = pathlib.Path(args.npz_dir)
    sidecar_dir = pathlib.Path(args.sidecar_dir)
    if transaction is not None:
        transaction.assert_held()
    _require_uncommitted_final_state(stats_root)
    _prepare_staging(staging)
    expected_run_names = tuple(cell["run_name"] for cell in cells)
    _validate_final_tree(
        staging,
        expected_run_names=expected_run_names,
        require_complete=False,
    )

    expected_counts = {}
    counts_by_arm = {}
    for i, cell in enumerate(cells):
        tag = f"[{i + 1}/{len(cells)}]"
        if cell["arm"] not in counts_by_arm:
            counts_by_arm[cell["arm"]] = _expected_prediction_counts(
                cell, split, npz_dir, sidecar_dir)
        counts = counts_by_arm[cell["arm"]]
        expected_counts[cell["run_name"]] = counts
        if _run_complete(staging / cell["run_name"]):
            _validate_staged_cell(cell, split, staging, counts)
            print(f"{tag} resume {cell['run_name']} (validated staged outputs)",
                  flush=True)
            continue
        print(f"{tag} score {cell['run_name']}", flush=True)
        score_run(cell, split, npz_dir, sidecar_dir, theta, staging)
        _validate_staged_cell(cell, split, staging, counts)

    floor_path = staging / "representation_floor_per_shot.csv"
    if _state_path_exists(floor_path) and (
            not floor_path.is_file() or floor_path.is_symlink()):
        raise RuntimeError(f"{floor_path}: floor staging path is redirected state")
    floor_counts = write_floor(
        split, npz_dir, sidecar_dir, theta, staging)
    strict_floor = getattr(args, "manifest_mode", "publication") == "publication"
    staged_floor = validate_representation_floor(
        floor_path,
        expected_shots=split.test,
        expected_n_slices=floor_counts,
        require_publication_count=strict_floor,
    )
    summaries = _verify_staging(cells, split, staging, expected_counts)
    _validate_final_tree(
        staging,
        expected_run_names=expected_run_names,
        require_complete=True,
    )

    if transaction is not None:
        transaction.assert_held()
    _require_uncommitted_final_state(stats_root)
    os.rename(staging, final_dir)
    _validate_final_tree(
        final_dir,
        expected_run_names=expected_run_names,
        require_complete=True,
    )
    final_floor = validate_representation_floor(
        final_dir / "representation_floor_per_shot.csv",
        expected_shots=split.test,
        expected_n_slices=floor_counts,
        require_publication_count=strict_floor,
    )
    if final_floor != staged_floor:
        raise RuntimeError(
            "representation floor hash changed during final-directory publication")

    if readiness is None:
        _atomic_json_dump({
            "marker": MARKER_NAME,
            "split": {"name": split.name, "version": split.version,
                      "n_test_shots": len(split.test),
                      "test_shots": [int(s) for s in split.test]},
            "config_sha256":
                cells[0]["artifact"]["run_fingerprint"]["config_sha256"],
            "arms": [str(a) for a in args.arms],
            "seeds": [int(s) for s in args.seeds],
            "run_prefix": args.run_prefix,
            "n_runs": len(summaries),
            "runs": summaries,
        }, marker)
    else:
        payload = _publication_marker_payload(
            args, readiness, final_floor, summaries)
        _verify_committed_publication(stats_root, payload)
        if transaction is not None:
            transaction.assert_held()
        if _state_path_exists(marker):
            raise RuntimeError(
                f"{marker} appeared during final transaction; refusing "
                "concurrent state")
        write_exclusive_destination(
            marker,
            canonical_work2_final_marker_bytes(payload),
            state_label="final transaction",
        )
    print(f"final test evaluated: {len(summaries)} runs x {len(split.test)} "
          f"shots -> {final_dir}\nmarker {marker} written last", flush=True)
    return 0


def run_final(args, split=None):
    """Validate readiness under the exclusive flock, then publish once."""
    _require_publication_invocation(args, operation="score")
    _require_production_matrix(args)
    manifest_mode = getattr(args, "manifest_mode", "publication")
    stats_root = pathlib.Path(args.stats_root)
    if manifest_mode == "publication":
        with ValidationFreezeLock(stats_root) as transaction:
            transaction.assert_held()
            readiness = _publication_final_readiness(args, split=split)
            npz_dir, sidecar_dir, split, theta = _preflight(
                args, split=readiness["split"])
            del npz_dir, sidecar_dir
            return _execute_final_transaction(
                args,
                split,
                readiness["cells"],
                theta,
                readiness=readiness,
                transaction=transaction,
            )
    if split is None:
        split = _load_top_level_split(args)
    _require_uncommitted_final_state(stats_root)
    _npz_dir, _sidecar_dir, split, theta = _preflight(args, split=split)
    cells = validate_matrix(args)
    return _execute_final_transaction(args, split, cells, theta)


def _verify_committed_publication(stats_root, marker):
    """Verify the marker-bound final tree without opening source targets."""
    stats_root = pathlib.Path(stats_root)
    final_dir = stats_root / FINAL_DIR_NAME
    expected_run_names = tuple(run["run_name"] for run in marker["runs"])
    _validate_final_tree(
        final_dir,
        expected_run_names=expected_run_names,
        require_complete=True,
    )
    for run in marker["runs"]:
        run_dir = final_dir / run["run_name"]
        if sha256_file(run_dir / "per_shot_metrics.csv") != \
                run["per_shot_metrics_sha256"]:
            raise RuntimeError(
                f"{run_dir}: metrics hash disagrees with final marker")
        if sha256_file(run_dir / "m3_pred.npz") != run["m3_pred_sha256"]:
            raise RuntimeError(
                f"{run_dir}: prediction hash disagrees with final marker")
    floor = validate_representation_floor(
        stats_root / marker["representation_floor"]["path"],
        expected_shots=marker["split"]["test"],
    )
    if floor != {
            key: marker["representation_floor"][key]
            for key in ("sha256", "header", "n_shots", "shots",
                        "shots_sha256", "n_slices")}:
        raise RuntimeError(
            "representation floor provenance disagrees with final marker")
    return floor


def verify_only(args, split=None):
    """Validate canonical readiness or committed state without test targets."""
    _require_publication_invocation(args, operation="verify")
    _require_production_matrix(args)
    stats_root = pathlib.Path(args.stats_root)
    marker_path = stats_root / MARKER_NAME
    manifest_mode = getattr(args, "manifest_mode", "publication")
    if manifest_mode == "publication":
        if split is None:
            split = _load_top_level_split(args)
        if _state_path_exists(marker_path):
            marker = validate_work2_final_marker(marker_path)
            _verify_committed_publication(stats_root, marker)
            print(f"already evaluated and verified: {marker_path}")
            return 0
        with ValidationFreezeLock(stats_root) as transaction:
            transaction.assert_held()
            readiness = _publication_final_readiness(args, split=split)
            _preflight(args, split=readiness["split"])
            cells = readiness["cells"]
            staging = stats_root / STAGING_NAME
            staged = sum(
                _run_complete(staging / cell["run_name"])
                for cell in cells
            )
            print(f"{len(cells)} runs validated against immutable readiness")
            if staging.exists():
                print(f"staging present: {staged}/{len(cells)} cells have both "
                      f"outputs under {staging}")
            return 0

    if split is None:
        split = _load_top_level_split(args)
    if _state_path_exists(marker_path):
        print(f"already evaluated: {marker_path} exists -- the final test is "
              "committed and will not rerun")
        return 0
    _preflight(args, split=split)
    cells = validate_matrix(args)
    staging = stats_root / STAGING_NAME
    staged = sum(_run_complete(staging / c["run_name"]) for c in cells)
    print(f"{len(cells)} runs validated (fingerprint + artifact contract)")
    if staging.exists():
        print(f"staging present: {staged}/{len(cells)} cells complete under "
              f"{staging}")
    return 0


def _arm_arg(value):
    arm = value.upper()
    if arm not in ARMS:
        raise argparse.ArgumentTypeError(
            f"arm must be one of {ARMS}, got {value!r}")
    return arm


def build_parser():
    ap = argparse.ArgumentParser(
        description="One-shot final test evaluation of the PF-observability "
                    "matrix (hash-gated, transactional)")
    ap.add_argument("--npz-dir", default=str(NPZ_DIR),
                    help="target dataset (default: ProjDB/datasets/NpzGeom)")
    ap.add_argument("--sidecar-dir", default=str(SIDECAR_DIR),
                    help="PF ref/actual sidecar "
                         "(default: ProjDB/datasets/NpzGeomPFObs)")
    ap.add_argument("--config", default=str(CONFIG),
                    help="shared arm config "
                         "(default: configs/dcs_pf_observability.yml)")
    ap.add_argument("--split", default=str(SPLIT),
                    help="frozen split manifest (default: the final "
                         "campaign manifest)")
    ap.add_argument("--manifest-mode", choices=("publication", "generic"),
                    default="publication",
                    help="strict publication bundle validation (default) or "
                         "explicit archived generic-manifest loading")
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="validation-artifact root (default: ProjDB/trains)")
    ap.add_argument("--stats-root", default=str(STATS_ROOT),
                    help="stats root holding validation/final state "
                         f"(default: {STATS_ROOT})")
    ap.add_argument("--audit-identity", default=str(AUDIT_IDENTITY),
                    help="canonical source_audit_identity.json directly under "
                         "the publication stats root")
    ap.add_argument("--arms", nargs="+", type=_arm_arg, default=list(ARMS),
                    help="arms to score (default: A B C D)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="seeds to score (default: 0 1 2 3 4)")
    ap.add_argument("--run-prefix", default="pfobs",
                    help="run_name prefix (default: pfobs -> pfobs_a_s0)")
    ap.add_argument("--allow-subset", action="store_true",
                    help="permit scoring a partial matrix (diagnostics "
                         "only); without it run_final refuses anything but "
                         "the full 4x5 production matrix")
    ap.add_argument("--verify-only", action="store_true",
                    help="validate the matrix and report transaction state; "
                         "never loads test targets, writes nothing")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.verify_only:
        return verify_only(args)
    return run_final(args)


if __name__ == "__main__":
    sys.exit(main())
