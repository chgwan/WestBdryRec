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
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.dcs_features import load_dcs_config, load_meta  # noqa: E402
from src.ml.pf_observability import ARMS, load_split, run_fingerprint  # noqa: E402
from src.ml.pfobs_infer import (  # noqa: E402
    build_model, floor_shot, load_artifact, predict_shot, scored_row_indices,
    score_shot,
)
from src.ml.pfobs_train import _available_shots, validate_base_contract  # noqa: E402
from src.ml.predictions import save_predictions  # noqa: E402
from src.ml.target import load_target  # noqa: E402
from src.proj_config import get_proj_config  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NPZ_DIR = REPO_ROOT / "ProjDB/datasets/NpzGeom"
SIDECAR_DIR = REPO_ROOT / "ProjDB/datasets/NpzGeomPFObs"
CONFIG = REPO_ROOT / "configs/dcs_pf_observability.yml"
SPLIT = REPO_ROOT / "configs/splits/communications_physics_campaign_v1.json"
OUT_ROOT = REPO_ROOT / "ProjDB/trains"
STATS_ROOT = get_proj_config().pfobs_stats_dir

FINAL_DIR_NAME = "final_test"
STAGING_NAME = FINAL_DIR_NAME + ".staging"
MARKER_NAME = "FINAL_TEST_EVALUATED.json"
# The one-shot transaction scores exactly this matrix; anything smaller needs
# an explicit --allow-subset (diagnostics only, never the committed result).
PRODUCTION_ARMS = tuple(ARMS)
PRODUCTION_SEEDS = (0, 1, 2, 3, 4)

PER_SHOT_HEADER = [
    "run", "arm", "seed", "shot", "n_slices", "mean_symmetric_mm", "p95_mm",
    "chamfer_rms_mm", "hausdorff_mm", "area_abs_m2", "centroid_mm",
    "elongation_abs", "triangularity_upper_abs", "triangularity_lower_abs",
    "ccc", "radii_mse", "centre_mse", "cold_start_n",
    "cold_start_mean_symmetric_mm"]
FLOOR_HEADER = [
    "shot", "n_slices", "mean_symmetric_mm", "p95_mm", "chamfer_rms_mm",
    "hausdorff_mm", "area_abs_m2", "centroid_mm", "elongation_abs",
    "triangularity_upper_abs", "triangularity_lower_abs"]


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


def _atomic_json_dump(obj, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def _preflight(args):
    """Symmetric config/split/dataset preflight -- no test target is opened."""
    npz_dir = pathlib.Path(args.npz_dir)
    sidecar_dir = pathlib.Path(args.sidecar_dir)
    validate_base_contract(load_dcs_config(args.config))
    split = load_split(args.split, available_shots=_available_shots(
        npz_dir, sidecar_dir))
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
        fingerprint = run_fingerprint(run["arm"], run["seed"], args.config,
                                      args.split, args.sidecar_dir,
                                      args.npz_dir)
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


def _run_complete(run_dir):
    """A staged cell counts as done only when BOTH outputs exist."""
    return ((run_dir / "m3_pred.npz").exists()
            and (run_dir / "per_shot_metrics.csv").exists())


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
    """The model-free representation floor, one row per frozen test shot."""
    rows = []
    for shot in split.test:
        T, bnd, _idx = _truth_rows(npz_dir, sidecar_dir, shot)
        rows.append({"shot": int(shot), **floor_shot(T, bnd, theta)})
    _write_csv(staging / "representation_floor_per_shot.csv", FLOOR_HEADER,
               rows)


def _verify_staging(cells, split, staging):
    """Every cell complete and every frozen test shot scored, or refuse."""
    summaries = []
    for cell in cells:
        run_dir = staging / cell["run_name"]
        if not _run_complete(run_dir):
            raise RuntimeError(f"{run_dir}: incomplete after scoring -- "
                               "refusing to commit the final test")
        with (run_dir / "per_shot_metrics.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        scored = {int(r["shot"]) for r in rows}
        if scored != {int(s) for s in split.test}:
            missing = sorted({int(s) for s in split.test} - scored)
            raise RuntimeError(
                f"{run_dir}: scored shots {sorted(scored)} != frozen test "
                f"shots -- missing {missing}")
        art = cell["artifact"]
        summaries.append({
            "run_name": cell["run_name"], "arm": art["arm"],
            "seed": int(art["seed"]), "n_shots": len(rows),
            "n_pred_rows": sum(int(r["n_slices"]) for r in rows),
            "best_val_mse": float(art["best_val_mse"]),
            "run_fingerprint": art["run_fingerprint"]})
    return summaries


def _require_production_matrix(args):
    """The one-shot transaction scores the FULL 4x5 production matrix.

    A completed subset would write the marker and consume the transaction,
    so anything but all four arms over all five production seeds is refused
    unless ``--allow-subset`` is passed explicitly.
    """
    if args.allow_subset:
        return
    if (sorted(args.arms) != sorted(PRODUCTION_ARMS)
            or sorted(args.seeds) != sorted(PRODUCTION_SEEDS)):
        raise RuntimeError(
            f"refusing a subset matrix (arms {args.arms}, seeds "
            f"{args.seeds}): the one-shot final test scores the full "
            f"{len(PRODUCTION_ARMS)}x{len(PRODUCTION_SEEDS)} production "
            "matrix -- pass --allow-subset explicitly for a diagnostic run")


def run_final(args):
    """The final transaction: validate all, stage, verify, rename, mark."""
    _require_production_matrix(args)
    stats_root = pathlib.Path(args.stats_root)
    final_dir = stats_root / FINAL_DIR_NAME
    staging = stats_root / STAGING_NAME
    marker = stats_root / MARKER_NAME
    if marker.exists():
        raise RuntimeError(
            f"{marker} exists: the final test was already evaluated -- a "
            "one-shot transaction never reruns")
    if final_dir.exists():
        raise RuntimeError(
            f"{final_dir} exists but its marker is missing -- resolve the "
            "half-finished transaction manually; never overwrite it")

    npz_dir, sidecar_dir, split, theta = _preflight(args)
    cells = validate_matrix(args)          # ALL cells before any test load
    staging.mkdir(parents=True, exist_ok=True)
    for i, cell in enumerate(cells):
        tag = f"[{i + 1}/{len(cells)}]"
        if _run_complete(staging / cell["run_name"]):
            print(f"{tag} resume {cell['run_name']} (staged outputs present)",
                  flush=True)
            continue
        print(f"{tag} score {cell['run_name']}", flush=True)
        score_run(cell, split, npz_dir, sidecar_dir, theta, staging)
    # the floor is model-free, so it is ALWAYS rewritten from the frozen
    # split: a stale floor left in staging by an earlier aborted run (a
    # different split, older code) can never be committed on existence alone
    write_floor(split, npz_dir, sidecar_dir, theta, staging)
    summaries = _verify_staging(cells, split, staging)

    os.rename(staging, final_dir)          # atomic: the target is absent
    _atomic_json_dump({                   # the marker is written LAST
        "marker": MARKER_NAME,
        "split": {"name": split.name, "version": split.version,
                  "n_test_shots": len(split.test),
                  "test_shots": [int(s) for s in split.test]},
        # every stored fingerprint was validated equal to the fresh one above
        "config_sha256":
            cells[0]["artifact"]["run_fingerprint"]["config_sha256"],
        "arms": [str(a) for a in args.arms],
        "seeds": [int(s) for s in args.seeds],
        "run_prefix": args.run_prefix,
        "n_runs": len(summaries),
        "runs": summaries,
    }, marker)
    print(f"final test evaluated: {len(summaries)} runs x {len(split.test)} "
          f"shots -> {final_dir}\nmarker {marker} written last", flush=True)
    return 0


def verify_only(args):
    """Validate the matrix and report transaction state; load no test target."""
    stats_root = pathlib.Path(args.stats_root)
    marker = stats_root / MARKER_NAME
    if marker.exists():
        print(f"already evaluated: {marker} exists -- the final test is "
              "committed and will not rerun")
        return 0
    _preflight(args)
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
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="validation-artifact root (default: ProjDB/trains)")
    ap.add_argument("--stats-root", default=str(STATS_ROOT),
                    help="stats root holding final_test and the marker "
                         f"(default: {STATS_ROOT})")
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
