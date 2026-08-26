# -*- coding: utf-8 -*-
"""The one-shot final test transaction for the PF-context matrix.

Loads NO test target until every cell of the requested matrix (production:
all 35 = 7 contexts x 5 seeds) validates: artifact + fingerprint JSON present,
the stored ``run_fingerprint`` equal to the freshly computed one, the fixed
base contract intact (Arm B width 21, ``pe = rope_time``), and the artifact's
claim scope neither ``smoke_only`` nor ``dry_run`` (Task 5's channels). A
missing cell, any hash mismatch, or a smoke/dry artifact aborts before a
single test shot is opened.

The transaction runs under the stats root (default
``ProjDB/Stats/pf_context``): everything is written into the sibling staging
directory ``ProjDB/Stats/pf_context.final_building`` (a crashed run is
disposable and resumable -- cells whose three output files already exist are
not rescored), then verified complete, then the staging directory is
atomically renamed to ``ProjDB/Stats/pf_context/final_test`` and only THEN is
``PFCTX_FINAL_TEST_EVALUATED.json`` written, last. An existing marker (or an
existing ``final_test`` directory) refuses the run; no CLI flag removes or
bypasses either. The 32-angle representation floor is read from the upstream
final scorer's table when present and recorded as provenance -- it is never
recomputed and never subtracted from any metric.

Devices: ``--devices 0 1 2 3`` makes the non-DDP parent spawn one worker
process per device, the artifacts partitioned deterministically across them
(worker ``w`` of ``n`` scores matrix entry ``w::n``, a pure function of the
ordered matrix). The single ``--devices cpu`` case is the parent-process
convenience path used by tests and diagnostics. After the workers finish, the
parent validates every prediction file, every metric file, every frozen test
shot, and identical row indices across contexts per shot before publishing.

``--verify-only`` performs the readiness checks without importing
``src.ml.pfctx_infer``, creating workers, or opening any test target.

Usage (from the repo root, after the matrix runner completed):
  python scripts/score_pf_context_sweep.py --split <frozen-manifest>
  python scripts/score_pf_context_sweep.py --split <frozen-manifest> \
    --devices 0 1 2 3
  python scripts/score_pf_context_sweep.py --split <frozen-manifest> \
    --devices 0 1 2 3 --verify-only
"""
import argparse
import csv
import datetime
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ml.pf_context import (  # noqa: E402
    CONTEXT_LEVELS, ContextLevel, context_level,
)
from src.ml.pf_observability import sha256_file  # noqa: E402
from src.ml.pfctx_train import (  # noqa: E402
    INPUT_WIDTH, N_OUT, REQUIRED_ARTIFACT_KEYS,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "ProjDB/datasets/NpzGeom"
SIDECAR_DIR = REPO_ROOT / "ProjDB/datasets/NpzGeomPFObs"
CONFIG = REPO_ROOT / "configs/dcs_pf_context_sweep.yml"
OUT_ROOT = REPO_ROOT / "ProjDB/trains"
STATS_ROOT = REPO_ROOT / "ProjDB/Stats/pf_context"

FINAL_DIR_NAME = "final_test"
BUILDING_SUFFIX = ".final_building"
MARKER_NAME = "PFCTX_FINAL_TEST_EVALUATED.json"
# The one-shot transaction scores exactly this matrix by default; anything a
# caller explicitly requests instead is scored as requested and recorded in
# the marker, but the production default is the full 35-cell matrix.
PRODUCTION_SEEDS = (0, 1, 2, 3, 4)
PRODUCTION_ARTIFACT_COUNT = len(CONTEXT_LEVELS) * len(PRODUCTION_SEEDS)
DEFAULT_DEVICES = ("0", "1", "2", "3")
# The upstream Essential-Work-2 final scorer's representation-floor table:
# read for provenance only, never recomputed or subtracted.
DEFAULT_FLOOR_CSV = (REPO_ROOT / "ProjDB/Stats/pf_observability"
                     / FINAL_DIR_NAME / "representation_floor_per_shot.csv")

WORKER_FLAG = "--score-partition"


def _load_runner():
    """scripts/ is not a package; load the validation runner by path so the
    matrix naming and fingerprint logic has exactly one implementation."""
    spec = importlib.util.spec_from_file_location(
        "run_pf_context_sweep",
        pathlib.Path(__file__).resolve().parent / "run_pf_context_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUNNER = _load_runner()


def pfctx_infer_module():
    """The ONE lazy import of the inference module: only the scoring path
    (workers and the single-CPU parent path) ever calls this, so
    ``--verify-only`` keeps it out of the process."""
    from src.ml import pfctx_infer
    return pfctx_infer


def building_root(args):
    """The staging directory: a SIBLING of the stats root, so the stats root
    itself only ever contains published content."""
    stats_root = pathlib.Path(args.stats_root)
    return stats_root.parent / (stats_root.name + BUILDING_SUFFIX)


def marker_path(args):
    return pathlib.Path(args.stats_root) / MARKER_NAME


def final_dir_path(args):
    return pathlib.Path(args.stats_root) / FINAL_DIR_NAME


def _atomic_json_dump(obj, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


# ── readiness: every fingerprint validated before any test target ─────
def _expected_entries(args):
    contexts = [x if isinstance(x, ContextLevel) else context_level(str(x))
                for x in args.contexts]
    return RUNNER.matrix_entries(contexts, [int(s) for s in args.seeds],
                                 prefix=str(args.run_prefix))


def _check_artifact_contract(run_name, artifact):
    """The pfctx_infer-free contract subset (--verify-only must never import
    the inference module); the inference primitive re-checks the same tokens
    through ``validate_artifact_contract``."""
    missing = [key for key in REQUIRED_ARTIFACT_KEYS if key not in artifact]
    if missing:
        raise ValueError(f"{run_name}: artifact is missing {missing}")
    tokens = {"study": "pf_context", "arm": "B", "model": "ActSeqAttn",
              "time_axis": "native_gmag_bnd"}
    for key, expected in tokens.items():
        if artifact[key] != expected:
            raise RuntimeError(
                f"{run_name}: artifact {key} must be {expected!r}, got "
                f"{artifact[key]!r}")
    if artifact["pe"] != "rope_time":
        raise RuntimeError(
            f"{run_name}: pe must be 'rope_time', got {artifact['pe']!r} -- "
            "the context sweep is only comparable under the fixed base model")
    if int(artifact["n_act"]) != INPUT_WIDTH:
        raise RuntimeError(
            f"{run_name}: the fixed Arm B input is {INPUT_WIDTH} columns, "
            f"got {int(artifact['n_act'])}")
    if int(artifact["n_out"]) != N_OUT:
        raise RuntimeError(
            f"{run_name}: n_out must be {N_OUT}, got {artifact['n_out']}")


def _check_entry_consistency(run_name, entry, artifact):
    if str(artifact["context_label"]) != entry.context.label:
        raise RuntimeError(
            f"{run_name}: artifact context_label "
            f"{artifact['context_label']!r} does not name its matrix cell "
            f"{entry.context.label!r}")
    if (float(artifact["context_seconds"]) != entry.context.seconds
            or int(artifact["nominal_samples"])
            != entry.context.nominal_samples):
        raise RuntimeError(
            f"{run_name}: artifact context geometry disagrees with the "
            "frozen grid")
    if int(artifact["seed"]) != int(entry.seed):
        raise RuntimeError(
            f"{run_name}: artifact seed {artifact['seed']} does not name its "
            f"matrix cell seed {entry.seed}")


def verify_final_readiness(args):
    """Every cell of the requested matrix stored, hash-valid, contract-clean,
    and scope-clean -- all before the caller touches any test target.

    Returns the validated cells (``entry`` / ``dir`` / ``artifact``). A
    missing cell is a ValueError naming the production count; a fingerprint
    mismatch is a RuntimeError (a stored run is never overwritten); a
    smoke/dry artifact is a RuntimeError extending the runner's channels.
    """
    split = RUNNER.load_context_split(args.split)
    entries = _expected_entries(args)
    out_root = pathlib.Path(args.out_root)
    cells, missing = [], []
    for entry in entries:
        run_dir = out_root / entry.run_name
        if not ((run_dir / "m3.pt").is_file()
                and (run_dir / "fingerprint.json").is_file()):
            missing.append(entry.run_name)
            continue
        artifact = torch.load(run_dir / "m3.pt", map_location="cpu",
                              weights_only=False)
        _check_artifact_contract(entry.run_name, artifact)
        _check_entry_consistency(entry.run_name, entry, artifact)
        scope = artifact.get("claim_scope")
        if scope in ("smoke_only", "dry_run"):
            raise RuntimeError(
                f"{entry.run_name}: stored artifact carries claim_scope "
                f"{scope!r} and can never be part of the final test -- "
                "smoke and dry-run channels are refused by the final scorer")
        stats = (artifact["feature_mean"], artifact["feature_std"],
                 artifact["target_mean"], artifact["target_std"])
        fresh = RUNNER.entry_payload(
            entry, split, pathlib.Path(args.config),
            pathlib.Path(args.target_dir), pathlib.Path(args.sidecar_dir),
            REPO_ROOT, stats)
        stored = json.loads(
            (run_dir / "fingerprint.json").read_text())
        compared = (set(stored) | set(fresh)) - {"run_name"}
        differing = [key for key in sorted(compared)
                     if stored.get(key) != fresh.get(key)]
        if differing:
            raise RuntimeError(
                f"{entry.run_name}: stored run disagrees with the final "
                f"scorer's freshly computed fingerprint ({', '.join(differing)}) "
                "-- never score a drifted matrix; move the directory aside "
                "or retrain it under the current pinned inputs")
        cells.append({"entry": entry, "dir": run_dir, "artifact": artifact})
    if missing:
        raise ValueError(
            f"final scorer requires all {len(entries)} artifacts of the "
            f"requested matrix (production: {PRODUCTION_ARTIFACT_COUNT} "
            f"artifacts) -- missing: {', '.join(missing)}")
    return cells


# ── scoring: deterministic partition, workers or the parent on CPU ────
def partition_entries(entries, n_workers):
    """Worker ``w`` of ``n`` scores ``entries[w::n]``.

    Deterministic by construction: a pure function of the ordered matrix and
    the worker count (no timing, no environment, no hash-order inputs), so
    the parent and every worker derive the identical disjoint partition, and
    re-deriving it in a fresh process yields the same assignment.
    """
    workers = int(n_workers)
    if workers < 1:
        raise ValueError("n_workers must be positive")
    return [list(entries)[i::workers] for i in range(workers)]


def _cell_dir(root, entry):
    return pathlib.Path(root) / entry.context.label / f"s{int(entry.seed)}"


def _cell_complete(cell_dir):
    return all((cell_dir / name).is_file() for name in (
        "m3_pred.npz", "per_shot_metrics.csv", "run_metadata.json"))


def _load_theta(target_dir):
    meta = json.loads(
        (pathlib.Path(target_dir) / "meta.json").read_text())
    if "theta_deg" not in meta:
        raise ValueError(
            f"{target_dir}/meta.json: no 'theta_deg' -- the uniform angle "
            "grid is required to rebuild contours")
    return np.deg2rad(np.asarray(meta["theta_deg"], float))


def _floor_provenance(args):
    """The upstream representation-floor table: located, hashed, counted --
    never recomputed, never subtracted from any metric."""
    path = pathlib.Path(getattr(args, "floor_csv", None) or DEFAULT_FLOOR_CSV)
    provenance = {"path": str(path), "present": path.is_file()}
    if provenance["present"]:
        with path.open(newline="") as fh:
            provenance["n_rows"] = max(
                sum(1 for _ in csv.DictReader(fh)), 0)
        provenance["sha256"] = sha256_file(path)
    return provenance


def score_entries(args, entries, device):
    """Score one partition of artifacts into the staging root (resumable:
    a cell whose three files already exist is not rescored)."""
    infer = pfctx_infer_module()
    split = RUNNER.load_context_split(args.split)
    theta = _load_theta(args.target_dir)
    floor = _floor_provenance(args)
    staging = building_root(args)
    staged = resumed = 0
    for entry in entries:
        cell = _cell_dir(staging, entry)
        if _cell_complete(cell):
            resumed += 1
            print(f"resume {entry.run_name} (staged outputs present)",
                  flush=True)
            continue
        artifact = torch.load(
            pathlib.Path(args.out_root) / entry.run_name / "m3.pt",
            map_location="cpu", weights_only=False)
        print(f"score {entry.run_name} on {device}", flush=True)
        infer.score_artifact(
            artifact, list(split.test), pathlib.Path(args.target_dir),
            pathlib.Path(args.sidecar_dir), theta, cell, device=device,
            floor=floor)
        staged += 1
    print(f"partition done: {staged} scored, {resumed} resumed", flush=True)
    return staged, resumed


def _worker_command(args, worker_index, n_workers, device):
    return [
        sys.executable, str(pathlib.Path(__file__).resolve()), WORKER_FLAG,
        "--worker-index", str(int(worker_index)),
        "--n-workers", str(int(n_workers)), "--device", str(device),
        "--split", str(args.split), "--config", str(args.config),
        "--target-dir", str(args.target_dir),
        "--sidecar-dir", str(args.sidecar_dir),
        "--out-root", str(args.out_root),
        "--stats-root", str(args.stats_root),
        "--contexts", *[x.label for x in args.contexts],
        "--seeds", *[str(int(s)) for s in args.seeds],
        "--run-prefix", str(args.run_prefix),
        "--floor-csv", str(getattr(args, "floor_csv", None)
                           or DEFAULT_FLOOR_CSV),
    ]


def _spawn_device_workers(args, devices):
    """One worker process per device over the deterministic partitions; the
    parent waits for all of them and fails the transaction on any failure."""
    commands = [_worker_command(args, index, len(devices), device)
                for index, device in enumerate(devices)]
    processes = [subprocess.Popen(
        command, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "OMP_NUM_THREADS": os.environ.get(
            "OMP_NUM_THREADS", "4"), "HDF5_USE_FILE_LOCKING": "FALSE"})
        for command in commands]
    failures = []
    for index, device, command, process in zip(
            range(len(devices)), devices, commands, processes):
        output, _ = process.communicate()
        if process.returncode != 0:
            failures.append((command, output or ""))
            print(f"device worker FAILED (worker {index}, device {device})",
                  flush=True)
            print(output, end="", flush=True)
        else:
            print(f"device worker ok: worker {index}, device {device}",
                  flush=True)
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(devices)} device workers failed -- "
            "the staging is incomplete and nothing will be published")


def _verify_staging(args, entries, split):
    """Every cell complete, every frozen test shot present in every cell, and
    identical row indices across contexts per shot -- or refuse to publish."""
    infer = pfctx_infer_module()
    staging = building_root(args)
    expected_shots = sorted(int(s) for s in split.test)
    reference_indices = {}
    summaries = []
    for entry in entries:
        cell = _cell_dir(staging, entry)
        absent = [name for name in ("m3_pred.npz", "per_shot_metrics.csv",
                                    "run_metadata.json")
                  if not (cell / name).is_file()]
        if absent:
            raise RuntimeError(
                f"{cell}: incomplete after scoring (missing {', '.join(absent)}) "
                "-- refusing to commit the final test")
        with (cell / "per_shot_metrics.csv").open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        scored = sorted(int(row["shot"]) for row in rows)
        if scored != expected_shots:
            missing_shots = sorted(set(expected_shots) - set(scored))
            raise RuntimeError(
                f"{cell}: scored shots {scored} != frozen test shots "
                f"{expected_shots} -- missing shot rows {missing_shots}")
        predictions = infer.load_final_predictions(cell / "m3_pred.npz")
        if sorted(predictions) != expected_shots:
            raise RuntimeError(
                f"{cell}: prediction shots {sorted(predictions)} != frozen "
                f"test shots {expected_shots} -- missing shot rows "
                f"{sorted(set(expected_shots) - set(predictions))}")
        for shot in expected_shots:
            rows_here = predictions[shot].row_index
            if shot in reference_indices:
                if not np.array_equal(rows_here,
                                      reference_indices[shot]):
                    raise RuntimeError(
                        f"shot {shot}: row indices differ across contexts "
                        f"({entry.run_name} vs an earlier artifact) -- every "
                        "context must score the identical common-valid rows")
            else:
                reference_indices[shot] = rows_here
        metadata = json.loads((cell / "run_metadata.json").read_text())
        summaries.append({
            "run_name": metadata["run_name"],
            "context": metadata["context_label"],
            "seed": int(metadata["seed"]),
            "n_shots": len(rows),
            "n_pred_rows": int(sum(int(row["n_slices"]) for row in rows)),
            "best_val_mse": float(metadata["best_val_mse"]),
            "fingerprints": metadata["fingerprints"],
        })
    return summaries


def _require_production_matrix(args):
    """The one-shot transaction must never be consumed by a subset.

    Under the production-default stats root the marker's no-rerun guarantee
    would make a subset matrix permanently block the full 35-cell run, so
    anything but the frozen 7x5 matrix is refused before any staging. A
    ``--stats-root`` pointing elsewhere (the fixtures' temporary roots, a
    diagnostic run) is the sanctioned escape and keeps the parameterized
    shape ``verify_final_readiness`` exposes.
    """
    if (pathlib.Path(args.stats_root).resolve()
            != pathlib.Path(STATS_ROOT).resolve()):
        return
    contexts = sorted(x.label for x in args.contexts)
    seeds = sorted(int(s) for s in args.seeds)
    if (contexts != sorted(x.label for x in CONTEXT_LEVELS)
            or seeds != sorted(PRODUCTION_SEEDS)):
        raise RuntimeError(
            f"refusing a subset matrix (contexts {contexts}, seeds {seeds}) "
            f"under the production stats root {STATS_ROOT}: the one-shot "
            "final test scores the full "
            f"{len(CONTEXT_LEVELS)}x{len(PRODUCTION_SEEDS)} production "
            "matrix and its marker can never be rerun -- point --stats-root "
            "at a separate root for a diagnostic run")


def run_final_transaction(args):
    """The final transaction: validate all, stage, verify, rename, mark."""
    stats_root = pathlib.Path(args.stats_root)
    marker = marker_path(args)
    final_dir = final_dir_path(args)
    building = building_root(args)
    if marker.exists():
        raise RuntimeError(
            f"{marker} exists: the final test was already evaluated -- a "
            "one-shot transaction never reruns and no flag bypasses it")
    if final_dir.exists():
        raise RuntimeError(
            f"{final_dir} exists but its marker is missing -- resolve the "
            "half-finished transaction manually; never overwrite it")
    _require_production_matrix(args)     # a subset never consumes the shot

    split = RUNNER.load_context_split(args.split)
    cells = verify_final_readiness(args)     # ALL cells before any test load
    entries = [cell["entry"] for cell in cells]
    building.mkdir(parents=True, exist_ok=True)
    devices = [str(device) for device in args.devices]
    if len(devices) == 1 and devices[0] == "cpu":
        score_entries(args, entries, "cpu")   # the parent-process CPU path
    else:
        _spawn_device_workers(args, devices)
    summaries = _verify_staging(args, entries, split)

    stats_root.mkdir(parents=True, exist_ok=True)   # the marker's home
    os.rename(building, final_dir)            # atomic: the target is absent
    _atomic_json_dump({                       # the marker is written LAST
        "marker": MARKER_NAME,
        "evaluated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "split": {"name": split.name, "version": int(split.version),
                  "n_test_shots": len(split.test),
                  "test_shots": [int(s) for s in split.test]},
        "config_sha256": cells[0]["artifact"]["fingerprints"]["config_sha256"],
        "source_sha256": cells[0]["artifact"]["fingerprints"]["source_sha256"],
        "contexts": [x.label for x in args.contexts],
        "seeds": [int(s) for s in args.seeds],
        "run_prefix": str(args.run_prefix),
        "devices": [str(d) for d in args.devices],
        "n_artifacts": len(summaries),
        "n_shots": len(split.test),
        "floor": _floor_provenance(args),
        "runs": summaries,
    }, marker)
    print(f"final test evaluated: {len(summaries)} artifacts x "
          f"{len(split.test)} shots -> {final_dir}\nmarker {marker} "
          "written last", flush=True)
    return 0


def verify_only(args):
    """Readiness checks and transaction state; no inference module, no
    workers, no test target."""
    marker = marker_path(args)
    if marker.exists():
        print(f"already evaluated: {marker} exists -- the final test is "
              "committed and will not rerun")
        return 0
    cells = verify_final_readiness(args)
    print(f"{len(cells)} artifacts validated (fingerprint + artifact "
          "contract)")
    building = building_root(args)
    if building.exists():
        staged = sum(_cell_complete(_cell_dir(building, cell["entry"]))
                     for cell in cells)
        print(f"staging present: {staged}/{len(cells)} cells complete under "
              f"{building}")
    return 0


def _worker_main(args):
    entries = _expected_entries(args)
    partition = partition_entries(entries, args.n_workers)[args.worker_index]
    staged, resumed = score_entries(args, partition, args.device)
    print(f"worker {args.worker_index}/{args.n_workers} device "
          f"{args.device}: {staged} scored, {resumed} resumed", flush=True)
    return 0


def _context_arg(value):
    return context_level(str(value))


def build_parser():
    with open(CONFIG) as fh:
        frozen = yaml.safe_load(fh)
    parser = argparse.ArgumentParser(
        description="One-shot final test transaction of the PF-context "
                    "matrix (hash-gated, atomic, multi-device)")
    parser.add_argument("--split", required=True,
                        help="frozen split manifest (the test-shot source)")
    parser.add_argument("--config", default=str(CONFIG),
                        help="frozen sweep configuration "
                             f"(default: {CONFIG})")
    parser.add_argument("--target-dir", default=str(TARGET_DIR),
                        help="target dataset (default: ProjDB/datasets/"
                             "NpzGeom)")
    parser.add_argument("--sidecar-dir", default=str(SIDECAR_DIR),
                        help="PF ref/actual sidecar (default: ProjDB/"
                             "datasets/NpzGeomPFObs)")
    parser.add_argument("--out-root", default=str(OUT_ROOT),
                        help="validation-artifact root (default: "
                             "ProjDB/trains)")
    parser.add_argument("--stats-root", default=str(STATS_ROOT),
                        help="stats root holding final_test and the marker "
                             f"(default: {STATS_ROOT})")
    parser.add_argument("--floor-csv", default=str(DEFAULT_FLOOR_CSV),
                        help="upstream representation-floor table, recorded "
                             "as provenance only (never recomputed or "
                             "subtracted)")
    parser.add_argument("--contexts", nargs="+", type=_context_arg,
                        default=[context_level(c["label"])
                                 for c in frozen["contexts"]],
                        help="context levels to score (default: the frozen "
                             "seven-level grid)")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[int(s) for s in frozen["seeds"]],
                        help="seeds to score (default: the frozen 0 1 2 3 4)")
    parser.add_argument("--run-prefix", default="pfctx",
                        help="run_name prefix (default: pfctx -> "
                             "pfctx_h0512_s0)")
    parser.add_argument("--devices", nargs="+", default=list(DEFAULT_DEVICES),
                        help="one scoring worker process per device "
                             "(default: 0 1 2 3; 'cpu' scores in the "
                             "parent)")
    parser.add_argument("--verify-only", action="store_true",
                        help="readiness checks only: no inference module, "
                             "no workers, no test-target access")
    parser.add_argument(WORKER_FLAG, action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--device", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--n-workers", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.score_partition:
        if args.worker_index is None or args.n_workers is None:
            raise SystemExit(f"{WORKER_FLAG} requires --worker-index and "
                             "--n-workers")
        return _worker_main(args)
    if args.verify_only:
        return verify_only(args)
    return run_final_transaction(args)


if __name__ == "__main__":
    sys.exit(main())
